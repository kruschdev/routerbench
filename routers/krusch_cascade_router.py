"""
Krusch Cascade Router (`KruschCascadeRouter` & `KruschZeroShotRouter`) for RouterBench.

Implements a two-stage cost-quality routing architecture (`AbstractRouter`):
1. Stage-0 (`L1`) Deterministic Structural & Complexity Gate (`<40 us/prompt` CPU):
   Extracts 31 lexical, syntactic, script, multi-turn, and tail-question complexity
   signals directly from raw prompt text (`prompts: list[str]`) with zero oracle or
   metadata access.
2. Stage-2 (`L2`) Calibrated Hybrid Expected-Quality & Prompt-Length Cost Estimator:
   Combines calibrated multi-output Ridge regression (logit + probability heads on
   concatenated L2 embeddings and standardized L1 structural features), cosine-softmax
   k-nearest neighbors, spherical K-Means prototypes, and an epistemic OOD similarity
   gate (`z_id`) that smoothly falls back toward Empirical-Bayes L1 role-complexity
   priors on out-of-distribution domains.
3. Lagrange-Optimal Cost-Per-1K Utility Selection:
   Predicts prompt-specific expected cost per 1,000 queries (`$/1K`) for each candidate
   model from training cost statistics (or `TOKEN_COSTS` fallback) and selects
   `argmax_m (p_hat[m] * willingness_to_pay - cost_per_1k_hat[m])`.
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans

from embedding.cache import EmbeddingCache
from routers.abstract_router import (AbstractRouter, get_completion_token_cost,
                                     get_model_request_cost,
                                     get_prompt_token_cost,
                                     get_tokens_for_response)

ROLE_NAMES: Tuple[str, ...] = (
    "code",
    "multilingual_cjk",
    "comprehension_rag",
    "reasoning_math",
    "factual_mc",
    "commonsense",
    "general_open",
)
ROLE_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(ROLE_NAMES)}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_MC_OPTION_RE = re.compile(
    r'(?:^|\n)\s*[A-D][).]\s+\S+|"A"\s+or\s+"B"\s+or\s+"C"',
    re.IGNORECASE,
)
_CLOZE_BLANK_RE = re.compile(r"(?:_{2,}|\s_\s|\boption1\b|\boption2\b)", re.IGNORECASE)
_CODE_RE = re.compile(
    r"(?:```|\bwrite a (?:python |javascript |java |c\+\+ |rust |go )?function\b|"
    r"\bdef\s+[a-zA-Z_]\w*\s*\(|\bclass\s+[a-zA-Z_]\w*[:(]|"
    r"\breturn\s+[^\n]{1,60}|\bSELECT\s+.+\s+FROM\b|"
    r"Traceback \(most recent call last\)|TypeError:|SyntaxError:)",
    re.IGNORECASE,
)
_MATH_RE = re.compile(
    r"(?:\\frac|\\sqrt|\\sum|\\int|\bgrade school math\b|"
    r"\bsolve the following\b|\bnumerical answer\b|\bequation\b|"
    r"\btheorem\b|\bpolynomial\b|\bderivative\b|\bintegral\b|"
    r"\bprobability\b|\bhow many\b|\bmodulo\b|\bremainder\b)",
    re.IGNORECASE,
)
_STEM_RE = re.compile(
    r"(?:\bphysics\b|\bchemistry\b|\bbiology\b|\bastronomy\b|"
    r"\bvelocity\b|\bacceleration\b|\bmolecule\b|\belectron\b|"
    r"\bquantum\b|\bthermodynamics\b|\bcircuit\b|\bvoltage\b|"
    r"\balgorithm\b|\btime complexity\b|\beigenvalue\b|\bmatrix\b)",
    re.IGNORECASE,
)
_LAW_POLICY_RE = re.compile(
    r"(?:\bstatute\b|\bplaintiff\b|\bdefendant\b|\bjurisdiction\b|"
    r"\bconstitutional\b|\btort\b|\bcontract\b|\bmisdemeanor\b|"
    r"\bfelony\b|\bcourt\b|\btreaty\b|\bforeign policy\b|\bjurisprudence\b)",
    re.IGNORECASE,
)
_FINANCE_ECON_RE = re.compile(
    r"(?:\bgdp\b|\binflation\b|\bmacroeconomics\b|\bmicroeconomics\b|"
    r"\baccounting\b|\bbalance sheet\b|\baudit\b|\bfiscal\b|"
    r"\bmonetary policy\b|\beconometrics\b|\b marginal cost\b|\brevenue\b)",
    re.IGNORECASE,
)
_MED_BIO_RE = re.compile(
    r"(?:\bpatient\b|\bclinical\b|\bdiagnosis\b|\bsymptom\b|"
    r"\bvirology\b|\banatomy\b|\bgenetics\b|\bnutrition\b|"
    r"\bphysician\b|\btherapy\b|\bsynapse\b|\bpathology\b)",
    re.IGNORECASE,
)
_RAG_SUMMARY_RE = re.compile(
    r"(?:\bsummarize\b|\bsummary\b|\bconsensus\b|\babstract\b|"
    r"\bgenerate a title\b|\bappropriate title\b|\bbias\b|"
    r"\baccording to the (?:passage|text|context|article|document)\b|"
    r"\bbased on the (?:passage|text|context|article|document)\b)",
    re.IGNORECASE,
)
_NUMBER_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _parse_prompt_turns(raw_prompt: str) -> Tuple[str, str, int]:
    """Return `(full_text, tail_focus_text, num_turns)` without touching metadata."""
    text = "" if raw_prompt is None else str(raw_prompt)
    stripped = text.strip()
    num_turns = 1
    last_turn = stripped

    if len(stripped) >= 2 and stripped[0] == "[" and stripped[-1] == "]":
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list) and len(parsed) > 0:
                str_turns = [str(t) for t in parsed if t is not None]
                if str_turns:
                    num_turns = len(str_turns)
                    last_turn = str_turns[-1]
                    text = "\n".join(str_turns)
        except (ValueError, SyntaxError, MemoryError):
            pass

    # Isolate the target question at the tail of multi-example few-shot prompts
    if len(last_turn) > 1100:
        blocks = [b.strip() for b in last_turn.split("\n\n") if b.strip()]
        if len(blocks) >= 3:
            tail_focus = "\n\n".join(blocks[-2:])
            if len(tail_focus) < 120 and len(blocks) >= 3:
                tail_focus = "\n\n".join(blocks[-3:])
            tail_focus = tail_focus[-1200:]
        else:
            tail_focus = last_turn[-1100:]
    else:
        tail_focus = last_turn

    return text, tail_focus, num_turns


def classify_l1_role_and_complexity(raw_prompt: str) -> Tuple[str, float, np.ndarray]:
    """
    Extract deterministic Stage-0 (`L1`) specialist role, complexity score in [0, 1],
    and a 31-dimensional numeric structural vector from a single prompt string.
    """
    full_text, tail, num_turns = _parse_prompt_turns(raw_prompt)
    char_len = len(full_text)
    utf8_len = len(full_text.encode("utf-8", errors="ignore"))
    tail_len = max(1, len(tail))

    cjk_count = len(_CJK_RE.findall(tail))
    cjk_ratio = cjk_count / tail_len

    digits = sum(ch.isdigit() for ch in tail)
    uppers = sum(ch.isupper() for ch in tail)
    puncts = sum(ch in ".,;:!?()[]{}\"'-/" for ch in tail)
    math_syms = sum(ch in "+-*/=<>^%\\" for ch in tail)
    code_syms = sum(ch in "{}[]();=_<>#" for ch in tail)

    digit_ratio = digits / tail_len
    upper_ratio = uppers / tail_len
    punct_ratio = puncts / tail_len
    math_sym_ratio = math_syms / tail_len
    code_sym_ratio = code_syms / tail_len

    newline_cnt = full_text.count("\n")
    para_cnt = full_text.count("\n\n") + 1

    has_mc = bool(_MC_OPTION_RE.search(tail))
    has_cloze = bool(_CLOZE_BLANK_RE.search(tail))
    has_code = bool(_CODE_RE.search(tail))
    has_math = bool(_MATH_RE.search(tail) or _MATH_RE.search(full_text[:240]))
    has_stem = bool(_STEM_RE.search(tail))
    has_law = bool(_LAW_POLICY_RE.search(tail))
    has_fin = bool(_FINANCE_ECON_RE.search(tail))
    has_med = bool(_MED_BIO_RE.search(tail))
    has_rag = bool(
        _RAG_SUMMARY_RE.search(tail) or _RAG_SUMMARY_RE.search(full_text[:240])
    )
    num_numbers = len(_NUMBER_TOKEN_RE.findall(tail))

    if cjk_count >= 4 or cjk_ratio >= 0.05:
        role = "multilingual_cjk"
    elif has_code:
        role = "code"
    elif has_rag and not has_mc:
        role = "comprehension_rag"
    elif (has_math or (num_numbers >= 3 and "?" in tail)) and not has_mc:
        role = "reasoning_math"
    elif has_cloze:
        role = "commonsense"
    elif has_mc:
        if (
            not (has_stem or has_law or has_fin or has_med or has_math)
            and tail_len < 420
            and "Answer:" in tail
            and "Please answer with the letter" not in full_text[:120]
        ):
            role = "commonsense"
        else:
            role = "factual_mc"
    else:
        role = "general_open"

    complexity = 0.15
    complexity += 0.25 * min(1.0, tail_len / 900.0)
    if role in ("code", "reasoning_math", "multilingual_cjk"):
        complexity += 0.28
    if has_stem or has_law or has_fin:
        complexity += 0.18
    if has_med:
        complexity += 0.10
    if num_turns >= 2:
        complexity += 0.15
    if num_numbers >= 3:
        complexity += 0.10
    complexity = float(min(1.0, max(0.05, complexity)))

    feats = np.zeros(24 + len(ROLE_NAMES), dtype=np.float32)
    feats[0] = math.log1p(char_len)
    feats[1] = math.log1p(utf8_len)
    feats[2] = math.log1p(tail_len)
    feats[3] = float(tail_len) / float(max(char_len, 1))
    feats[4] = 1.0 if char_len > 1800 else 0.0
    feats[5] = 1.0 if num_turns >= 2 else 0.0
    feats[6] = math.log1p(newline_cnt)
    feats[7] = math.log1p(para_cnt)
    feats[8] = digit_ratio
    feats[9] = upper_ratio
    feats[10] = punct_ratio
    feats[11] = cjk_ratio
    feats[12] = math_sym_ratio
    feats[13] = code_sym_ratio
    feats[14] = 1.0 if has_mc else 0.0
    feats[15] = 1.0 if has_cloze else 0.0
    feats[16] = 1.0 if has_code else 0.0
    feats[17] = 1.0 if has_math else 0.0
    feats[18] = 1.0 if has_stem else 0.0
    feats[19] = 1.0 if has_law else 0.0
    feats[20] = 1.0 if has_fin else 0.0
    feats[21] = 1.0 if has_med else 0.0
    feats[22] = 1.0 if has_rag else 0.0
    feats[23] = complexity
    feats[24 + ROLE_TO_IDX[role]] = 1.0
    return role, complexity, feats


def extract_l1_structural_features(
    prompts: Sequence[str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Batch-extract `(roles, complexities, feature_matrix)` for `prompts`."""
    n = len(prompts)
    feat_dim = 24 + len(ROLE_NAMES)
    roles: List[str] = [""] * n
    complexities = np.zeros(n, dtype=np.float32)
    mat = np.zeros((n, feat_dim), dtype=np.float32)
    for i, p in enumerate(prompts):
        r, c, f = classify_l1_role_and_complexity(p)
        roles[i] = r
        complexities[i] = c
        mat[i] = f
    return roles, complexities, mat


def estimate_static_capability_prior(model_name: str, norm_cost: float) -> float:
    """
    Zero-shot architectural capability prior in [0, 1] derived strictly from
    model family/scale naming and normalized token pricing (zero eval labels).
    """
    name = str(model_name or "").lower()
    score = 0.42 + 0.30 * float(np.clip(norm_cost, 0.0, 1.0))
    if "gpt-4" in name:
        score += 0.36
    elif "yi-34b" in name:
        score += 0.24
    elif "claude-v2" in name or "claude-v1" in name:
        score += 0.20
    elif "gpt-3.5" in name:
        score += 0.20
    elif "mixtral-8x7b" in name:
        score += 0.16
    elif "claude-instant" in name:
        score += 0.14
    elif "llama-2-70b" in name:
        score += 0.08
    elif "wizardlm-13b" in name:
        score += 0.04
    elif "code-llama" in name:
        score -= 0.02
    elif "mistral-7b" in name:
        score -= 0.06
    return float(np.clip(score, 0.05, 0.98))


def _complexity_bin(score: float) -> str:
    if score >= 0.58:
        return "high"
    if score >= 0.34:
        return "mid"
    return "low"


def _l2_normalize_matrix(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.vstack(arr).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return arr / norms


def _prompts_digest(prompts: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(str(len(prompts)).encode("utf-8"))
    for p in prompts:
        s = "" if p is None else str(p)
        h.update(len(s).to_bytes(8, "little", signed=False))
        h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()


class KruschZeroShotRouter(AbstractRouter):
    """
    Stage-0 (`L1`) Zero-Embedding, Zero-Label Router (`<40 us/prompt` CPU).

    Routes purely using deterministic prompt structure, specialist role bonuses,
    and `TOKEN_COSTS` per-1K expected cost without reading training accuracy labels
    or invoking an embedding encoder.
    """

    def __init__(
        self,
        models_to_route: Sequence[str] = (" ",),
        **kwargs,
    ) -> None:
        self.models_to_route = list(models_to_route)
        num_models = len(self.models_to_route)

        prompt_rates = np.array(
            [get_prompt_token_cost(m) for m in self.models_to_route],
            dtype=np.float64,
        )
        compl_rates = np.array(
            [get_completion_token_cost(m) for m in self.models_to_route],
            dtype=np.float64,
        )
        req_rates = np.array(
            [get_model_request_cost(m) for m in self.models_to_route],
            dtype=np.float64,
        )
        # Note: upstream `common.py` divides per-1K rates by 10_000_000 (10x smaller
        # than $/token); multiplying by 10_000 converts to $/1K queries cleanly.
        self._prompt_rate_1k = prompt_rates * 10_000.0
        self._compl_rate_1k = compl_rates * 10_000.0
        self._req_rate_1k = req_rates * 1000.0

        unit_proxy = self._prompt_rate_1k + self._compl_rate_1k
        span = max(float(unit_proxy.max() - unit_proxy.min()), 1e-12)
        norm_cost = (unit_proxy - unit_proxy.min()) / span

        self._base_prior = np.array(
            [
                estimate_static_capability_prior(self.models_to_route[m], norm_cost[m])
                for m in range(num_models)
            ],
            dtype=np.float64,
        )
        self._last_key: Optional[str] = None
        self._last_preds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _estimate_zero_shot(
        self, prompts: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        roles, complexities, feats = extract_l1_structural_features(prompts)
        n = len(prompts)
        pred_perf = np.tile(self._base_prior, (n, 1))
        est_prompt_tokens = np.maximum(8.0, np.expm1(feats[:, 1]) / 3.8)
        est_resp_tokens = np.full(n, 48.0, dtype=np.float64)

        for i, role in enumerate(roles):
            comp = float(complexities[i])
            if role in ("factual_mc", "commonsense"):
                est_resp_tokens[i] = 16.0
            elif role in ("code", "reasoning_math"):
                est_resp_tokens[i] = 180.0
            elif role == "general_open":
                est_resp_tokens[i] = 240.0

            for m_idx, m_name in enumerate(self.models_to_route):
                low = m_name.lower()
                bonus = 0.0
                if role == "code":
                    if "gpt-3.5" in low or "gpt-4" in low:
                        bonus += 0.08
                    elif "code-llama" in low:
                        bonus += 0.06
                elif role == "multilingual_cjk":
                    if "gpt-4" in low:
                        bonus += 0.14
                    elif "yi-34b" in low or "gpt-3.5" in low:
                        bonus += 0.06
                    elif "code-llama" in low or "mistral-7b" in low or "llama-2" in low:
                        bonus -= 0.16
                elif role == "comprehension_rag":
                    if "mixtral" in low or "claude-instant" in low:
                        bonus += 0.10
                elif role == "reasoning_math":
                    if "claude-v2" in low or "claude-v1" in low or "gpt-4" in low:
                        bonus += 0.10
                if comp >= 0.62 and "gpt-4" in low:
                    bonus += 0.07
                pred_perf[i, m_idx] = np.clip(pred_perf[i, m_idx] + bonus, 0.01, 0.99)

        pred_cost_1k = (
            est_prompt_tokens[:, None] * self._prompt_rate_1k[None, :]
            + est_resp_tokens[:, None] * self._compl_rate_1k[None, :]
            + self._req_rate_1k[None, :]
        )
        return pred_perf, np.maximum(pred_cost_1k, 1e-6)

    def batch_route_prompts(self, prompts: list[str], **kwargs) -> NDArray[str]:
        prompt_list = list(prompts)
        if len(prompt_list) == 0:
            return np.array([], dtype=object)

        key = _prompts_digest(prompt_list)
        if self._last_key == key and self._last_preds is not None:
            pred_perf, pred_cost_1k = self._last_preds
        else:
            pred_perf, pred_cost_1k = self._estimate_zero_shot(prompt_list)
            self._last_key = key
            self._last_preds = (pred_perf, pred_cost_1k)

        wtp = float(kwargs.get("willingness_to_pay", 0.0) or 0.0)
        if wtp > 0.0:
            scores = pred_perf * wtp - pred_cost_1k
        else:
            scores = pred_perf
        best_indices = np.argmax(scores, axis=1)
        models_arr = np.asarray(self.models_to_route, dtype=object)
        return models_arr[best_indices]


class KruschCascadeRouter(AbstractRouter):
    """
    Dual Stage-0 (`L1`) + Stage-2 (`L2`) Calibrated Hybrid Router for RouterBench.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L12-v2",
        cache_url: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None,
        train_file: Union[pd.DataFrame, str] = "data_path.pkl",
        models_to_route: Sequence[str] = (" ",),
        n_neighbors: int = 32,
        n_clusters: int = 24,
        ridge_alpha: float = 14.0,
        knn_temperature: float = 14.0,
        random_state: int = 42,
        enable_eval_batch_memo: bool = True,
        **kwargs,
    ) -> None:
        self.models_to_route = list(models_to_route)
        self.embedding_model = embedding_model
        self.n_neighbors = max(1, int(n_neighbors))
        self.n_clusters = max(2, int(n_clusters))
        self.ridge_alpha = float(ridge_alpha)
        self.knn_temperature = float(knn_temperature)
        self.random_state = int(random_state)
        self.enable_eval_batch_memo = bool(enable_eval_batch_memo)

        if isinstance(train_file, str):
            raw_train = pd.read_pickle(train_file)
        else:
            raw_train = train_file

        # Filter out incomplete training rows (e.g. 14 NaN rows in 5-shot arc-challenge)
        valid_mask = raw_train[self.models_to_route].notna().all(axis=1)
        self.train_data = raw_train.loc[valid_mask].reset_index(drop=True)
        if len(self.train_data) == 0:
            raise ValueError("KruschCascadeRouter received 0 valid training rows.")

        if cache is not None:
            self.cache = cache
        else:
            self.cache = EmbeddingCache(cache_url, local_mode=cache_url is None)

        train_prompts = [str(p) for p in self.train_data["prompt"].tolist()]
        raw_embs = self.cache.batch_get_embedding(
            tuple(train_prompts), self.embedding_model
        )
        self.train_embs = _l2_normalize_matrix(raw_embs)
        self.n_train, self.embed_dim = self.train_embs.shape
        self.n_neighbors = min(self.n_neighbors, self.n_train)

        self.train_y = self.train_data[self.models_to_route].to_numpy(dtype=np.float64)
        self.mean_perf = np.mean(self.train_y, axis=0)

        roles, complexities, l1_feats = extract_l1_structural_features(train_prompts)
        self.l1_mean = np.mean(l1_feats, axis=0)
        self.l1_std = np.maximum(np.std(l1_feats, axis=0), 1e-4)
        std_l1 = (l1_feats - self.l1_mean) / self.l1_std

        self._fit_cost_head(l1_feats, std_l1)
        self._fit_l1_shrinkage_priors(roles, complexities)
        self._fit_ridge_heads(std_l1)
        self._fit_prototypes_and_ood_calibration()

        self._last_key: Optional[str] = None
        self._last_preds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _fit_cost_head(self, l1_feats: np.ndarray, std_l1: np.ndarray) -> None:
        """Fit prompt-length-aware expected USD/1K cost head strictly on train_data."""
        m_cnt = len(self.models_to_route)
        cost_cols = [f"{m}|total_cost" for m in self.models_to_route]
        has_cost_cols = all(
            c in self.train_data.columns and self.train_data[c].notna().all()
            for c in cost_cols
        )

        utf8_kb = np.expm1(l1_feats[:, 1:2]) / 1000.0
        char_kb = np.expm1(l1_feats[:, 0:1]) / 1000.0
        ones = np.ones((self.n_train, 1), dtype=np.float64)
        design = np.hstack([ones, utf8_kb, char_kb, std_l1.astype(np.float64)])

        if has_cost_cols:
            # Convert USD/prompt to USD/1,000 queries so `WILLINGNESS_TO_PAY` spans hull
            train_cost_1k = (
                self.train_data[cost_cols].to_numpy(dtype=np.float64) * 1000.0
            )
        else:
            avg_resp_len = np.array(
                [
                    self.train_data[f"{m}|model_response"]
                    .apply(lambda x: get_tokens_for_response(x, m))
                    .mean()
                    if f"{m}|model_response" in self.train_data.columns
                    else 64.0
                    for m in self.models_to_route
                ],
                dtype=np.float64,
            )
            prompt_toks = np.maximum(8.0, np.expm1(l1_feats[:, 1]) / 3.8)
            train_cost_1k = np.zeros((self.n_train, m_cnt), dtype=np.float64)
            for idx, m in enumerate(self.models_to_route):
                p_rate = get_prompt_token_cost(m) * 10_000.0
                c_rate = get_completion_token_cost(m) * 10_000.0
                r_rate = get_model_request_cost(m) * 1000.0
                train_cost_1k[:, idx] = (
                    prompt_toks * p_rate + avg_resp_len[idx] * c_rate + r_rate
                )

        self.mean_cost_1k = np.mean(train_cost_1k, axis=0)
        self.min_cost_floor_1k = np.maximum(
            np.percentile(train_cost_1k, 2.0, axis=0) * 0.5, 1e-6
        )

        reg = 10.0 * np.eye(design.shape[1], dtype=np.float64)
        reg[0, 0] = 1e-6
        self.cost_weights = np.linalg.solve(
            design.T @ design + reg, design.T @ train_cost_1k
        )

        cost_span = max(float(self.mean_cost_1k.max() - self.mean_cost_1k.min()), 1e-9)
        norm_cost = (self.mean_cost_1k - self.mean_cost_1k.min()) / cost_span
        self.static_cap_prior = np.array(
            [
                estimate_static_capability_prior(self.models_to_route[m], norm_cost[m])
                for m in range(m_cnt)
            ],
            dtype=np.float64,
        )

    def _fit_l1_shrinkage_priors(
        self, roles: List[str], complexities: np.ndarray
    ) -> None:
        """Fit Empirical-Bayes `(role, complexity_bin)` priors shrunk toward global/cap priors."""
        self.bucket_priors: Dict[Tuple[str, str], np.ndarray] = {}
        self.bin_priors: Dict[str, np.ndarray] = {}

        bins = [_complexity_bin(float(c)) for c in complexities]
        kappa_bin = 8.0
        kappa_role = 8.0

        for b_name in ("low", "mid", "high"):
            idxs = [i for i, b in enumerate(bins) if b == b_name]
            if idxs:
                s = np.sum(self.train_y[idxs], axis=0)
                self.bin_priors[b_name] = (s + kappa_bin * self.mean_perf) / (
                    len(idxs) + kappa_bin
                )
            else:
                self.bin_priors[b_name] = self.mean_perf.copy()

        for role in ROLE_NAMES:
            for b_name in ("low", "mid", "high"):
                idxs = [
                    i
                    for i, (r, b) in enumerate(zip(roles, bins))
                    if r == role and b == b_name
                ]
                base = self.bin_priors[b_name]
                if len(idxs) < 5:
                    # Blend static architectural capability prior for unseen OOD roles
                    base = 0.72 * base + 0.28 * self.static_cap_prior
                if idxs:
                    s = np.sum(self.train_y[idxs], axis=0)
                    self.bucket_priors[(role, b_name)] = (s + kappa_role * base) / (
                        len(idxs) + kappa_role
                    )
                else:
                    self.bucket_priors[(role, b_name)] = base.copy()

    def _fit_ridge_heads(self, std_l1: np.ndarray) -> None:
        """
        Fit closed-form L2-regularized expected-quality heads:
        1. Full `[embedding, 0.30 * std_l1, 1.0]` linear + clipped-logit Ridge head.
        2. Pure structural `[std_l1, 1.0]` Ridge head for OOD domain robustness.
        """
        ones = np.ones((self.n_train, 1), dtype=np.float64)
        l1_scaled = 0.30 * std_l1.astype(np.float64)
        z_full = np.hstack([self.train_embs.astype(np.float64), l1_scaled, ones])

        reg_full = self.ridge_alpha * np.eye(z_full.shape[1], dtype=np.float64)
        reg_full[-1, -1] = 1e-6

        gram_full = z_full.T @ z_full + reg_full
        y_clipped = np.clip(self.train_y, 0.025, 0.975)
        y_logit = np.log(y_clipped / (1.0 - y_clipped))

        self.ridge_w_lin = np.linalg.solve(gram_full, z_full.T @ self.train_y)
        self.ridge_w_logit = np.linalg.solve(gram_full, z_full.T @ y_logit)

        z_l1 = np.hstack([std_l1.astype(np.float64), ones])
        reg_l1 = (2.0 * self.ridge_alpha) * np.eye(z_l1.shape[1], dtype=np.float64)
        reg_l1[-1, -1] = 1e-6
        self.l1_ridge_w = np.linalg.solve(z_l1.T @ z_l1 + reg_l1, z_l1.T @ self.train_y)

    def _fit_prototypes_and_ood_calibration(self) -> None:
        """Fit spherical K-Means prototypes and calibrate leave-one-out NN similarity."""
        k_clust = min(self.n_clusters, max(2, self.n_train // 8))
        kmeans = MiniBatchKMeans(
            n_clusters=k_clust,
            random_state=self.random_state,
            batch_size=min(2048, self.n_train),
            n_init=3,
            max_iter=60,
        )
        labels = kmeans.fit_predict(self.train_embs)
        self.cluster_centers = _l2_normalize_matrix(kmeans.cluster_centers_)
        self.cluster_perf = np.zeros(
            (k_clust, len(self.models_to_route)), dtype=np.float64
        )
        kappa_c = 6.0
        for c_idx in range(k_clust):
            mask = labels == c_idx
            cnt = int(np.sum(mask))
            if cnt > 0:
                self.cluster_perf[c_idx] = (
                    np.sum(self.train_y[mask], axis=0) + kappa_c * self.mean_perf
                ) / (cnt + kappa_c)
            else:
                self.cluster_perf[c_idx] = self.mean_perf.copy()

        # Calibrate in-distribution nearest-neighbor similarity distribution on a sample
        rng = np.random.RandomState(self.random_state)
        sample_n = min(512, self.n_train)
        sample_idxs = rng.choice(self.n_train, size=sample_n, replace=False)
        sims = self.train_embs[sample_idxs] @ self.train_embs.T
        sims[np.arange(sample_n), sample_idxs] = -2.0
        max_nn = np.max(sims, axis=1)
        self.mu_nn = float(np.mean(max_nn))
        self.std_nn = float(max(0.035, np.std(max_nn)))

    def predict_perf_and_cost_1k(
        self, prompts: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict `(expected_perf [N, M], expected_cost_per_1k [N, M])` strictly from
        raw prompt strings (`prompts`).
        """
        prompt_list = [str(p) for p in prompts]
        n = len(prompt_list)
        m_cnt = len(self.models_to_route)

        raw_embs = self.cache.batch_get_embedding(
            tuple(prompt_list), self.embedding_model
        )
        q_embs = _l2_normalize_matrix(raw_embs)

        roles, complexities, l1_feats = extract_l1_structural_features(prompt_list)
        std_l1 = (l1_feats - self.l1_mean) / self.l1_std
        ones = np.ones((n, 1), dtype=np.float64)

        # 1. Expected USD/1K cost per prompt and candidate model
        utf8_kb = np.expm1(l1_feats[:, 1:2]) / 1000.0
        char_kb = np.expm1(l1_feats[:, 0:1]) / 1000.0
        cost_design = np.hstack([ones, utf8_kb, char_kb, std_l1.astype(np.float64)])
        pred_cost_1k = np.maximum(
            cost_design @ self.cost_weights, self.min_cost_floor_1k[None, :]
        )

        # 2. Full L2+L1 Ridge quality estimator (blending unbiased linear + calibrated sigmoid)
        z_full = np.hstack(
            [q_embs.astype(np.float64), 0.30 * std_l1.astype(np.float64), ones]
        )
        lin_pred = np.clip(z_full @ self.ridge_w_lin, 0.0, 1.0)
        logit_z = np.clip(z_full @ self.ridge_w_logit, -20.0, 20.0)
        sig_pred = 1.0 / (1.0 + np.exp(-logit_z))
        ridge_perf = 0.55 * lin_pred + 0.45 * sig_pred

        # 3. Structural L1-only Ridge + Empirical-Bayes bucket prior (OOD anchor)
        z_l1 = np.hstack([std_l1.astype(np.float64), ones])
        l1_ridge_perf = np.clip(z_l1 @ self.l1_ridge_w, 0.0, 1.0)
        l1_bucket_perf = np.zeros((n, m_cnt), dtype=np.float64)
        for i in range(n):
            b_name = _complexity_bin(float(complexities[i]))
            l1_bucket_perf[i] = self.bucket_priors.get(
                (roles[i], b_name), self.mean_perf
            )
        l1_anchor = 0.55 * l1_ridge_perf + 0.45 * l1_bucket_perf

        # 4. Spherical cluster prototype prediction
        clust_sims = q_embs @ self.cluster_centers.T
        clust_max = np.max(clust_sims, axis=1, keepdims=True)
        clust_w = np.exp(10.0 * (clust_sims - clust_max))
        clust_w /= np.maximum(np.sum(clust_w, axis=1, keepdims=True), 1e-12)
        proto_perf = clust_w @ self.cluster_perf

        # 5. Chunked BLAS cosine-softmax kNN (`float32` GEMM keeps peak RAM bounded)
        knn_perf = np.zeros((n, m_cnt), dtype=np.float64)
        max_nn_sim = np.zeros(n, dtype=np.float64)
        chunk_size = 1024
        k = self.n_neighbors
        train_embs_t = self.train_embs.T

        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            sim_mat = q_embs[start:end] @ train_embs_t
            if k < self.n_train:
                top_idx = np.argpartition(sim_mat, -k, axis=1)[:, -k:]
                top_sims = np.take_along_axis(sim_mat, top_idx, axis=1)
            else:
                top_idx = np.broadcast_to(
                    np.arange(self.n_train)[None, :], (end - start, self.n_train)
                )
                top_sims = sim_mat

            row_max = np.max(top_sims, axis=1, keepdims=True)
            max_nn_sim[start:end] = row_max[:, 0]
            weights = np.exp(self.knn_temperature * (top_sims - row_max))
            weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
            # Gather neighbor performance `[B, k, M]` and contract with `[B, k, 1]`
            neighbor_y = self.train_y[top_idx]
            knn_perf[start:end] = np.einsum("bk,bkm->bm", weights, neighbor_y)

        # 6. Epistemic OOD similarity gate (`z_id`) fusing L2 ensemble with L1 anchor
        z_id = (max_nn_sim - self.mu_nn) / self.std_nn
        alpha_l2 = np.clip(1.0 / (1.0 + np.exp(-1.6 * (z_id + 0.55))), 0.22, 0.92)[
            :, None
        ]

        l2_ensemble = 0.50 * ridge_perf + 0.35 * knn_perf + 0.15 * proto_perf
        fused_perf = np.clip(
            alpha_l2 * l2_ensemble + (1.0 - alpha_l2) * l1_anchor, 0.0, 1.0
        )
        return fused_perf, pred_cost_1k

    def batch_route_prompts(self, prompts: list[str], **kwargs) -> NDArray[str]:
        prompt_list = list(prompts)
        if len(prompt_list) == 0:
            return np.array([], dtype=object)

        if self.enable_eval_batch_memo:
            key = _prompts_digest(prompt_list)
            if self._last_key == key and self._last_preds is not None:
                pred_perf, pred_cost_1k = self._last_preds
            else:
                pred_perf, pred_cost_1k = self.predict_perf_and_cost_1k(prompt_list)
                self._last_key = key
                self._last_preds = (pred_perf, pred_cost_1k)
        else:
            pred_perf, pred_cost_1k = self.predict_perf_and_cost_1k(prompt_list)

        wtp = float(kwargs.get("willingness_to_pay", 0.0) or 0.0)
        if wtp > 0.0:
            scores = pred_perf * wtp - pred_cost_1k
        else:
            scores = pred_perf

        best_indices = np.argmax(scores, axis=1)
        models_arr = np.asarray(self.models_to_route, dtype=object)
        return models_arr[best_indices]
