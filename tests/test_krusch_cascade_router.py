import hashlib
import json

import numpy as np
import pandas as pd

from embedding.cache import EmbeddingCache
from evaluation.AIQ import calc_AIQ, get_non_descreasing_convex_hull_of
from routers.krusch_cascade_router import (KruschCascadeRouter,
                                           KruschZeroShotRouter,
                                           classify_l1_role_and_complexity,
                                           extract_l1_structural_features)
from utils import WILLINGNESS_TO_PAY, parse_model_parameters


class _DeterministicMockCache:
    """Fast deterministic unit-vector embedding cache for unit tests."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def batch_get_embedding(self, prompts: tuple, embedding_model: str) -> np.ndarray:
        out = np.zeros((len(prompts), self.dim), dtype=np.float32)
        for i, p in enumerate(prompts):
            digest = hashlib.sha256(f"{embedding_model}::{p}".encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "little")
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            if "function" in p.lower() or "def " in p:
                vec[0] += 3.5
            elif "how many" in p.lower() or "solve" in p.lower():
                vec[1] += 3.5
            elif any("\u4e00" <= ch <= "\u9fff" for ch in p):
                vec[2] += 3.5
            else:
                vec[3] += 2.0
            vec /= np.linalg.norm(vec)
            out[i] = vec
        return out


def _make_synthetic_routerbench_df(
    n: int = 96, include_nans: bool = False
) -> pd.DataFrame:
    models = [
        "mistralai/mistral-7b-chat",
        "zero-one-ai/Yi-34B-Chat",
        "gpt-3.5-turbo-1106",
        "gpt-4-1106-preview",
    ]
    rows = []
    for i in range(n):
        bucket = i % 4
        if bucket == 0:
            prompt = f"['Write a python function to sort list {i} in ascending order.']"
            scores = [0.20, 0.50, 0.85, 0.90]
        elif bucket == 1:
            prompt = (
                f"['TASK: Solve the following grade school math problem. "
                f"Alice has {i + 12} apples and buys {2 * i + 3} more. How many apples?']"
            )
            scores = [0.10, 0.45, 0.55, 0.95]
        elif bucket == 2:
            prompt = f"['猜字谜：第{i}题，请根据汉字偏旁猜一个字。']"
            scores = [0.05, 0.80, 0.40, 0.88]
        else:
            prompt = (
                f"['Question {i}: Which planet is closest to the Sun?\\n"
                f"A) Mercury\\nB) Venus\\nC) Earth\\nD) Mars\\nAnswer:']"
            )
            scores = [0.75, 0.85, 0.82, 0.88]

        base_costs = [0.00004, 0.00018, 0.00024, 0.00330]
        row = {
            "sample_id": f"syn.{i}",
            "prompt": prompt,
            "eval_name": "synthetic",
            "oracle_model_to_route_to": "gpt-4-1106-preview",
        }
        for m, s, c in zip(models, scores, base_costs):
            row[m] = np.nan if (include_nans and i < 3) else float(s)
            row[f"{m}|total_cost"] = float(c * (1.0 + 0.001 * len(prompt)))
            row[f"{m}|model_response"] = "['A']"
        rows.append(row)
    return pd.DataFrame(rows)


def test_l1_role_and_tail_extraction():
    few_shot_mc = (
        "['Please answer with the letter.\\n\\n"
        + ("Example question?\\nA) 1\\nB) 2\\nC) 3\\nD) 4\\nAnswer: A\\n\\n" * 8)
        + "Under constitutional tort statute, is the plaintiff entitled to relief?\\n"
        + "A) Yes\\nB) No\\nC) Partial\\nD) Dismissed\\nAnswer:']"
    )
    role_mc, comp_mc, feats_mc = classify_l1_role_and_complexity(few_shot_mc)
    assert role_mc == "factual_mc"
    assert 0.05 <= comp_mc <= 1.0
    assert feats_mc.shape == (31,)
    assert feats_mc[19] == 1.0  # law/policy marker preserved from tail

    role_code, _, _ = classify_l1_role_and_complexity(
        "['Write a function to compute binary search on a sorted array.']"
    )
    assert role_code == "code"

    role_cjk, _, _ = classify_l1_role_and_complexity("['请解释唐诗三百首中的意象与平仄格律。']")
    assert role_cjk == "multilingual_cjk"

    roles, comps, mat = extract_l1_structural_features([few_shot_mc, "['hello']"])
    assert len(roles) == 2
    assert comps.shape == (2,)
    assert mat.shape == (2, 31)


def test_krusch_zero_shot_router_monotonic_cost():
    df = _make_synthetic_routerbench_df(32)
    models = [
        "mistralai/mistral-7b-chat",
        "zero-one-ai/Yi-34B-Chat",
        "gpt-3.5-turbo-1106",
        "gpt-4-1106-preview",
    ]
    router = KruschZeroShotRouter(models_to_route=models)
    prompts = df["prompt"].tolist()

    low_wtp_choices = router.batch_route_prompts(prompts, willingness_to_pay=1e-10)
    high_wtp_choices = router.batch_route_prompts(prompts, willingness_to_pay=1e9)

    assert len(low_wtp_choices) == len(prompts)
    assert set(low_wtp_choices).issubset(set(models))
    assert set(high_wtp_choices).issubset(set(models))

    low_cost = sum(df.loc[i, f"{m}|total_cost"] for i, m in enumerate(low_wtp_choices))
    high_cost = sum(
        df.loc[i, f"{m}|total_cost"] for i, m in enumerate(high_wtp_choices)
    )
    assert low_cost < high_cost


def test_krusch_cascade_router_nan_resilience_and_blind_prompts():
    df = _make_synthetic_routerbench_df(96, include_nans=True)
    models = [
        "mistralai/mistral-7b-chat",
        "zero-one-ai/Yi-34B-Chat",
        "gpt-3.5-turbo-1106",
        "gpt-4-1106-preview",
    ]
    train_df = df.iloc[:72].copy()
    eval_df = df.iloc[72:].copy()

    cache = _DeterministicMockCache(dim=32)
    router = KruschCascadeRouter(
        embedding_model="mock-embed:latest",
        cache=cache,
        train_file=train_df,
        models_to_route=models,
        n_neighbors=12,
        n_clusters=6,
        ridge_alpha=8.0,
    )
    # 3 NaN rows must be filtered cleanly during training
    assert len(router.train_data) == 69

    # Blind prompt list only (all ground-truth score/cost columns absent)
    blind_prompts = [str(p) for p in eval_df["prompt"].tolist()]
    costs_across_wtp = []
    perfs_across_wtp = []

    for wtp in [WILLINGNESS_TO_PAY[0], 1.0, 20.0, WILLINGNESS_TO_PAY[-1]]:
        routed = router.batch_route_prompts(blind_prompts, willingness_to_pay=wtp)
        assert routed.shape == (len(blind_prompts),)
        assert set(routed).issubset(set(models))
        c = float(
            np.sum([eval_df.iloc[i][f"{m}|total_cost"] for i, m in enumerate(routed)])
        )
        p = float(np.mean([eval_df.iloc[i][m] for i, m in enumerate(routed)]))
        costs_across_wtp.append(c)
        perfs_across_wtp.append(p)

    assert costs_across_wtp[0] < costs_across_wtp[-1]
    assert perfs_across_wtp[-1] >= perfs_across_wtp[0]

    # Verify calc_AIQ works cleanly without plotting
    res_df = pd.DataFrame(
        {
            "eval_name": ["synthetic"] * len(costs_across_wtp),
            "total_cost": costs_across_wtp,
            "performance": perfs_across_wtp,
        }
    )
    hull = get_non_descreasing_convex_hull_of(
        res_df[["total_cost", "performance"]].to_numpy()
    )
    assert hull.shape[1] == 2
    aiq_scores = calc_AIQ([res_df], plot_graph=False)
    assert len(aiq_scores) == 1
    assert 0.0 < aiq_scores[0] <= 1.0


def test_compact_npy_cache_and_colon_param_parsing(tmp_path):
    prompts = ("alpha prompt", "beta prompt")
    vecs = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    vec_file = tmp_path / "vecs.f32.npy"
    np.save(vec_file, vecs)
    idx_file = tmp_path / "index.json"
    idx_file.write_text(
        json.dumps(
            {
                "vectors_file": vec_file.name,
                "hash_to_row": {
                    hashlib.sha256(p.encode("utf-8")).hexdigest(): i
                    for i, p in enumerate(prompts)
                },
            }
        ),
        encoding="utf-8",
    )

    cache = EmbeddingCache(local_cache_path=str(idx_file), local_mode=True)
    loaded = cache.batch_get_embedding(prompts, "ollama/snowflake-arctic-embed2:latest")
    assert loaded.shape == (2, 4)
    assert np.allclose(loaded, vecs)

    parsed = parse_model_parameters(
        "krusch_cascade|neighbors:32|ridge_alpha:14.0"
        "|embedding:ollama/snowflake-arctic-embed2:latest"
        "|willingness_to_pay:10.0|performance"
    )
    assert parsed["routing_algorithm"] == "krusch_cascade"
    assert parsed["embedding"] == "ollama/snowflake-arctic-embed2:latest"
    assert parsed["value_type"] == "performance"


def test_evaluate_routers_local_pipeline_integration():
    from evaluate_routers import get_results_for_all_evals
    from evaluate_utils import (generate_krusch_routers,
                                generate_krusch_zero_shot_routers)
    from evaluation.utils import (model_metric_dict,
                                  transform_batch_df_to_wide_format)

    df = _make_synthetic_routerbench_df(64)
    # Include a row with `no_model_correct` to exercise oracle lookup safety
    df.loc[df.index[-1], "oracle_model_to_route_to"] = "no_model_correct"
    models = [
        "mistralai/mistral-7b-chat",
        "zero-one-ai/Yi-34B-Chat",
        "gpt-3.5-turbo-1106",
        "gpt-4-1106-preview",
    ]
    train_df = df.iloc[:44].copy()
    eval_df = df.iloc[44:].copy()

    cache = _DeterministicMockCache(dim=32)
    routers = generate_krusch_routers(
        [("ollama/snowflake-arctic-embed2:latest", 8, 14.0)],
        cache,
        train_df,
        0.7,
        models,
    )
    routers += generate_krusch_zero_shot_routers(0.7, models)

    agg_df = get_results_for_all_evals(
        eval_df,
        model_routers_and_names=routers,
        use_local=True,
    )
    assert "krusch_cascade" in set(agg_df["model_name"])
    assert "krusch_zero_shot" in set(agg_df["model_name"])
    assert "oracle" in set(agg_df["model_name"])

    subset_dict = {
        k: model_metric_dict[k] for k in ("krusch_cascade", "krusch_zero_shot")
    }
    wide_df = transform_batch_df_to_wide_format(agg_df, subset_dict)
    assert len(wide_df) == 1
    assert any(col.startswith("krusch_cascade|") for col in wide_df.columns)
    assert any(col.startswith("krusch_zero_shot|") for col in wide_df.columns)
