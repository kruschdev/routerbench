import hashlib
import json
import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import fsspec
import numpy as np
import pymongo
import requests
from tqdm import tqdm

_log = logging.getLogger(__name__)


def save_cache_to_disk(results, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with fsspec.open(path, "wb") as f:
        # save results as pickle
        f.write(pickle.dumps(results))
    return None


def _format_ollama_input(prompt: str, model_name: str, max_chars: int = 4000) -> str:
    raw = "" if prompt is None else str(prompt)
    if len(raw) > max_chars:
        raw = f"{raw[:1200]}\n...\n{raw[-2800:]}"
    if "arctic-embed2" in model_name.lower():
        return f"query: {raw}"
    return raw


class EmbeddingCache:
    def __init__(
        self,
        connection_string: str = None,
        local_cache_path: Optional[str] = None,
        local_mode: bool = False,
    ):
        self.local_mode = local_mode
        if not local_mode:
            self.client = pymongo.MongoClient(connection_string)
            self.db = self.client["embedding_cache"]
        self.local_cache = []
        self._local_hash_map = {}
        self._npy_vectors = None
        self._prompt_sha_to_row = None
        self.local_cache_path = local_cache_path
        if local_cache_path is not None and local_cache_path.endswith(".json"):
            self._load_compact_npy_cache(local_cache_path)

    def _load_compact_npy_cache(self, json_index_path: str) -> None:
        idx_path = Path(json_index_path)
        if not idx_path.exists():
            return
        meta = json.loads(idx_path.read_text(encoding="utf-8"))
        vec_path = idx_path.parent / meta["vectors_file"]
        if not vec_path.exists():
            return
        self._npy_vectors = np.load(vec_path, mmap_mode="r")
        self._prompt_sha_to_row = meta["hash_to_row"]

    def _index_local_cache_items(self, items: list) -> None:
        for item in items:
            self._local_hash_map[item["hash"]] = item["embedding"]

    def batch_get_embedding(self, prompts, embedding_model: str) -> np.ndarray:
        """
        Get the embeddings for a list or tuple of prompts.

        :param prompts: Prompts to get the embeddings for
        :param embedding_model: Embedding model name to use
        :return: Embeddings for the prompts
        """
        prompts_tuple = prompts if isinstance(prompts, tuple) else tuple(prompts)
        return self._batch_get_embedding_cached(prompts_tuple, embedding_model)

    @lru_cache(maxsize=1000)
    def _batch_get_embedding_cached(
        self, prompts: tuple, embedding_model: str
    ) -> np.ndarray:
        print(f"Getting embeddings for {len(prompts)} prompts")
        if self._npy_vectors is not None and self._prompt_sha_to_row is not None:
            p_hashes = [
                hashlib.sha256(str(p).encode("utf-8")).hexdigest() for p in prompts
            ]
            if all(ph in self._prompt_sha_to_row for ph in p_hashes):
                rows = [self._prompt_sha_to_row[ph] for ph in p_hashes]
                return np.asarray(self._npy_vectors[rows], dtype=np.float32)

        hashes = [self._compute_hash(prompt, embedding_model) for prompt in prompts]
        if self._local_hash_map and all(h in self._local_hash_map for h in hashes):
            embeddings = np.asarray([self._local_hash_map[h] for h in hashes])
            print(f"Got {len(embeddings)} embeddings from the local cache index")
            return embeddings

        if self.local_cache_path is not None and os.path.exists(self.local_cache_path):
            if not self.local_cache_path.endswith(".json"):
                with fsspec.open(self.local_cache_path, "rb") as f:
                    self.local_cache = pickle.loads(f.read())
                self._index_local_cache_items(self.local_cache)
                if all(h in self._local_hash_map for h in hashes):
                    embeddings = np.asarray([self._local_hash_map[h] for h in hashes])
                    print(
                        f"Got {len(embeddings)} embeddings from the local cache file "
                        f"{self.local_cache_path}"
                    )
                    return embeddings

        safe_model_tag = embedding_model.replace("/", "__").replace(":", "_")
        cache_candidates = [
            f"data/embedding_cache_{embedding_model}.pkl",
            f"data/embedding_cache_{safe_model_tag}.pkl",
        ]
        selected_disk_path = cache_candidates[-1]
        for disk_path in cache_candidates:
            if os.path.exists(disk_path):
                selected_disk_path = disk_path
                with fsspec.open(disk_path, "rb") as f:
                    self.local_cache = pickle.loads(f.read())
                self._index_local_cache_items(self.local_cache)
                if all(h in self._local_hash_map for h in hashes):
                    embeddings = np.asarray([self._local_hash_map[h] for h in hashes])
                    print(f"Got {len(embeddings)} embeddings from the local cache")
                    return embeddings

        if self.local_mode:
            unique_prompts = list(dict.fromkeys(prompts))
            missing_prompts = [
                p
                for p in unique_prompts
                if self._compute_hash(p, embedding_model) not in self._local_hash_map
            ]
            new_results = []
            if missing_prompts:
                computed = self._compute_embedding(missing_prompts, embedding_model)
                for i, prompt in enumerate(missing_prompts):
                    h = self._compute_hash(prompt, embedding_model)
                    item = {
                        "prompt": prompt,
                        "embedding": computed[i],
                        "hash": h,
                        "embedding_model": embedding_model,
                    }
                    new_results.append(item)
                    self._local_hash_map[h] = computed[i]
            self.local_cache += new_results
            save_cache_to_disk(self.local_cache, selected_disk_path)
            return np.asarray([self._local_hash_map[h] for h in hashes])

        # MongoDB cache branch
        print(f"Checking if {len(hashes)} prompts are in the cache")
        results = []
        search_hashes = list(set(hashes))
        for i in tqdm(range(0, len(search_hashes), 10000)):
            results += list(
                self.db.embedding_cache.find(
                    {"hash": {"$in": search_hashes[i : i + 10000]}}
                )
            )
        _log.info(
            f"Got {len(results)} cached embeddings for {len(prompts)} prompts from the cache"
        )
        if len(results) != len(search_hashes):
            self.batch_add_embedding(prompts, embedding_model)
            results = []
            for i in tqdm(range(0, len(search_hashes), 10000)):
                results += list(
                    self.db.embedding_cache.find(
                        {"hash": {"$in": search_hashes[i : i + 10000]}}
                    )
                )
        self._index_local_cache_items(results)
        if os.path.exists(selected_disk_path):
            with fsspec.open(selected_disk_path, "rb") as f:
                self.local_cache = pickle.loads(f.read())
            self.local_cache += results
            save_cache_to_disk(self.local_cache, selected_disk_path)
        else:
            save_cache_to_disk(results, selected_disk_path)

        return np.asarray([self._local_hash_map[h] for h in hashes])

    def batch_add_embedding(self, prompts: tuple, embedding_model: str) -> None:
        """
        Add the embeddings for a list of prompts to the cache

        :param prompts: Prompts to add the embeddings for, as a tuple of strings
        :param embedding_model: Embedding model to use
        """
        prompts = list(set(prompts))
        hashes = [self._compute_hash(prompt, embedding_model) for prompt in prompts]
        results = []
        for i in tqdm(range(0, len(hashes), 10000)):
            results += list(
                self.db.embedding_cache.find({"hash": {"$in": hashes[i : i + 10000]}})
            )
        cached_hashes = {result["hash"] for result in results}
        prompts = [
            prompt
            for prompt in prompts
            if self._compute_hash(prompt, embedding_model) not in cached_hashes
        ]
        if len(prompts) == 0:
            return
        _log.info(
            f"Adding embeddings for {len(prompts)} prompts to the cache, after filtering"
        )
        embeddings = self._compute_embedding(prompts, embedding_model)
        hashes = [self._compute_hash(prompt, embedding_model) for prompt in prompts]
        items_to_insert = []
        for i, prompt in tqdm(enumerate(prompts)):
            items_to_insert.append(
                {
                    "prompt": prompt,
                    "embedding": embeddings[i],
                    "hash": hashes[i],
                    "embedding_model": embedding_model,
                }
            )
            if len(items_to_insert) == 1000:
                self.db.embedding_cache.insert_many(items_to_insert)
                items_to_insert = []
        if len(items_to_insert) > 0:
            self.db.embedding_cache.insert_many(items_to_insert)
        _log.info(f"Added embeddings for {len(items_to_insert)} prompts to the cache")
        return None

    def _compute_embedding(self, prompts, embedding_model) -> list[list[float]]:
        if embedding_model.startswith("ollama/"):
            ollama_model = embedding_model.split("ollama/", 1)[1]
            return self._compute_ollama_embedding(list(prompts), ollama_model)
        return self._compute_sentence_transformers_embedding(
            list(prompts), embedding_model
        )

    @staticmethod
    def _compute_hash(prompt: str, embedding_model: str) -> str:
        return hashlib.sha256((prompt + embedding_model).encode()).hexdigest()

    @staticmethod
    def _compute_ollama_embedding(
        prompts: list[str], ollama_model: str, batch_size: int = 128
    ) -> list[list[float]]:
        base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        all_embeddings: list[list[float]] = []
        session = requests.Session()
        for start in range(0, len(prompts), batch_size):
            batch = [
                _format_ollama_input(p, ollama_model)
                for p in prompts[start : start + batch_size]
            ]
            resp = session.post(
                f"{base_url}/api/embed",
                json={"model": ollama_model, "input": batch},
                timeout=180,
            )
            resp.raise_for_status()
            vecs = np.asarray(resp.json()["embeddings"], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms > 1e-12, norms, 1.0)
            all_embeddings.extend((vecs / norms).tolist())
        return all_embeddings

    @staticmethod
    def _compute_sentence_transformers_embedding(
        prompts: list[str], embedding_model: str
    ) -> list[list[float]]:
        import sentence_transformers

        model = sentence_transformers.SentenceTransformer(embedding_model)
        return model.encode(prompts).tolist()
