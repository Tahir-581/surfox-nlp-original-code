import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from .nlp_serving_urls import (
    BGE_API_STYLE,
    BGE_ENCODE_URL,
    BGE_MODEL_ID,
    TRITON_BGE_INFER_URL,
    USE_FASTAPI_MODEL_WRAPPERS,
    USE_TRITON,
)


class BGEServiceError(RuntimeError):
    """Raised when the BGE embedding service is unavailable or returns invalid data."""


class BGETritonClient:
    """HTTP client for BGE embeddings (OpenAI, legacy FastAPI, or Triton)."""

    _session: Optional[requests.Session] = None
    _pool_size: Optional[int] = None
    _embedding_dim: Optional[int] = None

    @classmethod
    def _get_pool_size(cls) -> int:
        return max(1, int(os.getenv("BGE_REQUEST_CONCURRENCY", "64")))

    @classmethod
    def _get_session(cls) -> requests.Session:
        pool_size = cls._get_pool_size()
        if cls._session is None or cls._pool_size != pool_size:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            cls._session = session
            cls._pool_size = pool_size
        return cls._session

    @classmethod
    def _embedding_dim_or_default(cls, dim: int) -> int:
        if dim > 0:
            cls._embedding_dim = dim
        return cls._embedding_dim or 1024

    @classmethod
    def _empty_matrix(cls, rows: int) -> np.ndarray:
        cols = cls._embedding_dim or 1024
        return np.empty((rows, cols), dtype=np.float32)

    @staticmethod
    def _raise_service_error(exc: Exception, url: str, response_text: str = "") -> None:
        detail = f"BGE service request failed at {url}: {exc}"
        if response_text:
            detail = f"{detail} | response={response_text[:500]}"
        raise BGEServiceError(detail) from exc

    def _post_json(self, url: str, payload: dict, timeout: int) -> dict:
        try:
            r = self._get_session().post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            response_text = ""
            if exc.response is not None:
                response_text = exc.response.text or ""
            self._raise_service_error(exc, url, response_text)
        except requests.RequestException as exc:
            self._raise_service_error(exc, url)
        except ValueError as exc:
            self._raise_service_error(exc, url, "invalid JSON response")

    @classmethod
    def _parse_openai_embeddings(cls, data: dict, expected_rows: int) -> np.ndarray:
        items = data.get("data", [])
        if not isinstance(items, list) or len(items) != expected_rows:
            raise BGEServiceError("BGE OpenAI response missing or mismatched data[]")
        rows = []
        for item in sorted(items, key=lambda x: x.get("index", 0)):
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise BGEServiceError("BGE OpenAI response missing embedding vector")
            rows.append(emb)
        out = np.array(rows, dtype=np.float32)
        cls._embedding_dim_or_default(int(out.shape[1]) if out.ndim == 2 else 0)
        return out

    @staticmethod
    def _parse_legacy_embeddings(data: dict, expected_rows: int) -> np.ndarray:
        if isinstance(data, dict):
            for key in ("embeddings", "vectors", "results", "result"):
                value = data.get(key)
                if isinstance(value, list):
                    return np.array(value, dtype=np.float32)

        raise BGEServiceError("BGE legacy response missing embeddings")

    @staticmethod
    def _parse_triton_embeddings(data: dict, expected_rows: int) -> np.ndarray:
        output = None
        for out in data.get("outputs", []):
            if out.get("name") == "embeddings":
                output = out
                break
        if output is None:
            raise BGEServiceError("BGE Triton response missing embeddings output")
        shape = output.get("shape", [])
        flat = output.get("data", [])
        rows = int(shape[0]) if shape else expected_rows
        cols = int(shape[1]) if len(shape) > 1 else 1024
        return np.array(flat, dtype=np.float32).reshape(rows, cols)

    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        payload = {"model": BGE_MODEL_ID, "input": list(texts)}
        return self._parse_openai_embeddings(
            self._post_json(BGE_ENCODE_URL, payload, timeout=300),
            expected_rows=len(texts),
        )

    def _encode_legacy(self, texts: List[str], is_query: bool) -> np.ndarray:
        payload = {"texts": list(texts), "is_query": bool(is_query)}
        return self._parse_legacy_embeddings(
            self._post_json(BGE_ENCODE_URL, payload, timeout=300),
            expected_rows=len(texts),
        )

    def _encode_triton_chunk(self, chunk: List[str], is_query: bool) -> np.ndarray:
        payload = {
            "inputs": [
                {
                    "name": "text",
                    "datatype": "BYTES",
                    "shape": [len(chunk), 1],
                    "data": chunk,
                },
                {
                    "name": "is_query",
                    "datatype": "BOOL",
                    "shape": [len(chunk), 1],
                    "data": [bool(is_query)] * len(chunk),
                },
            ],
            "outputs": [{"name": "embeddings"}],
        }
        return self._parse_triton_embeddings(
            self._post_json(TRITON_BGE_INFER_URL, payload, timeout=300),
            expected_rows=len(chunk),
        )

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return self._empty_matrix(0)

        if USE_TRITON:
            max_batch = max(1, int(os.getenv("BGE_MAX_BATCH_SIZE", "64")))
            parallel_requests = max(1, int(os.getenv("BGE_PARALLEL_INFER_REQUESTS", "8")))
            chunks = [texts[i : i + max_batch] for i in range(0, len(texts), max_batch)]
            chunk_rows: List[np.ndarray] = [self._empty_matrix(0) for _ in chunks]

            def _encode_chunk(idx: int, chunk: List[str]) -> None:
                chunk_rows[idx] = self._encode_triton_chunk(chunk, is_query)
                self._embedding_dim_or_default(int(chunk_rows[idx].shape[1]))

            workers = min(parallel_requests, len(chunks) or 1)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_encode_chunk, i, c) for i, c in enumerate(chunks)]
                for fut in futures:
                    fut.result()
            return np.vstack(chunk_rows) if chunk_rows else self._empty_matrix(0)

        if USE_FASTAPI_MODEL_WRAPPERS and BGE_API_STYLE == "openai":
            max_batch = max(1, int(os.getenv("BGE_MAX_BATCH_SIZE", "64")))
            parallel_requests = max(1, int(os.getenv("BGE_PARALLEL_INFER_REQUESTS", "8")))
            chunks = [texts[i : i + max_batch] for i in range(0, len(texts), max_batch)]
            chunk_rows: List[np.ndarray] = [self._empty_matrix(0) for _ in chunks]

            def _encode_chunk(idx: int, chunk: List[str]) -> None:
                chunk_rows[idx] = self._encode_openai(chunk)

            workers = min(parallel_requests, len(chunks) or 1)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_encode_chunk, i, c) for i, c in enumerate(chunks)]
                for fut in futures:
                    fut.result()
            return np.vstack(chunk_rows) if chunk_rows else self._empty_matrix(0)

        if USE_FASTAPI_MODEL_WRAPPERS:
            return self._encode_legacy(texts, is_query)

        raise BGEServiceError("No BGE API mode configured (set SURF_MODEL_API_MODE=fastapi or triton)")
