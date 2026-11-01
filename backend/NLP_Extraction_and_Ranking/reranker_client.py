import os
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter

from .nlp_serving_urls import RERANK_MODEL_ID, RERANK_URL


class RerankerServiceError(RuntimeError):
    """Raised when the reranker service is unavailable or returns invalid data."""


class RerankerClient:
    """HTTP client for BGE reranker (vLLM /v2/rerank API)."""

    _session: Optional[requests.Session] = None
    _pool_size: Optional[int] = None

    @classmethod
    def _get_pool_size(cls) -> int:
        return max(1, int(os.getenv("RERANK_REQUEST_CONCURRENCY", "32")))

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

    @staticmethod
    def _raise_service_error(exc: Exception, url: str, response_text: str = "") -> None:
        detail = f"Reranker service request failed at {url}: {exc}"
        if response_text:
            detail = f"{detail} | response={response_text[:500]}"
        raise RerankerServiceError(detail) from exc

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

    @staticmethod
    def _parse_scores(data: dict, num_documents: int) -> List[float]:
        results = data.get("results", [])
        if not isinstance(results, list):
            raise RerankerServiceError("Reranker response missing results[]")
        scores = [0.0] * num_documents
        for item in results:
            idx = int(item.get("index", -1))
            if 0 <= idx < num_documents:
                scores[idx] = float(item.get("relevance_score", 0.0))
        return scores

    def _rerank_batch(self, query: str, documents: List[str]) -> List[float]:
        payload = {
            "model": RERANK_MODEL_ID,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        return self._parse_scores(
            self._post_json(RERANK_URL, payload, timeout=300),
            num_documents=len(documents),
        )

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
        if not (query or query.strip()):
            return [0.0] * len(documents)

        max_batch = max(1, int(os.getenv("RERANK_MAX_BATCH_SIZE", "128")))
        if len(documents) <= max_batch:
            return self._rerank_batch(query, documents)

        chunks = [documents[i : i + max_batch] for i in range(0, len(documents), max_batch)]
        all_scores: List[float] = []

        def _run_chunk(chunk: List[str]) -> List[float]:
            return self._rerank_batch(query, chunk)

        workers = min(
            max(1, int(os.getenv("RERANK_PARALLEL_REQUESTS", "4"))),
            len(chunks),
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for chunk_scores in executor.map(_run_chunk, chunks):
                all_scores.extend(chunk_scores)
        return all_scores
