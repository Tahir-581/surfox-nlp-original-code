import json
import os
import time
import requests
from collections import Counter
from typing import List, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor

from requests.adapters import HTTPAdapter

from .nlp_serving_urls import (
    GLINER_BATCH_URL,
    GLINER_PREDICT_URL,
    GLINER_THRESHOLD,
    TRITON_GLINER_INFER_URL,
    USE_TRITON,
    is_local_gliner_url,
)


def _default_parallel_batch_requests() -> int:
    """Local single-process GLiNER cannot safely serve many parallel /predict calls."""
    explicit = os.getenv("GLINER_PARALLEL_BATCH_REQUESTS", "").strip()
    if explicit:
        return max(1, int(explicit))
    return 1 if is_local_gliner_url() else 8


class GLiNERServiceError(RuntimeError):
    """Raised when the GLiNER service is unavailable or returns invalid data."""


class GLiNERClient:
    """Client for GLiNER served either by direct Triton or the FastAPI wrapper."""

    _session: Optional[requests.Session] = None
    _pool_size: Optional[int] = None

    @classmethod
    def _get_session(cls) -> requests.Session:
        # Keep-alive session to avoid TCP handshake cost per chunk.
        pool_size = max(1, int(os.getenv("GLINER_REQUEST_CONCURRENCY", "64")))
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
        detail = f"GLiNER service request failed at {url}: {exc}"
        if response_text:
            detail = f"{detail} | response={response_text[:500]}"
        raise GLiNERServiceError(detail) from exc

    def _post_json(self, url: str, payload: dict, timeout: int, retries: int = 0) -> dict:
        max_attempts = max(1, retries + 1)
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                r = self._get_session().post(url, json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as exc:
                last_exc = exc
                response_text = ""
                status = exc.response.status_code if exc.response is not None else None
                if exc.response is not None:
                    response_text = exc.response.text or ""
                if status in {500, 503} and attempt < max_attempts - 1:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                self._raise_service_error(exc, url, response_text)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                self._raise_service_error(exc, url)
            except ValueError as exc:
                self._raise_service_error(exc, url, "invalid JSON response")
        if last_exc is not None:
            self._raise_service_error(last_exc, url)
        raise GLiNERServiceError(f"GLiNER service request failed at {url}")

    @staticmethod
    def _parse_batch_entities_response(data, expected_len: int) -> List[list]:
        if not isinstance(data, dict):
            return []
        rows = data.get("entities_per_text")
        if not isinstance(rows, list) or len(rows) != expected_len:
            return []
        return [row if isinstance(row, list) else [] for row in rows]

    @staticmethod
    def _parse_entities_response(data) -> List[dict]:
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        for key in ("entities", "results", "result", "predictions"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        outputs = data.get("outputs", [])
        if not outputs:
            return []
        raw = outputs[0].get("data", [])
        if not raw:
            return []
        first = raw[0]
        decoded = first.decode("utf-8") if isinstance(first, (bytes, bytearray)) else first
        parsed = json.loads(decoded)
        return parsed if isinstance(parsed, list) else []

    def predict_entities(self, text, labels, threshold=GLINER_THRESHOLD):
        if not USE_TRITON:
            payload = {
                "text": text,
                "labels": list(labels),
                "threshold": float(threshold),
            }
            return self._parse_entities_response(
                self._post_json(GLINER_PREDICT_URL, payload, timeout=120, retries=2)
            )

        labels_json = json.dumps(list(labels))
        payload = {
            "inputs": [
                {
                    "name": "text",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [text],
                },
                {
                    "name": "labels",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [labels_json],
                },
                {
                    "name": "threshold",
                    "datatype": "FP32",
                    "shape": [1, 1],
                    "data": [float(threshold)],
                },
            ],
            "outputs": [{"name": "entities"}],
        }
        return self._parse_entities_response(
            self._post_json(TRITON_GLINER_INFER_URL, payload, timeout=120)
        )

    def predict_entities_batch(self, texts: List[str], labels, threshold=GLINER_THRESHOLD) -> List[list]:
        items = [t for t in list(texts) if isinstance(t, str)]
        if not items:
            return []

        if not USE_TRITON:
            labels_list = list(labels)
            payload = {
                "texts": items,
                "labels": labels_list,
                "threshold": float(threshold),
            }
            try:
                data = self._post_json(GLINER_BATCH_URL, payload, timeout=300, retries=2)
                parsed = self._parse_batch_entities_response(data, len(items))
                if parsed:
                    return parsed
            except GLiNERServiceError:
                pass

            return [
                self.predict_entities(item, labels_list, threshold=threshold) or []
                for item in items
            ]

        labels_json = json.dumps(list(labels))
        payload = {
            "inputs": [
                {
                    "name": "text",
                    "datatype": "BYTES",
                    "shape": [len(items), 1],
                    "data": items,
                },
                {
                    "name": "labels",
                    "datatype": "BYTES",
                    "shape": [len(items), 1],
                    "data": [labels_json] * len(items),
                },
                {
                    "name": "threshold",
                    "datatype": "FP32",
                    "shape": [len(items), 1],
                    "data": [float(threshold)] * len(items),
                },
            ],
            "outputs": [{"name": "entities"}],
        }
        resp = self._post_json(TRITON_GLINER_INFER_URL, payload, timeout=300)
        outputs = resp.get("outputs", [])
        if not outputs:
            return [[] for _ in items]
        rows = outputs[0].get("data", [])
        parsed: List[list] = []
        for row in rows:
            decoded = row.decode("utf-8") if isinstance(row, (bytes, bytearray)) else row
            parsed.append(json.loads(decoded))
        return parsed

ENTITY_LABELS = [
    # People & orgs
    "Person", "Organization", "Company", "Institution", "School", "University",
    "Author", "Artist", "Character", "Profession", "Role", "Group",
    # Places
    "Location", "City", "Country", "Address", "Region", "Place", "Building",
    "Landmark", "Continent", "Area", "Territory",
    # Time
    "Date", "Time", "Event", "Period", "Year", "Duration", "Era",
    # Things & products
    "Work of Art", "Consumer Good", "Product", "Brand", "Vehicle", "Tool",
    "Equipment", "Technology", "Software", "Food", "Dish", "Substance",
    "Chemical", "Material", "Object",
    # Abstract & concepts
    "Concept", "Theory", "Idea", "Topic", "Theme", "Category", "Type",
    "Attribute", "Quality", "Trait", "Behavior", "Activity", "Skill",
    "Method", "Technique", "Principle", "Requirement", "Feature", "Benefit",
    "Condition", "State", "Aspect", "Factor", "Criteria",
    # Legal, medical, science
    "Law", "Medical Condition", "Disease", "Scientific Term", "Award",
    "Language", "Percentage", "Price", "Phone Number",
    # Animals & nature (your domain)
    "Animal", "Animal Breed", "Dog Breed", "Breed", "Species", "Pet",
    "Nature", "Plant", "Environment", "Habitat",
    # Other
    "Title", "Position", "Facility", "Other", "Term", "Phrase", "Expression",
]

def _chunk_starts_and_ends(text, target_chunk_size=600, step_size=300):
    """Yield (start, end) indices so each chunk is aligned to word boundaries.
    This avoids cutting mid-word, which causes GLiNER to predict fragments like
    'ude', 'tion', 'attit', 'iven'.
    """
    n = len(text)
    start = 0
    while start < n:
        end = min(start + target_chunk_size, n)
        # If we're not at the end, back up to the last space so we don't cut a word
        if end < n and end > start:
            last_space = text.rfind(" ", start, end + 1)
            if last_space > start:
                end = last_space + 1
        yield start, end
        # Step forward; try to start at a word boundary
        start = start + step_size
        if start < n and text[start] not in " \n\t":
            next_space = text.find(" ", start, min(start + 80, n))
            if next_space != -1:
                start = next_space + 1
        if start >= n:
            break


def extract_entities_sliding_window(text, model, step_size=300, context_size=600):
    # Use a Counter where the key is a tuple of (text, label)
    # We'll aggregate case-insensitively: use a normalized (casefolded) text as the
    # counting key but remember the original forms so we can pick a canonical
    # display string (the most frequent original form seen).
    entity_counts = Counter()
    original_forms = {}

    PRONOUNS = {
        "i", "me", "you", "he", "him", "she", "her", "it",
        "we", "us", "they", "them", "my", "your", "his", "hers",
        "our", "their", "mine", "yours", "ours", "theirs",
        "who", "whom", "whose", "that", "this", "these", "those",
    }
    EXCLUDED_SINGLE_TOKENS = PRONOUNS | {"the", "a"}

    # Common word fragments GLiNER can emit when context is cut mid-word
    FRAGMENTS = {
        "tion", "ude", "ive", "nat", "iven", "attit", "ident", "itude",
        "ment", "ness", "ence", "ance", "ally", "cal", "ful", "ous",
        "ive", "ent", "ant", "est", "ity", "ive", "ly", "er", "ed",
        "al", "ic", "an", "or", "ar", "en", "on", "in", "at", "ed",
    }

    # Collect all non-empty chunks
    chunks: List[str] = []
    for start, end in _chunk_starts_and_ends(text, context_size, step_size):
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)

    chunk_batch_default = "4" if is_local_gliner_url() else "8"
    batch_size = int(os.getenv("GLINER_CHUNK_BATCH_SIZE", chunk_batch_default))
    batch_size = max(1, min(batch_size, 64))
    request_concurrency = _default_parallel_batch_requests()

    chunk_results: List[list] = []
    if hasattr(model, "predict_entities_batch"):
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        batch_outputs: List[List[list]] = [[] for _ in batches]

        def _run_batch(idx: int, batch: List[str]) -> None:
            out = model.predict_entities_batch(batch, ENTITY_LABELS, threshold=GLINER_THRESHOLD)
            if len(out) != len(batch):
                out = [model.predict_entities(c, ENTITY_LABELS, threshold=GLINER_THRESHOLD) for c in batch]
            batch_outputs[idx] = out

        max_workers = min(request_concurrency, len(batches) or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_batch, i, b) for i, b in enumerate(batches)]
            for fut in futures:
                fut.result()

        for out in batch_outputs:
            chunk_results.extend(out)
    else:
        chunk_results = [model.predict_entities(c, ENTITY_LABELS, threshold=GLINER_THRESHOLD) for c in chunks]

    for chunk_ents in chunk_results:
        for ent in chunk_ents:
            text_span = ent["text"].strip()

            # Pronoun / weak single-token filtering
            if " " not in text_span:
                token = text_span.replace("’", "'").strip("\"'()[]{}<>.,;:!?-")
                base = token.split("'")[0].lower()
                if base in EXCLUDED_SINGLE_TOKENS:
                    continue

            # Length filtering
            cleaned = "".join(ch for ch in text_span if ch.isalnum())
            if len(cleaned) <= 2:
                continue

            # Skip known word fragments (from mid-word chunk cuts) and suffix-like tokens
            base_lower = cleaned.lower()
            if base_lower in FRAGMENTS:
                continue
            if len(base_lower) <= 5 and (base_lower.endswith(("tion", "ude", "ive", "ment", "ness", "ence", "ance")) or base_lower.startswith(("nat", "ident", "iven"))):
                continue

            # Normalize text for aggregation (case-insensitive)
            norm_text = text_span.casefold()
            key = norm_text

            entity_counts[key] += 1

            if key not in original_forms:
                original_forms[key] = {"text": Counter(), "labels": Counter()}
            original_forms[key]["text"][text_span] += 1
            label = (ent.get("label") or "Other").strip() or "Other"
            original_forms[key]["labels"][label] += 1

    final_entities = []
    for norm_text, count in entity_counts.items():
        form_data = original_forms.get(norm_text)
        if form_data:
            display_text = form_data["text"].most_common(1)[0][0]
            label_val = form_data["labels"].most_common(1)[0][0] if form_data["labels"] else "Other"
        else:
            display_text = norm_text
            label_val = "Other"

        final_entities.append({
            "text": display_text,
            "count": count,
            "label": label_val,
        })
            
    # Sort by count (descending) so the most frequent appear first
    return sorted(final_entities, key=lambda x: x['count'], reverse=True)

def main():
    INPUT_FILE = "Dog_Breeds_to_Deter_Intruders_and_Keep_You_Safe/www.wisdompanel.com_27412.json"
    OUTPUT_DIR = "outputs"
    MODEL_NAME = "urchade/gliner_multi-v2.1" 
    
    if not os.path.isfile(INPUT_FILE):
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        text = data.get("content", "")

    if not text:
        raise SystemExit("No 'content' field found in JSON file")

    print(f"--- Loading LARGE Model: {MODEL_NAME} (via API on port 6000) ---")
    model = GLiNERClient()
    
    print(f"--- Deep Scanning & Aggregating {len(text)} characters... ---")
    entities = extract_entities_sliding_window(text, model)

    total_instances = sum(ent["count"] for ent in entities)
    
    print("\n[Deep Extraction Summary]")
    print("-" * 40)
    print(f"UNIQUE ENTITIES FOUND: {len(entities)}")
    print(f"TOTAL INSTANCES FOUND: {total_instances}\n")
    print("-" * 40)

    filename = os.path.splitext(os.path.basename(INPUT_FILE))[0] + "_ner.json"
    output_path = "outputs/gliner_output.json"
    result = {
        "source": os.path.basename(INPUT_FILE), 
        "unique_count": len(entities),
        "total_instances": total_instances,
        "entities": entities
    }

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, ensure_ascii=False, indent=2)

    print(f"Aggregated results saved to: {output_path}")

if __name__ == "__main__":
    main()
