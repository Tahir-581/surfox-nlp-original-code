# Surfox NLP — SERP competitor term extraction and ranking

An SEO and content-analysis tool. You give it a keyword or an article title. It collects the Google results, scrapes the competitor pages, pulls out the important terms with GLiNER, ranks and de-duplicates them with BGE embeddings and a reranker, and groups them into relevance tiers that a React dashboard displays.

## Pipeline

1. **SERP collection:** Playwright or Scrapling backends (`backend/serp_backends/`) with block and captcha detection and optional proxies.
2. **Page scraping:** title, meta description, headings, paragraphs and word counts.
3. **Term extraction:** GLiNER NER (`backend/NLP_Extraction_and_Ranking/`), served over FastAPI or Triton HTTP.
4. **Ranking:** BGE bi-encoder similarity to the query/title, plus an optional BGE cross-encoder reranker.
5. **De-duplication and clustering:** semantic and fuzzy de-duplication, then embedding-consensus tiering into Green / Orange / White buckets (`backend/nlp_embedding_tiers/`).
6. **Merge:** combines the terms from the competitor URLs you select into one keyword report.

The backend is a FastAPI app (`backend/main.py`) with JWT auth and PostgreSQL storage for searches, articles and users. The frontend is React (`frontend/`). A Dockerfile and docker-compose.yml are included.

## Run

```bash
cp .env.example .env                    # set DB, JWT secret, model service URLs
pip install -r backend/requirements.txt
playwright install chromium
python run_services.py                  # starts model server(s), backend and frontend
```

You can also use Docker: `docker compose up --build`.

Model services (GLiNER, BGE embeddings, BGE reranker) are located through the `SURF_*_API_URL` variables. `backend/serve_models.py` and `backend/NLP_Extraction_and_Ranking/model_server.py` can host them locally.

## Tests

```bash
cd backend && pytest -q
```

## Notes

- This repository contains code only. Scraped SERP results and generated outputs are not included.
- The domain-authority lookup expects a CSV of your own at `backend/Inputs/` (not included).

## License

MIT
