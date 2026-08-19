import json
import logging
import os
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import Json, RealDictCursor
except ImportError:  # pragma: no cover - handled at runtime with a clear error
    psycopg2 = None
    sql = None
    Json = None
    RealDictCursor = None


log = logging.getLogger(__name__)

DB_HOST = os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost"))
DB_PORT = int(os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432")))
DB_NAME = os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "serfox_db"))
DB_USER = os.getenv("POSTGRES_USER", os.getenv("DB_USER", "postgres"))
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "postgres"))
_ARTICLE_COLUMNS_READY = False

def _require_driver() -> None:
    if psycopg2 is None:
        raise RuntimeError("psycopg2-binary is not installed. Run: pip install -r backend/requirements.txt")


def _connect(dbname: Optional[str] = None):
    _require_driver()
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=dbname or DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def ensure_database_exists() -> None:
    with _connect("postgres") as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
            if cur.fetchone():
                return
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            log.info("Created PostgreSQL database %s", DB_NAME)


def init_db() -> None:
    ensure_database_exists()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS nlp_keyword_outputs (
                    id SERIAL PRIMARY KEY,
                    source_keyword TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL,
                    keywords_json JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("ALTER TABLE nlp_keyword_outputs ADD COLUMN IF NOT EXISTS keywords_json JSONB")
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'nlp_keyword_outputs'
                  AND column_name = 'json_output'
                """
            )
            if cur.fetchone():
                cur.execute(
                    """
                    UPDATE nlp_keyword_outputs
                    SET keywords_json = json_output
                    WHERE keywords_json IS NULL
                    """
                )
            cur.execute("UPDATE nlp_keyword_outputs SET keywords_json = '{}'::jsonb WHERE keywords_json IS NULL")
            cur.execute("ALTER TABLE nlp_keyword_outputs ALTER COLUMN keywords_json SET NOT NULL")
            cur.execute("ALTER TABLE nlp_keyword_outputs DROP COLUMN IF EXISTS selected_keywords")
            cur.execute("ALTER TABLE nlp_keyword_outputs DROP COLUMN IF EXISTS selected_at")
            cur.execute("ALTER TABLE nlp_keyword_outputs DROP COLUMN IF EXISTS json_output")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_nlp_keyword_outputs_source_keyword
                ON nlp_keyword_outputs (LOWER(source_keyword))
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_users (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    role TEXT NOT NULL DEFAULT 'writer',
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
                    verification_token TEXT,
                    reset_token TEXT,
                    reset_token_expiry TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    article_key TEXT NOT NULL UNIQUE,
                    session_id TEXT,
                    title TEXT NOT NULL,
                    keyword TEXT,
                    keywords_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    results JSONB NOT NULL DEFAULT '[]'::jsonb,
                    selected_urls JSONB NOT NULL DEFAULT '[]'::jsonb,
                    content_score INTEGER NOT NULL DEFAULT 0,
                    html TEXT NOT NULL DEFAULT '',
                    text_content TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'drafting',
                    assigned_to INTEGER REFERENCES app_users(id) ON DELETE SET NULL,
                    created_by INTEGER REFERENCES app_users(id) ON DELETE SET NULL,
                    updated_by INTEGER REFERENCES app_users(id) ON DELETE SET NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS article_revisions (
                    id SERIAL PRIMARY KEY,
                    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
                    diff_patch TEXT NOT NULL DEFAULT '',
                    changed_by INTEGER REFERENCES app_users(id) ON DELETE SET NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """ 
            )
            # Migrate existing table: add diff_patch if missing
            cur.execute("ALTER TABLE article_revisions ADD COLUMN IF NOT EXISTS diff_patch TEXT NOT NULL DEFAULT ''")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS article_permissions (
                    id SERIAL PRIMARY KEY,
                    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
                    can_edit BOOLEAN NOT NULL DEFAULT FALSE,
                    can_update BOOLEAN NOT NULL DEFAULT FALSE,
                    assigned_by INTEGER REFERENCES app_users(id) ON DELETE SET NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(article_id, user_id)
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_article_key ON articles (article_key)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_article_revisions_article_id ON article_revisions (article_id)")
            cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS content_score INTEGER NOT NULL DEFAULT 0")
            # Migrate existing app_users table with new auth columns
            cur.execute("ALTER TABLE app_users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT FALSE")
            cur.execute("ALTER TABLE app_users ADD COLUMN IF NOT EXISTS verification_token TEXT")
            cur.execute("ALTER TABLE app_users ADD COLUMN IF NOT EXISTS reset_token TEXT")
            cur.execute("ALTER TABLE app_users ADD COLUMN IF NOT EXISTS reset_token_expiry TIMESTAMPTZ")
            # Per-user search session storage
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS search_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
                    session_id TEXT NOT NULL UNIQUE,
                    keyword TEXT,
                    results JSONB NOT NULL DEFAULT '[]'::jsonb,
                    selected_urls JSONB NOT NULL DEFAULT '[]'::jsonb,
                    merge_output JSONB,
                    timing JSONB,
                    status TEXT NOT NULL DEFAULT 'completed',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_search_sessions_user_created
                ON search_sessions (user_id, created_at DESC)
                """
            )
            # Scope NLP keyword outputs per user
            cur.execute(
                "ALTER TABLE nlp_keyword_outputs ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES app_users(id) ON DELETE CASCADE"
            )
            cur.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'nlp_keyword_outputs_source_keyword_key'
                    ) THEN
                        ALTER TABLE nlp_keyword_outputs DROP CONSTRAINT nlp_keyword_outputs_source_keyword_key;
                    END IF;
                END $$;
                """
            )
            cur.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'nlp_keyword_outputs_user_id_source_keyword_key'
                    ) THEN
                        ALTER TABLE nlp_keyword_outputs
                        ADD CONSTRAINT nlp_keyword_outputs_user_id_source_keyword_key
                        UNIQUE (user_id, source_keyword);
                    END IF;
                END $$;
                """
            )
    log.info("PostgreSQL ready at %s:%s/%s", DB_HOST, DB_PORT, DB_NAME)


def ensure_article_columns() -> None:
    global _ARTICLE_COLUMNS_READY
    if _ARTICLE_COLUMNS_READY:
        return
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS keywords_json JSONB NOT NULL DEFAULT '[]'::jsonb")
            cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS results JSONB NOT NULL DEFAULT '[]'::jsonb")
            cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS selected_urls JSONB NOT NULL DEFAULT '[]'::jsonb")
            cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS content_score INTEGER NOT NULL DEFAULT 0")
            cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS html TEXT NOT NULL DEFAULT ''")
            cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS text_content TEXT NOT NULL DEFAULT ''")
            cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'drafting'")
            cur.execute("ALTER TABLE articles ADD COLUMN IF NOT EXISTS assigned_to INTEGER REFERENCES app_users(id) ON DELETE SET NULL")
    _ARTICLE_COLUMNS_READY = True


def _coerce_content_score(value) -> int:
    try:
        return max(0, min(100, int(round(float(value or 0)))))
    except (TypeError, ValueError):
        return 0


def _hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt.encode("utf-8"), 120000)
    return {"salt": salt, "hash": digest.hex()}


def create_user(name: str, email: str, password: str, role: str = "writer", verification_token: Optional[str] = None) -> Dict:
    password_data = _hash_password(password)
    email_verified = verification_token is None  # if no token, mark verified immediately
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO app_users (name, email, role, password_hash, password_salt, email_verified, verification_token)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, name, email, role, email_verified, created_at, updated_at
                """,
                (name.strip(), email.strip().lower(), role.strip() or "writer",
                 password_data["hash"], password_data["salt"], email_verified, verification_token),
            )
            return dict(cur.fetchone())


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM app_users WHERE email = %s", (email.strip().lower(),))
            user = cur.fetchone()
            if not user:
                return None
            password_data = _hash_password(password, user["password_salt"])
            if not secrets.compare_digest(password_data["hash"], user["password_hash"]):
                return None
            return {k: user[k] for k in ("id", "name", "email", "role", "email_verified", "created_at", "updated_at")}

def list_users() -> List[Dict]:
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, name, email, role, created_at FROM app_users ORDER BY name ASC")
            return [dict(row) for row in cur.fetchall()]


def verify_email_token(token: str) -> Optional[Dict]:
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, email FROM app_users WHERE verification_token = %s AND email_verified = FALSE",
                (token,)
            )
            user = cur.fetchone()
            if not user:
                return None
            cur.execute(
                "UPDATE app_users SET email_verified = TRUE, verification_token = NULL, updated_at = NOW() WHERE id = %s RETURNING id, name, email, role",
                (user["id"],)
            )
            return dict(cur.fetchone())


def set_reset_token(email: str, token: str) -> bool:
    from datetime import timezone, timedelta
    expiry = datetime.now(timezone.utc) + timedelta(hours=1)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE app_users SET reset_token = %s, reset_token_expiry = %s, updated_at = NOW() WHERE email = %s",
                (token, expiry, email.strip().lower())
            )
            return cur.rowcount > 0


def reset_password_with_token(token: str, new_password: str) -> Optional[Dict]:
    from datetime import timezone
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id FROM app_users WHERE reset_token = %s AND reset_token_expiry > %s",
                (token, datetime.now(timezone.utc))
            )
            user = cur.fetchone()
            if not user:
                return None
            password_data = _hash_password(new_password)
            cur.execute(
                """
                UPDATE app_users
                SET password_hash = %s, password_salt = %s, reset_token = NULL,
                    reset_token_expiry = NULL, updated_at = NOW()
                WHERE id = %s
                RETURNING id, name, email, role
                """,
                (password_data["hash"], password_data["salt"], user["id"])
            )
            return dict(cur.fetchone())


def update_user_role(user_id: int, role: str) -> Dict:
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE app_users
                SET role = %s, updated_at = NOW()
                WHERE id = %s
                RETURNING id, name, email, role, created_at, updated_at
                """,
                (role.strip(), user_id),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError("User not found")
            return dict(row)


def upsert_article(payload: Dict) -> Dict:
    article_key = (payload.get("article_key") or payload.get("key") or payload.get("session_id") or payload.get("title") or "untitled").strip()
    user_id = payload.get("user_id")
    ensure_article_columns()
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, html FROM articles WHERE article_key = %s", (article_key,))
            existing = cur.fetchone()
            cur.execute(
                """
                INSERT INTO articles (
                    article_key, session_id, title, keyword, keywords_json, results, selected_urls, content_score, html, text_content, status, assigned_to, created_by, updated_by
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (article_key)
                DO UPDATE SET
                    session_id = COALESCE(EXCLUDED.session_id, articles.session_id),
                    title = EXCLUDED.title,
                    keyword = COALESCE(EXCLUDED.keyword, articles.keyword),
                    keywords_json = EXCLUDED.keywords_json,
                    results = EXCLUDED.results,
                    selected_urls = EXCLUDED.selected_urls,
                    content_score = EXCLUDED.content_score,
                    html = EXCLUDED.html,
                    text_content = EXCLUDED.text_content,
                    status = COALESCE(EXCLUDED.status, articles.status),
                    assigned_to = COALESCE(EXCLUDED.assigned_to, articles.assigned_to),
                    updated_by = EXCLUDED.updated_by,
                    updated_at = NOW()
                RETURNING *
                """,
                (
                    article_key,
                    payload.get("session_id"),
                    payload.get("title") or "Untitled",
                    payload.get("keyword"),
                    Json(payload.get("keywords") or []),
                    Json(payload.get("results") or []),
                    Json(payload.get("selected_urls") or []),
                    _coerce_content_score(payload.get("content_score")),
                    payload.get("html") or "",
                    payload.get("text") or payload.get("text_content") or "",
                    payload.get("status") or "drafting",
                    payload.get("assigned_to"),
                    user_id,
                    user_id,
                ),
            )
            article = dict(cur.fetchone())
            # Only store revision if the article already existed and content changed
            # diff_patch is computed on the frontend to avoid storing duplicate full HTML
            diff_patch = payload.get("diff_patch", "")
            if existing and diff_patch:
                cur.execute(
                    """
                    INSERT INTO article_revisions (article_id, diff_patch, changed_by)
                    VALUES (%s, %s, %s)
                    """,
                    (article["id"], diff_patch, user_id),
                )
                # Cleanup: Enforce bounded storage by keeping only the last 30 revisions per article
                cur.execute(
                    """
                    DELETE FROM article_revisions
                    WHERE article_id = %s
                      AND id NOT IN (
                          SELECT id FROM article_revisions
                          WHERE article_id = %s
                          ORDER BY created_at DESC
                          LIMIT 30
                      )
                    """,
                    (article["id"], article["id"])
                )
            return article


def get_article(article_key: str) -> Optional[Dict]:
    ensure_article_columns()
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    article_key,
                    session_id,
                    title,
                    keyword,
                    keywords_json,
                    results,
                    selected_urls,
                    content_score,
                    html,
                    text_content,
                    status,
                    assigned_to,
                    created_at,
                    updated_at
                FROM articles
                WHERE article_key = %s
                """,
                (article_key,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def list_articles() -> List[Dict]:
    ensure_article_columns()
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    article_key,
                    session_id,
                    title,
                    keyword,
                    keywords_json,
                    content_score,
                    status,
                    assigned_to,
                    created_at,
                    updated_at
                FROM articles
                ORDER BY updated_at DESC
                """
            )
            return [dict(row) for row in cur.fetchall()]


def delete_article(article_key: str) -> bool:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM articles WHERE article_key = %s", (article_key,))
            return cur.rowcount > 0


def get_article_history(article_key: str) -> List[Dict]:
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM articles WHERE article_key = %s", (article_key,))
            article = cur.fetchone()
            if not article:
                return []
            cur.execute(
                """
                SELECT
                    r.id,
                    r.diff_patch,
                    r.created_at,
                    u.name AS changed_by_name,
                    u.role AS changed_by_role
                FROM article_revisions r
                LEFT JOIN app_users u ON u.id = r.changed_by
                WHERE r.article_id = %s
                ORDER BY r.created_at DESC
                LIMIT 100
                """,
                (article["id"],),
            )
            return [dict(row) for row in cur.fetchall()]


def assign_article_permission(article_key: str, user_id: int, can_edit: bool, can_update: bool, assigned_by: Optional[int] = None) -> Dict:
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id FROM articles WHERE article_key = %s", (article_key,))
            article = cur.fetchone()
            if not article:
                raise ValueError("Article not found")
            cur.execute(
                """
                INSERT INTO article_permissions (article_id, user_id, can_edit, can_update, assigned_by)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (article_id, user_id)
                DO UPDATE SET
                    can_edit = EXCLUDED.can_edit,
                    can_update = EXCLUDED.can_update,
                    assigned_by = EXCLUDED.assigned_by,
                    updated_at = NOW()
                RETURNING *
                """,
                (article["id"], user_id, can_edit, can_update, assigned_by),
            )
            return dict(cur.fetchone())


def get_user_by_id(user_id: int) -> Optional[Dict]:
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, name, email, role, email_verified, created_at, updated_at FROM app_users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def create_search_session(
    user_id: int,
    session_id: str,
    keyword: str,
    results: List,
    timing: Optional[Dict] = None,
    status: str = "completed",
) -> Dict:
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO search_sessions
                    (user_id, session_id, keyword, results, timing, status)
                VALUES
                    (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id)
                DO UPDATE SET
                    keyword = EXCLUDED.keyword,
                    results = EXCLUDED.results,
                    timing = EXCLUDED.timing,
                    status = EXCLUDED.status,
                    updated_at = NOW()
                RETURNING *
                """,
                (user_id, session_id, keyword, Json(results or []), Json(timing or {}), status),
            )
            return dict(cur.fetchone())


def update_search_session_merge(
    user_id: int,
    session_id: str,
    selected_urls: List,
    merge_output: Dict,
) -> Optional[Dict]:
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE search_sessions
                SET selected_urls = %s,
                    merge_output = %s,
                    updated_at = NOW()
                WHERE user_id = %s AND session_id = %s
                RETURNING *
                """,
                (Json(selected_urls or []), Json(merge_output or {}), user_id, session_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def list_search_sessions(user_id: int, limit: int = 50, offset: int = 0) -> List[Dict]:
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    id,
                    session_id,
                    keyword,
                    status,
                    jsonb_array_length(COALESCE(results, '[]'::jsonb)) AS result_count,
                    (merge_output IS NOT NULL AND merge_output != 'null'::jsonb) AS has_merge,
                    created_at,
                    updated_at
                FROM search_sessions
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (user_id, limit, offset),
            )
            return [dict(row) for row in cur.fetchall()]


def get_search_session(user_id: int, session_id: str) -> Optional[Dict]:
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT *
                FROM search_sessions
                WHERE user_id = %s AND session_id = %s
                """,
                (user_id, session_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def get_search_session_by_session_id(session_id: str) -> Optional[Dict]:
    """Load session by session_id only (for merge fallback when user is known)."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM search_sessions WHERE session_id = %s", (session_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def update_search_session_merge_output(
    user_id: int,
    session_id: str,
    merge_output: Dict,
) -> Optional[Dict]:
    """Update only merge_output JSON for an existing search session."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE search_sessions
                SET merge_output = %s,
                    updated_at = NOW()
                WHERE user_id = %s AND session_id = %s
                RETURNING *
                """,
                (Json(merge_output or {}), user_id, session_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def list_search_sessions_with_merge(
    user_id: Optional[int] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict]:
    """List sessions that have merge_output stored."""
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT user_id, session_id, keyword, merge_output, selected_urls
                FROM search_sessions
                WHERE merge_output IS NOT NULL
                  AND merge_output != 'null'::jsonb
            """
            params: List = []
            if user_id is not None:
                query += " AND user_id = %s"
                params.append(user_id)
            query += " ORDER BY updated_at DESC"
            if limit is not None:
                query += " LIMIT %s OFFSET %s"
                params.extend([limit, offset])
            cur.execute(query, tuple(params))
            return [dict(row) for row in cur.fetchall()]


def delete_search_session(user_id: int, session_id: str) -> bool:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM search_sessions WHERE user_id = %s AND session_id = %s",
                (user_id, session_id),
            )
            return cur.rowcount > 0


def upsert_keyword_json_output(
    source_keyword: str,
    file_name: str,
    payload: Dict,
    user_id: Optional[int] = None,
) -> Dict:
    source_keyword = (source_keyword or file_name or "search").strip()
    file_name = (file_name or f"{source_keyword}.json").strip()
    payload = payload or {}

    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if user_id is not None:
                cur.execute(
                    """
                    INSERT INTO nlp_keyword_outputs
                        (user_id, source_keyword, file_name, keywords_json)
                    VALUES
                        (%s, %s, %s, %s)
                    ON CONFLICT (user_id, source_keyword)
                    DO UPDATE SET
                        file_name = EXCLUDED.file_name,
                        keywords_json = EXCLUDED.keywords_json,
                        updated_at = NOW()
                    RETURNING
                        id,
                        user_id,
                        source_keyword,
                        file_name,
                        keywords_json,
                        created_at,
                        updated_at
                    """,
                    (user_id, source_keyword, file_name, Json(payload)),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO nlp_keyword_outputs
                        (source_keyword, file_name, keywords_json)
                    VALUES
                        (%s, %s, %s)
                    RETURNING
                        id,
                        user_id,
                        source_keyword,
                        file_name,
                        keywords_json,
                        created_at,
                        updated_at
                    """,
                    (source_keyword, file_name, Json(payload)),
                )
            return dict(cur.fetchone())


def import_json_outputs(json_outputs_dir: Path) -> int:
    total = 0
    for file_path in sorted(Path(json_outputs_dir).glob("*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        source_keyword = file_path.stem.replace("_", " ")
        upsert_keyword_json_output(source_keyword, file_path.name, payload)
        total += 1
    return total


def list_keyword_json_outputs(
    source_keyword: Optional[str] = None,
    user_id: Optional[int] = None,
) -> List[Dict]:
    params = []
    query = """
        SELECT
            id,
            user_id,
            source_keyword,
            file_name,
            keywords_json,
            created_at,
            updated_at
        FROM nlp_keyword_outputs
        WHERE 1=1
    """
    if user_id is not None:
        query += " AND user_id = %s"
        params.append(user_id)
    if source_keyword:
        query += " AND source_keyword = %s"
        params.append(source_keyword)
    query += " ORDER BY updated_at DESC, source_keyword ASC"

    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, tuple(params))
            return [dict(row) for row in cur.fetchall()]


def delete_keyword_json_output(output_id: int, user_id: Optional[int] = None) -> bool:
    with _connect() as conn:
        with conn.cursor() as cur:
            if user_id is not None:
                cur.execute(
                    "DELETE FROM nlp_keyword_outputs WHERE id = %s AND user_id = %s",
                    (output_id, user_id),
                )
            else:
                cur.execute("DELETE FROM nlp_keyword_outputs WHERE id = %s", (output_id,))
            return cur.rowcount > 0


def list_articles_for_user(user_id: int) -> List[Dict]:
    ensure_article_columns()
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT DISTINCT
                    a.id,
                    a.article_key,
                    a.session_id,
                    a.title,
                    a.keyword,
                    a.keywords_json,
                    a.content_score,
                    a.status,
                    a.assigned_to,
                    a.created_at,
                    a.updated_at
                FROM articles a
                LEFT JOIN article_permissions ap ON ap.article_id = a.id AND ap.user_id = %s
                WHERE a.created_by = %s OR ap.user_id = %s
                ORDER BY a.updated_at DESC
                """,
                (user_id, user_id, user_id),
            )
            return [dict(row) for row in cur.fetchall()]


def get_article_for_user(article_key: str, user_id: int) -> Optional[Dict]:
    ensure_article_columns()
    with _connect() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    a.id,
                    a.article_key,
                    a.session_id,
                    a.title,
                    a.keyword,
                    a.keywords_json,
                    a.results,
                    a.selected_urls,
                    a.content_score,
                    a.html,
                    a.text_content,
                    a.status,
                    a.assigned_to,
                    a.created_at,
                    a.updated_at
                FROM articles a
                LEFT JOIN article_permissions ap ON ap.article_id = a.id AND ap.user_id = %s
                WHERE a.article_key = %s
                  AND (a.created_by = %s OR ap.user_id = %s)
                """,
                (user_id, article_key, user_id, user_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None
