"""
CodeLens AI — Database engine, session factory, and Supabase client wiring.

Connection priority:
  1. DATABASE_URL / SUPABASE_DB_URL  (direct PostgreSQL URI — preferred for SQLAlchemy)
  2. SUPABASE_URL if it is already a postgresql:// URI
  3. Raises a clear configuration error otherwise

SUPABASE_URL + SUPABASE_KEY (or SUPABASE_ANON_KEY / SUPABASE_SERVICE_ROLE_KEY)
are also exposed for the Supabase Python client (Auth, Storage, Realtime).
"""

from __future__ import annotations

import os
from collections.abc import Generator
from functools import lru_cache
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

# Canonical env keys
ENV_DATABASE_URL = "DATABASE_URL"
ENV_SUPABASE_DB_URL = "SUPABASE_DB_URL"
ENV_SUPABASE_URL = "SUPABASE_URL"
ENV_SUPABASE_KEY = "SUPABASE_KEY"
ENV_SUPABASE_ANON_KEY = "SUPABASE_ANON_KEY"
ENV_SUPABASE_SERVICE_ROLE_KEY = "SUPABASE_SERVICE_ROLE_KEY"


def resolve_database_url() -> str:
    """
    Resolve the SQLAlchemy PostgreSQL connection string from the environment.

    Supports both a dedicated DB URI and a mis-set SUPABASE_URL that already
    contains a postgresql:// scheme (some teams store the pooler URI there).
    """
    for key in (ENV_DATABASE_URL, ENV_SUPABASE_DB_URL):
        value = (os.getenv(key) or "").strip()
        if value:
            return value

    supabase_url = (os.getenv(ENV_SUPABASE_URL) or "").strip()
    if supabase_url.startswith(("postgresql://", "postgres://", "postgresql+psycopg2://")):
        return supabase_url

    raise EnvironmentError(
        "No PostgreSQL connection string found. Set one of:\n"
        "  DATABASE_URL=postgresql://postgres:PASSWORD@db.PROJECT.supabase.co:5432/postgres\n"
        "  SUPABASE_DB_URL=postgresql://...\n"
        "Optionally also set SUPABASE_URL and SUPABASE_KEY for the Supabase client API."
    )


def _normalize_sqlalchemy_url(url: str) -> str:
    """
    Ensure the URL uses a driver SQLAlchemy understands with psycopg2.

    Also strips Prisma/Supabase helper query params (e.g. pgbouncer=true)
    that are not valid libpq connection options.
    """
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://") :]
    elif url.startswith("postgresql://") and "+psycopg2" not in url and "+psycopg" not in url:
        url = "postgresql+psycopg2://" + url[len("postgresql://") :]

    parsed = urlparse(url)
    if parsed.query:
        # Keep only libpq-recognized options; drop helpers like pgbouncer=true
        libpq_allowed = {
            "host",
            "port",
            "user",
            "password",
            "dbname",
            "connect_timeout",
            "sslmode",
            "sslcert",
            "sslkey",
            "sslrootcert",
            "application_name",
            "options",
            "channel_binding",
            "gssencmode",
            "krbsrvname",
            "service",
            "target_session_attrs",
        }
        filtered = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if k.lower() in libpq_allowed
        ]
        url = urlunparse(parsed._replace(query=urlencode(filtered)))

    return url


@lru_cache(maxsize=1)
def get_engine(*, echo: Optional[bool] = None) -> Engine:
    """Create (once) and return the shared SQLAlchemy Engine."""
    url = _normalize_sqlalchemy_url(resolve_database_url())
    if echo is None:
        echo = os.getenv("SQL_ECHO", "").lower() in {"1", "true", "yes"}

    return create_engine(
        url,
        echo=echo,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        future=True,
    )


def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI-compatible session dependency.

    Callers own transaction boundaries (commit/rollback). The session is
    always closed when the request finishes.
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def resolve_supabase_key() -> Optional[str]:
    """Prefer explicit SUPABASE_KEY, then service role, then anon key."""
    for key in (ENV_SUPABASE_KEY, ENV_SUPABASE_SERVICE_ROLE_KEY, ENV_SUPABASE_ANON_KEY):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return None


@lru_cache(maxsize=1)
def get_supabase_client() -> Any:
    """
    Lazily construct the Supabase Python client for Auth/Storage/Realtime.

    Requires SUPABASE_URL (https://xxxx.supabase.co) and a project API key.
    """
    from supabase import create_client

    url = (os.getenv(ENV_SUPABASE_URL) or "").strip()
    key = resolve_supabase_key()

    if not url or url.startswith(("postgresql://", "postgres://")):
        raise EnvironmentError(
            "SUPABASE_URL must be the HTTPS project URL "
            "(e.g. https://xxxx.supabase.co), not a Postgres URI."
        )
    if not key:
        raise EnvironmentError(
            "Missing Supabase API key. Set SUPABASE_KEY, "
            "SUPABASE_SERVICE_ROLE_KEY, or SUPABASE_ANON_KEY."
        )

    return create_client(url, key)


def ping_database() -> bool:
    """Lightweight connectivity check against the configured Postgres instance."""
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    return True