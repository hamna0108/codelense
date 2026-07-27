"""
CodeLens AI — Apply ORM metadata to the connected PostgreSQL / Supabase database.

Usage:
    .\\.venv\\Scripts\\python.exe init_db.py
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from sqlalchemy import inspect

load_dotenv()


def main() -> int:
    print("=" * 60)
    print("CodeLens AI - Database Initialization")
    print("=" * 60)

    try:
        from database import get_engine, resolve_database_url
        from models import Base

        # Import models so all tables register on Base.metadata
        import models  # noqa: F401

        url = resolve_database_url()
        # Redact password for console output
        safe_url = url
        if "@" in url:
            scheme, rest = url.split("://", 1)
            creds, host = rest.split("@", 1)
            user = creds.split(":", 1)[0]
            safe_url = f"{scheme}://{user}:***@{host}"

        print(f"Target: {safe_url}")
        engine = get_engine()

        print("Creating tables from SQLAlchemy metadata...")
        Base.metadata.create_all(bind=engine)

        inspector = inspect(engine)
        tables = sorted(inspector.get_table_names())
        expected = {"instructors", "assignments", "submissions", "grading_reports"}
        present = expected.intersection(tables)

        print("\nTables present in database:")
        for name in tables:
            marker = "OK" if name in expected else "  "
            print(f"  [{marker}] {name}")

        missing = expected - present
        if missing:
            print(f"\nFAILED — missing tables: {', '.join(sorted(missing))}")
            return 1

        print("\nSUCCESS — core CodeLens schema is ready.")
        print(
            "Tables: instructors, assignments, submissions, grading_reports"
        )
        return 0

    except EnvironmentError as exc:
        print(f"\nFAILED — configuration error:\n{exc}")
        return 1
    except Exception as exc:
        print(f"\nFAILED — could not create tables:\n{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
