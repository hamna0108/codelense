"""
CodeLens AI — .env diagnostic.

Run this from the SAME terminal / virtualenv you use to launch uvicorn:

    .\\.venv\\Scripts\\python.exe check_env.py

It tells you, in order:
  1. Which .env file python-dotenv actually found (if any)
  2. Whether SUPABASE_URL / SUPABASE_ANON_KEY / DATABASE_URL / GEMINI_API_KEY
     are visible to Python right now, and their length (never prints the
     full secret — just enough to confirm it's non-empty and roughly the
     right shape)
"""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv

print("=" * 60)
print("CodeLens AI - Environment Diagnostic")
print("=" * 60)

print(f"\nCurrent working directory:\n  {os.getcwd()}")

dotenv_path = find_dotenv(usecwd=False)
print(f"\n.env file dotenv would load (search from this script's location):")
print(f"  {dotenv_path or '(NOT FOUND — this is almost certainly the bug)'}")

loaded = load_dotenv(override=False)
print(f"\nload_dotenv() reported success: {loaded}")


def report(name: str, *, is_url: bool = False) -> None:
    value = os.getenv(name)
    if value is None:
        print(f"  {name:22s} -> NOT SET")
    elif not value.strip():
        print(f"  {name:22s} -> SET BUT EMPTY")
    else:
        v = value.strip()
        if is_url:
            preview = v
        else:
            preview = f"{v[:6]}...{v[-4:]} (length {len(v)})"
        print(f"  {name:22s} -> OK  {preview}")


print("\nVariables Python actually sees right now:")
report("GEMINI_API_KEY")
report("DATABASE_URL")
report("DIRECT_URL")
report("SUPABASE_URL", is_url=True)
report("SUPABASE_ANON_KEY")

print("\n" + "=" * 60)
if not dotenv_path:
    print(
        "-> dotenv could not find a .env file at all from this location.\n"
        "   Make sure a file literally named '.env' (not '.env.txt') sits\n"
        "   next to main.py/database.py, and that your editor didn't save\n"
        "   it with a hidden extension."
    )
elif not (os.getenv("SUPABASE_URL") or "").strip() or not (os.getenv("SUPABASE_ANON_KEY") or "").strip():
    print(
        "-> dotenv found a .env file, but SUPABASE_URL / SUPABASE_ANON_KEY\n"
        "   still didn't come through. Open that exact file in a plain text\n"
        "   viewer (not just VS Code's syntax-highlighted view) and check for:\n"
        "     - the variable name spelled slightly differently\n"
        "     - a stray space before/after the '=' sign\n"
        "     - the long JWT value accidentally containing a real line break\n"
        "       (common if it was pasted into an editor with word-wrap and\n"
        "       Enter was pressed, or copied from a source that inserted one)\n"
        "     - a second .env file elsewhere on the path being found instead"
    )
else:
    print(
        "-> Both variables are visible to Python right here, right now.\n"
        "   If /api/config still shows auth_configured=false in the browser,\n"
        "   the running uvicorn process was started BEFORE this and needs a\n"
        "   full stop + restart (not just --reload), or it's a different\n"
        "   venv/interpreter than the one running this script."
    )
print("=" * 60)