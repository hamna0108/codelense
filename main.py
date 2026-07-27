"""
CodeLens AI — FastAPI application: auth-protected assignment management,
bulk submission ingestion, and static frontend serving.

Auth model
----------
The frontend authenticates directly against Supabase Auth (via the
supabase-js SDK) and holds a session there. It sends the resulting access
token on every API call as:

    Authorization: Bearer <supabase_access_token>

`get_current_instructor` verifies that token against Supabase's Auth API
(GoTrue) on every request — no local JWT secret needed — and transparently
provisions a matching `instructors` row on first use, keyed by the same UUID
Supabase Auth uses for the user. This is what lets us "eliminate all manual
database inserts": nobody ever runs SQL to create an instructor row by hand.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database import get_db, get_supabase_client
from models import Assignment, GradingReport, Instructor
from submission_pipeline import process_bulk_submissions

app = FastAPI(
    title="CodeLens AI",
    description="Structural code grading and pedagogical analysis platform",
    version="0.3.0",
)

# index.html is expected to sit next to this file (same directory).
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"


@app.on_event("startup")
def _print_config_status_on_boot() -> None:
    """
    Print exactly what this running process sees for the Supabase config,
    straight to the uvicorn console, the moment it starts. No browser, no
    curl, no caching — if this print is wrong, the .env truly isn't being
    picked up by THIS process; if it's right here but the browser still
    disagrees, the browser is talking to something other than this server.
    """
    url = (os.getenv("SUPABASE_URL") or "").strip()
    anon_key = (os.getenv("SUPABASE_ANON_KEY") or "").strip()
    print("=" * 60)
    print("CodeLens AI - startup config check")
    print(f"  SUPABASE_URL      : {url or '(MISSING)'}")
    print(f"  SUPABASE_ANON_KEY : {'set (' + str(len(anon_key)) + ' chars)' if anon_key else '(MISSING)'}")
    print(f"  auth_configured   : {bool(url and anon_key)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def get_current_instructor(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> Instructor:
    """
    Resolve the calling instructor from a Supabase Auth access token.

    Verifies the bearer token against Supabase's Auth API (this makes a
    network call to Supabase, but requires no local JWT secret and always
    reflects live session/revocation state). Auto-provisions the matching
    `instructors` row on first sight of a given Supabase user id.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token. Include 'Authorization: Bearer <access_token>'.",
        )

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token.",
        )

    try:
        supabase = get_supabase_client()
        user_response = supabase.auth.get_user(token)
        user = getattr(user_response, "user", None)
    except Exception as exc:  # noqa: BLE001 — any verification failure = unauthorized
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session. Please log in again.",
        ) from exc

    if user is None or not getattr(user, "id", None) or not getattr(user, "email", None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session. Please log in again.",
        )

    instructor = db.get(Instructor, user.id)
    if instructor is not None:
        return instructor

    # First time we've seen this Supabase user — provision their row now.
    metadata = getattr(user, "user_metadata", None) or {}
    display_name = metadata.get("name") or metadata.get("full_name") or user.email.split("@")[0]

    instructor = Instructor(id=user.id, email=user.email, name=display_name)
    db.add(instructor)
    try:
        db.commit()
    except IntegrityError:
        # Lost a race with a concurrent request provisioning the same row.
        db.rollback()
        instructor = db.get(Instructor, user.id)
        if instructor is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not provision instructor account.",
            )
    else:
        db.refresh(instructor)

    return instructor


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PublicConfigOut(BaseModel):
    supabase_url: str | None = None
    supabase_anon_key: str | None = None
    auth_configured: bool


class AssignmentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    rubric_text: str = Field(..., min_length=1)


class AssignmentOut(BaseModel):
    id: str
    title: str
    rubric_text: str
    created_at: str


class FileResultOut(BaseModel):
    archive_path: str
    student_identifier: str
    file_name: str
    success: bool
    submission_id: str | None = None
    report_id: str | None = None
    score: int | None = None
    error: str | None = None


class BulkUploadResponse(BaseModel):
    assignment_id: str
    instructor_email: str
    total_python_files: int
    graded: int
    failed: int
    results: list[FileResultOut] = Field(default_factory=list)


class GradingReportOut(BaseModel):
    report_id: str
    submission_id: str
    student_identifier: str
    file_name: str
    score: int
    status: str
    structural_metrics: dict[str, Any]
    requirements_checked: list[dict[str, Any]]
    pedagogical_feedback: str
    generated_at: str


# ---------------------------------------------------------------------------
# Health / config / static frontend
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config", response_model=PublicConfigOut, include_in_schema=False)
def public_config(response: Response) -> PublicConfigOut:
    """
    Public, non-secret configuration the frontend needs to initialize the
    Supabase JS client. Only ever returns the ANON/publishable key — never
    the service role key — since this response is readable by anyone.

    Explicitly marked no-store: this must never be served stale from a
    browser cache after an operator sets/changes the env vars and restarts.
    """
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    url = (os.getenv("SUPABASE_URL") or "").strip() or None
    anon_key = (os.getenv("SUPABASE_ANON_KEY") or "").strip() or None
    return PublicConfigOut(
        supabase_url=url,
        supabase_anon_key=anon_key,
        auth_configured=bool(url and anon_key),
    )


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    """Serve the single-page CodeLens AI frontend at the site root."""
    if not INDEX_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="index.html not found next to main.py",
        )
    return FileResponse(INDEX_FILE)


# ---------------------------------------------------------------------------
# Assignments
# ---------------------------------------------------------------------------


@app.get("/api/assignments", response_model=list[AssignmentOut])
def list_assignments(
    current_instructor: Instructor = Depends(get_current_instructor),
    db: Session = Depends(get_db),
) -> list[AssignmentOut]:
    """List assignments owned by the logged-in instructor, newest first."""
    rows = db.scalars(
        select(Assignment)
        .where(Assignment.instructor_id == current_instructor.id)
        .order_by(Assignment.created_at.desc())
    ).all()
    return [
        AssignmentOut(
            id=str(a.id),
            title=a.title,
            rubric_text=a.rubric_text,
            created_at=a.created_at.isoformat(),
        )
        for a in rows
    ]


@app.post(
    "/api/assignments",
    response_model=AssignmentOut,
    status_code=status.HTTP_201_CREATED,
)
def create_assignment(
    payload: AssignmentCreate,
    current_instructor: Instructor = Depends(get_current_instructor),
    db: Session = Depends(get_db),
) -> AssignmentOut:
    """Create a new assignment owned by the logged-in instructor — no SQL required."""
    assignment = Assignment(
        instructor_id=current_instructor.id,
        title=payload.title.strip(),
        rubric_text=payload.rubric_text.strip(),
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return AssignmentOut(
        id=str(assignment.id),
        title=assignment.title,
        rubric_text=assignment.rubric_text,
        created_at=assignment.created_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Bulk upload + grading
# ---------------------------------------------------------------------------


@app.post(
    "/api/assignments/{assignment_id}/upload-bulk",
    response_model=BulkUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_bulk_submissions(
    assignment_id: UUID,
    file: UploadFile = File(..., description="ZIP archive of student .py submissions"),
    current_instructor: Instructor = Depends(get_current_instructor),
    db: Session = Depends(get_db),
) -> BulkUploadResponse:
    """
    Ingest a ZIP of student Python files, grade each in-memory via AST + Gemini,
    and persist Submissions + GradingReports.

    Ownership of the assignment is verified against the authenticated
    instructor's session — there is no separate instructor_email form field
    to trust anymore; the bearer token is the only source of identity.
    """
    filename = (file.filename or "").lower()
    if not filename.endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a .zip archive",
        )

    assignment = db.get(Assignment, assignment_id)
    if assignment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assignment {assignment_id} not found",
        )
    if assignment.instructor_id != current_instructor.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not own this assignment",
        )

    zip_bytes = await file.read()
    try:
        summary = process_bulk_submissions(db, assignment=assignment, zip_bytes=zip_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except EnvironmentError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return BulkUploadResponse(
        assignment_id=str(assignment.id),
        instructor_email=current_instructor.email,
        total_python_files=summary.total_python_files,
        graded=summary.graded,
        failed=summary.failed,
        results=[FileResultOut(**r) for r in summary.to_dict()["results"]],
    )


@app.get(
    "/api/reports/{report_id}",
    response_model=GradingReportOut,
)
def get_report(
    report_id: UUID,
    current_instructor: Instructor = Depends(get_current_instructor),
    db: Session = Depends(get_db),
) -> GradingReportOut:
    """
    Fetch the full grading report (pedagogical feedback, per-requirement
    breakdown, and structural metrics) for the feedback modal. Restricted to
    the instructor who owns the parent assignment.
    """
    report = db.get(GradingReport, report_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Grading report {report_id} not found",
        )

    submission = report.submission
    if submission.assignment.instructor_id != current_instructor.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this report",
        )

    return GradingReportOut(
        report_id=str(report.id),
        submission_id=str(submission.id),
        student_identifier=submission.student_identifier,
        file_name=submission.file_name,
        score=report.total_score,
        status=submission.status.value,
        structural_metrics=report.structural_metrics,
        requirements_checked=report.requirements_checked,
        pedagogical_feedback=report.pedagogical_feedback,
        generated_at=report.generated_at.isoformat(),
    )