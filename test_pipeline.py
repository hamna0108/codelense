"""
CodeLens AI — End-to-end integration test for the bulk ZIP upload pipeline.

What this does:
  1. Creates (or reuses) a test Instructor + a fresh Assignment in Supabase.
  2. Builds an in-memory ZIP containing 2 sample student .py submissions.
  3. Runs process_bulk_submissions() directly against the live DB session.
  4. Queries Supabase to confirm rows landed in `submissions` and
     `grading_reports`, and prints them.
  5. Cleans up the Assignment it created (cascades to Submissions +
     GradingReports) so re-running this script doesn't pile up test data.

Usage:
    .\\.venv\\Scripts\\python.exe test_pipeline.py

By default this uses a lightweight FAKE grader (no Gemini call, no
GEMINI_API_KEY required) so the test is fast, free, and deterministic —
it's testing the *pipeline/persistence* logic, not Gemini's grading quality.

Set CODELENS_TEST_USE_REAL_GEMINI=1 to instead exercise the real
AIEvaluationEngine (requires GEMINI_API_KEY in your .env).
"""

from __future__ import annotations

import io
import os
import sys
import uuid
import zipfile
from typing import Any

from sqlalchemy import select

from database import get_session_factory
from models import Assignment, GradingReport, Instructor, Submission, SubmissionStatus
from submission_pipeline import process_bulk_submissions

TEST_INSTRUCTOR_EMAIL = "codelens-pipeline-test@example.com"

STUDENT_FILE_ALICE = '''
"""Alice's submission — simple inheritance + polymorphism."""


class Shape:
    def __init__(self, name):
        self.name = name

    def describe(self):
        return f"{self.name} is a shape"


class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius

    def describe(self):
        return f"{self.name} with radius {self.radius}"


def total_area(circles):
    total = 0
    for c in circles:
        total = total + (3.14159 * c.radius * c.radius)
    return total
'''

STUDENT_FILE_BOB = '''
"""Bob's submission — recursion example."""


class MathHelper:
    def factorial(self, n):
        if n <= 1:
            return 1
        return n * self.factorial(n - 1)

    def is_even(self, n):
        if n % 2 == 0:
            return True
        return False


def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
'''

SAMPLE_RUBRIC = """
Lab Rubric — OOP & Recursion Foundations
1. Base class with at least one subclass.
2. At least one recursive method or function.
3. Reasonable encapsulation via self.
"""


class FakeGradingEngine:
    """
    Drop-in stand-in for AIEvaluationEngine.evaluate_submission().

    Returns a canned, schema-shaped result so the pipeline/persistence layer
    can be tested without a network call or API key.
    """

    provider = "fake"

    def evaluate_submission(self, blueprint: dict[str, Any], rubric: str) -> dict[str, Any]:
        classes = blueprint.get("classes", [])
        return {
            "score": 88,
            "requirements_checked": [
                {
                    "requirement_name": "Base class with subclass",
                    "status": "met" if len(classes) >= 1 else "unmet",
                    "deduction": 0,
                    "technical_rationale": f"Blueprint reports {len(classes)} class(es).",
                }
            ],
            "pedagogical_feedback": "Nice structure — fake grader output for pipeline testing.",
        }


def build_test_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("alice/submission.py", STUDENT_FILE_ALICE)
        zf.writestr("bob/submission.py", STUDENT_FILE_BOB)
    return buf.getvalue()


def get_or_create_test_instructor(db) -> Instructor:
    instructor = db.scalar(
        select(Instructor).where(Instructor.email == TEST_INSTRUCTOR_EMAIL)
    )
    if instructor is None:
        instructor = Instructor(email=TEST_INSTRUCTOR_EMAIL, name="Pipeline Test Instructor")
        db.add(instructor)
        db.commit()
        db.refresh(instructor)
        print(f"Created test instructor: {instructor.email} ({instructor.id})")
    else:
        print(f"Reusing existing test instructor: {instructor.email} ({instructor.id})")
    return instructor


def create_test_assignment(db, instructor: Instructor) -> Assignment:
    assignment = Assignment(
        instructor_id=instructor.id,
        title=f"Pipeline Smoke Test {uuid.uuid4().hex[:8]}",
        rubric_text=SAMPLE_RUBRIC,
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    print(f"Created test assignment: {assignment.title} ({assignment.id})")
    return assignment


def main() -> int:
    print("=" * 60)
    print("CodeLens AI - Bulk Upload Pipeline Integration Test")
    print("=" * 60)

    SessionLocal = get_session_factory()
    db = SessionLocal()

    assignment: Assignment | None = None
    exit_code = 0

    try:
        instructor = get_or_create_test_instructor(db)
        assignment = create_test_assignment(db, instructor)

        zip_bytes = build_test_zip()
        print(f"\nBuilt in-memory ZIP: {len(zip_bytes)} bytes, 2 student files")

        use_real_gemini = os.getenv("CODELENS_TEST_USE_REAL_GEMINI") == "1"
        if use_real_gemini:
            from ai_grader import AIEvaluationEngine

            print("Using REAL AIEvaluationEngine (Gemini) — requires GEMINI_API_KEY")
            grader = AIEvaluationEngine()
        else:
            print("Using FakeGradingEngine (no Gemini call, no API key needed)")
            grader = FakeGradingEngine()

        print("\n--- Running process_bulk_submissions() ---")
        summary = process_bulk_submissions(
            db,
            assignment=assignment,
            zip_bytes=zip_bytes,
            grader=grader,
        )

        print(f"\nTotal files:  {summary.total_python_files}")
        print(f"Graded:       {summary.graded}")
        print(f"Failed:       {summary.failed}")
        for r in summary.results:
            status_tag = "OK" if r.success else "FAIL"
            print(
                f"  [{status_tag}] {r.student_identifier}/{r.file_name} "
                f"submission_id={r.submission_id} score={r.score} error={r.error}"
            )

        assert summary.total_python_files == 2, "expected 2 python files in test zip"
        assert summary.graded == 2, f"expected 2 graded, got {summary.graded}"
        assert summary.failed == 0, f"expected 0 failed, got {summary.failed}"

        print("\n--- Verifying rows directly in Supabase ---")
        db.expire_all()
        submissions = (
            db.scalars(
                select(Submission).where(Submission.assignment_id == assignment.id)
            )
            .all()
        )
        assert len(submissions) == 2, f"expected 2 submissions in DB, found {len(submissions)}"

        for sub in submissions:
            print(
                f"  Submission {sub.id} | student={sub.student_identifier} | "
                f"status={sub.status.value}"
            )
            assert sub.status == SubmissionStatus.graded, (
                f"expected status=Graded for {sub.student_identifier}, "
                f"got {sub.status.value}"
            )

            report = db.scalar(
                select(GradingReport).where(GradingReport.submission_id == sub.id)
            )
            assert report is not None, f"missing GradingReport for submission {sub.id}"
            print(
                f"    GradingReport {report.id} | score={report.total_score} | "
                f"requirements={len(report.requirements_checked)}"
            )

        print("\n" + "=" * 60)
        print("SUCCESS — Submissions and GradingReports verified in Supabase.")
        print("=" * 60)

    except AssertionError as exc:
        print(f"\nFAILED — assertion error: {exc}")
        exit_code = 1
    except EnvironmentError as exc:
        print(f"\nFAILED — configuration error: {exc}")
        exit_code = 1
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED — unexpected error: {type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        # Cleanup: deleting the Assignment cascades to Submissions and
        # GradingReports (ondelete="CASCADE" in models.py), so re-running
        # this script repeatedly won't accumulate test data. The test
        # Instructor is left in place so it can be reused next run.
        if assignment is not None:
            try:
                db.delete(assignment)
                db.commit()
                print(f"\nCleaned up test assignment {assignment.id}")
            except Exception as cleanup_exc:  # noqa: BLE001
                db.rollback()
                print(f"\nWARNING — cleanup failed: {cleanup_exc}")
        db.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())