"""
CodeLens AI — In-memory ZIP extraction and per-student project grading orchestration.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ai_grader import AIEvaluationEngine
from models import Assignment, GradingReport, Submission, SubmissionStatus
from parser_engine import CodeBlueprintExtractor

_SKIP_PATH_MARKERS = (
    "__macosx",
    ".git/",
    ".git\\",
    "__pycache__",
    ".idea/",
    ".vscode/",
    ".ds_store",
)

_MAX_ZIP_BYTES = 25 * 1024 * 1024  # 25 MiB archive cap
_MAX_MEMBER_BYTES = 1 * 1024 * 1024  # 1 MiB per .py file


@dataclass
class ExtractedPythonFile:
    archive_path: str
    file_name: str
    student_identifier: str
    source_code: str


@dataclass
class StudentProject:
    student_identifier: str
    files: list[ExtractedPythonFile] = field(default_factory=list)

    @property
    # Combines all files belonging to this student into a single source payload
    def combined_source_code(self) -> str:
        return "\n\n".join(
            f"# === FILE: {f.file_name} ===\n{f.source_code}" for f in self.files
        )

    @property
    def file_names_str(self) -> str:
        return ", ".join(f.file_name for f in self.files)


@dataclass
class FileProcessingResult:
    archive_path: str
    student_identifier: str
    file_name: str
    success: bool
    submission_id: Optional[str] = None
    report_id: Optional[str] = None
    score: Optional[int] = None
    error: Optional[str] = None


@dataclass
class BulkProcessingSummary:
    assignment_id: str
    total_python_files: int
    graded: int = 0
    failed: int = 0
    results: list[FileProcessingResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _should_skip_member(name: str) -> bool:
    lowered = name.replace("\\", "/").lower()
    if not lowered.endswith(".py"):
        return True
    if any(marker in lowered for marker in _SKIP_PATH_MARKERS):
        return True
    base = PurePosixPath(lowered).name
    if base.startswith("."):
        return True
    return False


def _infer_student_identifier(archive_path: str) -> str:
    path = PurePosixPath(archive_path.replace("\\", "/"))
    parts = [p for p in path.parts if p not in ("", ".")]
    if len(parts) >= 2:
        if len(parts) >= 3 and parts[0].lower() in {"submissions", "students", "src"}:
            return parts[1]
        return parts[0]
    return path.stem or "unknown_student"


def extract_python_files_from_zip(zip_bytes: bytes) -> list[ExtractedPythonFile]:
    if not zip_bytes:
        raise ValueError("Uploaded ZIP archive is empty")
    if len(zip_bytes) > _MAX_ZIP_BYTES:
        raise ValueError(
            f"ZIP archive exceeds {_MAX_ZIP_BYTES // (1024 * 1024)} MiB size limit"
        )

    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("Uploaded file is not a valid ZIP archive") from exc

    extracted: list[ExtractedPythonFile] = []
    with archive:
        for info in archive.infolist():
            if info.is_dir() or _should_skip_member(info.filename):
                continue
            if info.file_size > _MAX_MEMBER_BYTES:
                raise ValueError(
                    f"Member {info.filename!r} exceeds "
                    f"{_MAX_MEMBER_BYTES // (1024 * 1024)} MiB size limit"
                )

            normalized = PurePosixPath(info.filename.replace("\\", "/"))
            if ".." in normalized.parts:
                continue

            raw = archive.read(info)
            try:
                source = raw.decode("utf-8")
            except UnicodeDecodeError:
                source = raw.decode("latin-1")

            extracted.append(
                ExtractedPythonFile(
                    archive_path=info.filename.replace("\\", "/"),
                    file_name=normalized.name,
                    student_identifier=_infer_student_identifier(info.filename),
                    source_code=source,
                )
            )

    return extracted


def process_bulk_submissions(
    db: Session,
    *,
    assignment: Assignment,
    zip_bytes: bytes,
    grader: Optional[AIEvaluationEngine] = None,
) -> BulkProcessingSummary:
    """
    Extract files, group them by student folder/identifier into single projects,
    and grade each student project as a unified codebase.
    """
    files = extract_python_files_from_zip(zip_bytes)
    
    # Group files by student identifier
    student_map: dict[str, StudentProject] = {}
    for f in files:
        if f.student_identifier not in student_map:
            student_map[f.student_identifier] = StudentProject(student_identifier=f.student_identifier)
        student_map[f.student_identifier].files.append(f)

    summary = BulkProcessingSummary(
        assignment_id=str(assignment.id),
        total_python_files=len(files),
    )
    engine = grader or AIEvaluationEngine()
    rubric = assignment.rubric_text

    for student_id, project in student_map.items():
        result = _process_student_project(
            db,
            assignment_id=assignment.id,
            project=project,
            rubric=rubric,
            engine=engine,
        )
        summary.results.append(result)
        if result.success:
            summary.graded += 1
        else:
            summary.failed += 1

    return summary


def _process_student_project(
    db: Session,
    *,
    assignment_id: UUID,
    project: StudentProject,
    rubric: str,
    engine: AIEvaluationEngine,
) -> FileProcessingResult:
    primary_file_name = project.files[0].file_name if len(project.files) == 1 else f"{project.student_identifier}_project"
    archive_path_repr = project.files[0].archive_path

    submission = Submission(
        assignment_id=assignment_id,
        student_identifier=project.student_identifier,
        source_code_path=f"zip://multi_file_project",
        file_name=primary_file_name,
        status=SubmissionStatus.pending,
    )
    try:
        db.add(submission)
        db.commit()
        db.refresh(submission)
    except IntegrityError as exc:
        db.rollback()
        return FileProcessingResult(
            archive_path=archive_path_repr,
            student_identifier=project.student_identifier,
            file_name=primary_file_name,
            success=False,
            error=f"IntegrityError: submission already exists ({exc.orig})",
        )
    except Exception as exc:
        db.rollback()
        return FileProcessingResult(
            archive_path=archive_path_repr,
            student_identifier=project.student_identifier,
            file_name=primary_file_name,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    submission_id = submission.id

    try:
        # Analyze the combined source code of all files in the student's project folder
        combined_code = project.combined_source_code
        blueprint = CodeBlueprintExtractor.analyze(combined_code)
        
        evaluation = engine.evaluate_submission(blueprint, rubric)
        metrics = blueprint.get("design_metrics") or {}

        report = GradingReport(
            submission_id=submission_id,
            total_score=int(evaluation["score"]),
            structural_metrics=metrics,
            requirements_checked=evaluation.get("requirements_checked") or [],
            pedagogical_feedback=evaluation.get("pedagogical_feedback") or "",
        )
        db.add(report)

        db_submission = db.get(Submission, submission_id)
        db_submission.status = SubmissionStatus.graded
        db.commit()
        db.refresh(report)

        return FileProcessingResult(
            archive_path=archive_path_repr,
            student_identifier=project.student_identifier,
            file_name=primary_file_name,
            success=True,
            submission_id=str(submission_id),
            report_id=str(report.id),
            score=report.total_score,
        )
    except Exception as exc:
        db.rollback()
        try:
            db_submission = db.get(Submission, submission_id)
            if db_submission is not None:
                db_submission.status = SubmissionStatus.failed
                db.commit()
        except Exception:
            db.rollback()

        return FileProcessingResult(
            archive_path=archive_path_repr,
            student_identifier=project.student_identifier,
            file_name=primary_file_name,
            success=False,
            submission_id=str(submission_id),
            error=f"{type(exc).__name__}: {exc}",
        )