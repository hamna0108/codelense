"""
CodeLens AI — ORM domain models for the educational SaaS platform.
SQLAlchemy 2.0 declarative mappings with UUID keys and JSON columns.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Shared declarative base for all CodeLens ORM models."""


# Use JSONB on PostgreSQL; fall back to generic JSON elsewhere (e.g. local SQLite).
JsonType = JSON().with_variant(JSONB(), "postgresql")


class SubmissionStatus(str, enum.Enum):
    pending = "Pending"
    graded = "Graded"
    failed = "Failed"


class Instructor(Base):
    """
    Course instructor / educator account.

    `id` is intentionally a string UUID so it can mirror Supabase Auth `auth.users.id`.
    """

    __tablename__ = "instructors"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=utcnow,
    )

    assignments: Mapped[list[Assignment]] = relationship(
        back_populates="instructor",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Instructor id={self.id!r} email={self.email!r}>"


class Assignment(Base):
    """Lab / homework assignment owned by an instructor, including the grading rubric."""

    __tablename__ = "assignments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    instructor_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("instructors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    rubric_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_oop_rules: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=utcnow,
    )

    instructor: Mapped[Instructor] = relationship(back_populates="assignments")
    submissions: Mapped[list[Submission]] = relationship(
        back_populates="assignment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Assignment id={self.id!r} title={self.title!r}>"


class Submission(Base):
    """A student's uploaded source submission against a specific assignment."""

    __tablename__ = "submissions"
    __table_args__ = (
        UniqueConstraint(
            "assignment_id",
            "student_identifier",
            "file_name",
            name="uq_submission_assignment_student_file",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    assignment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("assignments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    student_identifier: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    source_code_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[SubmissionStatus] = mapped_column(
        Enum(
            SubmissionStatus,
            name="submission_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=SubmissionStatus.pending,
        server_default=SubmissionStatus.pending.value,
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=utcnow,
    )

    assignment: Mapped[Assignment] = relationship(back_populates="submissions")
    grading_report: Mapped[Optional[GradingReport]] = relationship(
        back_populates="submission",
        cascade="all, delete-orphan",
        uselist=False,
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"<Submission id={self.id!r} student={self.student_identifier!r} "
            f"status={self.status.value!r}>"
        )


class GradingReport(Base):
    """
    One-to-one AI grading artifact for a submission.

    `requirements_checked` mirrors the Gemini EvaluationResult schema;
    `structural_metrics` stores AST design metrics (LOC, complexity, counts).
    """

    __tablename__ = "grading_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    submission_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    total_score: Mapped[int] = mapped_column(Integer, nullable=False)
    structural_metrics: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False)
    requirements_checked: Mapped[list[dict[str, Any]]] = mapped_column(JsonType, nullable=False)
    pedagogical_feedback: Mapped[str] = mapped_column(Text, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=utcnow,
    )

    submission: Mapped[Submission] = relationship(back_populates="grading_report")

    def __repr__(self) -> str:
        return f"<GradingReport id={self.id!r} score={self.total_score}>"