"""
CodeLens AI — LLM-backed pedagogical evaluation engine.
Grades structural blueprints against a rubric using Google Gemini
with strict JSON / structured-output enforcement.
"""

from __future__ import annotations

import json
import os
import warnings
from enum import Enum
from typing import Any, Literal

# google.generativeai is deprecated upstream but still requested for this module.
warnings.filterwarnings(
    "ignore",
    message=r".*google\.generativeai.*",
    category=FutureWarning,
)

import google.generativeai as genai
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# Load .env if present; existing process env vars are not overridden.
load_dotenv()

# gemini-1.5-flash has been retired. Use the current Flash alias so the
# free-tier endpoint stays available as Google rotates model IDs.
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"


SYSTEM_PROMPT = """\
You are an expert, highly encouraging university Teaching Assistant grading \
an Object-Oriented Programming or Data Structures lab submission.

ROLE & TONE
- Be rigorous about structural compliance, but supportive and mentorship-focused.
- Write pedagogical_feedback in Markdown: celebrate what was done well, then \
  give concrete next steps for design patterns, encapsulation, and clarity.
- Never shame the student; coach them toward better architecture.

EVIDENCE RULES (NON-NEGOTIABLE)
- You MUST judge the submission using ONLY the factual data in the provided \
  structural blueprint JSON. That blueprint was extracted statically from the \
  student's source via AST analysis.
- NEVER invent, assume, or hallucinate classes, methods, arguments, inheritance \
  edges, recursion flags, or metrics that are not present in the blueprint.
- If a rubric item cannot be verified from the blueprint, mark it "unmet" or \
  "partial" and explain which blueprint fields were missing — do not guess.
- Cite specific class names, method names, bases, and design_metrics fields \
  in every technical_rationale.

SCORING
- Start from 100 and apply integer deductions per unmet/partial requirement.
- "met" => deduction 0; "partial" => modest deduction; "unmet" => larger deduction.
- Final score must be an integer clamped to [0, 100] and consistent with the \
  sum of deductions (score ≈ 100 - sum(deductions), floored at 0).

OUTPUT
- Return ONLY a response that conforms to the required JSON schema.
- Do not include prose outside the schema fields.
"""


class RequirementStatus(str, Enum):
    met = "met"
    unmet = "unmet"
    partial = "partial"


class RequirementCheck(BaseModel):
    requirement_name: str = Field(..., description="Short name of the rubric requirement")
    status: Literal["met", "unmet", "partial"] = Field(
        ..., description="Whether the requirement is met, unmet, or partially met"
    )
    deduction: int = Field(..., description="Points deducted for this item (0-100)")
    technical_rationale: str = Field(
        ...,
        description="Evidence-based explanation citing blueprint classes/methods/metrics",
    )

    @field_validator("deduction")
    @classmethod
    def clamp_deduction(cls, value: int) -> int:
        return max(0, min(100, int(value)))


class EvaluationResult(BaseModel):
    score: int = Field(..., description="Overall score from 0 to 100")
    requirements_checked: list[RequirementCheck] = Field(
        ..., description="Per-requirement grading outcomes"
    )
    pedagogical_feedback: str = Field(
        ...,
        description="Markdown mentorship feedback on structure and design patterns",
    )

    @field_validator("score")
    @classmethod
    def clamp_score(cls, value: int) -> int:
        return max(0, min(100, int(value)))


def _gemini_compatible_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Convert a Pydantic model into a Gemini response_schema dict.

    google.generativeai's Schema proto rejects several JSON Schema keywords
    that Pydantic emits (minimum/maximum, $defs/$ref, additionalProperties).
    This inlines refs and strips unsupported keys while preserving structure.
    """
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {}) or raw.pop("definitions", {}) or {}

    def _resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        if not isinstance(node, dict):
            return node

        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            return _resolve(defs[ref_name])

        allowed = {
            "type",
            "format",
            "description",
            "properties",
            "required",
            "items",
            "enum",
            "nullable",
        }
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key not in allowed:
                continue
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {k: _resolve(v) for k, v in value.items()}
            elif key == "items":
                cleaned[key] = _resolve(value)
            else:
                cleaned[key] = _resolve(value)

        # Gemini expects uppercase type names in some SDK paths; leave as-is
        # when already lowercase JSON Schema — the SDK normalizes them.
        return cleaned

    return _resolve(raw)


class AIEvaluationEngine:
    """
    LLM evaluation engine for CodeLens structural blueprints.

    Uses Google Gemini (gemini-1.5-flash) with GEMINI_API_KEY and
    structured JSON output via response_mime_type + response_schema.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.provider = "gemini"

        if not self.api_key:
            raise EnvironmentError(
                "No LLM API key found. Set GEMINI_API_KEY in the environment "
                "or a .env file."
            )

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT,
        )

    def evaluate_submission(self, blueprint: dict[str, Any], rubric: str) -> dict[str, Any]:
        """
        Grade a structural blueprint against a pedagogical rubric.

        Returns a plain dict matching EvaluationResult's JSON schema.
        """
        if not isinstance(blueprint, dict):
            raise TypeError("blueprint must be a dict")
        if blueprint.get("success") is False:
            raise ValueError(
                "Cannot evaluate a failed blueprint. "
                f"Parser error: {blueprint.get('error')}"
            )
        if not rubric or not rubric.strip():
            raise ValueError("rubric must be a non-empty string")

        user_prompt = self._build_user_prompt(blueprint, rubric)
        result = self._evaluate_gemini(user_prompt)
        return result.model_dump()

    @staticmethod
    def _build_user_prompt(blueprint: dict[str, Any], rubric: str) -> str:
        return (
            "## Grading Rubric\n"
            f"{rubric.strip()}\n\n"
            "## Structural Blueprint (AST-extracted facts — sole evidence source)\n"
            "```json\n"
            f"{json.dumps(blueprint, indent=2)}\n"
            "```\n\n"
            "Evaluate the blueprint against every rubric requirement. "
            "Use only the JSON facts above."
        )

    def _evaluate_gemini(self, user_prompt: str) -> EvaluationResult:
        """Call Gemini with strict JSON schema enforcement, then validate via Pydantic."""
        generation_config = genai.GenerationConfig(
            temperature=0.2,
            response_mime_type="application/json",
            # Schema derived from our Pydantic EvaluationResult / RequirementCheck models
            response_schema=_gemini_compatible_schema(EvaluationResult),
        )

        response = self._model.generate_content(
            user_prompt,
            generation_config=generation_config,
        )

        raw_text = (response.text or "").strip()
        if not raw_text:
            raise RuntimeError("Gemini returned an empty response")

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini returned non-JSON content: {raw_text[:500]}"
            ) from exc

        return EvaluationResult.model_validate(payload)
