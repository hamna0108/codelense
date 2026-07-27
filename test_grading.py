"""
CodeLens AI — Integration test: AST blueprint -> AI pedagogical grader.
"""

from __future__ import annotations

import json
import sys

from ai_grader import AIEvaluationEngine
from parser_engine import CodeBlueprintExtractor


SAMPLE_STUDENT_CODE = '''
"""Student homework — animal zoo simulator (messy but intentional)."""

import math   # unused on purpose


class Animal:
    """Base creature in the zoo."""

    def __init__(self, name, energy=100):
        self.name = name
        self.energy = energy
        self.tags = []

    def speak(self):
        return "..."

    def rest(self, hours):
        recovered = hours * 10
        self.energy = self.energy + recovered
        if self.energy > 100:
            self.energy = 100
        return self.energy


class Dog(Animal):
    """A dog that can bark and fetch recursively."""

    def __init__(self, name, breed="mutt"):
        super().__init__(name)
        self.breed = breed
        self.toys = []

    def speak(self):  # polymorphism override
        bark = "Woof"
        for i in range(3):
            bark = bark + "!"
        return bark

    def fetch(self, distance):
        """Recursive fetch — keeps going until distance is tiny."""
        if distance <= 0:
            return 0
        steps = 1
        remaining = distance - 1
        return steps + self.fetch(remaining)

    def train(self, tricks):
        learned = []
        try:
            for trick in tricks:
                if trick not in learned:
                    learned.append(trick)
                    self.toys.append(trick)
        except TypeError:
            return []
        while len(learned) > 5:
            learned.pop()
        return learned


class Cat(Animal):
    def speak(self):
        mood = "meow"
        if self.energy < 30:
            mood = "hiss"
        elif self.energy > 80:
            mood = "purr"
        return mood


def feed_all(animals, food_amount=5):
    """Standalone helper — feed every animal in the list."""
    fed = 0
    for animal in animals:
        if animal.energy < 90:
            animal.energy = animal.energy + food_amount
            fed = fed + 1
    return fed


def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''


SAMPLE_RUBRIC = """
Lab Rubric — OOP & Recursion Foundations

1. Base class with inheritance: Must define a clear base class and at least one
   subclass that inherits from it (inheritance chain visible in the blueprint).
2. Recursion: Must use recursion at least once (a method or function marked
   is_recursive=true in the blueprint).
3. Method complexity: Prefer methods with low branching; flag any method whose
   local structure suggests high cyclomatic complexity (e.g. nested loops +
   try/except + while). Overall module cyclomatic_complexity_approx should
   ideally stay modest; deduct if individual methods appear overly branched.
4. Encapsulation: Prefer methods that operate on instance state via self and
   avoid exposing everything as loose globals. Deduct if critical behavior is
   only in standalone functions with no class ownership, or if constructors
   are missing where inheritance hierarchies expect them.
5. Polymorphism: Prefer overridden methods with the same name across the
   inheritance hierarchy (e.g. speak on base and subclasses).
"""


def main() -> None:
    print("=" * 60)
    print("CodeLens AI - Grading Integration Verification")
    print("=" * 60)

    blueprint = CodeBlueprintExtractor.analyze(SAMPLE_STUDENT_CODE)
    if not blueprint.get("success"):
        print("Parser failed:", json.dumps(blueprint, indent=2))
        sys.exit(1)

    print("\n--- Blueprint metrics ---")
    print(json.dumps(blueprint["design_metrics"], indent=2))

    print("\n--- Invoking AIEvaluationEngine ---")
    try:
        engine = AIEvaluationEngine()
    except EnvironmentError as exc:
        print(f"ERROR: {exc}")
        print(
            "\nCreate a .env file in the project root with:\n"
            "  GEMINI_API_KEY=your-google-ai-studio-key"
        )
        sys.exit(1)

    print(f"Provider: {engine.provider}")
    evaluation = engine.evaluate_submission(blueprint, SAMPLE_RUBRIC)

    print("\n--- Raw evaluation JSON ---")
    print(json.dumps(evaluation, indent=2))
    print("=" * 60)
    print(f"Score: {evaluation['score']}/100")


if __name__ == "__main__":
    main()
