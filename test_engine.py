"""
CodeLens AI — Verification script for CodeBlueprintExtractor.
Parses a messy student-style OOP sample and prints the architectural JSON.
"""

from __future__ import annotations

import json

from parser_engine import CodeBlueprintExtractor


# Messy sample student script: inheritance, polymorphism, loops, recursion.
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
        # recurse via self.fetch
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


def broken_syntax_demo():
    x = 1
    return x
'''


def main() -> None:
    print("=" * 60)
    print("CodeLens AI - Blueprint Extraction Verification")
    print("=" * 60)

    result = CodeBlueprintExtractor.analyze(SAMPLE_STUDENT_CODE)

    print(json.dumps(result, indent=2))

    print("=" * 60)
    if result.get("success"):
        metrics = result["design_metrics"]
        print(
            f"OK — {metrics['total_classes']} classes, "
            f"{metrics['total_standalone_functions']} standalone functions, "
            f"{metrics['total_methods']} methods, "
            f"complexity~={metrics['cyclomatic_complexity_approx']}, "
            f"{metrics['total_lines_of_code']} LOC"
        )
    else:
        print("FAILED —", result.get("error"))

    # Also verify SyntaxError handling
    print("\n--- SyntaxError handling check ---")
    bad = CodeBlueprintExtractor.analyze("def broken(\n  pass")
    print(json.dumps(bad, indent=2))


if __name__ == "__main__":
    main()
