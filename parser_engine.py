"""
CodeLens AI — Core AST Analysis Engine
Deterministic structural extraction from Python source without execution.
"""

from __future__ import annotations

import ast
import json
from typing import Any


class _LocalVarCollector(ast.NodeVisitor):
    """Collect names assigned in a function body (excluding nested defs)."""

    def __init__(self) -> None:
        self.locals: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.locals.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        pass


class _ReturnChecker(ast.NodeVisitor):
    """Detect presence of a return statement within a function body."""

    def __init__(self) -> None:
        self.has_return = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Return(self, node: ast.Return) -> None:
        self.has_return = True


class _RecursionChecker(ast.NodeVisitor):
    """Detect whether a function calls itself by name."""

    def __init__(self, func_name: str) -> None:
        self.func_name = func_name
        self.is_recursive = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == self.func_name:
            self.is_recursive = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr == self.func_name:
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id == "self":
                self.is_recursive = True
        self.generic_visit(node)


class _StatementInspector(ast.NodeVisitor):
    """Extract internal structural elements like instance variables and statement types."""

    def __init__(self) -> None:
        self.instance_variables: set[str] = set()
        self.statements_used: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                self.instance_variables.add(target.attr)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        target = node.target
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            self.instance_variables.add(target.attr)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.statements_used.add("If")
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.statements_used.add("Try")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.statements_used.add("Loop")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.statements_used.add("Loop")
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.statements_used.add("Raise")
        self.generic_visit(node)


class _ComplexityCounter(ast.NodeVisitor):
    """Approximate cyclomatic complexity via branching constructs."""

    BRANCH_NODES = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ExceptHandler,
        ast.With,
        ast.AsyncWith,
        ast.Assert,
        ast.comprehension,
    )

    def __init__(self) -> None:
        self.count = 0

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self.BRANCH_NODES):
            self.count += 1
        if isinstance(node, ast.BoolOp):
            self.count += max(len(node.values) - 1, 0)
        if isinstance(node, (ast.IfExp,)):
            self.count += 1
        super().generic_visit(node)


def _extract_args(args: ast.arguments) -> list[dict[str, Any]]:
    """Serialize function/method parameters into a JSON-friendly list."""
    result: list[dict[str, Any]] = []

    for arg in args.posonlyargs:
        result.append({"name": arg.arg, "kind": "positional_only"})

    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        entry: dict[str, Any] = {"name": arg.arg, "kind": "positional"}
        if i >= defaults_offset:
            entry["has_default"] = True
        result.append(entry)

    if args.vararg:
        result.append({"name": args.vararg.arg, "kind": "varargs"})

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        entry: dict[str, Any] = {"name": arg.arg, "kind": "keyword_only"}
        if default is not None:
            entry["has_default"] = True
        result.append(entry)

    if args.kwarg:
        result.append({"name": args.kwarg.arg, "kind": "kwargs"})

    return result


def _analyze_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """Build a structural map for a single function or method."""
    collector = _LocalVarCollector()
    for child in node.body:
        collector.visit(child)

    param_names = {a.arg for a in node.args.args}
    param_names.update(a.arg for a in node.args.posonlyargs)
    param_names.update(a.arg for a in node.args.kwonlyargs)
    if node.args.vararg:
        param_names.add(node.args.vararg.arg)
    if node.args.kwarg:
        param_names.add(node.args.kwarg.arg)
    local_vars = collector.locals - param_names

    return_checker = _ReturnChecker()
    for child in node.body:
        return_checker.visit(child)

    recursion_checker = _RecursionChecker(node.name)
    for child in node.body:
        recursion_checker.visit(child)

    statement_inspector = _StatementInspector()
    for child in node.body:
        statement_inspector.visit(child)

    return {
        "name": node.name,
        "arguments": _extract_args(node.args),
        "local_variable_count": len(local_vars),
        "local_variables": sorted(local_vars),
        "has_return": return_checker.has_return,
        "is_recursive": recursion_checker.is_recursive,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "docstring": ast.get_docstring(node),
        "lineno": node.lineno,
        "statement_types_used": sorted(list(statement_inspector.statements_used)),
    }


class CodeBlueprintExtractor(ast.NodeVisitor):
    """
    Walk a Python AST and produce a deterministic, JSON-serializable
    structural blueprint of the source module.
    """

    def __init__(self) -> None:
        self.classes: list[dict[str, Any]] = []
        self.functions: list[dict[str, Any]] = []
        self.complexity = 0
        self._source_lines = 0
        self._class_stack: list[dict[str, Any]] = []

    @classmethod
    def analyze(cls, source: str) -> dict[str, Any]:
        extractor = cls()
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return {
                "success": False,
                "error": {
                    "type": "SyntaxError",
                    "message": exc.msg,
                    "line": exc.lineno,
                    "column": exc.offset,
                    "text": (exc.text or "").rstrip("\n") if exc.text else None,
                },
            }

        extractor._source_lines = len(source.splitlines())
        complexity_counter = _ComplexityCounter()
        complexity_counter.visit(tree)
        extractor.complexity = complexity_counter.count

        extractor.visit(tree)
        return extractor.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": True,
            "classes": self.classes,
            "functions": self.functions,
            "design_metrics": {
                "total_lines_of_code": self._source_lines,
                "total_classes": len(self.classes),
                "total_functions": len(self.functions)
                + sum(len(c["methods"]) for c in self.classes),
                "total_standalone_functions": len(self.functions),
                "total_methods": sum(len(c["methods"]) for c in self.classes),
                "cyclomatic_complexity_approx": self.complexity,
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [_base_name(b) for b in node.bases]

        class_info: dict[str, Any] = {
            "name": node.name,
            "bases": bases,
            "docstring": ast.get_docstring(node),
            "methods": [],
            "instance_variables": [],
            "lineno": node.lineno,
        }

        self._class_stack.append(class_info)
        
        all_instance_vars = set()
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_data = _analyze_location_or_method = _analyze_function(child)
                class_info["methods"].append(method_data)
                
                # Collect instance variables found inside methods (like __init__)
                inspector = _StatementInspector()
                for body_child in child.body:
                    inspector.visit(body_child)
                all_instance_vars.update(inspector.instance_variables)

            elif isinstance(child, ast.ClassDef):
                self.visit(child)

        class_info["instance_variables"] = sorted(list(all_instance_vars))
        self._class_stack.pop()
        self.classes.append(class_info)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self._class_stack:
            self.functions.append(_analyze_function(node))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if not self._class_stack:
            self.functions.append(_analyze_function(node))


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _base_name(node.value)
        return f"{value}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{_base_name(node.value)}[...]"
    return ast.dump(node)