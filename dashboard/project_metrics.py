from __future__ import annotations

import ast
import json
import math
import statistics
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from radon.complexity import cc_rank, cc_visit
from radon.metrics import h_visit, mi_rank, mi_visit
from radon.raw import analyze as raw_analyze


DEFAULT_EXCLUDED_DIRECTORIES = {
    ".git",
    ".github",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "migrations",
    "node_modules",
    "site-packages",
}


@dataclass
class FunctionMetrics:
    file_path: str
    qualified_name: str
    name: str
    function_type: str
    line_start: int
    line_end: int
    loc: int

    cyclomatic_complexity: int = 0
    complexity_rank: str = "A"

    parameter_count: int = 0
    positional_parameters: int = 0
    keyword_only_parameters: int = 0
    default_parameter_count: int = 0
    variadic_parameter_count: int = 0

    local_variable_count: int = 0
    return_count: int = 0
    yield_count: int = 0

    decision_count: int = 0
    if_count: int = 0
    match_count: int = 0
    match_case_count: int = 0
    ternary_count: int = 0

    loop_count: int = 0
    for_count: int = 0
    while_count: int = 0
    nested_loop_count: int = 0

    try_count: int = 0
    exception_handler_count: int = 0
    raise_count: int = 0
    assert_count: int = 0

    boolean_operator_count: int = 0
    comparison_count: int = 0
    comprehension_count: int = 0

    call_count: int = 0
    unique_called_functions: int = 0
    recursive_call_count: int = 0

    max_nesting_depth: int = 0
    average_nesting_depth: float = 0.0

    decorator_count: int = 0
    docstring_present: bool = False
    async_function: bool = False


@dataclass
class ClassMetrics:
    file_path: str
    qualified_name: str
    name: str
    line_start: int
    line_end: int
    loc: int

    method_count: int = 0
    async_method_count: int = 0
    property_count: int = 0
    class_variable_count: int = 0
    base_class_count: int = 0
    decorator_count: int = 0

    weighted_methods_per_class: int = 0
    average_method_complexity: float = 0.0
    maximum_method_complexity: int = 0

    public_method_count: int = 0
    private_method_count: int = 0
    special_method_count: int = 0

    docstring_present: bool = False


@dataclass
class FileMetrics:
    file_path: str
    file_name: str
    extension: str

    parse_successful: bool = True
    error: str | None = None

    loc: int = 0
    sloc: int = 0
    logical_loc: int = 0
    comments: int = 0
    multi_line_comments: int = 0
    blank_lines: int = 0
    source_and_comment_lines: int = 0
    comment_percentage: float = 0.0

    maintainability_index: float = 0.0
    maintainability_rank: str = "C"

    total_cyclomatic_complexity: int = 0
    average_cyclomatic_complexity: float = 0.0
    maximum_cyclomatic_complexity: int = 0
    complexity_density_per_100_sloc: float = 0.0

    function_count: int = 0
    async_function_count: int = 0
    class_count: int = 0
    method_count: int = 0

    import_count: int = 0
    unique_import_count: int = 0
    imports: list[str] = field(default_factory=list)

    global_variable_count: int = 0
    lambda_count: int = 0

    decision_count: int = 0
    loop_count: int = 0
    return_count: int = 0
    exception_handler_count: int = 0
    boolean_operator_count: int = 0
    call_count: int = 0
    maximum_nesting_depth: int = 0

    halstead_distinct_operators: float = 0.0
    halstead_distinct_operands: float = 0.0
    halstead_total_operators: float = 0.0
    halstead_total_operands: float = 0.0
    halstead_vocabulary: float = 0.0
    halstead_length: float = 0.0
    halstead_calculated_length: float = 0.0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    halstead_time: float = 0.0
    halstead_estimated_bugs: float = 0.0

    functions: list[FunctionMetrics] = field(default_factory=list)
    classes: list[ClassMetrics] = field(default_factory=list)


class FunctionASTVisitor(ast.NodeVisitor):
    """
    Extract structural metrics from a single function.

    Nested functions and nested classes are deliberately skipped so their
    metrics are not incorrectly attributed to the parent function.
    """



    def __init__(self, function_name: str) -> None:
        self.function_name = function_name

        self.local_variables: set[str] = set()
        self.called_functions: list[str] = []

        self.return_count = 0
        self.yield_count = 0

        self.if_count = 0
        self.match_count = 0
        self.match_case_count = 0
        self.ternary_count = 0

        self.for_count = 0
        self.while_count = 0
        self.nested_loop_count = 0
        self.current_loop_depth = 0

        self.try_count = 0
        self.exception_handler_count = 0
        self.raise_count = 0
        self.assert_count = 0

        self.boolean_operator_count = 0
        self.comparison_count = 0
        self.comprehension_count = 0

        self.call_count = 0
        self.recursive_call_count = 0

        self.current_nesting = 0
        self.nesting_depths: list[int] = []
        self.max_nesting_depth = 0

        self._root_function_visited = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if self._root_function_visited:
            return

        self._root_function_visited = True
        self._visit_function_body(node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self._root_function_visited:
            return

        self._root_function_visited = True
        self._visit_function_body(node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Do not include nested class internals in the current function.
        return

    def _visit_function_body(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.visit(statement)

    def _visit_nested_node(self, node: ast.AST) -> None:
        self.current_nesting += 1
        self.nesting_depths.append(self.current_nesting)
        self.max_nesting_depth = max(
            self.max_nesting_depth,
            self.current_nesting,
        )

        self.generic_visit(node)
        self.current_nesting -= 1

    def visit_If(self, node: ast.If) -> None:
        self.if_count += 1
        self._visit_nested_node(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.ternary_count += 1
        self.generic_visit(node)

    def visit_Match(self, node: ast.AST) -> None:
        self.match_count += 1
        self.match_case_count += len(node.cases)
        self._visit_nested_node(node)

    def visit_For(self, node: ast.For) -> None:
        self.for_count += 1

        if self.current_loop_depth > 0:
            self.nested_loop_count += 1

        self.current_loop_depth += 1
        self._visit_nested_node(node)
        self.current_loop_depth -= 1

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.for_count += 1

        if self.current_loop_depth > 0:
            self.nested_loop_count += 1

        self.current_loop_depth += 1
        self._visit_nested_node(node)
        self.current_loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.while_count += 1

        if self.current_loop_depth > 0:
            self.nested_loop_count += 1

        self.current_loop_depth += 1
        self._visit_nested_node(node)
        self.current_loop_depth -= 1

    def visit_Try(self, node: ast.Try) -> None:
        self.try_count += 1
        self.exception_handler_count += len(node.handlers)
        self._visit_nested_node(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_nested_node(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_nested_node(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.return_count += 1
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:
        self.yield_count += 1
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.yield_count += 1
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.raise_count += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.assert_count += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # `a and b and c` has two Boolean operations.
        self.boolean_operator_count += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.comparison_count += len(node.ops)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.comprehension_count += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.comprehension_count += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.comprehension_count += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.comprehension_count += 1
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.local_variables.add(node.id)

    def visit_Call(self, node: ast.Call) -> None:
        self.call_count += 1
        called_name = get_callable_name(node.func)

        if called_name:
            self.called_functions.append(called_name)

            if (
                called_name == self.function_name
                or called_name.endswith(f".{self.function_name}")
            ):
                self.recursive_call_count += 1

        self.generic_visit(node)


def get_callable_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = get_callable_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr

    return None


def get_node_end_line(node: ast.AST) -> int:
    return int(getattr(node, "end_lineno", getattr(node, "lineno", 0)))


def percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    lower_value = ordered[lower]
    upper_value = ordered[upper]
    fraction = position - lower

    return float(lower_value + (upper_value - lower_value) * fraction)


def safe_mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return statistics.mean(values_list) if values_list else 0.0


def safe_median(values: Iterable[float]) -> float:
    values_list = list(values)
    return statistics.median(values_list) if values_list else 0.0


def safe_stdev(values: Iterable[float]) -> float:
    values_list = list(values)
    return statistics.stdev(values_list) if len(values_list) >= 2 else 0.0


def round_number(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def collect_python_files(
    project_path: Path,
    excluded_directories: set[str],
) -> list[Path]:
    files: list[Path] = []

    for file_path in project_path.rglob("*.py"):
        if any(part in excluded_directories for part in file_path.parts):
            continue

        if file_path.is_file():
            files.append(file_path)

    return sorted(files)


def parameter_metrics(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, int]:
    arguments = node.args

    positional_parameters = (
        len(arguments.posonlyargs)
        + len(arguments.args)
    )

    keyword_only_parameters = len(arguments.kwonlyargs)

    variadic_parameter_count = int(arguments.vararg is not None)
    variadic_parameter_count += int(arguments.kwarg is not None)

    default_parameter_count = len(arguments.defaults)
    default_parameter_count += sum(
        default is not None
        for default in arguments.kw_defaults
    )

    return {
        "parameter_count": (
            positional_parameters
            + keyword_only_parameters
            + variadic_parameter_count
        ),
        "positional_parameters": positional_parameters,
        "keyword_only_parameters": keyword_only_parameters,
        "variadic_parameter_count": variadic_parameter_count,
        "default_parameter_count": default_parameter_count,
    }


def build_radon_complexity_lookup(source: str) -> dict[tuple[str, int], int]:
    """
    Map `(function_name, line_number)` to Cyclomatic Complexity.
    """

    lookup: dict[tuple[str, int], int] = {}

    for block in cc_visit(source):
        if hasattr(block, "methods"):
            for method in block.methods:
                lookup[(method.name, method.lineno)] = method.complexity
        else:
            lookup[(block.name, block.lineno)] = block.complexity

    return lookup


def extract_function_metrics(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    relative_file_path: str,
    qualified_name: str,
    function_type: str,
    complexity_lookup: dict[tuple[str, int], int],
) -> FunctionMetrics:
    visitor = FunctionASTVisitor(node.name)
    visitor.visit(node)

    parameters = parameter_metrics(node)
    complexity = complexity_lookup.get((node.name, node.lineno), 1)

    line_end = get_node_end_line(node)
    loc = max(1, line_end - node.lineno + 1)

    decision_count = (
        visitor.if_count
        + visitor.match_case_count
        + visitor.ternary_count
        + visitor.comparison_count
    )

    loop_count = visitor.for_count + visitor.while_count

    average_nesting = safe_mean(visitor.nesting_depths)

    return FunctionMetrics(
        file_path=relative_file_path,
        qualified_name=qualified_name,
        name=node.name,
        function_type=function_type,
        line_start=node.lineno,
        line_end=line_end,
        loc=loc,
        cyclomatic_complexity=complexity,
        complexity_rank=cc_rank(complexity),
        parameter_count=parameters["parameter_count"],
        positional_parameters=parameters["positional_parameters"],
        keyword_only_parameters=parameters["keyword_only_parameters"],
        default_parameter_count=parameters["default_parameter_count"],
        variadic_parameter_count=parameters["variadic_parameter_count"],
        local_variable_count=len(visitor.local_variables),
        return_count=visitor.return_count,
        yield_count=visitor.yield_count,
        decision_count=decision_count,
        if_count=visitor.if_count,
        match_count=visitor.match_count,
        match_case_count=visitor.match_case_count,
        ternary_count=visitor.ternary_count,
        loop_count=loop_count,
        for_count=visitor.for_count,
        while_count=visitor.while_count,
        nested_loop_count=visitor.nested_loop_count,
        try_count=visitor.try_count,
        exception_handler_count=visitor.exception_handler_count,
        raise_count=visitor.raise_count,
        assert_count=visitor.assert_count,
        boolean_operator_count=visitor.boolean_operator_count,
        comparison_count=visitor.comparison_count,
        comprehension_count=visitor.comprehension_count,
        call_count=visitor.call_count,
        unique_called_functions=len(set(visitor.called_functions)),
        recursive_call_count=visitor.recursive_call_count,
        max_nesting_depth=visitor.max_nesting_depth,
        average_nesting_depth=round_number(average_nesting),
        decorator_count=len(node.decorator_list),
        docstring_present=ast.get_docstring(node) is not None,
        async_function=isinstance(node, ast.AsyncFunctionDef),
    )


def classify_method_name(name: str) -> str:
    if name.startswith("__") and name.endswith("__"):
        return "special"

    if name.startswith("_"):
        return "private"

    return "public"


def extract_classes_and_functions(
    tree: ast.AST,
    relative_file_path: str,
    complexity_lookup: dict[tuple[str, int], int],
) -> tuple[list[FunctionMetrics], list[ClassMetrics]]:
    functions: list[FunctionMetrics] = []
    classes: list[ClassMetrics] = []

    def visit_scope(
        body: list[ast.stmt],
        parent_names: list[str],
        current_class: str | None = None,
    ) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                class_qualified_name = ".".join(
                    [*parent_names, node.name]
                )

                class_methods: list[FunctionMetrics] = []

                for child in node.body:
                    if isinstance(
                        child,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    ):
                        method_qualified_name = (
                            f"{class_qualified_name}.{child.name}"
                        )

                        method_metrics = extract_function_metrics(
                            node=child,
                            relative_file_path=relative_file_path,
                            qualified_name=method_qualified_name,
                            function_type="method",
                            complexity_lookup=complexity_lookup,
                        )

                        functions.append(method_metrics)
                        class_methods.append(method_metrics)

                        # Analyse functions nested inside methods.
                        visit_scope(
                            child.body,
                            [*parent_names, node.name, child.name],
                            current_class=node.name,
                        )

                class_variables = {
                    target.id
                    for child in node.body
                    if isinstance(child, (ast.Assign, ast.AnnAssign))
                    for target in (
                        child.targets
                        if isinstance(child, ast.Assign)
                        else [child.target]
                    )
                    if isinstance(target, ast.Name)
                }

                property_count = sum(
                    any(
                        (
                            isinstance(decorator, ast.Name)
                            and decorator.id == "property"
                        )
                        or (
                            isinstance(decorator, ast.Attribute)
                            and decorator.attr == "setter"
                        )
                        for decorator in child.decorator_list
                    )
                    for child in node.body
                    if isinstance(
                        child,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                )

                method_complexities = [
                    method.cyclomatic_complexity
                    for method in class_methods
                ]

                method_name_types = Counter(
                    classify_method_name(method.name)
                    for method in class_methods
                )

                classes.append(
                    ClassMetrics(
                        file_path=relative_file_path,
                        qualified_name=class_qualified_name,
                        name=node.name,
                        line_start=node.lineno,
                        line_end=get_node_end_line(node),
                        loc=max(
                            1,
                            get_node_end_line(node) - node.lineno + 1,
                        ),
                        method_count=len(class_methods),
                        async_method_count=sum(
                            method.async_function
                            for method in class_methods
                        ),
                        property_count=property_count,
                        class_variable_count=len(class_variables),
                        base_class_count=len(node.bases),
                        decorator_count=len(node.decorator_list),
                        weighted_methods_per_class=sum(
                            method_complexities
                        ),
                        average_method_complexity=round_number(
                            safe_mean(method_complexities)
                        ),
                        maximum_method_complexity=max(
                            method_complexities,
                            default=0,
                        ),
                        public_method_count=method_name_types["public"],
                        private_method_count=method_name_types["private"],
                        special_method_count=method_name_types["special"],
                        docstring_present=ast.get_docstring(node) is not None,
                    )
                )

                # Analyse nested classes.
                nested_class_nodes = [
                    child
                    for child in node.body
                    if isinstance(child, ast.ClassDef)
                ]

                visit_scope(
                    nested_class_nodes,
                    [*parent_names, node.name],
                    current_class=node.name,
                )

            elif isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                # Methods were already handled while processing the class.
                if current_class is not None:
                    continue

                qualified_name = ".".join([*parent_names, node.name])

                functions.append(
                    extract_function_metrics(
                        node=node,
                        relative_file_path=relative_file_path,
                        qualified_name=qualified_name,
                        function_type=(
                            "nested_function"
                            if parent_names
                            else "function"
                        ),
                        complexity_lookup=complexity_lookup,
                    )
                )

                visit_scope(
                    node.body,
                    [*parent_names, node.name],
                    current_class=None,
                )

    module_body = getattr(tree, "body", [])
    visit_scope(module_body, [])

    return functions, classes


def extract_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return imports


def count_module_variables(tree: ast.AST) -> int:
    variables: set[str] = set()

    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables.add(target.id)

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                variables.add(node.target.id)

    return len(variables)


def extract_file_metrics(
    project_path: Path,
    file_path: Path,
) -> FileMetrics:
    relative_path = str(file_path.relative_to(project_path))

    result = FileMetrics(
        file_path=relative_path,
        file_name=file_path.name,
        extension=file_path.suffix.lower(),
    )

    try:
        source = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        tree = ast.parse(source, filename=str(file_path))

        raw = raw_analyze(source)
        result.loc = raw.loc
        result.sloc = raw.sloc
        result.logical_loc = raw.lloc
        result.comments = raw.comments
        result.multi_line_comments = raw.multi
        result.blank_lines = raw.blank
        result.source_and_comment_lines = raw.single_comments

        comment_lines = raw.comments + raw.multi

        result.comment_percentage = round_number(
            (comment_lines / raw.loc * 100)
            if raw.loc
            else 0.0
        )

        result.maintainability_index = round_number(
            mi_visit(source, multi=True)
        )
        result.maintainability_rank = mi_rank(
            result.maintainability_index
        )

        halstead = h_visit(source).total

        result.halstead_distinct_operators = round_number(
            halstead.h1
        )
        result.halstead_distinct_operands = round_number(
            halstead.h2
        )
        result.halstead_total_operators = round_number(
            halstead.N1
        )
        result.halstead_total_operands = round_number(
            halstead.N2
        )
        result.halstead_vocabulary = round_number(
            halstead.vocabulary
        )
        result.halstead_length = round_number(
            halstead.length
        )
        result.halstead_calculated_length = round_number(
            halstead.calculated_length
        )
        result.halstead_volume = round_number(
            halstead.volume
        )
        result.halstead_difficulty = round_number(
            halstead.difficulty
        )
        result.halstead_effort = round_number(
            halstead.effort
        )
        result.halstead_time = round_number(
            halstead.time
        )
        result.halstead_estimated_bugs = round_number(
            halstead.bugs
        )

        complexity_lookup = build_radon_complexity_lookup(source)

        functions, classes = extract_classes_and_functions(
            tree=tree,
            relative_file_path=relative_path,
            complexity_lookup=complexity_lookup,
        )

        imports = extract_imports(tree)
        function_complexities = [
            function.cyclomatic_complexity
            for function in functions
        ]

        result.functions = functions
        result.classes = classes

        result.function_count = len(functions)
        result.async_function_count = sum(
            function.async_function
            for function in functions
        )
        result.class_count = len(classes)
        result.method_count = sum(
            function.function_type == "method"
            for function in functions
        )

        result.total_cyclomatic_complexity = sum(
            function_complexities
        )
        result.average_cyclomatic_complexity = round_number(
            safe_mean(function_complexities)
        )
        result.maximum_cyclomatic_complexity = max(
            function_complexities,
            default=0,
        )
        result.complexity_density_per_100_sloc = round_number(
            (
                result.total_cyclomatic_complexity
                / result.sloc
                * 100
            )
            if result.sloc
            else 0.0
        )

        result.import_count = len(imports)
        result.unique_import_count = len(set(imports))
        result.imports = sorted(set(imports))

        result.global_variable_count = count_module_variables(tree)
        result.lambda_count = sum(
            isinstance(node, ast.Lambda)
            for node in ast.walk(tree)
        )

        result.decision_count = sum(
            function.decision_count
            for function in functions
        )
        result.loop_count = sum(
            function.loop_count
            for function in functions
        )
        result.return_count = sum(
            function.return_count
            for function in functions
        )
        result.exception_handler_count = sum(
            function.exception_handler_count
            for function in functions
        )
        result.boolean_operator_count = sum(
            function.boolean_operator_count
            for function in functions
        )
        result.call_count = sum(
            function.call_count
            for function in functions
        )
        result.maximum_nesting_depth = max(
            (
                function.max_nesting_depth
                for function in functions
            ),
            default=0,
        )

    except (SyntaxError, UnicodeError, OSError, ValueError) as exc:
        result.parse_successful = False
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def run_git_command(
    project_path: Path,
    arguments: list[str],
) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_path), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )

        if completed.returncode != 0:
            return None

        return completed.stdout.strip()

    except (OSError, subprocess.SubprocessError):
        return None


def extract_git_metrics(project_path: Path) -> dict[str, Any]:
    is_repository = (
        run_git_command(
            project_path,
            ["rev-parse", "--is-inside-work-tree"],
        )
        == "true"
    )

    if not is_repository:
        return {
            "is_git_repository": False,
        }

    commit_count_text = run_git_command(
        project_path,
        ["rev-list", "--count", "HEAD"],
    )

    authors_text = run_git_command(
        project_path,
        ["shortlog", "-sne", "HEAD"],
    )

    branches_text = run_git_command(
        project_path,
        ["branch", "--format=%(refname:short)"],
    )

    tags_text = run_git_command(
        project_path,
        ["tag", "--list"],
    )

    first_commit_date = run_git_command(
        project_path,
        [
            "log",
            "--reverse",
            "--format=%aI",
            "-1",
        ],
    )

    latest_commit_date = run_git_command(
        project_path,
        [
            "log",
            "-1",
            "--format=%aI",
        ],
    )

    latest_commit_hash = run_git_command(
        project_path,
        [
            "rev-parse",
            "HEAD",
        ],
    )

    numstat_text = run_git_command(
        project_path,
        [
            "log",
            "--numstat",
            "--format=",
        ],
    )

    lines_added = 0
    lines_deleted = 0
    changed_file_entries = 0

    if numstat_text:
        for line in numstat_text.splitlines():
            parts = line.split("\t")

            if len(parts) < 3:
                continue

            added, deleted = parts[0], parts[1]

            # Binary files are represented by "-".
            if added.isdigit():
                lines_added += int(added)

            if deleted.isdigit():
                lines_deleted += int(deleted)

            changed_file_entries += 1

    contributors = 0

    if authors_text:
        contributors = len(
            [
                line
                for line in authors_text.splitlines()
                if line.strip()
            ]
        )

    branches = (
        [line for line in branches_text.splitlines() if line]
        if branches_text
        else []
    )

    tags = (
        [line for line in tags_text.splitlines() if line]
        if tags_text
        else []
    )

    return {
        "is_git_repository": True,
        "commit_count": (
            int(commit_count_text)
            if commit_count_text
            else 0
        ),
        "contributor_count": contributors,
        "branch_count": len(branches),
        "tag_count": len(tags),
        "branches": branches,
        "tags": tags,
        "first_commit_date": first_commit_date,
        "latest_commit_date": latest_commit_date,
        "latest_commit_hash": latest_commit_hash,
        "historical_lines_added": lines_added,
        "historical_lines_deleted": lines_deleted,
        "historical_total_churn": lines_added + lines_deleted,
        "historical_changed_file_entries": changed_file_entries,
    }


def build_project_summary(
    file_metrics: list[FileMetrics],
) -> dict[str, Any]:
    successful_files = [
        file
        for file in file_metrics
        if file.parse_successful
    ]

    all_functions = [
        function
        for file in successful_files
        for function in file.functions
    ]

    all_classes = [
        class_metric
        for file in successful_files
        for class_metric in file.classes
    ]

    complexities = [
        function.cyclomatic_complexity
        for function in all_functions
    ]

    maintainability_values = [
        file.maintainability_index
        for file in successful_files
    ]

    nesting_values = [
        function.max_nesting_depth
        for function in all_functions
    ]

    total_sloc = sum(file.sloc for file in successful_files)
    total_complexity = sum(complexities)

    complexity_bands = {
        "low_1_to_5": sum(1 <= value <= 5 for value in complexities),
        "moderate_6_to_10": sum(
            6 <= value <= 10
            for value in complexities
        ),
        "high_11_to_20": sum(
            11 <= value <= 20
            for value in complexities
        ),
        "very_high_21_to_50": sum(
            21 <= value <= 50
            for value in complexities
        ),
        "extreme_over_50": sum(
            value > 50
            for value in complexities
        ),
    }

    top_complex_functions = sorted(
        all_functions,
        key=lambda item: (
            item.cyclomatic_complexity,
            item.max_nesting_depth,
            item.loc,
        ),
        reverse=True,
    )[:20]

    top_complex_files = sorted(
        successful_files,
        key=lambda item: (
            item.total_cyclomatic_complexity,
            item.maximum_cyclomatic_complexity,
        ),
        reverse=True,
    )[:20]

    return {
        "file_count": len(file_metrics),
        "successfully_parsed_file_count": len(successful_files),
        "failed_file_count": len(file_metrics) - len(successful_files),

        "function_count": len(all_functions),
        "method_count": sum(
            function.function_type == "method"
            for function in all_functions
        ),
        "async_function_count": sum(
            function.async_function
            for function in all_functions
        ),
        "class_count": len(all_classes),

        "total_loc": sum(file.loc for file in successful_files),
        "total_sloc": total_sloc,
        "total_logical_loc": sum(
            file.logical_loc
            for file in successful_files
        ),
        "total_comment_lines": sum(
            file.comments + file.multi_line_comments
            for file in successful_files
        ),
        "total_blank_lines": sum(
            file.blank_lines
            for file in successful_files
        ),

        "total_cyclomatic_complexity": total_complexity,
        "average_cyclomatic_complexity": round_number(
            safe_mean(complexities)
        ),
        "median_cyclomatic_complexity": round_number(
            safe_median(complexities)
        ),
        "minimum_cyclomatic_complexity": min(
            complexities,
            default=0,
        ),
        "maximum_cyclomatic_complexity": max(
            complexities,
            default=0,
        ),
        "cyclomatic_complexity_standard_deviation": round_number(
            safe_stdev(complexities)
        ),

        "cyclomatic_complexity_p75": round_number(
            percentile(complexities, 0.75)
        ),
        "cyclomatic_complexity_p90": round_number(
            percentile(complexities, 0.90)
        ),
        "cyclomatic_complexity_p95": round_number(
            percentile(complexities, 0.95)
        ),
        "cyclomatic_complexity_p99": round_number(
            percentile(complexities, 0.99)
        ),

        "complexity_density_per_100_sloc": round_number(
            total_complexity / total_sloc * 100
            if total_sloc
            else 0.0
        ),

        "complexity_distribution": complexity_bands,
        "high_complexity_function_count_cc_over_10": sum(
            complexity > 10
            for complexity in complexities
        ),
        "critical_complexity_function_count_cc_over_20": sum(
            complexity > 20
            for complexity in complexities
        ),

        "average_maintainability_index": round_number(
            safe_mean(maintainability_values)
        ),
        "minimum_maintainability_index": round_number(
            min(maintainability_values, default=0.0)
        ),
        "maximum_maintainability_index": round_number(
            max(maintainability_values, default=0.0)
        ),

        "average_maximum_nesting_depth": round_number(
            safe_mean(nesting_values)
        ),
        "maximum_nesting_depth": max(
            nesting_values,
            default=0,
        ),

        "total_decisions": sum(
            function.decision_count
            for function in all_functions
        ),
        "total_loops": sum(
            function.loop_count
            for function in all_functions
        ),
        "total_nested_loops": sum(
            function.nested_loop_count
            for function in all_functions
        ),
        "total_returns": sum(
            function.return_count
            for function in all_functions
        ),
        "total_exception_handlers": sum(
            function.exception_handler_count
            for function in all_functions
        ),
        "total_boolean_operations": sum(
            function.boolean_operator_count
            for function in all_functions
        ),
        "total_function_calls": sum(
            function.call_count
            for function in all_functions
        ),
        "recursive_function_count": sum(
            function.recursive_call_count > 0
            for function in all_functions
        ),

        "total_halstead_volume": round_number(
            sum(
                file.halstead_volume
                for file in successful_files
            )
        ),
        "total_halstead_effort": round_number(
            sum(
                file.halstead_effort
                for file in successful_files
            )
        ),
        "total_halstead_estimated_bugs": round_number(
            sum(
                file.halstead_estimated_bugs
                for file in successful_files
            )
        ),

        "top_complex_functions": [
            {
                "file_path": function.file_path,
                "qualified_name": function.qualified_name,
                "line_start": function.line_start,
                "cyclomatic_complexity":
                    function.cyclomatic_complexity,
                "complexity_rank": function.complexity_rank,
                "loc": function.loc,
                "max_nesting_depth":
                    function.max_nesting_depth,
                "parameter_count": function.parameter_count,
            }
            for function in top_complex_functions
        ],

        "top_complex_files": [
            {
                "file_path": file.file_path,
                "total_cyclomatic_complexity":
                    file.total_cyclomatic_complexity,
                "average_cyclomatic_complexity":
                    file.average_cyclomatic_complexity,
                "maximum_cyclomatic_complexity":
                    file.maximum_cyclomatic_complexity,
                "maintainability_index":
                    file.maintainability_index,
                "sloc": file.sloc,
            }
            for file in top_complex_files
        ],
    }


def extract_project_metrics(
    project_directory: str | Path,
    *,
    excluded_directories: set[str] | None = None,
    include_git_metrics: bool = True,
) -> dict[str, Any]:
    """
    Extract technical-debt and complexity metrics from a Python project.

    Parameters
    ----------
    project_directory:
        Path to the project root.

    excluded_directories:
        Directory names that should not be scanned.

    include_git_metrics:
        Whether to calculate Git history metrics.

    Returns
    -------
    dict
        A JSON-serialisable dictionary containing repository-, file-,
        class-, and function-level metrics.
    """

    project_path = Path(project_directory).expanduser().resolve()

    if not project_path.exists():
        raise FileNotFoundError(
            f"Project directory does not exist: {project_path}"
        )

    if not project_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {project_path}"
        )

    excluded = (
        DEFAULT_EXCLUDED_DIRECTORIES.copy()
        if excluded_directories is None
        else excluded_directories
    )

    python_files = collect_python_files(
        project_path,
        excluded,
    )

    file_metrics = [
        extract_file_metrics(project_path, file_path)
        for file_path in python_files
    ]

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": project_path.name,
            "path": str(project_path),
            "language": "Python",
            "excluded_directories": sorted(excluded),
        },
        "summary": build_project_summary(file_metrics),
        "git": (
            extract_git_metrics(project_path)
            if include_git_metrics
            else {"included": False}
        ),
        "files": [
            asdict(file)
            for file in file_metrics
        ],
    }

    return report


def save_metrics_json(
    metrics: dict[str, Any],
    output_file: str | Path,
) -> Path:
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Extract complexity and technical-debt metrics "
            "from a Python project."
        )
    )

    parser.add_argument(
        "project_directory",
        help="Path to the Python project",
    )

    parser.add_argument(
        "--output",
        default="project_metrics.json",
        help="Output JSON path",
    )

    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Do not extract Git history metrics",
    )

    arguments = parser.parse_args()

    project_metrics = extract_project_metrics(
        arguments.project_directory,
        include_git_metrics=not arguments.no_git,
    )

    output_path = save_metrics_json(
        project_metrics,
        arguments.output,
    )

    summary = project_metrics["summary"]

    print(f"Metrics saved to: {output_path}")
    print(f"Files analysed: {summary['file_count']}")
    print(f"Functions: {summary['function_count']}")
    print(f"Classes: {summary['class_count']}")
    print(
        "Average CC: "
        f"{summary['average_cyclomatic_complexity']}"
    )
    print(
        "Maximum CC: "
        f"{summary['maximum_cyclomatic_complexity']}"
    )
    print(
        "Average MI: "
        f"{summary['average_maintainability_index']}"
    )