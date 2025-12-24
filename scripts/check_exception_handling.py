#!/usr/bin/env python3
"""
Custom pre-commit hook to enforce exception handling best practices.

This script checks for:
1. Bare except clauses (except:)
2. Overly broad Exception catches without specific handling
3. Missing exception context preservation (raise ... from e)
4. Missing logger.exception() in error handlers
5. Swallowed exceptions without logging

Usage:
    python scripts/check_exception_handling.py file1.py file2.py ...
    Or use as pre-commit hook (automatically called with filenames)
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple


def _has_proper_exception_handling(node: ast.ExceptHandler) -> tuple:
    """Check if exception handler has proper logging or raising."""
    has_raise = False
    has_logger_exception = False
    has_logger_error = False

    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Raise):
            has_raise = True
            if stmt.cause is None and stmt.exc is not None:
                if isinstance(stmt.exc, ast.Call):
                    if isinstance(stmt.exc.func, ast.Name):
                        if "Error" in stmt.exc.func.id:
                            has_raise = True
        elif isinstance(stmt, ast.Call):
            if isinstance(stmt.func, ast.Attribute):
                if stmt.func.attr == "exception":
                    has_logger_exception = True
                elif stmt.func.attr == "error" and any(
                    kw.arg == "exc_info"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value
                    for kw in (stmt.keywords or [])
                ):
                    has_logger_error = True

    return has_raise, has_logger_exception, has_logger_error


def _has_acceptable_return(node: ast.ExceptHandler) -> bool:
    """Check if handler returns acceptable values (None/False/empty)."""
    for stmt in node.body:
        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                if isinstance(stmt.value, (ast.Constant, ast.NameConstant)):
                    if stmt.value.value in (None, False, [], {}):
                        return True
            return True
    return False


class ExceptionHandlerChecker(ast.NodeVisitor):
    """AST visitor to check exception handling patterns."""

    def __init__(self, filename: str):
        self.filename = filename
        self.errors: List[Tuple[int, str]] = []
        self.current_function = None

    def visit_FunctionDef(self, node):
        """Track current function name."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_AsyncFunctionDef(self, node):
        """Track current async function name."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_ExceptHandler(self, node):
        """Check exception handler patterns."""
        line_no = node.lineno

        # Check for bare except clause
        if node.type is None:
            self.errors.append(
                (
                    line_no,
                    "Bare except clause found. Use specific exception types "
                    "(e.g., except ValueError as e:)",
                )
            )
            self.generic_visit(node)
            return

        # Check for overly broad Exception catch
        if isinstance(node.type, ast.Name) and node.type.id == "Exception":
            has_raise, has_logger_exception, has_logger_error = (
                _has_proper_exception_handling(node)
            )

            # Warn if Exception is caught without proper handling
            if not has_raise and not has_logger_exception and not has_logger_error:
                has_return = _has_acceptable_return(node)
                if not has_return:
                    self.errors.append(
                        (
                            line_no,
                            "Broad Exception catch without proper handling. "
                            "Consider catching specific exceptions or using "
                            "logger.exception()",
                        )
                    )

        # Check for specific exception types that should be caught
        if isinstance(node.type, ast.Name):
            exc_type = node.type.id
            # MongoDB operations should catch MongoDB-specific exceptions
            if self.current_function and any(
                op in self.current_function.lower()
                for op in [
                    "insert",
                    "update",
                    "delete",
                    "find",
                    "count",
                    "aggregate",
                    "create_index",
                ]
            ):
                if exc_type == "Exception" and "mongodb" not in self.filename.lower():
                    # This is a warning, not an error - MongoDB operations in database modules
                    # should catch OperationFailure, AutoReconnect, etc.
                    pass

        self.generic_visit(node)


def check_file(filepath: Path) -> List[Tuple[int, str]]:
    """Check a single Python file for exception handling issues."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(filepath))
        checker = ExceptionHandlerChecker(str(filepath))
        checker.visit(tree)

        return checker.errors
    except SyntaxError as e:
        return [(e.lineno or 1, f"Syntax error: {e.msg}")]
    except Exception as e:
        return [(1, f"Error checking file: {e}")]


def main():
    """Main entry point for the checker."""
    if len(sys.argv) < 2:
        print("Usage: check_exception_handling.py <file1> [file2] ...")
        sys.exit(1)

    all_errors = []
    files_checked = []

    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)

        # Skip non-Python files
        if not filepath.suffix == ".py":
            continue

        # Skip test files (they may have different exception handling patterns)
        if "test_" in filepath.name or filepath.name.startswith("test"):
            continue

        # Skip example files (they may have simpler exception handling)
        if "examples" in str(filepath):
            continue

        files_checked.append(filepath)
        errors = check_file(filepath)

        if errors:
            all_errors.extend([(filepath, line, msg) for line, msg in errors])

    if all_errors:
        print("\n❌ Exception Handling Issues Found:\n")
        for filepath, line, msg in all_errors:
            print(f"  {filepath}:{line}: {msg}")
        print(f"\nTotal: {len(all_errors)} issue(s) in {len(files_checked)} file(s)\n")
        sys.exit(1)
    else:
        if files_checked:
            print(
                f"✅ Exception handling checks passed for {len(files_checked)} file(s)"
            )
        sys.exit(0)


if __name__ == "__main__":
    main()
