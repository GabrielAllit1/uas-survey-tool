import ast
import os
import logging
import argparse
import json
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class AmbiguousIfVisitor(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.issues = []

    def visit_If(self, node: ast.If):
        test = node.test
        if isinstance(test, ast.Compare):
            return
        if isinstance(test, (ast.Name, ast.Subscript, ast.Attribute, ast.Call)):
            if isinstance(test, ast.Call) and isinstance(test.func, ast.Attribute):
                safe_attrs = {"any", "all", "size", "shape", "notna", "isna"}
                if test.func.attr in safe_attrs:
                    return
            self.issues.append((self.filename, node.lineno, ast.unparse(test).strip()))
        self.generic_visit(node)

def scan_dir_with_ast(root="."):
    total = 0
    issues = []
    exclude_dirs = ['Lib', 'site-packages', '.venv', 'venv', '__pycache__', 'build', 'dist']
    for dirpath, _, files in os.walk(root):
        if any(excluded in dirpath for excluded in exclude_dirs):
            logger.debug(f"Skipping directory: {dirpath}")
            continue
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=path)
                    visitor = AmbiguousIfVisitor(path)
                    visitor.visit(tree)
                    for fn, ln, expr in visitor.issues:
                        issue = f"{fn}:{ln} ➜ ambiguous boolean test: `if {expr}:`"
                        print(issue)
                        issues.append(issue)
                    total += len(visitor.issues)
                except SyntaxError as e:
                    logger.warning(f"Skipping {path} due to syntax error: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to parse {path}: {str(e)}")
                    continue
    print(f"\nTotal issues found: {total}")
    return total, issues

def main():
    parser = argparse.ArgumentParser(
        description="Scan Python files for ambiguous NumPy boolean tests"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan"
    )
    parser.add_argument(
        "--exit-zero",
        action="store_true",
        help="Always exit 0 (for non-CI runs)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for each issue"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for results (e.g., report.json)"
    )
    args = parser.parse_args()

    total, issues = scan_dir_with_ast(args.root)
    if args.verbose and issues:
        for issue in issues:
            print(issue)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({"total_issues": total, "details": issues}, f, indent=2)
            print(f"Results saved to {args.output}")
    if total and not args.exit_zero:
        print(f"\n🛑 Found {total} ambiguous condition(s)")
        sys.exit(1)
    else:
        print(f"\n✅ Scan complete. Found {total} issues.")
        sys.exit(0)

if __name__ == "__main__":
    main()