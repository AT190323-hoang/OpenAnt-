#!/usr/bin/env python3
"""
Java Parser Pipeline Orchestrator (Stage 1-4 + reachability filter)

Wires RepositoryScanner -> FunctionExtractor -> CallGraphBuilder ->
UnitGenerator, then applies the shared (language-agnostic)
entry-point/reachability filter. This is the subprocess entry point
core/parser_adapter.py's ``_parse_java()`` (once registered) invokes.

Required CLI interface (per adding-a-parser.md):
    python test_pipeline.py <repo_path> --output <dir>
        --processing-level <all|reachable|codeql|exploitable>
        [--skip-tests] [--name <name>]
    Exit 0 on success, non-zero on failure.

``codeql``/``exploitable`` levels need infrastructure this parser does not
implement (a CodeQL query pass, an LLM exploitability classifier) -- they
degrade to ``reachable`` with a printed warning rather than erroring, the
same posture the Ruby orchestrator takes for ``exploitable`` without
``--llm --agentic``.
"""

import argparse
import os
import sys
from datetime import datetime
from enum import Enum
from typing import Set

# MUST run before any `from utilities...` / `from parsers...` import: the
# adapter invokes this script as a subprocess with cwd=<openant-core root>
# and no PYTHONPATH, so an import above this line dies with
# ModuleNotFoundError on the exact invocation the product uses (a fixture
# that pre-sets PYTHONPATH would not catch that ordering bug).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utilities.file_io import write_json  # noqa: E402
from utilities.agentic_enhancer import (  # noqa: E402
    EntryPointDetector,
    ReachabilityAnalyzer,
    library_seed_ids,
)

from repository_scanner import RepositoryScanner  # noqa: E402
from function_extractor import FunctionExtractor  # noqa: E402
from call_graph_builder import CallGraphBuilder  # noqa: E402
from unit_generator import UnitGenerator  # noqa: E402


class ProcessingLevel(Enum):
    ALL = "all"
    REACHABLE = "reachable"
    CODEQL = "codeql"
    EXPLOITABLE = "exploitable"


class JavaPipeline:
    def __init__(self, repo_path: str, output_dir: str, processing_level: ProcessingLevel,
                 skip_tests: bool = True, depth: int = 3, name: str = None,
                 library_mode: bool = False):
        self.repo_path = os.path.abspath(repo_path)
        self.output_dir = output_dir or os.path.join(os.getcwd(), "java_output")
        self.processing_level = processing_level
        self.skip_tests = skip_tests
        self.depth = depth
        self.dataset_name = name
        self.library_mode = library_mode

        self.dataset_file = None
        self.analyzer_output_file = None
        self.entry_points: Set[str] = set()
        self.reachable_units: Set[str] = set()

    def run(self) -> bool:
        os.makedirs(self.output_dir, exist_ok=True)

        print("=" * 60, file=sys.stderr)
        print("STAGE: java_parser_pipeline", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        try:
            print("  [1/4] Scanning repository for Java files...", file=sys.stderr)
            scanner = RepositoryScanner(self.repo_path, {"skip_tests": self.skip_tests})
            scan_result = scanner.scan()
            print(f"         Found {scan_result['statistics']['total_files']} files", file=sys.stderr)
            write_json(os.path.join(self.output_dir, "scan_results.json"), scan_result)

            print("  [2/4] Extracting classes/methods via tree-sitter...", file=sys.stderr)
            extractor = FunctionExtractor(self.repo_path, scan_result)
            extract_result = extractor.extract()
            print(
                f"         Extracted {extract_result['statistics']['total_functions']} functions, "
                f"{extract_result['statistics']['total_classes']} classes/interfaces "
                f"({extract_result['statistics']['files_with_errors']} file errors)",
                file=sys.stderr,
            )

            print("  [3/4] Building interface-aware call graph...", file=sys.stderr)
            builder = CallGraphBuilder(extract_result)
            graph_result = builder.build()
            gstats = graph_result["statistics"]
            print(
                f"         {gstats['total_edges']} edges, avg out-degree {gstats['avg_out_degree']:.2f}, "
                f"{gstats['isolated_functions']} isolated",
                file=sys.stderr,
            )

            print("  [4/4] Generating dataset units...", file=sys.stderr)
            gen_options = {"max_depth": self.depth}
            if self.dataset_name:
                gen_options["dataset_name"] = self.dataset_name
            generator = UnitGenerator(graph_result, gen_options)
            dataset = generator.generate_units()
            print(f"         Generated {dataset['statistics']['total_units']} units", file=sys.stderr)

            self.dataset_file = os.path.join(self.output_dir, "dataset.json")
            self.analyzer_output_file = os.path.join(self.output_dir, "analyzer_output.json")
            write_json(self.dataset_file, dataset)
            write_json(self.analyzer_output_file, generator.generate_analyzer_output())
            write_json(os.path.join(self.output_dir, "call_graph.json"), graph_result)

            if self.processing_level in (
                ProcessingLevel.REACHABLE, ProcessingLevel.CODEQL, ProcessingLevel.EXPLOITABLE,
            ):
                if self.processing_level != ProcessingLevel.REACHABLE:
                    print(
                        f"  [Warning] --processing-level {self.processing_level.value} needs "
                        "infrastructure this parser does not implement (CodeQL / LLM "
                        "exploitability classification) -- filtering by reachability only.",
                        file=sys.stderr,
                    )
                self._apply_reachability_filter()

            return True

        except Exception as exc:  # noqa: BLE001
            print(f"[java-pipeline] FAILED: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return False

    def _apply_reachability_filter(self) -> None:
        print("=" * 60, file=sys.stderr)
        print("STAGE: reachability_filter (static analysis)", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

        from utilities.file_io import read_json

        analyzer = read_json(self.analyzer_output_file)
        functions = analyzer.get("functions", {})

        normalized_functions = {
            func_id: {
                "name": f.get("name", ""),
                "unitType": f.get("unitType", "method"),
                "code": f.get("code", ""),
                "filePath": f.get("filePath", ""),
                "startLine": f.get("startLine", 0),
                "endLine": f.get("endLine", 0),
                "isExported": f.get("isExported", True),
                "isSingleton": f.get("isSingleton", False),
            }
            for func_id, f in functions.items()
        }

        dataset = read_json(self.dataset_file)
        call_graph = {}
        reverse_call_graph = {}
        for unit in dataset.get("units", []):
            unit_id = unit.get("id")
            meta = unit.get("metadata", {})
            if meta.get("direct_calls"):
                call_graph[unit_id] = meta["direct_calls"]
            if meta.get("direct_callers"):
                reverse_call_graph[unit_id] = meta["direct_callers"]

        detector = EntryPointDetector(normalized_functions, call_graph)
        self.entry_points = detector.detect_entry_points()
        if self.library_mode:
            self.entry_points = self.entry_points | library_seed_ids(normalized_functions)

        reachability = ReachabilityAnalyzer(
            functions=normalized_functions,
            reverse_call_graph=reverse_call_graph,
            entry_points=self.entry_points,
        )
        self.reachable_units = reachability.get_all_reachable()

        units = dataset.get("units", [])
        original_count = len(units)

        # No entry points detected => reachable set is empty => every unit
        # would be pruned, silently blacking out the dataset. Degrade to
        # keep-all + warn (mirrors core/parser_adapter.py's own safety net).
        if not self.entry_points and units:
            print(
                "  [Warning] No entry points detected -- keeping all units "
                "unfiltered to avoid a silent blackout.",
                file=sys.stderr,
            )
            self.reachable_units = {u.get("id", "") for u in units}

        filtered_units = []
        for u in units:
            unit_id = u.get("id", "")
            if unit_id in self.reachable_units:
                u["reachable"] = True
                u["is_entry_point"] = unit_id in self.entry_points
                filtered_units.append(u)

        dataset["units"] = filtered_units
        dataset.setdefault("metadata", {})["reachability_filter"] = {
            "original_units": original_count,
            "entry_points": len(self.entry_points),
            "reachable_units": len(filtered_units),
            "filtered_out": original_count - len(filtered_units),
        }
        write_json(self.dataset_file, dataset)
        print(
            f"  Entry points: {len(self.entry_points)}  "
            f"Units: {original_count} -> {len(filtered_units)}",
            file=sys.stderr,
        )


def main():
    argp = argparse.ArgumentParser(description="Run the Java parser pipeline on a repository")
    argp.add_argument("repo_path")
    argp.add_argument("--output", "-o", default=None)
    argp.add_argument(
        "--processing-level",
        choices=["all", "reachable", "codeql", "exploitable"],
        default="reachable",
    )
    argp.add_argument("--skip-tests", action="store_true")
    argp.add_argument("--depth", "-d", type=int, default=3)
    argp.add_argument("--name", "-n", default=None)
    argp.add_argument("--library-mode", action="store_true")
    args = argp.parse_args()

    if not os.path.exists(args.repo_path):
        print(f"Error: repository not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    pipeline = JavaPipeline(
        args.repo_path,
        args.output,
        ProcessingLevel(args.processing_level),
        skip_tests=args.skip_tests,
        depth=args.depth,
        name=args.name,
        library_mode=args.library_mode,
    )
    ok = pipeline.run()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
