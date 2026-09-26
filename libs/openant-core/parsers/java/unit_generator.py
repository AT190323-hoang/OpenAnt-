#!/usr/bin/env python3
"""
Unit Generator for Java Codebases (Stage 4)

Builds the standardized dataset.json / analyzer_output.json OpenAnt
downstream stages consume. Follows the same shape as
parsers/ruby/unit_generator.py (the reference implementation named in
adding-a-parser.md), adapted for Java's field vocabulary (param_count,
is_varargs, is_exported from visibility).
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from core.file_boundary import neutralize_boundaries
from utilities.file_io import open_utf8, read_json, write_json

FILE_BOUNDARY = "\n\n// ========== File Boundary ==========\n\n"


class UnitGenerator:
    """Stage 4: turns (functions, call_graph, reverse_call_graph) into
    self-contained analysis units with dependency context inlined."""

    def __init__(self, call_graph_data: Dict, options: Optional[Dict] = None):
        options = options or {}
        self.functions: Dict[str, Dict] = call_graph_data.get("functions", {})
        self.call_graph: Dict[str, List[str]] = call_graph_data.get("call_graph", {})
        self.reverse_call_graph: Dict[str, List[str]] = call_graph_data.get("reverse_call_graph", {})
        self.repo_path = call_graph_data.get("repository", "")

        self.max_depth = options.get("max_depth", 3)
        self.dataset_name = options.get("dataset_name", Path(self.repo_path).name if self.repo_path else "dataset")

        self.units: List[Dict] = []
        self.statistics = {
            "total_units": 0,
            "by_type": {},
            "units_with_upstream": 0,
            "units_with_downstream": 0,
            "units_enhanced": 0,
            "avg_upstream": 0,
            "avg_downstream": 0,
        }

    def get_dependencies(self, func_id: str, depth: Optional[int] = None) -> List[str]:
        max_d = depth if depth is not None else self.max_depth
        dependencies: List[str] = []
        visited = {func_id}
        queue = [(func_id, 0)]
        while queue:
            current_id, current_depth = queue.pop(0)
            if current_depth >= max_d:
                continue
            for called_id in self.call_graph.get(current_id, []):
                if called_id not in visited:
                    visited.add(called_id)
                    dependencies.append(called_id)
                    queue.append((called_id, current_depth + 1))
        return dependencies

    def get_callers(self, func_id: str, depth: Optional[int] = None) -> List[str]:
        max_d = depth if depth is not None else self.max_depth
        callers: List[str] = []
        visited = {func_id}
        queue = [(func_id, 0)]
        while queue:
            current_id, current_depth = queue.pop(0)
            if current_depth >= max_d:
                continue
            for caller_id in self.reverse_call_graph.get(current_id, []):
                if caller_id not in visited:
                    visited.add(caller_id)
                    callers.append(caller_id)
                    queue.append((caller_id, current_depth + 1))
        return callers

    def assemble_enhanced_code(self, func_data: Dict, upstream_deps: List[Dict],
                                downstream_callers: List[Dict]) -> str:
        parts = []
        included_code: Set[str] = set()

        primary_code = func_data.get("code", "")
        parts.append(primary_code)
        included_code.add(primary_code)

        for dep in upstream_deps:
            dep_code = dep.get("code", "")
            if dep_code and dep_code not in included_code:
                parts.append(dep_code)
                included_code.add(dep_code)

        for caller in downstream_callers:
            caller_code = caller.get("code", "")
            if caller_code and caller_code not in included_code:
                parts.append(caller_code)
                included_code.add(caller_code)

        # Neutralize before joining: each part is scanned-repository source
        # and may otherwise forge the separator (see core/file_boundary.py).
        return FILE_BOUNDARY.join(neutralize_boundaries(p) for p in parts)

    def collect_files_included(self, primary_file: str, upstream_deps: List[Dict],
                                downstream_callers: List[Dict]) -> List[str]:
        files: Set[str] = {primary_file}
        for dep in upstream_deps:
            fp = dep.get("file_path", "")
            if fp:
                files.add(fp)
        for caller in downstream_callers:
            fp = caller.get("file_path", "")
            if fp:
                files.add(fp)
        return sorted(files)

    def create_unit(self, func_id: str, func_data: Dict) -> Dict:
        file_path = func_data.get("file_path", "")
        func_name = func_data.get("name", "")
        class_name = func_data.get("class_name")
        module_name = func_data.get("module_name")
        unit_type = func_data.get("unit_type", "method")

        upstream_ids = self.get_dependencies(func_id)
        upstream_deps = [
            {
                "id": dep_id,
                "name": self.functions[dep_id].get("name"),
                "code": self.functions[dep_id].get("code", ""),
                "file_path": self.functions[dep_id].get("file_path", ""),
                "unit_type": self.functions[dep_id].get("unit_type", "method"),
                "class_name": self.functions[dep_id].get("class_name"),
            }
            for dep_id in upstream_ids if dep_id in self.functions
        ]

        caller_ids = self.get_callers(func_id)
        downstream_callers = [
            {
                "id": caller_id,
                "name": self.functions[caller_id].get("name"),
                "code": self.functions[caller_id].get("code", ""),
                "file_path": self.functions[caller_id].get("file_path", ""),
                "unit_type": self.functions[caller_id].get("unit_type", "method"),
                "class_name": self.functions[caller_id].get("class_name"),
            }
            for caller_id in caller_ids if caller_id in self.functions
        ]

        enhanced_code = self.assemble_enhanced_code(func_data, upstream_deps, downstream_callers)
        files_included = self.collect_files_included(file_path, upstream_deps, downstream_callers)
        has_deps_inlined = bool(upstream_deps or downstream_callers)

        direct_calls = self.call_graph.get(func_id, [])
        direct_callers = self.reverse_call_graph.get(func_id, [])

        return {
            "id": func_id,
            "unit_type": unit_type,
            "code": {
                "primary_code": enhanced_code,
                "primary_origin": {
                    "file_path": file_path,
                    "start_line": func_data.get("start_line"),
                    "end_line": func_data.get("end_line"),
                    "function_name": func_name,
                    "class_name": class_name,
                    "deps_inlined": has_deps_inlined,
                    "files_included": files_included,
                    "original_length": len(func_data.get("code", "")),
                    "enhanced_length": len(enhanced_code),
                },
                "dependencies": [],
                "dependency_metadata": {
                    "depth": self.max_depth,
                    "total_upstream": len(upstream_deps),
                    "total_downstream": len(downstream_callers),
                    "direct_calls": len(direct_calls),
                    "direct_callers": len(direct_callers),
                },
            },
            "ground_truth": {
                "status": "UNKNOWN",
                "vulnerability_types": [],
                "issues": [],
                "annotation_source": None,
                "annotation_key": None,
                "notes": None,
            },
            "metadata": {
                "is_static": func_data.get("is_static", False),
                "is_exported": func_data.get("is_exported", True),
                "module_name": module_name,
                "parameters": func_data.get("parameters", []),
                "param_count": func_data.get("param_count", 0),
                "is_varargs": func_data.get("is_varargs", False),
                "annotations": func_data.get("annotations", []),
                "generator": "java_unit_generator.py",
                "direct_calls": direct_calls,
                "direct_callers": direct_callers,
            },
        }

    def update_statistics(self, unit: Dict) -> None:
        self.statistics["total_units"] += 1
        unit_type = unit.get("unit_type", "method")
        self.statistics["by_type"][unit_type] = self.statistics["by_type"].get(unit_type, 0) + 1

        dep_meta = unit.get("code", {}).get("dependency_metadata", {})
        if dep_meta.get("total_upstream", 0) > 0:
            self.statistics["units_with_upstream"] += 1
        if dep_meta.get("total_downstream", 0) > 0:
            self.statistics["units_with_downstream"] += 1
        if unit.get("code", {}).get("primary_origin", {}).get("deps_inlined", False):
            self.statistics["units_enhanced"] += 1

    def generate_units(self) -> Dict:
        total_upstream = 0
        total_downstream = 0

        for func_id, func_data in self.functions.items():
            if func_data.get("is_abstract") and func_data.get("unit_type") != "route_handler":
                # An interface/abstract method DECLARATION has no body to
                # analyze for a vulnerability -- skip it as a unit, but it
                # remains a valid dispatch TARGET in the call graph (an
                # abstract route_handler is unusual but not impossible via
                # an annotated interface default; keep those).
                continue
            unit = self.create_unit(func_id, func_data)
            self.units.append(unit)
            self.update_statistics(unit)
            dep_meta = unit["code"]["dependency_metadata"]
            total_upstream += dep_meta["total_upstream"]
            total_downstream += dep_meta["total_downstream"]

        if self.statistics["total_units"] > 0:
            self.statistics["avg_upstream"] = round(total_upstream / self.statistics["total_units"], 2)
            self.statistics["avg_downstream"] = round(total_downstream / self.statistics["total_units"], 2)

        return {
            "name": self.dataset_name,
            "repository": self.repo_path,
            "units": self.units,
            "statistics": self.statistics,
            "metadata": {
                "generator": "java_unit_generator.py",
                "generated_at": datetime.now().isoformat(),
                "dependency_depth": self.max_depth,
            },
        }

    def generate_analyzer_output(self) -> Dict:
        functions = {}
        for func_id, func_data in self.functions.items():
            functions[func_id] = {
                "name": func_data.get("name", ""),
                "unitType": func_data.get("unit_type", "method"),
                "code": func_data.get("code", ""),
                "filePath": func_data.get("file_path", ""),
                "startLine": func_data.get("start_line", 0),
                "endLine": func_data.get("end_line", 0),
                "isExported": func_data.get("is_exported", True),
                "isSingleton": func_data.get("is_static", False),
                "moduleName": func_data.get("module_name"),
                "parameters": func_data.get("parameters", []),
                "className": func_data.get("class_name"),
            }
        return {
            "repository": self.repo_path,
            "functions": functions,
            "call_graph": self.call_graph,
            "reverse_call_graph": self.reverse_call_graph,
        }


def main():
    import argparse

    argp = argparse.ArgumentParser(description="Generate analysis units from Java call graph data")
    argp.add_argument("input_file")
    argp.add_argument("--output", "-o")
    argp.add_argument("--analyzer-output")
    argp.add_argument("--depth", "-d", type=int, default=3)
    argp.add_argument("--name", "-n")
    args = argp.parse_args()

    call_graph_data = read_json(args.input_file)
    options = {"max_depth": args.depth}
    if args.name:
        options["dataset_name"] = args.name

    generator = UnitGenerator(call_graph_data, options)
    result = generator.generate_units()

    output = json.dumps(result, indent=2)
    if args.output:
        with open_utf8(args.output, "w") as f:
            f.write(output)
    else:
        print(output)

    if args.analyzer_output:
        write_json(args.analyzer_output, generator.generate_analyzer_output())


if __name__ == "__main__":
    main()
