#!/usr/bin/env python3
"""
Function Extractor for Java Codebases (Stage 2)

Uses tree-sitter-java to extract methods, constructors, classes, interfaces,
enums, and records into the shared extractor schema (see
``adding-a-parser.md`` in the OpenAnt repo root for the canonical fields).

Node types below were verified against the INSTALLED grammar
(tree-sitter-java 0.23.5) by dumping a real parse tree, not guessed from a
filtered example — see the field names used via ``child_by_field_name``.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import tree_sitter_java as ts_java
from tree_sitter import Language, Parser

from utilities.file_io import open_utf8, read_repo_file

JAVA_LANGUAGE = Language(ts_java.language())

# Spring MVC / JAX-RS annotations whose PRESENCE on a method marks it as an
# HTTP entry point regardless of static/instance. Matched on the bare
# (unqualified) annotation identifier — `@org.foo.GetMapping` and
# `@GetMapping` both end in `GetMapping`.
_ROUTE_ANNOTATIONS = {
    "RequestMapping", "GetMapping", "PostMapping", "PutMapping",
    "DeleteMapping", "PatchMapping",  # Spring MVC
    "GET", "POST", "PUT", "DELETE", "PATCH", "Path",  # JAX-RS
}
_TEST_ANNOTATIONS = {"Test", "ParameterizedTest", "RepeatedTest", "TestFactory"}

# javax.servlet.http.HttpServlet lifecycle methods. A method with this exact
# name is classified as a route handler WITHOUT checking that the class
# actually extends HttpServlet (that would need the inheritance graph, built
# one pass later). Over-seeding a same-named private helper as an entry
# point is a false "reachable" for one unit, not a missed edge — the safe
# direction per this project's own reachability philosophy.
_SERVLET_METHOD_NAMES = {"doGet", "doPost", "doPut", "doDelete", "doHead", "doOptions", "service"}

# Declaration node types that introduce a "type" (class/interface/enum/record)
# scope for qualified-name and inheritance purposes.
_TYPE_DECL_NODES = {
    "class_declaration", "interface_declaration", "enum_declaration",
    "record_declaration", "annotation_type_declaration",
}
_METHOD_DECL_NODES = {"method_declaration", "constructor_declaration"}


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _bare_type_name(type_text: str) -> str:
    """Strip generic args and array brackets: 'List<Foo>' -> 'List', 'Foo[]' -> 'Foo'."""
    t = type_text.strip()
    if "<" in t:
        t = t[: t.index("<")]
    return t.replace("[]", "").strip()


class FunctionExtractor:
    """Stage 2: walks the tree-sitter AST of every .java file, extracting
    functions (methods/constructors), classes/interfaces (with their
    extends/implements clauses), and per-file imports."""

    def __init__(self, repo_path: str, scan_results: Dict):
        self.repo_path = Path(repo_path).resolve()
        self.scan_results = scan_results
        self.parser = Parser(JAVA_LANGUAGE)

        self.functions: Dict[str, Dict] = {}
        self.classes: Dict[str, Dict] = {}
        self.imports: Dict[str, Dict[str, str]] = {}
        # bare type name -> list of bare super/interface type names (direct).
        self.inheritance: Dict[str, List[str]] = {}
        self.stats = {
            "total_functions": 0,
            "total_classes": 0,
            "files_processed": 0,
            "files_with_errors": 0,
        }

    # -- top level -----------------------------------------------------

    def extract(self) -> Dict:
        for file_info in self.scan_results.get("files", []):
            rel_path = file_info["path"]
            try:
                self._extract_file(rel_path)
                self.stats["files_processed"] += 1
            except Exception as exc:  # noqa: BLE001 - one bad file must not abort the scan
                print(f"[java-extractor] error parsing {rel_path}: {exc}", file=sys.stderr)
                self.stats["files_with_errors"] += 1

        self.stats["total_functions"] = len(self.functions)
        self.stats["total_classes"] = len(self.classes)

        return {
            "repository": str(self.repo_path),
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "inheritance": self.inheritance,
            "statistics": self.stats,
        }

    def _extract_file(self, rel_path: str) -> None:
        abs_path = self.repo_path / rel_path
        source = read_repo_file(abs_path)
        if isinstance(source, str):
            source = source.encode("utf-8", errors="replace")

        tree = self.parser.parse(source)
        root = tree.root_node

        package = ""
        pkg_node = next((c for c in root.children if c.type == "package_declaration"), None)
        if pkg_node is not None:
            # package_declaration's identifier/scoped_identifier child carries the name.
            name_node = next(
                (c for c in pkg_node.children if c.type in ("identifier", "scoped_identifier")),
                None,
            )
            if name_node is not None:
                package = _node_text(name_node, source)

        file_imports: Dict[str, str] = {}
        for imp in (c for c in root.children if c.type == "import_declaration"):
            name_node = next(
                (c for c in imp.children if c.type in ("identifier", "scoped_identifier")),
                None,
            )
            if name_node is None:
                continue
            fq = _node_text(name_node, source)
            simple = fq.rsplit(".", 1)[-1]
            file_imports[simple] = fq
        self.imports[rel_path] = file_imports

        for child in root.children:
            if child.type in _TYPE_DECL_NODES:
                self._walk_type(child, source, rel_path, package, enclosing_qname=None)

    # -- type declarations (class/interface/enum/record, recursive for nesting) --

    def _walk_type(self, node, source: bytes, rel_path: str, package: str,
                    enclosing_qname: Optional[str]) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        bare_name = _node_text(name_node, source)
        qualified_name = f"{enclosing_qname}.{bare_name}" if enclosing_qname else bare_name
        is_interface = node.type == "interface_declaration"
        is_annotation_type = node.type == "annotation_type_declaration"

        supers: List[str] = []
        superclass_node = node.child_by_field_name("superclass")
        if superclass_node is not None:
            supers.extend(self._type_list_bare_names(superclass_node, source))

        # Classes/enums/records use field "interfaces" (a super_interfaces
        # node) for `implements`. interface_declaration's own `extends` (an
        # interface CAN extend several) is NOT exposed as a named field in
        # this grammar version -- verified empirically: child_by_field_name
        # returns None for both "extends_interfaces" and "interfaces" on an
        # interface_declaration node, so it must be matched by child TYPE.
        iface_node = node.child_by_field_name("interfaces")
        if iface_node is not None:
            supers.extend(self._type_list_bare_names(iface_node, source))
        ext_iface_node = next((c for c in node.children if c.type == "extends_interfaces"), None)
        if ext_iface_node is not None:
            supers.extend(self._type_list_bare_names(ext_iface_node, source))

        if supers:
            self.inheritance.setdefault(bare_name, [])
            for s in supers:
                if s not in self.inheritance[bare_name]:
                    self.inheritance[bare_name].append(s)

        body_node = node.child_by_field_name("body")
        fields: Dict[str, str] = {}
        if body_node is not None:
            for member in body_node.children:
                if member.type == "field_declaration":
                    type_node = member.child_by_field_name("type")
                    bare_type = _bare_type_name(_node_text(type_node, source)) if type_node else None
                    for decl in member.children:
                        if decl.type == "variable_declarator":
                            dname = decl.child_by_field_name("name")
                            if dname is not None and bare_type is not None:
                                fields[_node_text(dname, source)] = bare_type

        self.classes[bare_name] = {
            "name": bare_name,
            "qualified_name": qualified_name,
            "file_path": rel_path,
            "package": package,
            "is_interface": is_interface or is_annotation_type,
            "extends": supers,
            "fields": fields,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

        if body_node is None:
            return
        for member in body_node.children:
            if member.type in _METHOD_DECL_NODES:
                self._extract_method(member, source, rel_path, package, bare_name, qualified_name)
            elif member.type in _TYPE_DECL_NODES:
                # Nested class/interface/enum — recurse with the qualified name
                # of the enclosing type so its members get "Outer.Inner.method".
                self._walk_type(member, source, rel_path, package, enclosing_qname=qualified_name)
            elif member.type == "enum_body_declarations":
                # Enum constant bodies (`FOO { ... }`) and the enum's own
                # methods after the `;` both live inside `enum_body`, with
                # extra members grouped under `enum_body_declarations`.
                for sub in member.children:
                    if sub.type in _METHOD_DECL_NODES:
                        self._extract_method(sub, source, rel_path, package, bare_name, qualified_name)

    def _type_list_bare_names(self, field_node, source: bytes) -> List[str]:
        """Walk a `superclass`/`interfaces`/`extends_interfaces` field node down
        to its `type_identifier`/`generic_type` leaves and return bare names."""
        names: List[str] = []

        def visit(n):
            if n.type in ("type_identifier",):
                names.append(_bare_type_name(_node_text(n, source)))
                return
            if n.type == "generic_type":
                base = n.child_by_field_name("type") or (n.children[0] if n.children else None)
                if base is not None:
                    names.append(_bare_type_name(_node_text(base, source)))
                return
            if n.type == "scoped_type_identifier":
                # `pkg.Outer.Inner` as a supertype — keep only the last segment.
                names.append(_bare_type_name(_node_text(n, source).rsplit(".", 1)[-1]))
                return
            for c in n.children:
                visit(c)

        visit(field_node)
        return names

    # -- methods / constructors ------------------------------------------

    def _extract_method(self, node, source: bytes, rel_path: str, package: str,
                         class_name: str, class_qname: str) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        method_name = _node_text(name_node, source)
        qualified_name = f"{class_qname}.{method_name}"
        func_id = f"{rel_path}:{qualified_name}"

        modifiers_node = next((c for c in node.children if c.type == "modifiers"), None)
        annotations: Set[str] = set()
        is_static = False
        is_private = False
        if modifiers_node is not None:
            for m in modifiers_node.children:
                if m.type in ("marker_annotation", "annotation"):
                    ann_name_node = m.child_by_field_name("name")
                    if ann_name_node is not None:
                        annotations.add(_node_text(ann_name_node, source).rsplit(".", 1)[-1])
                elif m.type == "static":
                    is_static = True
                elif m.type == "private":
                    is_private = True

        params_node = node.child_by_field_name("parameters")
        param_names: List[str] = []
        is_varargs = False
        if params_node is not None:
            for p in params_node.children:
                if p.type == "formal_parameter":
                    pname = p.child_by_field_name("name")
                    if pname is not None:
                        param_names.append(_node_text(pname, source))
                elif p.type == "spread_parameter":
                    is_varargs = True
                    # spread_parameter has no named fields (verified empirically);
                    # its children are [type, "...", variable_declarator] where
                    # variable_declarator (no initializer here) is just the name.
                    decl = next((c for c in p.children if c.type == "variable_declarator"), None)
                    if decl is not None:
                        param_names.append(_node_text(decl, source))

        is_constructor = node.type == "constructor_declaration"

        if annotations & _ROUTE_ANNOTATIONS:
            unit_type = "route_handler"
        elif annotations & _TEST_ANNOTATIONS:
            unit_type = "test"
        elif is_constructor:
            unit_type = "constructor"
        elif method_name in _SERVLET_METHOD_NAMES:
            unit_type = "route_handler"
        elif is_static:
            unit_type = "singleton_method"
        else:
            unit_type = "method"

        body_node = node.child_by_field_name("body")

        self.functions[func_id] = {
            "name": method_name,
            "qualified_name": qualified_name,
            "file_path": rel_path,
            "module_name": package,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "code": _node_text(node, source),
            "class_name": class_name,
            "parameters": param_names,
            "param_count": len(param_names),
            "is_varargs": is_varargs,
            "is_static": is_static,
            "is_exported": not is_private,
            "is_abstract": body_node is None,
            "unit_type": unit_type,
            "annotations": sorted(annotations),
        }


def main():
    import argparse

    argp = argparse.ArgumentParser(description="Extract functions from a Java repository")
    argp.add_argument("repo_path")
    argp.add_argument("--scan-results", required=True, help="Path to scan_results.json from Stage 1")
    argp.add_argument("--output", "-o")
    args = argp.parse_args()

    with open_utf8(args.scan_results) as f:
        scan_results = json.load(f)

    extractor = FunctionExtractor(args.repo_path, scan_results)
    result = extractor.extract()

    output = json.dumps(result, indent=2)
    if args.output:
        with open_utf8(args.output, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
