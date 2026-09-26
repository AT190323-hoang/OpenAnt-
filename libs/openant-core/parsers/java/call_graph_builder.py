#!/usr/bin/env python3
"""
Call Graph Builder for Java Codebases (Stage 3)

Builds bidirectional call graphs, with INTERFACE-AWARE dispatch: a call on a
receiver typed as an interface (or abstract/base class) must reach every
concrete implementor's method, not just a same-name coincidence. This is the
"dispatch resolution" work adding-a-parser.md calls out as the highest-
leverage part of a call-graph builder for a language with explicit
interfaces (Rust traits / Swift protocols / Java interfaces are the same
shape) -- the design here (a reverse transitive "conformer closure" consulted
at every member-call site) mirrors the Swift parser's
``_conformer_closure``/``_dispatch_types``, adapted to Java's simpler
(non-labeled-argument) overload model.

Known, accepted simplifications relative to the Rust/Swift reference
implementations (documented rather than silently absent):
  - Overload resolution is by ARGUMENT COUNT only (no parameter-type
    matching) -- Java's type system would need a much larger type-inference
    pass to do properly; arity-only narrowing still resolves the common case
    and falls back to the full candidate set (recall-first) when arity alone
    is ambiguous, same policy as the Swift builder's `_match_overloads`.
  - A chained call's receiver type (`foo().bar()`) is NOT tracked -- no
    return-type index is built. `bar()` resolves via the capped global
    bare-name fallback instead of a typed dispatch. A real miss, not a
    crash; documented for the same reason Swift's `_func_return` exists but
    is deliberately not replicated here (scope).
  - `new SomeInterface() { ... }` (anonymous-class implementations) are not
    registered as synthetic conformers -- their methods parse as ordinary
    class members but the anonymous class itself never becomes a dispatch
    target. A known gap, not a silent wrong answer.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import tree_sitter_java as ts_java
from tree_sitter import Language, Node, Parser

from utilities.file_io import open_utf8

JAVA_LANGUAGE = Language(ts_java.language())

# java.lang.Object / java.lang.String / java.util.Collection member names so
# common that resolving them via the UNTYPED global fallback would fan out
# into every user-defined method that happens to share the name (toString,
# equals, get, add...). Real calls to these almost always target the JDK,
# not repo code -- excluding them from the untyped path is a precision
# choice; a call on a KNOWN-typed receiver (e.g. a field typed as a repo
# class named `Builder` with a `build()` method) is unaffected, since typed
# dispatch narrows by declaring type first and never reaches this filter.
_COMMON_JDK_METHOD_NAMES = frozenset({
    "toString", "equals", "hashCode", "clone", "getClass",
    "get", "set", "put", "add", "remove", "contains", "size", "isEmpty",
    "iterator", "forEach", "stream", "of", "asList", "builder", "build",
    "getOrDefault", "compute", "merge", "keySet", "values", "entrySet",
    "close", "toArray", "sort", "compareTo", "valueOf", "format", "parse",
    "append", "length", "substring", "split", "trim", "replace", "matches",
})

# Cap on how many candidates a global (untyped-receiver) bare-name fallback
# may fan out to before it is discarded rather than emitted. An unresolved
# receiver calling a common short name (`run`, `execute`, `process`) can
# otherwise match dozens of unrelated classes -- past this cap the call
# looks less like "the compiler resolved this uniquely" and more like "the
# name is generic", so emitting nothing is the safer (lower phantom-edge)
# choice; a missed edge here is a real recall cost, traded deliberately.
_GLOBAL_FALLBACK_MAX_CANDIDATES = 5


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _bare_type_name(type_text: str) -> str:
    t = type_text.strip()
    if "<" in t:
        t = t[: t.index("<")]
    return t.replace("[]", "").strip()


class CallGraphBuilder:
    """Stage 3: resolves method_invocation / object_creation_expression call
    sites to function IDs, using interface/inheritance-aware dispatch."""

    def __init__(self, extractor_output: Dict):
        self.functions: Dict[str, Dict] = extractor_output.get("functions", {})
        self.classes: Dict[str, Dict] = extractor_output.get("classes", {})
        self.inheritance: Dict[str, List[str]] = extractor_output.get("inheritance", {})
        self.repository = extractor_output.get("repository", "")

        self._repo_declared: Set[str] = set(self.classes.keys())
        self._inherit_closure = self._compute_closure(self.inheritance)
        self._conformer_closure = self._compute_reverse_closure(self.inheritance)

        self._name_to_ids: Dict[str, List[str]] = defaultdict(list)
        for func_id, info in self.functions.items():
            self._name_to_ids[info["name"]].append(func_id)

        self.parser = Parser(JAVA_LANGUAGE)
        self.call_graph: Dict[str, List[str]] = {}
        self.reverse_call_graph: Dict[str, List[str]] = {}
        self.stats_extra: Dict[str, int] = {
            "global_fallback_used": 0,
            "global_fallback_capped": 0,
            "sites_total": 0,
            "sites_resolved": 0,
        }

    # -- closures ---------------------------------------------------------

    def _compute_closure(self, direct: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """type -> ALL transitive supertypes/interfaces (forward)."""
        closure: Dict[str, Set[str]] = {}
        for t in direct:
            seen: Set[str] = set()
            stack = list(direct.get(t, ()))
            while stack:
                s = stack.pop()
                if s in seen:
                    continue
                seen.add(s)
                stack.extend(direct.get(s, ()))
            closure[t] = seen
        return closure

    def _compute_reverse_closure(self, direct: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """type -> ALL transitive subtypes/implementors (reverse edges) --
        this is what lets a call on an interface-typed receiver reach every
        concrete implementation."""
        reverse: Dict[str, Set[str]] = defaultdict(set)
        for sub, supers in direct.items():
            for sup in supers:
                reverse[sup].add(sub)
        closure: Dict[str, Set[str]] = {}
        for t in reverse:
            seen: Set[str] = set()
            stack = list(reverse.get(t, ()))
            while stack:
                s = stack.pop()
                if s in seen:
                    continue
                seen.add(s)
                stack.extend(reverse.get(s, ()))
            closure[t] = seen
        return closure

    def _dispatch_types(self, recv_type: str, is_super: bool) -> Set[str]:
        """Declaring types accepted for a member dispatch on `recv_type`.

        Non-super: recv_type itself + its supertypes/interfaces (inherited
        default/abstract methods) + its transitive implementors/subclasses
        (dynamic dispatch -- an interface-typed receiver reaches every
        concrete override). Super: supertypes ONLY, never recv_type itself
        or its implementors -- `super.m()` must reach the PARENT's method.
        """
        if is_super:
            return set(self._inherit_closure.get(recv_type, set()))
        return (
            {recv_type}
            | self._inherit_closure.get(recv_type, set())
            | self._conformer_closure.get(recv_type, set())
        )

    # -- overload / arity matching -----------------------------------------

    def _arity_compatible(self, info: Dict, arg_count: int) -> bool:
        if info.get("is_varargs"):
            return arg_count >= max(0, info["param_count"] - 1)
        return info["param_count"] == arg_count

    def _match_candidates(self, accept: Set[str], method_name: str, arg_count: int) -> List[str]:
        candidates = [
            fid for fid in self._name_to_ids.get(method_name, [])
            if self.functions[fid]["class_name"] in accept
        ]
        if len(candidates) <= 1:
            return candidates
        # Overload narrowing by argument count. If arity disambiguates,
        # prefer the narrowed set; if NONE match (e.g. a call built from a
        # macro/generated arg list this pass can't count reliably), keep the
        # full candidate set -- for a METHOD call, a missed edge (an
        # unreachable finding never gets a caller/callee trace) is the
        # expensive direction, so recall wins the tie (same policy as the
        # Swift/Rust builders' `_match_overloads`).
        narrowed = [c for c in candidates if self._arity_compatible(self.functions[c], arg_count)]
        return narrowed if narrowed else candidates

    def _match_constructors(self, type_name: str, arg_count: int) -> List[str]:
        accept = self._dispatch_types(type_name, is_super=False)
        candidates = [
            fid for fid, info in self.functions.items()
            if info["unit_type"] == "constructor" and info["class_name"] in accept
        ]
        if len(candidates) <= 1:
            return candidates
        narrowed = [c for c in candidates if self._arity_compatible(self.functions[c], arg_count)]
        return narrowed if narrowed else candidates

    def _global_fallback(self, method_name: str, arg_count: int) -> List[str]:
        """Untyped-receiver resolution: match by bare name across the WHOLE
        repo, capped to avoid phantom fan-out on a generic name."""
        if method_name in _COMMON_JDK_METHOD_NAMES:
            return []
        candidates = list(self._name_to_ids.get(method_name, []))
        if not candidates:
            return []
        narrowed = [c for c in candidates if self._arity_compatible(self.functions[c], arg_count)]
        pool = narrowed if narrowed else candidates
        self.stats_extra["global_fallback_used"] += 1
        if len(pool) > _GLOBAL_FALLBACK_MAX_CANDIDATES:
            self.stats_extra["global_fallback_capped"] += 1
            return []
        return pool

    # -- call-site extraction ----------------------------------------------

    def _collect_var_types(self, method_node: Node, source: bytes, caller_class: str) -> Dict[str, str]:
        """name -> bare declared type, from this method's own parameters and
        local variable declarations, merged over the enclosing class's
        fields (locals/params shadow fields on name collision -- real Java
        scoping)."""
        var_types: Dict[str, str] = {}
        class_info = self.classes.get(caller_class)
        if class_info:
            var_types.update(class_info.get("fields", {}))

        params_node = method_node.child_by_field_name("parameters")
        if params_node is not None:
            for p in params_node.children:
                if p.type == "formal_parameter":
                    tnode = p.child_by_field_name("type")
                    nnode = p.child_by_field_name("name")
                    if tnode is not None and nnode is not None:
                        var_types[_node_text(nnode, source)] = _bare_type_name(_node_text(tnode, source))
                elif p.type == "spread_parameter":
                    tnode = next((c for c in p.children if "type" in c.type or c.type == "type_identifier"), None)
                    decl = next((c for c in p.children if c.type == "variable_declarator"), None)
                    if tnode is not None and decl is not None:
                        var_types[_node_text(decl, source)] = _bare_type_name(_node_text(tnode, source))

        def visit_locals(n: Node):
            if n.type == "local_variable_declaration":
                tnode = n.child_by_field_name("type")
                bare_type = _bare_type_name(_node_text(tnode, source)) if tnode else None
                if bare_type:
                    for decl in n.children:
                        if decl.type == "variable_declarator":
                            dname = decl.child_by_field_name("name")
                            if dname is not None:
                                var_types[_node_text(dname, source)] = bare_type
            for c in n.children:
                visit_locals(c)

        body = method_node.child_by_field_name("body")
        if body is not None:
            visit_locals(body)
        return var_types

    def _find_call_sites(self, method_node: Node) -> List[Node]:
        """method_invocation and object_creation_expression nodes anywhere
        in the method body (including inside lambdas/anonymous blocks --
        those are real call sites too)."""
        sites: List[Node] = []
        body = method_node.child_by_field_name("body")
        if body is None:
            return sites

        def visit(n: Node):
            if n.type in ("method_invocation", "object_creation_expression"):
                sites.append(n)
            for c in n.children:
                visit(c)

        visit(body)
        return sites

    def _arg_count(self, args_node: Optional[Node]) -> int:
        if args_node is None:
            return 0
        return sum(1 for c in args_node.children if c.type not in ("(", ")", ","))

    def _resolve_site(self, site: Node, source: bytes, caller_class: str,
                       var_types: Dict[str, str]) -> List[str]:
        if site.type == "object_creation_expression":
            type_node = site.child_by_field_name("type")
            if type_node is None:
                return []
            type_name = _bare_type_name(_node_text(type_node, source))
            arg_count = self._arg_count(site.child_by_field_name("arguments"))
            return self._match_constructors(type_name, arg_count)

        # method_invocation
        name_node = site.child_by_field_name("name")
        if name_node is None:
            return []
        method_name = _node_text(name_node, source)
        arg_count = self._arg_count(site.child_by_field_name("arguments"))
        obj = site.child_by_field_name("object")

        if obj is None or obj.type == "this":
            accept = self._dispatch_types(caller_class, is_super=False)
            return self._match_candidates(accept, method_name, arg_count)

        if obj.type == "super":
            accept = self._dispatch_types(caller_class, is_super=True)
            return self._match_candidates(accept, method_name, arg_count)

        if obj.type == "identifier":
            ident = _node_text(obj, source)
            if ident in var_types:
                accept = self._dispatch_types(var_types[ident], is_super=False)
                matched = self._match_candidates(accept, method_name, arg_count)
                if matched:
                    return matched
                # A typed receiver whose type has no matching method in-repo
                # (e.g. a JDK type stored in a repo-typed var, or a bare-name
                # collision) -- fall through to the untyped path rather than
                # silently dropping the edge.
            if ident in self.classes:
                accept = self._dispatch_types(ident, is_super=False)
                matched = self._match_candidates(accept, method_name, arg_count)
                if matched:
                    return matched
            return self._global_fallback(method_name, arg_count)

        if obj.type == "object_creation_expression":
            # `new Foo().bar()` -- the constructed type IS the receiver type,
            # no var-type lookup needed (cheap, common idiom worth a direct
            # branch rather than falling through to the untyped path).
            ctor_type_node = obj.child_by_field_name("type")
            if ctor_type_node is not None:
                recv_type = _bare_type_name(_node_text(ctor_type_node, source))
                accept = self._dispatch_types(recv_type, is_super=False)
                matched = self._match_candidates(accept, method_name, arg_count)
                if matched:
                    return matched
            return self._global_fallback(method_name, arg_count)

        if obj.type == "field_access":
            field_node = obj.child_by_field_name("field")
            if field_node is not None:
                fname = _node_text(field_node, source)
                if fname in var_types:
                    accept = self._dispatch_types(var_types[fname], is_super=False)
                    matched = self._match_candidates(accept, method_name, arg_count)
                    if matched:
                        return matched
            return self._global_fallback(method_name, arg_count)

        # Chained call (`foo().bar()`), array access, cast expression, etc. --
        # no return-type tracking (documented simplification); fall back to
        # the capped untyped path.
        return self._global_fallback(method_name, arg_count)

    # -- driver -------------------------------------------------------------

    def build_call_graph(self) -> None:
        call_graph: Dict[str, List[str]] = defaultdict(list)
        reverse_call_graph: Dict[str, List[str]] = defaultdict(list)

        for func_id, info in self.functions.items():
            if info.get("is_abstract"):
                continue  # no body to scan (interface/abstract method decl)
            code = info["code"]
            try:
                tree = self.parser.parse(code.encode("utf-8", errors="replace"))
            except Exception:
                continue
            root = tree.root_node
            method_node = next(
                (c for c in root.children if c.type in ("method_declaration", "constructor_declaration")),
                None,
            )
            if method_node is None:
                continue
            source = code.encode("utf-8", errors="replace")
            caller_class = info["class_name"]
            var_types = self._collect_var_types(method_node, source, caller_class)

            targets: Set[str] = set()
            for site in self._find_call_sites(method_node):
                self.stats_extra["sites_total"] += 1
                resolved = self._resolve_site(site, source, caller_class, var_types)
                if resolved:
                    self.stats_extra["sites_resolved"] += 1
                for t in resolved:
                    if t != func_id:
                        targets.add(t)

            if targets:
                call_graph[func_id] = sorted(targets)
                for t in targets:
                    reverse_call_graph[t].append(func_id)

        for k in reverse_call_graph:
            reverse_call_graph[k] = sorted(set(reverse_call_graph[k]))

        self.call_graph = dict(call_graph)
        self.reverse_call_graph = dict(reverse_call_graph)

    def get_dependencies(self, func_id: str, depth: Optional[int] = None) -> List[str]:
        return self._bfs(func_id, self.call_graph, depth)

    def get_callers(self, func_id: str, depth: Optional[int] = None) -> List[str]:
        return self._bfs(func_id, self.reverse_call_graph, depth)

    def _bfs(self, func_id: str, graph: Dict[str, List[str]], depth: Optional[int]) -> List[str]:
        visited: Set[str] = set()
        frontier = [func_id]
        d = 0
        while frontier and (depth is None or d < depth):
            nxt: List[str] = []
            for f in frontier:
                for n in graph.get(f, []):
                    if n not in visited and n != func_id:
                        visited.add(n)
                        nxt.append(n)
            frontier = nxt
            d += 1
        return sorted(visited)

    def get_statistics(self) -> Dict:
        total_edges = sum(len(v) for v in self.call_graph.values())
        out_degrees = [len(v) for v in self.call_graph.values()]
        isolated = sum(
            1 for fid in self.functions
            if not self.call_graph.get(fid) and not self.reverse_call_graph.get(fid)
        )
        return {
            "total_functions": len(self.functions),
            "total_edges": total_edges,
            "avg_out_degree": (total_edges / len(out_degrees)) if out_degrees else 0.0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "isolated_functions": isolated,
            **self.stats_extra,
        }

    def build(self) -> Dict:
        self.build_call_graph()
        return self.export()

    def export(self) -> Dict:
        return {
            "repository": self.repository,
            "functions": self.functions,
            "classes": self.classes,
            "imports": {},
            "call_graph": self.call_graph,
            "reverse_call_graph": self.reverse_call_graph,
            "statistics": self.get_statistics(),
        }


def main():
    import argparse

    argp = argparse.ArgumentParser(description="Build a call graph for a Java repository")
    argp.add_argument("extractor_output", help="Path to function_extractor.py output JSON")
    argp.add_argument("--output", "-o")
    args = argp.parse_args()

    with open_utf8(args.extractor_output) as f:
        extractor_output = json.load(f)

    builder = CallGraphBuilder(extractor_output)
    result = builder.build()

    output = json.dumps(result, indent=2)
    if args.output:
        with open_utf8(args.output, "w") as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
