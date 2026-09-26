#!/usr/bin/env python3
"""
Repository Scanner for Java Codebases

Enumerates all Java source files in a repository. Stage 1 of the Java parser.

Usage:
    python repository_scanner.py <repo_path> [--output <file>]

Output (JSON): see parsers/ruby/repository_scanner.py for the shared schema
(``repository``, ``scan_time``, ``files``, ``statistics``).
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from core.repo_walk import ExcludedDirRecorder, walk_repository
from utilities.file_io import open_utf8


class RepositoryScanner:
    """Stage 1 of the Java parser pipeline.

    Traversal itself lives in ``core/repo_walk.py`` (shared across parsers);
    this class only decides which files are Java sources, which are tests,
    and which directories are Java-specific noise (build output, dependency
    caches).
    """

    def __init__(self, repo_path: str, options: Optional[Dict] = None):
        self.repo_path = Path(repo_path).resolve()
        options = options or {}

        # Pruned by name alone: nothing legitimate is ever called this.
        self.exclude_patterns: Set[str] = set(options.get('exclude_patterns', [
            '.git',
            '.gradle',
            '.mvn',
            '.idea',
            'node_modules',
        ]))

        # Build-output directory names are ALSO valid Java package names
        # (`vag.mb.out`, `com.x.build`), so pruning them by name silently drops
        # a whole package. They are pruned only when they sit directly under a
        # build-project root (a directory holding one of `project_markers`).
        self.build_dir_names: Set[str] = set(options.get('build_dir_names', [
            'target',        # Maven
            'build',         # Gradle
            'out',           # IntelliJ
            'bin',
            'dist',
        ]))
        self.project_markers = (
            'pom.xml', 'build.gradle', 'build.gradle.kts',
            'settings.gradle', 'settings.gradle.kts', 'build.xml',
        )
        self._is_project_root_cache: Dict[str, bool] = {}

        self.source_extensions: Set[str] = set(options.get('source_extensions', ['.java']))
        self.skip_tests = options.get('skip_tests', False)

        # Maven/Gradle test conventions: a dedicated `src/test/...` source
        # root (distinct from `src/main/...`), plus the JUnit/TestNG/Surefire/
        # Failsafe filename conventions (`FooTest.java`, `FooTests.java`,
        # `TestFoo.java`, `FooIT.java` for integration tests picked up by
        # failsafe). Anchored to the basename / path component so a
        # legitimate class like `Contest.java` or `TestUtils.java`-adjacent
        # production code isn't misclassified — only the FULL basename
        # (minus `.java`) matching one of these shapes counts.
        self.test_dir_segments = {'test', 'tests'}
        self.test_file_prefixes = ('Test',)
        self.test_file_suffixes = ('Test', 'Tests', 'IT', 'ITCase')

        self.stats: Dict = {}
        self.files: List[Dict] = []
        self._excluded_recorder = ExcludedDirRecorder(self.exclude_patterns)

    def should_exclude_directory(self, dir_name: str) -> bool:
        if dir_name in self.exclude_patterns:
            return True
        if dir_name.startswith('.'):
            return True
        return False

    def _is_project_root(self, rel_dir_parts: tuple) -> bool:
        key = "/".join(rel_dir_parts)
        cached = self._is_project_root_cache.get(key)
        if cached is None:
            d = self.repo_path.joinpath(*rel_dir_parts)
            cached = any((d / m).is_file() for m in self.project_markers)
            self._is_project_root_cache[key] = cached
        return cached

    def is_build_output_path(self, relative_path: str) -> bool:
        """True if any ancestor dir is a build-output dir directly under a project root."""
        parts = Path(relative_path).parts[:-1]
        return any(
            name in self.build_dir_names and self._is_project_root(parts[:i])
            for i, name in enumerate(parts)
        )

    def is_source_file(self, file_name: str) -> bool:
        ext = os.path.splitext(file_name)[1].lower()
        return ext in self.source_extensions

    def is_test_file(self, relative_path: str) -> bool:
        p = Path(relative_path)
        # `src/test/java/...` (Maven/Gradle standard layout) — a `test`
        # segment anywhere in the directory chain, not just adjacent to
        # `src`, since multi-module repos vary (`module/src/test/...`).
        if any(part.lower() in self.test_dir_segments for part in p.parts[:-1]):
            return True
        stem = p.stem  # basename without .java
        if stem.startswith(self.test_file_prefixes):
            return True
        if stem.endswith(self.test_file_suffixes):
            return True
        return False

    def scan_directory(self, dir_path: Path) -> None:
        def _on_file(entry: Path, entry_relative: str) -> None:
            if not self.is_source_file(entry.name):
                return
            if self.is_build_output_path(entry_relative):
                self.stats['build_output_files_skipped'] = self.stats.get('build_output_files_skipped', 0) + 1
                return
            if self.skip_tests and self.is_test_file(entry_relative):
                self.stats['test_files_skipped'] = self.stats.get('test_files_skipped', 0) + 1
                return
            try:
                file_size = entry.stat().st_size
            except OSError:
                file_size = 0
            self.files.append({'path': entry_relative, 'size': file_size})
            self.stats['total_files'] = self.stats.get('total_files', 0) + 1
            self.stats['total_size_bytes'] = self.stats.get('total_size_bytes', 0) + file_size

        walk_repository(
            dir_path,
            should_exclude_directory=self.should_exclude_directory,
            on_file=_on_file,
            stats=self.stats,
            note_excluded=self._excluded_recorder.note,
        )

    def scan(self) -> Dict:
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
        if not self.repo_path.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {self.repo_path}")

        self.files = []
        self.stats = {
            'total_files': 0,
            'total_size_bytes': 0,
            'directories_scanned': 0,
            'directories_excluded': 0,
            'test_files_skipped': 0,
            'directories_read_failed': 0,
        }
        self._excluded_recorder = ExcludedDirRecorder(self.exclude_patterns)

        self.scan_directory(self.repo_path)
        self._excluded_recorder.merge_into(self.stats)
        self.files.sort(key=lambda f: f['path'])

        return {
            'repository': str(self.repo_path),
            'scan_time': datetime.now().isoformat(),
            'files': self.files,
            'statistics': self.stats,
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Scan a Java repository for source files')
    parser.add_argument('repo_path')
    parser.add_argument('--output', '-o')
    parser.add_argument('--skip-tests', action='store_true')
    args = parser.parse_args()

    scanner = RepositoryScanner(args.repo_path, {'skip_tests': args.skip_tests})
    result = scanner.scan()

    output = json.dumps(result, indent=2)
    if args.output:
        with open_utf8(args.output, 'w') as f:
            f.write(output)
    else:
        print(output)


if __name__ == '__main__':
    main()
