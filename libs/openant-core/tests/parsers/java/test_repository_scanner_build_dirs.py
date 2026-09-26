"""Java scanner: build-output dirs are pruned, but a Java PACKAGE that happens
to be named like one (`vag.mb.out`) must not be.

Regression: a real Spring service keeps all its code under
src/main/java/vag/mb/out/..., and the name-only exclusion of `out` (IntelliJ
build dir) silently dropped every source file -> 0 units, no error.
"""
import sys
from pathlib import Path

import pytest

_CORE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CORE_ROOT))

from parsers.java.repository_scanner import RepositoryScanner


def _touch(root: Path, rel: str, text: str = "class X {}") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _scan(root: Path) -> set:
    result = RepositoryScanner(str(root)).scan()
    return {f["path"] for f in result["files"]}


@pytest.mark.parametrize("pkg", ["out", "build", "target", "bin", "dist"])
def test_package_named_like_a_build_dir_is_scanned(tmp_path, pkg):
    _touch(tmp_path, f"vag/mb/{pkg}/controller/A.java")
    assert _scan(tmp_path) == {f"vag/mb/{pkg}/controller/A.java"}


def test_package_named_like_a_build_dir_is_scanned_even_at_project_root_src(tmp_path):
    _touch(tmp_path, "pom.xml", "<project/>")
    _touch(tmp_path, "src/main/java/com/x/out/A.java")
    assert _scan(tmp_path) == {"src/main/java/com/x/out/A.java"}


@pytest.mark.parametrize("marker", ["pom.xml", "build.gradle", "build.gradle.kts"])
@pytest.mark.parametrize("out", ["target", "build", "out", "bin", "dist"])
def test_build_output_next_to_project_marker_is_pruned(tmp_path, marker, out):
    _touch(tmp_path, marker, "")
    _touch(tmp_path, "src/main/java/A.java")
    _touch(tmp_path, f"{out}/generated/G.java")
    assert _scan(tmp_path) == {"src/main/java/A.java"}


def test_build_output_of_a_submodule_is_pruned(tmp_path):
    _touch(tmp_path, "pom.xml", "")
    _touch(tmp_path, "mod/pom.xml", "")
    _touch(tmp_path, "mod/src/main/java/A.java")
    _touch(tmp_path, "mod/target/generated-sources/G.java")
    assert _scan(tmp_path) == {"mod/src/main/java/A.java"}


@pytest.mark.parametrize("d", [".git", ".gradle", ".mvn", ".idea", "node_modules", ".hidden"])
def test_unambiguous_noise_dirs_are_still_pruned_by_name(tmp_path, d):
    _touch(tmp_path, "A.java")
    _touch(tmp_path, f"{d}/B.java")
    assert _scan(tmp_path) == {"A.java"}
