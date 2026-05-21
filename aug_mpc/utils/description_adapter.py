import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


_PKG_FIND_PATTERNS = (
    re.compile(r"\$\(find\s+([A-Za-z0-9_\-]+)\)"),
    re.compile(r"\$\(find-pkg-share\s+([A-Za-z0-9_\-]+)\)"),
)

_TEXT_SUFFIXES = {
    ".xacro",
    ".urdf",
    ".srdf",
    ".xml",
    ".launch",
    ".yaml",
    ".yml",
}


def prepare_xrdf_input(
        xacro_path: str,
        dump_path: str = "/tmp") -> Tuple[str, Dict[str, str]]:
    """Return an xacro path that can be resolved without ROS package install.

    Some upstream description packages, such as PAL's Talos description, use
    `$(find package)` inside xacro files. IBRIDO usually points directly to
    source-tree xacros, so those packages may not be present in the ament
    index. When `$(find ...)` is detected, this function creates a temporary
    adapted copy of the relevant source packages and rewrites those includes to
    absolute paths inside the adapted copy.
    """

    xacro_path = os.path.abspath(os.path.expanduser(xacro_path))
    source_text = _read_text_if_possible(xacro_path)
    if "$(" not in source_text:
        return xacro_path, _discover_packages(_workspace_src_roots(xacro_path))

    src_roots = _workspace_src_roots(xacro_path)
    package_map = _discover_packages(src_roots)
    package_name, package_root = _package_for_path(xacro_path, package_map)
    required_packages = _dependency_closure({package_name}, package_map)

    adapted_root = tempfile.mkdtemp(
        prefix="augmpc_xrdf_",
        dir=dump_path if os.path.isdir(dump_path) else None)

    adapted_package_map = {}
    for pkg in sorted(pkg for pkg in required_packages if pkg in package_map):
        adapted_package_map[pkg] = os.path.join(adapted_root, pkg)
        shutil.copytree(
            package_map[pkg],
            adapted_package_map[pkg],
            ignore=_copy_ignore)

    for pkg_path in adapted_package_map.values():
        _rewrite_package_finds(pkg_path, adapted_package_map)

    rel_path = os.path.relpath(xacro_path, package_root)
    adapted_xacro_path = os.path.join(adapted_package_map[package_name], rel_path)
    return adapted_xacro_path, package_map


def rewrite_package_uris(xrdf_path: str, package_map: Dict[str, str]) -> None:
    """Rewrite package:// mesh/resource URIs in a generated URDF/SRDF file."""

    path = Path(xrdf_path)
    text = _read_text_if_possible(str(path))
    if "package://" not in text:
        return

    for pkg, pkg_path in package_map.items():
        text = text.replace(f"package://{pkg}/", f"{pkg_path}/")

    path.write_text(text)


def _workspace_src_roots(reference_path: str) -> List[str]:
    roots = []

    env_root = os.environ.get("IBRIDO_WS_SRC")
    if env_root:
        roots.append(env_root)

    path_parts = Path(reference_path).parts
    if "src" in path_parts:
        src_idx = len(path_parts) - 1 - list(reversed(path_parts)).index("src")
        roots.append(os.path.join(*path_parts[:src_idx + 1]))

    home_root = os.path.expanduser("~/ibrido_ws/src")
    roots.append(home_root)

    deduped = []
    for root in roots:
        root = os.path.abspath(os.path.expanduser(root))
        if os.path.isdir(root) and root not in deduped:
            deduped.append(root)

    return deduped


def _discover_packages(src_roots: List[str]) -> Dict[str, str]:
    package_map = {}
    ignored_dirs = {".git", "build", "install", "log", "__pycache__"}

    for src_root in src_roots:
        for root, dirs, files in os.walk(src_root):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            if "package.xml" not in files:
                continue
            package_name = _package_name_from_xml(os.path.join(root, "package.xml"))
            if package_name and package_name not in package_map:
                package_map[package_name] = root

    return package_map


def _package_name_from_xml(package_xml: str) -> Optional[str]:
    try:
        root = ET.parse(package_xml).getroot()
    except ET.ParseError:
        return None

    name = root.find("name")
    if name is None or name.text is None:
        return None

    return name.text.strip()


def _package_for_path(
        path: str,
        package_map: Dict[str, str]) -> Tuple[str, str]:
    path = os.path.abspath(path)
    best_pkg = None
    best_root = None

    for pkg, root in package_map.items():
        root_abs = os.path.abspath(root)
        try:
            common = os.path.commonpath([path, root_abs])
        except ValueError:
            continue
        if common != root_abs:
            continue
        if best_root is None or len(root_abs) > len(best_root):
            best_pkg = pkg
            best_root = root_abs

    if best_pkg is None or best_root is None:
        raise FileNotFoundError(f"No package.xml ancestor found for {path}")

    return best_pkg, best_root


def _dependency_closure(
        package_names: Set[str],
        package_map: Dict[str, str]) -> Set[str]:
    required = set(package_names)
    pending = list(package_names)

    while pending:
        package_name = pending.pop()
        package_root = package_map.get(package_name)
        if package_root is None:
            continue

        refs = _package_find_refs(package_root)
        for ref in refs:
            if ref not in required:
                required.add(ref)
                pending.append(ref)

    return required


def _package_find_refs(package_root: str) -> Set[str]:
    refs = set()
    for root, dirs, files in os.walk(package_root):
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "build", "install", "log", "__pycache__"}
        ]
        for fname in files:
            path = os.path.join(root, fname)
            if not _is_text_candidate(path):
                continue
            refs.update(_find_refs_in_text(_read_text_if_possible(path)))

    return refs


def _find_refs_in_text(text: str) -> Set[str]:
    refs = set()
    for pattern in _PKG_FIND_PATTERNS:
        refs.update(pattern.findall(text))
    return refs


def _rewrite_package_finds(
        package_root: str,
        adapted_package_map: Dict[str, str]) -> None:
    for root, dirs, files in os.walk(package_root):
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "build", "install", "log", "__pycache__"}
        ]
        for fname in files:
            path = os.path.join(root, fname)
            if not _is_text_candidate(path):
                continue

            text = _read_text_if_possible(path)
            if "$(" not in text:
                continue

            rewritten = text
            for pattern in _PKG_FIND_PATTERNS:
                rewritten = pattern.sub(
                    lambda match: adapted_package_map.get(
                        match.group(1),
                        match.group(0)),
                    rewritten)

            if rewritten != text:
                Path(path).write_text(rewritten)


def _is_text_candidate(path: str) -> bool:
    suffixes = Path(path).suffixes
    return any(suffix in _TEXT_SUFFIXES for suffix in suffixes)


def _read_text_if_possible(path: str) -> str:
    try:
        return Path(path).read_text()
    except UnicodeDecodeError:
        return ""


def _copy_ignore(dir_path: str, names: List[str]) -> Set[str]:
    ignored = {
        ".git",
        "build",
        "install",
        "log",
        "__pycache__",
    }
    return {name for name in names if name in ignored}
