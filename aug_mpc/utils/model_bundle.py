from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _default_workspace_src() -> Path:
    return Path(os.environ.get("AUGMPC_WORKSPACE_SRC", str(Path.home() / "ibrido_ws" / "src"))).expanduser()


def _run_git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(args, cwd=str(cwd), text=True).strip()


def collect_workspace_git_state(src_root: str | Path | None = None) -> list[dict[str, object]]:
    src_path = Path(src_root) if src_root is not None else _default_workspace_src()
    src_path = src_path.expanduser().resolve()
    repos: list[dict[str, object]] = []
    if not src_path.is_dir():
        return repos

    for entry in sorted(src_path.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir() or not (entry / ".git").exists():
            continue
        try:
            commit = _run_git(["git", "rev-parse", "HEAD"], entry)
            branch = _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], entry)
            try:
                remote = _run_git(["git", "remote", "get-url", "origin"], entry)
            except subprocess.CalledProcessError:
                remote = ""
            dirty = bool(_run_git(["git", "status", "--porcelain"], entry))
        except subprocess.CalledProcessError:
            continue
        repos.append(
            {
                "name": entry.name,
                "commit": commit,
                "branch": branch,
                "remote": remote,
                "dirty": dirty,
            }
        )
    return repos



def load_existing_framework_repos(bundle_dir: str | Path) -> list[dict[str, object]] | None:
    if yaml is None:
        return None
    manifest_path = Path(bundle_dir).resolve() / "bundle.yaml"
    if not manifest_path.is_file():
        return None
    try:
        data = yaml.safe_load(manifest_path.read_text()) or {}
    except Exception:
        return None
    repos = ((data.get("framework") or {}).get("repos"))
    if not isinstance(repos, dict):
        return None
    parsed = []
    for name in sorted(repos.keys(), key=str.lower):
        entry = repos.get(name) or {}
        parsed.append({
            "name": name,
            "commit": entry.get("commit", ""),
            "branch": entry.get("branch", ""),
            "remote": entry.get("remote", ""),
            "dirty": bool(entry.get("dirty", False)),
        })
    return parsed


def load_run_metadata_repos(run_metadata_dir: str | Path) -> list[dict[str, object]] | None:
    if yaml is None:
        return None
    metadata_path = Path(run_metadata_dir).resolve()
    repos_path = metadata_path / "git" / "repos.yaml"
    if not repos_path.is_file():
        return None
    try:
        data = yaml.safe_load(repos_path.read_text()) or {}
    except Exception:
        return None
    repos = data.get("repos")
    if not isinstance(repos, dict):
        return None

    parsed = []
    for name in sorted(repos.keys(), key=str.lower):
        entry = repos.get(name) or {}
        repo = {
            "name": name,
            "commit": entry.get("commit", ""),
            "branch": entry.get("branch", ""),
            "remote": entry.get("remote", ""),
            "dirty": bool(entry.get("dirty", False)),
        }
        if entry.get("status"):
            repo["status"] = f"run_metadata/{entry['status']}"
        if entry.get("patch"):
            repo["patch"] = f"run_metadata/{entry['patch']}"
        parsed.append(repo)
    return parsed

def find_preserved_training_cfgs(bundle_dir: str | Path) -> list[str]:
    bundle_path = Path(bundle_dir).resolve()
    cfgs = []
    for cfg in sorted(bundle_path.glob("ibrido_run_*/training_cfg_*")):
        if cfg.is_file():
            cfgs.append(cfg.relative_to(bundle_path).as_posix())
    return cfgs


def infer_checkpoint_file(bundle_dir: str | Path) -> str:
    bundle_path = Path(bundle_dir).resolve()
    bundle_name = bundle_path.name
    preferred = bundle_path / f"{bundle_name}_model"
    if preferred.is_file():
        return preferred.name

    candidates = sorted(
        p.name for p in bundle_path.iterdir() if p.is_file() and p.name.endswith("_model")
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file ending with '_model' found in {bundle_path}")
    raise RuntimeError(
        f"Multiple checkpoint candidates found in {bundle_path}: {', '.join(candidates)}"
    )


def _copy_run_metadata(
    bundle_path: Path,
    run_metadata_dir: str | Path | None,
) -> dict[str, str] | None:
    if run_metadata_dir is None:
        return None

    src = Path(run_metadata_dir).expanduser().resolve()
    if not src.is_dir():
        raise NotADirectoryError(f"Run metadata directory does not exist: {src}")

    dst = bundle_path / "run_metadata"
    if src != dst.resolve():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    manifest = dst / "run_manifest.yaml"
    refs = {"metadata_dir": dst.relative_to(bundle_path).as_posix()}
    if manifest.is_file():
        refs["run_manifest"] = manifest.relative_to(bundle_path).as_posix()
    resolved_config = dst / "resolved_config.yaml"
    if resolved_config.is_file():
        refs["resolved_config"] = resolved_config.relative_to(bundle_path).as_posix()
    launch_dir = dst / "launch"
    for script in sorted(launch_dir.glob("*.sh")):
        command_name = script.stem
        if script.is_file():
            refs[f"launch_{command_name}"] = script.relative_to(bundle_path).as_posix()
    return refs


def _dump_bundle_yaml(
    bundle_name: str,
    checkpoint_file: str,
    training_cfgs: Iterable[str],
    repos: list[dict[str, object]],
    run_metadata: dict[str, str] | None = None,
) -> str:
    lines: list[str] = []
    if run_metadata is None:
        lines.append("bundle_format: augmpc_model_bundle_v1")
    else:
        lines.append("bundle_format: augmpc_model_bundle_v2")
    lines.append(f"bundle_name: {bundle_name}")
    lines.append(f"checkpoint_file: {checkpoint_file}")
    if run_metadata is not None:
        if run_metadata.get("run_manifest"):
            lines.append(f"run_metadata: {run_metadata['run_manifest']}")
        else:
            lines.append(f"run_metadata: {run_metadata['metadata_dir']}")
        if run_metadata.get("resolved_config"):
            lines.append(f"resolved_config: {run_metadata['resolved_config']}")
        launch_commands = {
            key.removeprefix("launch_"): value
            for key, value in run_metadata.items()
            if key.startswith("launch_")
        }
        if launch_commands:
            lines.append("launch_commands:")
            for name in sorted(launch_commands.keys()):
                lines.append(f"  {name}: {launch_commands[name]}")
    training_cfgs = list(training_cfgs)
    if training_cfgs:
        lines.append("preserved_training_cfgs:")
        for cfg in training_cfgs:
            lines.append(f"  - {cfg}")
    else:
        lines.append("preserved_training_cfgs: []")
    lines.append("framework:")
    if repos:
        lines.append("  repos:")
    else:
        lines.append("  repos: {}")
    for repo in repos:
        lines.append(f"    {repo['name']}:")
        lines.append(f"      commit: {repo['commit']}")
        lines.append(f"      branch: {repo['branch']}")
        lines.append(f"      remote: {repo['remote']}")
        lines.append(f"      dirty: {'true' if repo['dirty'] else 'false'}")
        if repo.get("status"):
            lines.append(f"      status: {repo['status']}")
        if repo.get("patch"):
            lines.append(f"      patch: {repo['patch']}")
    return "\n".join(lines) + "\n"


def write_bundle_manifest(
    bundle_dir: str | Path,
    checkpoint_file: str | None = None,
    src_root: str | Path | None = None,
    preserve_existing_framework: bool = False,
    run_metadata_dir: str | Path | None = None,
) -> Path:
    bundle_path = Path(bundle_dir).resolve()
    if not bundle_path.is_dir():
        raise NotADirectoryError(f"Bundle directory does not exist: {bundle_path}")

    checkpoint_name = checkpoint_file or infer_checkpoint_file(bundle_path)
    training_cfgs = find_preserved_training_cfgs(bundle_path)
    repos = None
    run_metadata = _copy_run_metadata(
        bundle_path,
        run_metadata_dir or os.environ.get("IBRIDO_RUN_META_DIR"),
    )
    if run_metadata is not None:
        copied_metadata_path = bundle_path / run_metadata["metadata_dir"]
        repos = load_run_metadata_repos(copied_metadata_path)
    if preserve_existing_framework:
        repos = load_existing_framework_repos(bundle_path)
    if repos is None:
        repos = collect_workspace_git_state(src_root)
    manifest = _dump_bundle_yaml(
        bundle_path.name,
        checkpoint_name,
        training_cfgs,
        repos,
        run_metadata=run_metadata,
    )

    out = bundle_path / "bundle.yaml"
    out.write_text(manifest)
    return out
