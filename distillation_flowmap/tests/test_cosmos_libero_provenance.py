import json
import subprocess
import sys
from pathlib import Path

import pytest

from distillation_flowmap.cosmos_libero_provenance import (
    ProvenanceError,
    build_artifact_lock,
    canonical_json,
    fingerprint_git_repository,
    resolve_formal_provenance,
    verify_artifact_lock,
)


def _git_repo(root: Path) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Fixture"], check=True
    )
    (root / "source.py").write_text("REVISION = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return root


def _artifact(root: Path, *, kind: str) -> tuple[Path, tuple[str, ...], tuple[str, ...]]:
    root.mkdir(parents=True)
    if kind == "dataset":
        compact = (
            "meta/info.json",
            "meta/tasks.jsonl",
            "meta/episodes.jsonl",
            "meta/episodes_ori.jsonl",
            "meta/episodes_stats.jsonl",
            "empty_emb.pt",
        )
        large = ("data/chunk-000/episode_000000.parquet",)
    elif kind == "teacher":
        compact = ("config.json", "libero_dataset_statistics.json")
        large = ("Cosmos-Policy-LIBERO-Predict2-2B.pt", "libero_t5_embeddings.pkl")
    elif kind == "video_vae":
        compact = ("transformer/config.json",)
        large = ("transformer/diffusion_pytorch_model.safetensors",)
    else:
        compact = (
            "config.json",
            "model_index.json",
            "scheduler/scheduler_config.json",
            "tokenizer/tokenizer_config.json",
        )
        large = ("model-480p-16fps.pt", "tokenizer/tokenizer.pth")
    for relative in compact:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"path": relative, "revision": 1}), encoding="utf-8")
    for relative in large:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("large:" + relative + ":v1").encode())
    return root, compact, large


def _fixture(tmp_path: Path):
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    artifacts = {}
    for key, kind in (
        ("dataset", "dataset"),
        ("teacher", "teacher"),
        ("video_vae", "video_vae"),
        ("local_model", "local_model"),
    ):
        root, compact, large = _artifact(tmp_path / key, kind=kind)
        lock = build_artifact_lock(
            root, compact_paths=compact, large_paths=large, immutable_store=False
        )
        lock_path = lock_root / f"{key}.lock.json"
        lock_path.write_text(canonical_json(lock) + "\n", encoding="utf-8")
        artifacts[key] = {"root": root, "lock": lock_path, "compact": compact, "large": large}

    site_packages = tmp_path / "site-packages"
    metadata = site_packages / "torch-2.7.0.dist-info" / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: torch\nVersion: 2.7.0\n", encoding="utf-8")
    return {
        "artifacts": artifacts,
        "flashwam_repo": _git_repo(tmp_path / "flashwam"),
        "cosmos_repo": _git_repo(tmp_path / "cosmos"),
        "worker_python": Path(sys.executable),
        "worker_site_packages": site_packages,
        "extra_pythonpath": (),
    }


def _resolve(fixture, *, verify_large=True):
    artifacts = fixture["artifacts"]
    return resolve_formal_provenance(
        dataset_root=artifacts["dataset"]["root"],
        dataset_lock=artifacts["dataset"]["lock"],
        teacher_root=artifacts["teacher"]["root"],
        teacher_lock=artifacts["teacher"]["lock"],
        video_vae_root=artifacts["video_vae"]["root"],
        video_vae_lock=artifacts["video_vae"]["lock"],
        local_model_root=artifacts["local_model"]["root"],
        local_model_lock=artifacts["local_model"]["lock"],
        flashwam_repo=fixture["flashwam_repo"],
        cosmos_repo=fixture["cosmos_repo"],
        worker_python=fixture["worker_python"],
        worker_site_packages=fixture["worker_site_packages"],
        extra_pythonpath=fixture["extra_pythonpath"],
        verify_large_artifact_digests=verify_large,
    )


@pytest.mark.parametrize(
    ("artifact", "entry_class"),
    (
        ("dataset", "compact"),
        ("dataset", "large"),
        ("teacher", "compact"),
        ("teacher", "large"),
        ("video_vae", "compact"),
        ("local_model", "large"),
    ),
)
def test_same_path_artifact_mutation_fails_against_existing_lock(
    tmp_path, artifact, entry_class
):
    fixture = _fixture(tmp_path)
    _resolve(fixture)
    item = fixture["artifacts"][artifact]
    path = item["root"] / item[entry_class][0]
    old = path.read_bytes()
    path.write_bytes((b"X" + old[1:]) if old else b"X")
    with pytest.raises(ProvenanceError, match="digest|size|lock"):
        _resolve(fixture)


def test_refreshing_lock_after_same_path_mutation_changes_identity(tmp_path):
    fixture = _fixture(tmp_path)
    before = canonical_json(_resolve(fixture))
    item = fixture["artifacts"]["teacher"]
    weight = item["root"] / item["large"][0]
    weight.write_bytes(b"teacher-replacement")
    lock = build_artifact_lock(
        item["root"],
        compact_paths=item["compact"],
        large_paths=item["large"],
        immutable_store=False,
    )
    item["lock"].write_text(canonical_json(lock) + "\n", encoding="utf-8")
    after = canonical_json(_resolve(fixture))
    assert before != after


def test_formal_provenance_rejects_dirty_source_and_head_change_changes_identity(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    before = canonical_json(_resolve(fixture))
    source = fixture["cosmos_repo"] / "source.py"
    source.write_text("REVISION = 2\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="dirty"):
        _resolve(fixture)
    subprocess.run(["git", "-C", str(fixture["cosmos_repo"]), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(fixture["cosmos_repo"]), "commit", "-qm", "revision 2"],
        check=True,
    )
    after = canonical_json(_resolve(fixture))
    assert before != after


def test_worker_package_metadata_mutation_changes_identity(tmp_path):
    fixture = _fixture(tmp_path)
    before = canonical_json(_resolve(fixture))
    metadata = (
        fixture["worker_site_packages"] / "torch-2.7.0.dist-info" / "METADATA"
    )
    metadata.write_text("Name: torch\nVersion: 2.7.1\n", encoding="utf-8")
    after = canonical_json(_resolve(fixture))
    assert before != after


def test_dry_run_resolver_is_read_only_and_returns_bounded_identity(tmp_path):
    fixture = _fixture(tmp_path)
    indexes = []
    for key in ("flashwam_repo", "cosmos_repo"):
        git_dir = subprocess.run(
            ["git", "-C", str(fixture[key]), "rev-parse", "--git-dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dir = Path(git_dir)
        if not git_dir.is_absolute():
            git_dir = fixture[key] / git_dir
        index = git_dir.resolve() / "index"
        indexes.append(
            (
                index,
                index.read_bytes(),
                index.stat().st_size,
                index.stat().st_mtime_ns,
                index.stat().st_ctime_ns,
            )
        )
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    payload = json.loads(canonical_json(_resolve(fixture)))
    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert before == after
    for index, contents, size, mtime_ns, ctime_ns in indexes:
        stat_result = index.stat()
        assert index.read_bytes() == contents
        assert stat_result.st_size == size
        assert stat_result.st_mtime_ns == mtime_ns
        assert stat_result.st_ctime_ns == ctime_ns
        assert not index.with_name("index.lock").exists()
    assert payload["schema"] == "flashwam_cosmos_provenance_v1"
    assert payload["dataset"]["tree_digest"]
    assert payload["teacher"]["tree_digest"]
    assert payload["flashwam_repo"]["kind"] == "git-clean"
    assert payload["cosmos_repo"]["kind"] == "git-clean"
    assert payload["worker"]["identity_sha256"]


def test_large_artifact_lock_requires_immutable_store_or_explicit_verification(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    with pytest.raises(ProvenanceError, match="large|immutable|verify"):
        _resolve(fixture, verify_large=False)


@pytest.mark.parametrize("malformed", ("false", 0, 1, None))
def test_artifact_lock_requires_type_exact_immutable_store_bool(
    tmp_path, malformed
):
    fixture = _fixture(tmp_path)
    item = fixture["artifacts"]["dataset"]
    payload = json.loads(item["lock"].read_text(encoding="utf-8"))
    payload["immutable_store"] = malformed
    item["lock"].write_text(canonical_json(payload) + "\n", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="immutable_store|boolean"):
        verify_artifact_lock(
            item["root"],
            item["lock"],
            verify_large_artifact_digests=True,
        )


def test_build_artifact_lock_rejects_non_boolean_immutable_store(tmp_path):
    root, compact, large = _artifact(tmp_path / "artifact", kind="dataset")
    with pytest.raises(ProvenanceError, match="immutable_store|boolean"):
        build_artifact_lock(
            root,
            compact_paths=compact,
            large_paths=large,
            immutable_store="false",
        )


def test_unattested_immutable_claim_cannot_skip_large_digest(tmp_path):
    fixture = _fixture(tmp_path)
    item = fixture["artifacts"]["dataset"]
    payload = json.loads(item["lock"].read_text(encoding="utf-8"))
    payload["immutable_store"] = True
    item["lock"].write_text(canonical_json(payload) + "\n", encoding="utf-8")

    large = item["root"] / item["large"][0]
    original = large.read_bytes()
    replacement = bytes((value ^ 0xFF) for value in original)
    assert len(replacement) == len(original)
    large.write_bytes(replacement)

    with pytest.raises(ProvenanceError, match="large|immutable|verify|attest"):
        verify_artifact_lock(
            item["root"],
            item["lock"],
            verify_large_artifact_digests=False,
        )


def test_git_fingerprint_disables_optional_locks_and_index_refresh(
    tmp_path, monkeypatch
):
    repo = _git_repo(tmp_path / "repo")
    calls = []
    original = subprocess.run

    def recording_run(*args, **kwargs):
        if args and isinstance(args[0], list) and args[0][:1] == ["git"]:
            calls.append((tuple(args[0]), dict(kwargs.get("env") or {})))
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    fingerprint_git_repository(repo, purpose="fixture")

    assert calls
    for command, environment in calls:
        assert environment["GIT_OPTIONAL_LOCKS"] == "0"
        assert ("-c", "core.refreshIndex=false") == command[1:3]
        assert ("-c", "core.fsmonitor=false") == command[3:5]
