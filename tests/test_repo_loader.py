"""Tests for repo config schema and loader."""

import pytest
import tempfile
from pathlib import Path
import yaml

from scripts.repo.loader import (
    load_repo_config,
    VALID_LANGUAGES,
    VALID_EXTRACTION_UNITS,
    DEFAULT_INCLUDE,
)


def test_load_minimal_config():
    with tempfile.TemporaryDirectory() as tmp:
        real_path = Path(tmp)
        cfg_path = Path(tmp) / "repo.yaml"
        cfg_path.write_text(yaml.dump({"name": "test-repo", "path": str(real_path)}))
        config = load_repo_config(cfg_path)
        assert config.name == "test-repo"


def test_load_full_config():
    with tempfile.TemporaryDirectory() as tmp:
        real_path = Path(tmp)
        cfg_path = Path(tmp) / "repo.yaml"
        data = {
            "repo": {
                "name": "full-repo",
                "path": str(real_path),
                "language": "python",
                "include": ["src/**/*.py"],
                "extraction": {"units": ["functions", "classes"]},
            }
        }
        cfg_path.write_text(yaml.dump(data))
        config = load_repo_config(cfg_path)
        assert config.name == "full-repo"


def test_missing_required_fields():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "repo.yaml"
        cfg_path.write_text(yaml.dump({"name": "only-name"}))
        with pytest.raises(ValueError, match="path"):
            load_repo_config(cfg_path)


def test_unsupported_language():
    with tempfile.TemporaryDirectory() as tmp:
        real_path = Path(tmp)
        cfg_path = Path(tmp) / "repo.yaml"
        cfg_path.write_text(yaml.dump({"name": "bad", "path": str(real_path), "language": "java"}))
        with pytest.raises(ValueError, match="Unsupported language"):
            load_repo_config(cfg_path)


def test_invalid_extraction_unit():
    with tempfile.TemporaryDirectory() as tmp:
        real_path = Path(tmp)
        cfg_path = Path(tmp) / "repo.yaml"
        data = {"name": "x", "path": str(real_path), "extraction": {"units": ["invalid"]}}
        cfg_path.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="Invalid extraction unit"):
            load_repo_config(cfg_path)


def test_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_repo_config("/nonexistent/path/repo.yaml")


def test_invalid_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "repo.yaml"
        cfg_path.write_text("name: bad\n  path: [unbalanced")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_repo_config(cfg_path)


def test_non_list_include_exclude():
    with tempfile.TemporaryDirectory() as tmp:
        real_path = Path(tmp)
        cfg_path = Path(tmp) / "repo.yaml"
        data = {"name": "coerce", "path": str(real_path), "include": "**/*.py"}
        cfg_path.write_text(yaml.dump(data))
        config = load_repo_config(cfg_path)
        assert isinstance(config.include, list)


def test_constants_exported():
    assert "python" in VALID_LANGUAGES
    assert "functions" in VALID_EXTRACTION_UNITS


def test_cli_smoke_test_entrypoint():
    from scripts.repo import loader
    assert hasattr(loader, "load_repo_config")


# ---------------------------------------------------------------------------
# Milestone 1.3: url / branch / commit fields
# ---------------------------------------------------------------------------

def test_url_only_config_is_valid():
    """A config with url and no path must not raise (no local-dir check)."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "repo.yaml"
        cfg_path.write_text(yaml.dump({"name": "url-repo", "url": "https://example.com/repo.git"}))
        config = load_repo_config(cfg_path)
        assert config.name == "url-repo"
        assert config.url == "https://example.com/repo.git"
        assert config.path is None


def test_url_branch_commit_parsed_onto_config():
    """url, branch, and commit are all parsed correctly from flat yaml."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "repo.yaml"
        data = {
            "name": "git-repo",
            "url": "https://github.com/org/repo.git",
            "branch": "main",
            "commit": "abc1234",
        }
        cfg_path.write_text(yaml.dump(data))
        config = load_repo_config(cfg_path)
        assert config.url == "https://github.com/org/repo.git"
        assert config.branch == "main"
        assert config.commit == "abc1234"


def test_url_branch_commit_parsed_nested():
    """url, branch, and commit are parsed from nested 'repo:' yaml."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "repo.yaml"
        data = {
            "repo": {
                "name": "nested-git-repo",
                "url": "https://github.com/org/repo.git",
                "branch": "develop",
                "commit": "deadbeef",
            }
        }
        cfg_path.write_text(yaml.dump(data))
        config = load_repo_config(cfg_path)
        assert config.url == "https://github.com/org/repo.git"
        assert config.branch == "develop"
        assert config.commit == "deadbeef"


def test_neither_path_nor_url_raises_value_error():
    """A config missing both path and url must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "repo.yaml"
        cfg_path.write_text(yaml.dump({"name": "no-source"}))
        with pytest.raises(ValueError, match="path"):
            load_repo_config(cfg_path)


def test_path_only_config_still_works():
    """Existing path-only configs must continue to work exactly as before."""
    with tempfile.TemporaryDirectory() as tmp:
        real_path = Path(tmp)
        cfg_path = real_path / "repo.yaml"
        cfg_path.write_text(yaml.dump({"name": "path-repo", "path": str(real_path)}))
        config = load_repo_config(cfg_path)
        assert config.name == "path-repo"
        assert config.path == str(real_path)
        assert config.url is None
        assert config.branch is None
        assert config.commit is None


def test_url_skips_local_dir_check_for_nonexistent_path():
    """When url is set, no local-path existence check is performed."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "repo.yaml"
        # path intentionally points to a non-existent directory — must not raise
        cfg_path.write_text(yaml.dump({
            "name": "url-no-path-check",
            "url": "https://example.com/r.git",
            "path": "/nonexistent/does-not-exist-xyz",
        }))
        # url present → local-dir check skipped; should not raise
        config = load_repo_config(cfg_path)
        assert config.url == "https://example.com/r.git"


def test_empty_string_url_no_path_raises():
    """An empty-string url with no path counts as absent and must raise."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "repo.yaml"
        cfg_path.write_text(yaml.dump({"name": "empty-url", "url": ""}))
        with pytest.raises(ValueError, match="path"):
            load_repo_config(cfg_path)


# ---------------------------------------------------------------------------
# Fix #1: wrapped+sibling form — extraction/generation as top-level siblings
# ---------------------------------------------------------------------------

def test_wrapped_sibling_extraction_and_generation():
    """repo: block + sibling extraction:/generation: sections are honored."""
    with tempfile.TemporaryDirectory() as tmp:
        real_path = Path(tmp)
        cfg_path = real_path / "repo.yaml"
        data = {
            "repo": {
                "name": "sibling-repo",
                "path": str(real_path),
            },
            "extraction": {
                "units": ["classes"],
            },
            "generation": {
                "target_records": 7,
                "holdout_ratio": 0.3,
            },
        }
        cfg_path.write_text(yaml.dump(data))
        config = load_repo_config(cfg_path)
        assert config.extraction_units == ["classes"]
        assert config.target_records == 7
        assert config.holdout_ratio == 0.3


# ---------------------------------------------------------------------------
# Fix #4b: relative path resolved against config file's directory
# ---------------------------------------------------------------------------

def test_relative_path_resolved_against_config_dir():
    """A relative path: in repo.yaml is resolved against the config file's dir."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_dir = Path(tmp) / "configs"
        cfg_dir.mkdir()
        # Create a real subdirectory next to the config.
        repo_subdir = cfg_dir / "myrepo"
        repo_subdir.mkdir()
        cfg_path = cfg_dir / "repo.yaml"
        # Use a relative path (just the subdir name).
        cfg_path.write_text(yaml.dump({"name": "rel-repo", "path": "myrepo"}))
        config = load_repo_config(cfg_path)
        assert config.path == str(repo_subdir.resolve())


def test_absolute_path_unchanged():
    """An absolute path: in repo.yaml is stored as-is (not re-resolved)."""
    with tempfile.TemporaryDirectory() as tmp:
        real_path = Path(tmp)
        cfg_path = real_path / "repo.yaml"
        cfg_path.write_text(yaml.dump({"name": "abs-repo", "path": str(real_path)}))
        config = load_repo_config(cfg_path)
        assert config.path == str(real_path)
