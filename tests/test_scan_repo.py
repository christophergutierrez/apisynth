"""Tests for the repository scanner (Milestone 1.2 + 1.3)."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from scripts.repo.loader import RepoConfig, load_repo_config
from scripts.repo.scan_repo import scan_repo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_repo"


def _make_config(path, include=None, exclude=None):
    """Build a minimal RepoConfig without touching disk."""
    return RepoConfig(
        name="test",
        path=str(path),
        include=include or ["**/*.py"],
        exclude=exclude or [],
    )


# ---------------------------------------------------------------------------
# Original tests (must remain passing)
# ---------------------------------------------------------------------------

def test_scan_basic_repo():
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)
        (repo_path / "example.py").write_text("""
def hello():
    return "world"

class Foo:
    def bar(self):
        pass
""")

        cfg_path = repo_path / "repo.yaml"
        cfg_path.write_text(f"""
name: test-scan
path: {repo_path}
""")

        config = load_repo_config(cfg_path)
        units = scan_repo(config)

        assert any(u["type"] == "function" and u["name"] == "hello" for u in units)
        assert any(u["type"] == "class" and u["name"] == "Foo" for u in units)


def test_scan_respects_exclude():
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)
        (repo_path / "main.py").write_text("def main(): pass")
        (repo_path / "ignored.py").write_text("def ignored(): pass")

        cfg_path = repo_path / "repo.yaml"
        cfg_path.write_text(f"""
name: exclude-test
path: {repo_path}
exclude: ["ignored.py"]
""")

        config = load_repo_config(cfg_path)
        units = scan_repo(config)

        names = [u["name"] for u in units]
        assert "main" in names
        assert "ignored" not in names


# ---------------------------------------------------------------------------
# Fixture-based tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_repo_config():
    """RepoConfig pointing at the sample_repo fixture (all files)."""
    return _make_config(FIXTURES_DIR)


def test_fixture_exists():
    assert FIXTURES_DIR.is_dir(), f"Fixture directory not found: {FIXTURES_DIR}"


def test_fixture_yields_at_least_30_units(sample_repo_config):
    units = scan_repo(sample_repo_config)
    assert len(units) >= 30, (
        f"Expected >=30 units, got {len(units)}. "
        f"Breakdown: {_type_breakdown(units)}"
    )


def _type_breakdown(units):
    from collections import Counter
    return dict(Counter(u["type"] for u in units))


def test_fixture_has_all_unit_types(sample_repo_config):
    units = scan_repo(sample_repo_config)
    types = {u["type"] for u in units}
    assert "function" in types, "Expected top-level functions"
    assert "method" in types, "Expected methods"
    assert "class" in types, "Expected classes"
    assert "api_call" in types, "Expected API call sites"


# ---------------------------------------------------------------------------
# Method vs function distinction (using AST parentage, not heuristics)
# ---------------------------------------------------------------------------

def test_method_vs_function_distinction():
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)
        (repo_path / "mixed.py").write_text("""
def top_level():
    pass

class MyClass:
    def my_method(self):
        pass

    def another_method(self):
        pass
""")
        config = _make_config(repo_path)
        units = scan_repo(config)

        top_level = [u for u in units if u["name"] == "top_level"]
        assert len(top_level) == 1
        assert top_level[0]["type"] == "function"
        assert "class" not in top_level[0]

        my_method = [u for u in units if u["name"] == "my_method"]
        assert len(my_method) == 1
        assert my_method[0]["type"] == "method"
        assert my_method[0]["class"] == "MyClass"

        another = [u for u in units if u["name"] == "another_method"]
        assert len(another) == 1
        assert another[0]["type"] == "method"
        assert another[0]["class"] == "MyClass"


def test_method_carries_parent_class_name():
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)
        (repo_path / "cls.py").write_text("""
class Alpha:
    def alpha_method(self): pass

class Beta:
    def beta_method(self): pass
""")
        config = _make_config(repo_path)
        units = scan_repo(config)

        alpha_m = next(u for u in units if u["name"] == "alpha_method")
        beta_m = next(u for u in units if u["name"] == "beta_method")
        assert alpha_m["class"] == "Alpha"
        assert beta_m["class"] == "Beta"


# ---------------------------------------------------------------------------
# Include / exclude glob patterns — exercises ** semantics
# ---------------------------------------------------------------------------

def test_exclude_glob_star_star_excludes_subdir():
    """Units from excluded/ subdir must be absent when excluded with ** glob."""
    config = _make_config(FIXTURES_DIR, exclude=["excluded/**"])
    units = scan_repo(config)
    names = {u["name"] for u in units}
    # These are defined only in excluded/secret.py
    assert "this_should_not_appear" not in names
    assert "AlsoExcluded" not in names
    assert "hidden_method" not in names


def test_exclude_glob_double_star_pattern():
    """Exclude pattern **/excluded/*.py should drop files in that dir."""
    config = _make_config(FIXTURES_DIR, exclude=["**/excluded/*.py"])
    units = scan_repo(config)
    names = {u["name"] for u in units}
    assert "this_should_not_appear" not in names


def test_include_only_specific_subdir():
    """Include only core/**/*.py — api and utils units must be absent."""
    config = _make_config(FIXTURES_DIR, include=["core/**/*.py"])
    units = scan_repo(config)
    # core/models.py defines create_user — must be present
    assert any(u["name"] == "create_user" for u in units)
    # api/client.py defines fetch_weather — must be absent
    assert not any(u["name"] == "fetch_weather" for u in units)


def test_included_file_present_excluded_file_absent_star_star():
    """Combine include=core/**/*.py with exclude=**/auth.py."""
    config = _make_config(
        FIXTURES_DIR,
        include=["core/**/*.py"],
        exclude=["**/auth.py"],
    )
    units = scan_repo(config)
    names = {u["name"] for u in units}
    # models.py should be scanned
    assert "create_user" in names
    # auth.py should be excluded
    assert "verify_token" not in names
    assert "AuthClient" not in names


def test_dot_prefixed_exclude_pattern_not_mangled(tmp_path):
    """Dot-prefixed exclude patterns (.hidden/**, .env) must match correctly.

    Regression for an `str.lstrip("./")` bug that stripped a character SET and
    mangled patterns like ".env" -> "env" or ".github/**" -> "github/**".
    """
    # A dot-directory and a dot-file, plus a normal file that must survive.
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "secret.py").write_text("def hidden_secret(): pass")
    (tmp_path / ".env.py").write_text("def env_thing(): pass")
    (tmp_path / "keep.py").write_text("def keep_me(): pass")

    config = _make_config(tmp_path, exclude=[".hidden/**", ".env.py"])
    units = scan_repo(config)
    names = {u["name"] for u in units}

    # Dot-prefixed patterns must actually exclude their targets.
    assert "hidden_secret" not in names
    assert "env_thing" not in names
    # The normal file must still be scanned (proves we did not over-exclude).
    assert "keep_me" in names


# ---------------------------------------------------------------------------
# Graceful error handling
# ---------------------------------------------------------------------------

def test_unparsable_file_is_skipped():
    """A file with invalid syntax must be skipped; other units still returned."""
    # bad_syntax.py is in FIXTURES_DIR root — it has deliberate syntax errors
    config = _make_config(FIXTURES_DIR)
    units = scan_repo(config)  # must not raise
    # Other files should still be scanned
    assert len(units) >= 1


def test_unparsable_file_does_not_crash(tmp_path):
    """Verify graceful skip for a fresh unparsable file alongside valid ones."""
    (tmp_path / "good.py").write_text("def good(): pass")
    (tmp_path / "bad.py").write_text("def broken(\n")  # syntax error

    config = _make_config(tmp_path)
    units = scan_repo(config)  # must not raise

    names = [u["name"] for u in units]
    assert "good" in names
    assert "broken" not in names


def test_permission_error_file_is_skipped(tmp_path):
    """Files that can't be read (permission denied) must be skipped gracefully."""
    import os

    (tmp_path / "readable.py").write_text("def readable(): pass")
    unreadable = tmp_path / "unreadable.py"
    unreadable.write_text("def secret(): pass")
    os.chmod(unreadable, 0o000)

    try:
        config = _make_config(tmp_path)
        units = scan_repo(config)  # must not raise
        names = [u["name"] for u in units]
        assert "readable" in names
        # "secret" may or may not appear depending on whether we run as root
    finally:
        os.chmod(unreadable, 0o644)  # restore so cleanup works


# ---------------------------------------------------------------------------
# API call detection
# ---------------------------------------------------------------------------

def test_api_calls_detected():
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)
        (repo_path / "calls.py").write_text("""
import requests

def do_stuff():
    resp = requests.get("https://example.com")
    resp2 = requests.post("https://example.com", json={})
""")
        config = _make_config(repo_path)
        units = scan_repo(config)
        api_calls = [u for u in units if u["type"] == "api_call"]
        assert any("requests.get" in u["name"] for u in api_calls)
        assert any("requests.post" in u["name"] for u in api_calls)


def test_httpx_calls_detected():
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)
        (repo_path / "async_calls.py").write_text("""
import httpx

async def fetch(url: str):
    async with httpx.AsyncClient() as client:
        return await client.get(url)
""")
        config = _make_config(repo_path)
        units = scan_repo(config)
        api_calls = [u for u in units if u["type"] == "api_call"]
        assert any(u["name"] for u in api_calls)


def test_session_calls_detected():
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)
        (repo_path / "session_calls.py").write_text("""
import requests

session = requests.Session()

def upload(data):
    session.post("https://example.com/upload", json=data)
    session.get("https://example.com/check")
""")
        config = _make_config(repo_path)
        units = scan_repo(config)
        api_calls = [u for u in units if u["type"] == "api_call"]
        assert any("post" in u["name"] for u in api_calls)
        assert any("get" in u["name"] for u in api_calls)


def test_non_client_verb_calls_not_flagged():
    """Plain `.get()`/`.call()` on non-client receivers must NOT be api_calls.

    Regression: the verb heuristic previously flagged ANY `.get()/.call()` etc.
    as an api_call, polluting output with dict/queue accesses.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)
        (repo_path / "noise.py").write_text("""
def process(d, q, cb):
    val = d.get("k")            # dict access, not an API call
    item = q.get()              # queue access, not an API call
    result = cb.call()          # generic callable, not an API call
    cfg = self.cache.get("x")   # cache access, not an API call
    return val, item, result, cfg
""")
        config = _make_config(repo_path)
        units = scan_repo(config)
        api_calls = [u for u in units if u["type"] == "api_call"]
        names = {u["name"] for u in api_calls}
        assert "d.get" not in names
        assert "q.get" not in names
        assert "cb.call" not in names
        assert "cache.get" not in names
        # None of these noise calls should be flagged at all.
        assert api_calls == [], f"Unexpected api_calls: {api_calls}"


# ---------------------------------------------------------------------------
# Async function support
# ---------------------------------------------------------------------------

def test_async_functions_extracted():
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)
        (repo_path / "async_mod.py").write_text("""
async def async_top():
    pass

class AsyncWorker:
    async def run(self):
        pass
""")
        config = _make_config(repo_path)
        units = scan_repo(config)

        assert any(u["name"] == "async_top" and u["type"] == "function" for u in units)
        assert any(u["name"] == "run" and u["type"] == "method" for u in units)


# ---------------------------------------------------------------------------
# Unit count / type breakdown for the fixture (informational assertion)
# ---------------------------------------------------------------------------

def test_fixture_type_breakdown(sample_repo_config):
    """Checks minimum counts per type from the fixture."""
    units = scan_repo(sample_repo_config)
    breakdown = _type_breakdown(units)

    assert breakdown.get("class", 0) >= 5, f"Need >=5 classes, got {breakdown}"
    assert breakdown.get("method", 0) >= 10, f"Need >=10 methods, got {breakdown}"
    assert breakdown.get("function", 0) >= 5, f"Need >=5 functions, got {breakdown}"
    assert breakdown.get("api_call", 0) >= 3, f"Need >=3 api_calls, got {breakdown}"


# ---------------------------------------------------------------------------
# Milestone 1.3: clone-to-temp via url
# ---------------------------------------------------------------------------

def _git_available():
    """Return True if git is available on PATH."""
    try:
        result = subprocess.run(["git", "--version"], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _make_git_repo(tmp):
    """Create a minimal local git repo with a function and a class."""
    repo = os.path.join(tmp, "src_repo")
    os.makedirs(repo)
    # Write two Python files
    with open(os.path.join(repo, "module.py"), "w") as f:
        f.write(
            "def greet(name):\n"
            "    return f'Hello {name}'\n"
            "\n"
            "class Greeter:\n"
            "    def say_hello(self):\n"
            "        return greet('world')\n"
        )
    subprocess.run(["git", "init", "-q", repo], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", repo, "add", "."], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-q", "-m", "init"], check=True)
    return repo


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_clone_url_returns_expected_units():
    """scan_repo with url clones the repo and returns the correct units."""
    with tempfile.TemporaryDirectory() as tmp:
        src_repo = _make_git_repo(tmp)
        config = RepoConfig(name="clone-test", url=src_repo)
        units = scan_repo(config)
        names = {u["name"] for u in units}
        assert "greet" in names
        assert "Greeter" in names
        assert "say_hello" in names
        types = {u["type"] for u in units}
        assert "function" in types
        assert "class" in types
        assert "method" in types


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_clone_branch_checkout():
    """scan_repo with url+branch must clone the named branch ref.

    The source repo's default HEAD does NOT contain feature_func (we switch
    back to the default branch after creating the feature branch), so finding
    feature_func proves --branch drove the checkout, not the source's HEAD.
    """
    with tempfile.TemporaryDirectory() as tmp:
        src_repo = _make_git_repo(tmp)
        # Capture the default branch name before creating a new one.
        default_branch = subprocess.run(
            ["git", "-C", src_repo, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        # Create a new branch with an extra file
        subprocess.run(
            ["git", "-C", src_repo, "checkout", "-q", "-b", "feature"],
            check=True,
        )
        with open(os.path.join(src_repo, "extra.py"), "w") as f:
            f.write("def feature_func(): pass\n")
        subprocess.run(["git", "-C", src_repo, "add", "."], check=True)
        subprocess.run(
            ["git", "-C", src_repo, "commit", "-q", "-m", "add feature"],
            check=True,
        )

        # Switch source HEAD back to the default branch so its HEAD does NOT
        # contain feature_func — the branch ref is the only place it exists.
        subprocess.run(
            ["git", "-C", src_repo, "checkout", "-q", default_branch],
            check=True,
        )

        config = RepoConfig(name="branch-test", url=src_repo, branch="feature")
        units = scan_repo(config)
        names = {u["name"] for u in units}
        assert "feature_func" in names  # proves --branch feature was used
        assert "greet" in names  # original file still present


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_clone_commit_checkout():
    """scan_repo with url+commit checks out the exact SHA."""
    with tempfile.TemporaryDirectory() as tmp:
        src_repo = _make_git_repo(tmp)
        # Capture the initial commit SHA
        result = subprocess.run(
            ["git", "-C", src_repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        initial_sha = result.stdout.strip()

        # Add another commit on top
        with open(os.path.join(src_repo, "new_file.py"), "w") as f:
            f.write("def new_func(): pass\n")
        subprocess.run(["git", "-C", src_repo, "add", "."], check=True)
        subprocess.run(
            ["git", "-C", src_repo, "commit", "-q", "-m", "second commit"],
            check=True,
        )

        # Clone at the initial commit — new_func must NOT be present
        config = RepoConfig(name="commit-test", url=src_repo, commit=initial_sha)
        units = scan_repo(config)
        names = {u["name"] for u in units}
        assert "greet" in names
        assert "new_func" not in names


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_clone_bad_url_raises_runtime_error():
    """scan_repo with a nonexistent url raises RuntimeError."""
    config = RepoConfig(name="bad-url", url="/nonexistent/repo.git")
    with pytest.raises(RuntimeError, match="git clone failed"):
        scan_repo(config)


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_clone_dash_prefixed_url_rejected(tmp_path):
    """A url starting with '-' must be rejected before any clone proceeds.

    Guards against argument-injection (e.g. --upload-pack=...). The guard
    raises before mkdtemp/clone, so no clone side-effect can occur.
    """
    sentinel = tmp_path / "sentinel"
    config = RepoConfig(name="dash-url", url=f"--upload-pack=touch {sentinel}")
    with pytest.raises((ValueError, RuntimeError)):
        scan_repo(config)
    # No clone proceeded → the injected command never ran.
    assert not sentinel.exists()


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_clone_temp_dir_cleaned_up_on_success():
    """After scan_repo completes, the temp clone dir must be removed."""
    import gc
    with tempfile.TemporaryDirectory() as tmp:
        src_repo = _make_git_repo(tmp)

        # Intercept mkdtemp to track the temp dir path
        created_dirs = []
        import tempfile as _tf
        original_mkdtemp = _tf.mkdtemp

        def tracking_mkdtemp(*args, **kwargs):
            d = original_mkdtemp(*args, **kwargs)
            created_dirs.append(d)
            return d

        import scripts.repo.scan_repo as scan_mod
        original = scan_mod.tempfile.mkdtemp
        scan_mod.tempfile.mkdtemp = tracking_mkdtemp
        try:
            config = RepoConfig(name="cleanup-test", url=src_repo)
            scan_repo(config)
        finally:
            scan_mod.tempfile.mkdtemp = original

        # All tracked temp dirs must have been removed
        for d in created_dirs:
            assert not Path(d).exists(), f"Temp dir {d} was not cleaned up"


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_clone_temp_dir_cleaned_up_on_failure():
    """After a failed clone, the temp dir must still be removed."""
    import scripts.repo.scan_repo as scan_mod

    created_dirs = []
    original_mkdtemp = scan_mod.tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        d = original_mkdtemp(*args, **kwargs)
        created_dirs.append(d)
        return d

    scan_mod.tempfile.mkdtemp = tracking_mkdtemp
    try:
        config = RepoConfig(name="fail-cleanup", url="/nonexistent/path.git")
        with pytest.raises(RuntimeError):
            scan_repo(config)
    finally:
        scan_mod.tempfile.mkdtemp = original_mkdtemp

    for d in created_dirs:
        assert not Path(d).exists(), f"Temp dir {d} was not cleaned up after failure"


@pytest.mark.skipif(not _git_available(), reason="git not available")
def test_path_only_config_unchanged_behavior():
    """A config with only path (no url) must behave exactly as before."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp)
        (repo_path / "example.py").write_text(
            "def hello(): pass\n\nclass World:\n    def greet(self): pass\n"
        )
        config = RepoConfig(name="path-only", path=str(repo_path))
        units = scan_repo(config)
        names = {u["name"] for u in units}
        assert "hello" in names
        assert "World" in names


# ---------------------------------------------------------------------------
# Milestone 3.3: is_stub detection in scan_repo
# ---------------------------------------------------------------------------

from scripts.repo.scan_repo import _is_stub_body
import ast as _ast


def _parse_func(src: str):
    """Parse src as a module and return the first FunctionDef node."""
    tree = _ast.parse(src)
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            return node
    raise AssertionError(f"No function found in: {src!r}")


class TestIsStubBody:
    """Unit tests for the _is_stub_body helper."""

    def test_pass_body_is_stub(self):
        node = _parse_func("def f(): pass")
        assert _is_stub_body(node) is True

    def test_ellipsis_body_is_stub(self):
        node = _parse_func("def f(): ...")
        assert _is_stub_body(node) is True

    def test_raise_not_implemented_bare_is_stub(self):
        node = _parse_func("def f(): raise NotImplementedError")
        assert _is_stub_body(node) is True

    def test_raise_not_implemented_call_is_stub(self):
        node = _parse_func("def f(): raise NotImplementedError('not yet')")
        assert _is_stub_body(node) is True

    def test_docstring_only_is_stub(self):
        src = 'def f():\n    """Docstring only."""\n'
        node = _parse_func(src)
        assert _is_stub_body(node) is True

    def test_docstring_then_pass_is_stub(self):
        src = 'def f():\n    """Docstring."""\n    pass\n'
        node = _parse_func(src)
        assert _is_stub_body(node) is True

    def test_docstring_then_ellipsis_is_stub(self):
        src = 'def f():\n    """Docstring."""\n    ...\n'
        node = _parse_func(src)
        assert _is_stub_body(node) is True

    def test_docstring_then_raise_is_stub(self):
        src = 'def f():\n    """Docstring."""\n    raise NotImplementedError\n'
        node = _parse_func(src)
        assert _is_stub_body(node) is True

    def test_real_body_is_not_stub(self):
        node = _parse_func("def f(x): return x + 1")
        assert _is_stub_body(node) is False

    def test_assignment_body_is_not_stub(self):
        node = _parse_func("def f():\n    x = 1\n    return x\n")
        assert _is_stub_body(node) is False

    def test_raise_other_exception_is_not_stub(self):
        node = _parse_func("def f(): raise ValueError('bad')")
        assert _is_stub_body(node) is False


class TestScanRepoIsStubKey:
    """Integration: scan_repo sets is_stub ONLY on qualifying units."""

    def test_pass_method_gets_is_stub(self, tmp_path):
        (tmp_path / "stubs.py").write_text(
            "class A:\n"
            "    def stub_method(self): pass\n"
        )
        config = _make_config(tmp_path)
        units = scan_repo(config)
        stub_m = next(u for u in units if u["name"] == "stub_method")
        assert stub_m.get("is_stub") is True

    def test_ellipsis_function_gets_is_stub(self, tmp_path):
        (tmp_path / "stubs.py").write_text("def stub_fn(): ...\n")
        config = _make_config(tmp_path)
        units = scan_repo(config)
        stub_f = next(u for u in units if u["name"] == "stub_fn")
        assert stub_f.get("is_stub") is True

    def test_raise_not_implemented_method_gets_is_stub(self, tmp_path):
        (tmp_path / "stubs.py").write_text(
            "class B:\n"
            "    def abstract_method(self):\n"
            "        raise NotImplementedError\n"
        )
        config = _make_config(tmp_path)
        units = scan_repo(config)
        m = next(u for u in units if u["name"] == "abstract_method")
        assert m.get("is_stub") is True

    def test_docstring_only_method_gets_is_stub(self, tmp_path):
        (tmp_path / "stubs.py").write_text(
            'class C:\n'
            '    def doc_only(self):\n'
            '        """Docstring only."""\n'
        )
        config = _make_config(tmp_path)
        units = scan_repo(config)
        m = next(u for u in units if u["name"] == "doc_only")
        assert m.get("is_stub") is True

    def test_real_body_function_has_no_is_stub_key(self, tmp_path):
        (tmp_path / "real.py").write_text("def compute(x): return x * 2\n")
        config = _make_config(tmp_path)
        units = scan_repo(config)
        f = next(u for u in units if u["name"] == "compute")
        assert "is_stub" not in f, "Non-stub unit must NOT carry is_stub key"

    def test_real_body_method_has_no_is_stub_key(self, tmp_path):
        (tmp_path / "real.py").write_text(
            "class D:\n"
            "    def do_work(self, x):\n"
            "        return x + 1\n"
        )
        config = _make_config(tmp_path)
        units = scan_repo(config)
        m = next(u for u in units if u["name"] == "do_work")
        assert "is_stub" not in m, "Non-stub method must NOT carry is_stub key"

    def test_class_unit_never_has_is_stub(self, tmp_path):
        (tmp_path / "cls.py").write_text("class MyClass: pass\n")
        config = _make_config(tmp_path)
        units = scan_repo(config)
        cls_unit = next(u for u in units if u["type"] == "class")
        assert "is_stub" not in cls_unit

    def test_api_call_unit_never_has_is_stub(self, tmp_path):
        (tmp_path / "api.py").write_text(
            "import requests\n"
            "def fetch():\n"
            "    requests.get('https://example.com')\n"
        )
        config = _make_config(tmp_path)
        units = scan_repo(config)
        api_units = [u for u in units if u["type"] == "api_call"]
        assert api_units, "Expected at least one api_call unit"
        for u in api_units:
            assert "is_stub" not in u

    def test_scan_repo_still_returns_all_units_including_stubs(self, tmp_path):
        """scan_repo must NOT filter — all units are returned regardless of is_stub."""
        (tmp_path / "mixed.py").write_text(
            "def real(x): return x\n"
            "def stub(): pass\n"
        )
        config = _make_config(tmp_path)
        units = scan_repo(config)
        names = {u["name"] for u in units}
        assert "real" in names
        assert "stub" in names
