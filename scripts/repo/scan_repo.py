"""Repository scanner for apisynth.

Scans a local Python repository and extracts code units (functions, methods,
classes, and best-effort API call sites) for synthetic data generation.
"""

from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import List, Dict, Any
import ast
import copy
import fnmatch
import shutil
import subprocess
import tempfile


# Attribute names considered to make an API call when used as the receiver of a
# .get/.post/… call (best-effort heuristic, not exhaustive).
_API_CALL_METHODS = frozenset({
    "get", "post", "put", "patch", "delete", "head", "options", "request",
    "fetch", "send", "call", "invoke",
})

# Module-level names commonly associated with HTTP clients.
_API_MODULES = frozenset({
    "requests", "httpx", "aiohttp", "urllib", "urllib3", "session",
    "client", "http_client",
})

# Substrings that suggest a receiver is an HTTP/API client. Used to gate verb
# matches (e.g. ``.get()``) so that ``dict.get(k)`` / ``queue.get()`` are not
# mistaken for API call sites.
_CLIENT_HINTS = ("session", "client", "http", "api", "requests", "httpx", "aiohttp")


def _looks_like_client(receiver_name: str) -> bool:
    """Best-effort check that *receiver_name* looks like an HTTP/API client."""
    lowered = receiver_name.lower()
    return any(hint in lowered for hint in _CLIENT_HINTS)


def _matches_any(rel_posix: str, patterns: List[str]) -> bool:
    """Return True if *rel_posix* (repo-relative POSIX path) matches any pattern.

    Patterns follow glob semantics:
    - ``**/*.py`` matches py files at any depth
    - ``excluded/**`` matches everything under a directory
    - ``foo.py`` matches only the root-level file ``foo.py``

    Strategy:
    1. Use ``PurePosixPath.match`` which natively understands ``**`` in Python 3.12+.
       To handle the edge-case where the relative path is shallower than the leading
       ``**/`` prefix implies (e.g. ``excluded/secret.py`` vs ``**/excluded/*.py``),
       we also test against a fake-rooted path ``/x/<rel_posix>`` so that ``**``
       can match zero real segments.
    2. Fall back to ``fnmatch`` for simple patterns without ``**``.
    """
    # Fake-root path lets PurePosixPath.match handle ** matching zero segments.
    fake_rooted = PurePosixPath("/x") / rel_posix
    pure_rel = PurePosixPath(rel_posix)

    for pat in patterns:
        # Normalise: strip a leading "./" prefix (exact prefix, not a char set —
        # lstrip("./") would mangle dot-prefixed patterns like ".env" or ".github/**").
        if pat.startswith("./"):
            pat = pat[2:]

        # PurePosixPath.match supports ** natively (Python 3.12).
        if fake_rooted.match(pat) or pure_rel.match(pat):
            return True

        # fnmatch fallback for simple patterns without ** (e.g. "*.py", "foo.py")
        if "**" not in pat:
            if fnmatch.fnmatch(rel_posix, pat):
                return True
            # Filename-only pattern (no directory separator)
            if "/" not in pat and fnmatch.fnmatch(pure_rel.name, pat):
                return True

    return False


def _collect_files(repo_path: Path, include_patterns: List[str]) -> List[Path]:
    """Collect all files under *repo_path* that match any include pattern."""
    files: set = set()
    for pat in include_patterns:
        # Use pathlib glob which understands ** natively.
        for f in repo_path.glob(pat):
            if f.is_file():
                files.add(f)
    return sorted(files)


def _is_excluded(rel: Path, exclude_patterns: List[str]) -> bool:
    """Return True if the repo-relative path matches any exclude pattern."""
    rel_posix = rel.as_posix()
    return _matches_any(rel_posix, exclude_patterns)


def _render_signature(node) -> str:
    """Full def signature 'name(<params>)' — params include self/cls for methods."""
    return f"{node.name}({ast.unparse(node.args)})"


def _drop_leading_self(arguments):
    """Return a copy of an ast.arguments with a leading self/cls positional removed."""
    args = copy.deepcopy(arguments)
    lead = args.posonlyargs or args.args
    if lead and lead[0].arg in ("self", "cls"):
        if args.posonlyargs:
            args.posonlyargs = args.posonlyargs[1:]
        else:
            args.args = args.args[1:]
    return args


def _method_kind(node) -> str | None:
    """Return 'static', 'class', or None based on bare decorator names.

    Only bare ``ast.Name`` decorators for ``staticmethod`` / ``classmethod``
    are recognised; ``ast.Attribute`` or ``ast.Call`` decorators (e.g.
    ``module.staticmethod`` or ``@some_decorator()``) return None.
    """
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            if dec.id == "staticmethod":
                return "static"
            if dec.id == "classmethod":
                return "class"
    return None


def _is_stub_body(node) -> bool:
    """Return True when a function/method body is a stub.

    A stub body is one that, after stripping a leading docstring statement,
    is empty OR consists solely of:
      - ``pass`` statements
      - Ellipsis expressions (``...``)
      - ``raise NotImplementedError`` or ``raise NotImplementedError(...)``

    The leading docstring statement (if present) is an ``ast.Expr`` whose
    ``.value`` is an ``ast.Constant`` with a ``str`` value.
    """
    body = list(node.body)
    # Strip leading docstring.
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    if not body:
        return True

    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is ...
        ):
            continue
        if isinstance(stmt, ast.Raise) and stmt.exc is not None:
            exc = stmt.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                continue
            if (
                isinstance(exc, ast.Call)
                and isinstance(exc.func, ast.Name)
                and exc.func.id == "NotImplementedError"
            ):
                continue
        # Any other statement → not a stub.
        return False

    return True


def _docstring_summary(node):
    """First non-empty line of the node's docstring (<=200 chars), or None."""
    raw = ast.get_docstring(node, clean=True)
    if not raw:
        return None
    first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
    return first[:200] or None


def _extract_api_calls(tree: ast.AST, rel_str: str) -> List[Dict[str, Any]]:
    """Walk *tree* and emit best-effort API call site units.

    Recognises two common patterns:
    1. ``module.method(...)`` where *module* is in ``_API_MODULES``
       e.g. ``requests.get(url)``
    2. ``receiver.method(...)`` where *method* is an HTTP verb AND the receiver
       name looks client-like (see :func:`_looks_like_client`)
       e.g. ``self.session.get(url)``, ``client.post(...)``

    The receiver gate on pattern 2 avoids false positives such as ``dict.get(k)``
    or ``queue.get()`` which are not API calls.
    """
    units: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        method_name = func.attr
        # Determine the receiver name (best-effort: works for simple names and
        # one-level attribute access like self.session)
        receiver = func.value
        if isinstance(receiver, ast.Name):
            receiver_name = receiver.id
        elif isinstance(receiver, ast.Attribute):
            receiver_name = receiver.attr
        else:
            continue

        is_api_module = receiver_name in _API_MODULES
        # A verb match only counts when the receiver looks like an HTTP client,
        # so plain dict/queue ``.get()``/``.call()`` calls are not flagged.
        is_api_method = (
            method_name in _API_CALL_METHODS and _looks_like_client(receiver_name)
        )

        if is_api_module or is_api_method:
            called = f"{receiver_name}.{method_name}"
            units.append({
                "type": "api_call",
                "name": called,
                "file": rel_str,
                "lineno": node.lineno,
            })
    return units


def _extract_units(tree: ast.AST, rel_str: str) -> List[Dict[str, Any]]:
    """Extract functions, methods, classes, and API calls from a parsed AST.

    Methods are distinguished from top-level functions using the AST structure:
    only :class:`ast.FunctionDef` / :class:`ast.AsyncFunctionDef` nodes that
    are *direct children of a ClassDef body* are tagged as methods.
    """
    units: List[Dict[str, Any]] = []

    # First pass: build a set of function node ids that are methods, and record
    # their parent class name.
    method_parent: Dict[int, str] = {}  # id(node) -> class_name
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_parent[id(child)] = node.name

    # Second pass: emit class, function, and method units.
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            unit: Dict[str, Any] = {
                "type": "class",
                "name": node.name,
                "file": rel_str,
                "lineno": node.lineno,
            }
            doc = _docstring_summary(node)
            if doc is not None:
                unit["doc"] = doc
            # Class signature: instantiation form — find __init__ and drop self.
            try:
                init_node = next(
                    (
                        child for child in node.body
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == "__init__"
                    ),
                    None,
                )
                if init_node is not None:
                    unit["signature"] = (
                        f"{node.name}({ast.unparse(_drop_leading_self(init_node.args))})"
                    )
            except Exception:
                pass
            units.append(unit)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if id(node) in method_parent:
                unit = {
                    "type": "method",
                    "name": node.name,
                    "file": rel_str,
                    "lineno": node.lineno,
                    "class": method_parent[id(node)],
                }
                doc = _docstring_summary(node)
                if doc is not None:
                    unit["doc"] = doc
                try:
                    unit["signature"] = _render_signature(node)
                except Exception:
                    pass
                try:
                    unit["call_signature"] = (
                        f"{node.name}({ast.unparse(_drop_leading_self(node.args))})"
                    )
                except Exception:
                    pass
                # Additive: only set method_kind for static/classmethod nodes;
                # plain instance methods carry no 'method_kind' key.
                kind = _method_kind(node)
                if kind is not None:
                    unit["method_kind"] = kind
                # Additive: only set is_stub when the body qualifies as a stub;
                # non-stub units carry no 'is_stub' key.
                if _is_stub_body(node):
                    unit["is_stub"] = True
                units.append(unit)
            else:
                unit = {
                    "type": "function",
                    "name": node.name,
                    "file": rel_str,
                    "lineno": node.lineno,
                }
                doc = _docstring_summary(node)
                if doc is not None:
                    unit["doc"] = doc
                try:
                    unit["signature"] = _render_signature(node)
                except Exception:
                    pass
                # Additive: only set is_stub when the body qualifies as a stub;
                # non-stub units carry no 'is_stub' key.
                if _is_stub_body(node):
                    unit["is_stub"] = True
                units.append(unit)

    # Third pass: emit API call sites.
    units.extend(_extract_api_calls(tree, rel_str))

    return units


def _reject_dash_prefixed(value, label):
    """Reject operands that begin with '-' (defense against arg injection)."""
    if isinstance(value, str) and value.startswith("-"):
        raise ValueError(f"{label} must not start with '-': {value}")


@contextmanager
def _repo_workdir(config):
    """Context manager that yields a local Path to scan.

    If config.url is set, clones the remote (or local-path) repository into a
    temporary directory, checks out the requested ref, yields the clone path,
    and always cleans up the temp dir on exit.

    If config.url is not set, simply yields Path(config.path) — nothing is
    cloned and nothing is cleaned up.

    Ref precedence: when both ``commit`` and ``branch`` are set, ``commit``
    takes precedence and ``branch`` is ignored. The ``branch``/``commit``
    fields are only honored when ``url`` is set; for local-path configs they
    are ignored entirely.
    """
    url = getattr(config, "url", None)
    if url:
        branch = getattr(config, "branch", None)
        commit = getattr(config, "commit", None)

        # Defense in depth: reject any user operand that could be parsed as a
        # git option (the ``--`` terminator below is the primary guard).
        _reject_dash_prefixed(url, "url")
        _reject_dash_prefixed(branch, "branch")
        _reject_dash_prefixed(commit, "commit")

        tmp_dir = tempfile.mkdtemp()
        try:
            if commit:
                # Full clone needed to be able to checkout arbitrary SHAs.
                result = subprocess.run(
                    ["git", "clone", "--", url, tmp_dir],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"git clone failed for {url!r}: {result.stderr.strip()}"
                    )
                # NOTE: for `git checkout`, `--` separates revisions from
                # pathspecs, so the commit (a revision) must come BEFORE `--`.
                # Option-injection is already prevented by the dash-prefix
                # guard above; the trailing `--` ensures nothing is parsed as
                # a pathspec.
                result = subprocess.run(
                    ["git", "-C", tmp_dir, "checkout", commit, "--"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"git checkout {commit!r} failed: {result.stderr.strip()}"
                    )
            elif branch:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", branch, "--", url, tmp_dir],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"git clone --branch {branch!r} failed for {url!r}: {result.stderr.strip()}"
                    )
            else:
                result = subprocess.run(
                    ["git", "clone", "--depth", "1", "--", url, tmp_dir],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"git clone failed for {url!r}: {result.stderr.strip()}"
                    )

            yield Path(tmp_dir)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        yield Path(config.path)


def scan_repo(config) -> List[Dict[str, Any]]:
    """Scan a local Python repository and return a list of code units.

    Each unit is a dict with at minimum the keys:
      ``type``, ``name``, ``file``, ``lineno``

    Method units additionally carry a ``class`` key naming the parent class.

    Args:
        config: A :class:`~scripts.repo.loader.RepoConfig` instance.

    Returns:
        Sorted list of code unit dicts.
    """
    include_patterns: List[str] = config.include or ["**/*.py"]
    exclude_patterns: List[str] = config.exclude or []

    with _repo_workdir(config) as repo_path:
        repo_path = repo_path.resolve()
        py_files = _collect_files(repo_path, include_patterns)

        units: List[Dict[str, Any]] = []

        for py_file in py_files:
            rel = py_file.relative_to(repo_path)

            if _is_excluded(rel, exclude_patterns):
                continue

            try:
                source = py_file.read_text(encoding="utf-8")
            except (OSError, PermissionError, UnicodeDecodeError):
                # Skip binary, permission-denied, or non-UTF-8 files silently.
                continue

            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError:
                # Skip unparsable files without crashing.
                continue

            rel_str = rel.as_posix()
            units.extend(_extract_units(tree, rel_str))

    return units
