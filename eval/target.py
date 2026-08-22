"""Resolves, imports, and verifies the RAG project under test.

This suite evaluates ANY RAG project -- what's fixed across every run is
the dataset (see eval/msmarco.py); what varies is the target's own
retrieval and generation code. Verification is done by actually
IMPORTING the target's embedder/generator modules and checking the
required functions exist on them -- not by checking for expected file
names on disk. A filename check is a proxy that breaks the moment
someone's project is laid out differently (a flat main.py instead of an
app/ package, for instance) even though the actual required functions
might be right there under a different name -- and, worse, it can drift
out of sync with what's actually required (this suite shipped exactly
that bug once already: the launcher scripts checked for app/config.py
after eval/target.py itself had already stopped requiring it). Importing
the real module and checking real attributes can't drift like that --
it's always checking the actual thing that matters.

REQUIRED interface (see also TARGET_INTERFACE.md at this repo's root for
the full spec with examples):

  <embedder module>  (default "app.embedder", override EVAL_EMBEDDER_MODULE)
    embed(texts: list[str]) -> array-like, shape (len(texts), dim)
    embed_one(text: str) -> array-like, shape (dim,)
    get_model() -> anything (called once; only its side effect of loading
                   the model matters -- the return value is unused)

  <generator module>  (default "app.generator", override EVAL_GENERATOR_MODULE)
    generate_answer(query: str, results: list[<context object>]) -> <answer object>
      Each context object needs `.text` and `.source` attributes (plain
      duck typing -- eval/pipeline.py builds its own simple object with
      those two fields, not the target's own context class).
      The returned answer object needs `.text: str`, `.grounded: bool`,
      `.generation_ms: float`, `.model: str`.

If your project doesn't use an app/ package at all -- say, a flat
main.py at the project root defining embed/embed_one/generate_answer
directly -- point at it with:
    EVAL_EMBEDDER_MODULE=main  EVAL_GENERATOR_MODULE=main
(same module twice is fine if both live in one file). Whatever module
name you give must be importable once the target root is on sys.path,
which the resolution below arranges.

OPTIONAL, with suite-owned fallbacks if absent -- see each module's own
docstring for the exact fallback value and how to override it:

  app.config.GENERATION_BACKEND     -- worker-count safety clamp (see
                                        eval/pipeline.py)
  app.config.LATENCY_BUDGET_MS      -- retrieval latency budget for the
                                        report (see eval/checks/latency.py)
  app.config.GENERATION_MODEL /
  app.config.LOCAL_GENERATION_MODEL -- cosmetic "model" label in the
                                        report only

These optional items DO still come from a fixed location, app.config --
unlike the two required modules above, there's no override for this one,
since it's read defensively with getattr() and a default rather than
required to exist at all (see optional_config() below). If your project
doesn't have an app/config.py, or has one under a different name, these
three items simply fall back to their defaults; nothing breaks.

How the target ROOT DIRECTORY (not the module names within it) is
located, in order:
  1. --rag-root CLI flag (highest priority) -- exactly this one path, no
     fallback if it turns out incompatible (an explicit instruction that
     fails should say so, not silently try something else).
  2. RAG_PROJECT_ROOT environment variable -- same, exactly one path.
  3. If neither is set, two candidates are tried in order, and the first
     one that actually verifies (real import, real required functions)
     wins:
       a. This suite's own directory's parent -- i.e., wherever the
          eval/ folder itself was placed. This is the common case now:
          drop the eval/ folder (plus run.sh/run.ps1) directly into your
          RAG project's root and just run the script from there, no env
          var needed at all.
       b. A sibling directory named "RAG" next to wherever this repo's
          own folder lives (i.e. ../RAG) -- backward compatible with
          running this suite as its own separate repo, cloned alongside
          the target project.
     If neither candidate verifies, the error lists both attempts and
     what failed about each.
"""
import importlib
import os
import sys
from pathlib import Path
from typing import Any

_THIS_REPO_ROOT = Path(__file__).resolve().parent.parent
_injected = False
_resolved_root: Path | None = None

EMBEDDER_MODULE = os.environ.get("EVAL_EMBEDDER_MODULE", "app.embedder")
GENERATOR_MODULE = os.environ.get("EVAL_GENERATOR_MODULE", "app.generator")

_REQUIRED_EMBEDDER_ATTRS = ("embed", "embed_one", "get_model")
_REQUIRED_GENERATOR_ATTRS = ("generate_answer",)


class TargetNotFound(RuntimeError):
    pass


def _candidate_roots(cli_arg: str | None) -> list[tuple[str, Path]]:
    """Ordered (label, path) candidates. An explicit --rag-root or
    RAG_PROJECT_ROOT means exactly one candidate -- no silent fallback
    past an explicit instruction that turns out wrong."""
    if cli_arg:
        return [("--rag-root", Path(cli_arg).resolve())]
    if os.environ.get("RAG_PROJECT_ROOT"):
        return [("RAG_PROJECT_ROOT env var", Path(os.environ["RAG_PROJECT_ROOT"]).resolve())]
    return [
        ("this suite's own parent directory (eval/ dropped into your project)", _THIS_REPO_ROOT),
        ("sibling ../RAG (separate-repo layout)", _THIS_REPO_ROOT.parent / "RAG"),
    ]


def _check_module(module_name: str, required_attrs: tuple[str, ...], env_var: str) -> None:
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        raise TargetNotFound(
            f"could not import '{module_name}': {e} (set {env_var} if your project uses a "
            f"different module path, e.g. {env_var}=main for a flat main.py)"
        ) from e
    missing = [a for a in required_attrs if not hasattr(mod, a)]
    if missing:
        raise TargetNotFound(f"'{module_name}' is missing required attribute(s): {', '.join(missing)}")


def verify_target(cli_arg: str | None = None) -> Path:
    """The one real entry point: finds the target root and confirms it's
    actually usable by importing EMBEDDER_MODULE / GENERATOR_MODULE and
    checking every required attribute is present on each -- real
    verification of the real interface, not a proxy check against
    expected file names. Call this once, early (eval/runner.py does, before
    downloading the dataset) -- every other module in this suite that
    needs the target assumes this already ran and just imports directly.
    """
    global _injected, _resolved_root
    if _injected:
        return _resolved_root

    errors: list[str] = []
    for label, root in _candidate_roots(cli_arg):
        if not root.is_dir():
            errors.append(f"  [{label}] '{root}' is not a directory")
            continue

        modules_before = set(sys.modules)
        sys.path.insert(0, str(root))
        try:
            _check_module(EMBEDDER_MODULE, _REQUIRED_EMBEDDER_ATTRS, "EVAL_EMBEDDER_MODULE")
            _check_module(GENERATOR_MODULE, _REQUIRED_GENERATOR_ATTRS, "EVAL_GENERATOR_MODULE")
        except TargetNotFound as e:
            sys.path.remove(str(root))
            # Undo any partial imports this failed attempt cached, so a
            # same-named module under the next candidate root isn't
            # shadowed by a stale entry left behind by this one.
            for name in set(sys.modules) - modules_before:
                del sys.modules[name]
            errors.append(f"  [{label}] '{root}': {e}")
            continue

        _injected = True
        _resolved_root = root
        return root

    raise TargetNotFound(
        "Could not find a compatible RAG project. Tried:\n"
        + "\n".join(errors)
        + "\n\nSee TARGET_INTERFACE.md for the required interface. Point at your project "
        "explicitly with --rag-root <path>, or set the RAG_PROJECT_ROOT environment variable, e.g.:\n"
        '  $env:RAG_PROJECT_ROOT = "C:\\path\\to\\your-project"   # PowerShell\n'
        "  export RAG_PROJECT_ROOT=/path/to/your-project          # bash"
    )


def load_target(cli_arg: str | None = None) -> Path:
    """Resolves and verifies the target if that hasn't happened yet in
    this process, otherwise returns the already-resolved root. Safe to
    call from anywhere (eval/judge.py, eval/index_build.py, etc.) without
    re-running verification every time."""
    return verify_target(cli_arg)


def get_embedder():
    load_target()
    return importlib.import_module(EMBEDDER_MODULE)


def get_generator():
    load_target()
    return importlib.import_module(GENERATOR_MODULE)


def optional_config(name: str, default: Any) -> Any:
    """Reads app.config.<name> from the target if it exists, else returns
    `default`. Used for every OPTIONAL interface item listed above --
    lets this suite run against targets that don't define config the same
    way this suite's own original target project does, without crashing."""
    load_target()
    try:
        import app.config as target_config
    except ImportError:
        return default
    return getattr(target_config, name, default)
