"""Sphinx configuration for QuSpect.

QuSpect — A quantum computing SDK for research and education.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

sys.path.insert(0, os.path.abspath("../.."))

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _version_from_git() -> str | None:
    """Derive a setuptools-scm style version from ``git describe``.

    ``vX.Y.Z`` exactly on a tag gives ``X.Y.Z``; ``N`` commits after it gives
    ``X.Y.(Z+1).devN+gSHA`` (setuptools-scm's default "guess next dev"
    scheme).  Returns ``None`` when git or a ``v*`` tag is unavailable.
    """
    try:
        described = subprocess.run(
            ["git", "describe", "--tags", "--long", "--match", "v*"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.fullmatch(r"v(.+)-(\d+)-(g[0-9a-f]+)", described)
    if match is None:
        return None
    tag, distance, sha = match.group(1), int(match.group(2)), match.group(3)
    if distance == 0:
        return tag
    release_match = re.fullmatch(r"(\d+(?:\.\d+)*)", tag)
    if release_match is None:
        return f"{tag}.post{distance}+{sha}"
    parts = tag.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return f"{'.'.join(parts)}.dev{distance}+{sha}"


def _detect_release() -> str:
    """Return the version of the source tree being documented.

    The docs import the source tree directly (see ``sys.path`` above), so the
    git tag is preferred over installed metadata, which can be stale for an
    editable install.  Falls back to the installed distribution's version and
    finally to ``0.0.0+unknown``.
    """
    git_version = _version_from_git()
    if git_version is not None:
        return git_version
    try:
        return _dist_version("quspect")
    except PackageNotFoundError:
        return "0.0.0+unknown"


# -- Project information ---------------------------------------------------
project = "QuSpect"
copyright = "2026, Quspect"
author = "Quspect"
release = _detect_release()
version = ".".join(release.split(".")[:2])

# -- General configuration -------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
]

templates_path = ["_templates"]
exclude_patterns = []

language = "en"
locale_dirs = ["locale/"]
gettext_compact = False

# -- Napoleon settings -----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Autodoc settings ------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# -- Intersphinx settings --------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- HTML output -----------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "QuSpect Documentation"
html_short_title = "QuSpect"
html_show_sphinx = True
html_show_sourcelink = True
