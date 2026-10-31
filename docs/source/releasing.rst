Releasing
=========

This page is for maintainers.  Releases are fully automated by GitHub Actions:
pushing a version tag builds, tests and publishes the wheels and the source
distribution to PyPI.

How the release pipeline works
------------------------------

* ``.github/workflows/wheels.yml`` builds wheels with cibuildwheel for
  CPython 3.10 -- 3.15 on Linux x86_64 and Linux aarch64 (``manylinux_2_28``),
  Windows AMD64 and macOS arm64 (deployment target 14.0), plus the sdist.
  Every wheel is installed into a clean environment and tested with
  ``tools/wheel_smoke.py`` and the ``tests/native`` / ``tests/optimization``
  suites, with and without ``QUSPECT_DISABLE_AVX2=1``.  Every (platform,
  Python version) pair is built in its own parallel job; the version list is
  read from the ``build`` key of ``[tool.cibuildwheel]`` in
  ``pyproject.toml``.  A separate job runs the newest Python's Linux x86_64
  wheel under the Intel Software Development Emulator on an emulated CPU
  without AVX to prove the AVX2 module is never loaded there.
* Linux aarch64 wheels are built natively on GitHub's ``ubuntu-24.04-arm``
  runners, not under QEMU emulation: the workflow sets ``CIBW_ARCHS_LINUX``
  from the runner architecture, so each Linux runner builds only its own
  architecture.  The workflow fails if a Linux wheel's platform tags do not
  include ``manylinux_2_28_<arch>`` or contain any tag for another
  architecture or a newer glibc (auditwheel may add an older-glibc alias such
  as ``manylinux_2_27_x86_64``, which is accepted), if the extension modules
  are not built for that architecture, if ``_core_scalar`` is missing, if
  ``_core_avx2`` is missing on x86_64 or present on aarch64, or if libgomp is
  not bundled in ``quspect.libs/``.
* macOS wheels are built on ``macos-15`` runners but still target macOS 14.
  ``tools/build_libomp_macos.sh`` builds libomp from pinned LLVM sources with
  deployment target 14.0 (a Homebrew bottle requires the runner's macOS
  version, and delocate would raise the wheel tag to ``macosx_15_0``), and the
  workflow fails if any macOS wheel's platform tag is not
  ``macosx_14_0_arm64`` or a bundled binary requires a newer macOS.
* On every pull request to ``main`` the workflow builds only the oldest and
  newest supported Python (3.10 and 3.15) on all four platforms (8 wheels).
  Releases and manual runs (*Run workflow* in the Actions tab, with the
  ``full`` input left checked) build all 24 wheels: 6 Python versions x 4
  platforms.
* ``.github/workflows/release.yml`` runs on tags matching ``v*``.  It calls
  ``wheels.yml`` with ``full: true``, checks that exactly 24 wheels and one
  sdist were built and that the tag version matches the version of every
  built artifact, publishes to PyPI with Trusted Publishing, and creates a
  GitHub Release with the artifacts attached.

The package version is derived from the git tag by setuptools-scm; there is no
version number to edit in the source tree.

One-time setup
--------------

1. On PyPI, add a *Trusted Publisher* for the project ``quspect`` (for the
   first release, use "Add a new pending publisher" under your account's
   publishing settings):

   * Owner: ``QuSpect``
   * Repository name: ``quspect``
   * Workflow name: ``release.yml``
   * Environment name: ``pypi``

2. In the GitHub repository settings, create an environment named ``pypi``.
   Optionally restrict it to tags matching ``v*`` and add required reviewers.

No API token or repository secret is needed.

Cutting a release
-----------------

1. Make sure ``main`` is green and contains everything for the release.
2. Tag the commit and push the tag:

   .. code-block:: bash

      git checkout main
      git pull
      git tag vX.Y.Z
      git push origin vX.Y.Z

3. Watch the *Release* workflow in the Actions tab.  When it finishes, the new
   version is on https://pypi.org/project/quspect/ and a GitHub Release named
   after the tag exists.

Tags must follow PEP 440 after the leading ``v`` (for example ``v0.2.0`` or
``v0.2.0rc1``); pre-release tags create a GitHub pre-release.  A PyPI version
can never be re-uploaded, so if a release fails after publishing, fix the
problem and release a new version.

Publishing the documentation
----------------------------

The documentation is hosted on Read the Docs at
https://quspect.readthedocs.io/en/latest/ (Chinese:
https://quspect.readthedocs.io/zh-cn/latest/).

* Read the Docs rebuilds the ``latest`` version on every push to ``main`` and,
  once tag versions are activated in the project's *Versions* settings, a
  versioned copy for every ``v*`` tag (``stable`` follows the newest release
  tag).  Nothing in the release workflow needs to trigger it.
* The build is configured by ``.readthedocs.yaml`` in the repository root.  It
  installs only ``docs/requirements.txt`` (Sphinx, the theme, NumPy and SciPy)
  and does **not** compile the C++ extension: ``docs/source/conf.py`` puts the
  repository root on ``sys.path`` and every autodoc'd module imports without
  the native engine.  Add new documentation dependencies to
  ``docs/requirements.txt``.
* The documented version is derived from ``git describe`` in the same way as
  setuptools-scm, so a tag build shows ``X.Y.Z`` and ``main`` shows the next
  development version.
* The Chinese documentation is a second Read the Docs project with language
  ``zh_CN``, linked to the main project under *Admin -> Translations*.  It
  uses the same configuration file; Read the Docs passes
  ``-D language=zh_CN`` and Sphinx compiles ``docs/source/locale`` PO files on
  the fly, so no ``.mo`` files are committed.

One-time setup on readthedocs.org:

1. Import ``QuSpect/quspect`` (grant Read the Docs access to the ``QuSpect``
   GitHub organization if it is not listed).  Keep the project slug
   ``quspect`` and the language *English*.
2. Import the same repository a second time as a separate project (for
   example slug ``quspect-zh-cn``) with language *Simplified Chinese*, then
   add it as a translation of ``quspect`` in the main project's
   *Admin -> Translations*.
3. Optionally enable *Build pull requests for this project* in both projects
   to get preview builds on pull requests.

To preview locally, run ``python -m pip install -r docs/requirements.txt``
and then ``sphinx-build -b html docs/source docs/build/html`` (add
``-D language=zh_CN`` for the Chinese version).
