import sys

import nox

nox.options.default_venv_backend = "none"
nox.options.sessions = ["lint", "typecheck", "tests"]


@nox.session
def lint(session: nox.Session) -> None:
    """Run Ruff linter and format check."""
    session.run(sys.executable, "-m", "ruff", "check", ".")
    session.run(sys.executable, "-m", "ruff", "format", "--check", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    """Run Mypy strict type checking."""
    session.run(sys.executable, "-m", "mypy", "src", "tests")


@nox.session
def tests(session: nox.Session) -> None:
    """Run the pytest test suite."""
    session.run(sys.executable, "-m", "pytest", "tests/", *session.posargs)
