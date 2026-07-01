"""Task 1 bootstrap smoke: the package imports and exposes a version string."""

import sutradhar


def test_package_imports() -> None:
    assert sutradhar is not None


def test_version_is_string() -> None:
    assert isinstance(sutradhar.__version__, str)
    assert sutradhar.__version__
