"""Offline smoke suite: no API calls, model downloads, or paid inference."""
from pathlib import Path
import sys


def main():
    try:
        import pytest
    except ImportError:
        print('Install test dependencies first: python -m pip install -e ".[test]"')
        return 2
    return pytest.main([str(Path(__file__).parent / "tests"), "-q", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

