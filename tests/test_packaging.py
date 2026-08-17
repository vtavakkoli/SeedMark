"""Packaging and release-contract tests."""

from __future__ import annotations

import re
from pathlib import Path
import tomllib
import unittest

from seedmark import __version__

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$")


class PackagingTests(unittest.TestCase):
    def test_public_version_is_semver(self) -> None:
        self.assertRegex(__version__, SEMVER)

    def test_real_llm_requirements_match_pyproject(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        optional = set(project["project"]["optional-dependencies"]["real-llm"])
        requirements = {
            line.strip()
            for line in (ROOT / "requirements" / "real-llm.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(optional, requirements)


if __name__ == "__main__":
    unittest.main()
