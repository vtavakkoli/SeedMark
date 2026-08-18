"""Repository-level contract tests for the default Docker Compose experience."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ComposeContractTests(unittest.TestCase):
    def test_default_compose_runs_real_chat_demo(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("- qwen-demo", compose)
        self.assertIn("- --question", compose)
        self.assertIn("- What is AI?", compose)
        self.assertIn("- --max-new-tokens", compose)
        self.assertIn("working_dir: /workspace/results/qwen", compose)
        self.assertIn('"8081:8081"', compose)

    def test_toy_services_are_opt_in(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertGreaterEqual(compose.count('profiles: ["toy"]'), 2)
        self.assertNotIn('profiles: ["real-llm"]', compose)


if __name__ == "__main__":
    unittest.main()
