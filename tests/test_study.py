"""Protocol tests for paired seeds, immutable sources, and explicit budgets."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiments import study


class ProtocolTests(unittest.TestCase):
    def test_seed_roles_and_common_draws(self):
        a = np.random.default_rng(study.seed_for("calibration", 2)).normal(size=20)
        b = np.random.default_rng(study.seed_for("calibration", 2)).normal(size=20)
        np.testing.assert_array_equal(a, b)
        c = np.random.default_rng(study.seed_for("calibration", 2, 1)).normal(size=20)
        self.assertFalse(np.array_equal(a, c))

    def test_provenance_refuses_changed_source_and_keeps_initial_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            study.provenance(folder)
            before = (folder / "provenance.json").read_bytes()
            manifest = json.loads(before)
            self.assertIn("src/ssgmm/estimator.py", manifest["source_sha256"])
            study.provenance(folder)
            self.assertEqual(before, (folder / "provenance.json").read_bytes())
            (folder / "source/src/ssgmm/estimator.py").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                study.provenance(folder)

    def test_reject_undefined_mcse_and_invalid_budgets(self):
        for args in [["--repeats", "1"], ["--steps", "1"], ["--jobs", "0"]]:
            with patch("sys.argv", ["study", *args]), self.assertRaises(SystemExit) as failure:
                study.main()
            self.assertEqual(failure.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
