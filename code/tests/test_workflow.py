from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workflow import (
    _run_attached_assets,
    get_streams_to_analyze,
    main,
    resolve_session_inputs,
)


SESSION_ID = "853781_2026-07-01_12-20-31"


class ResolveSessionInputsTests(unittest.TestCase):
    def test_resolves_one_complete_session_pair(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / f"ecephys_{SESSION_ID}"
            clipped = raw / "ecephys" / "ecephys_clipped"
            settings = clipped / "Record Node 101" / "settings.xml"
            settings.parent.mkdir(parents=True)
            settings.write_text("<SETTINGS />", encoding="utf-8")
            trials = clipped / "session.opto.csv"
            trials.write_text("index,type\n", encoding="utf-8")

            sorted_path = root / f"ecephys_{SESSION_ID}_sorted_2026-07-16_16-50-35"
            (sorted_path / "curated").mkdir(parents=True)
            (sorted_path / "postprocessed").mkdir()

            resolved = resolve_session_inputs(raw, sorted_path)

            self.assertEqual(clipped, resolved[0])
            self.assertEqual(sorted_path, resolved[1])
            self.assertEqual(trials, resolved[2])
            self.assertEqual(SESSION_ID, resolved[3])

    def test_rejects_mismatched_sessions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            clipped = root / f"ecephys_{SESSION_ID}" / "ecephys_clipped"
            clipped.mkdir(parents=True)
            sorted_path = root / "ecephys_999999_2026-07-01_12-20-31_sorted"
            sorted_path.mkdir()

            with self.assertRaisesRegex(ValueError, "does not match"):
                resolve_session_inputs(clipped, sorted_path)


class StreamSelectionTests(unittest.TestCase):
    def test_matches_probe_letter_and_keeps_shanks(self):
        streams = ["ProbeA-1", "ProbeA-2", "ProbeB-1"]

        selected = get_streams_to_analyze(streams, ["Probe A"])

        self.assertEqual(
            [("ProbeA-1", "Probe A"), ("ProbeA-2", "Probe A")], selected
        )


class EntryPointTests(unittest.TestCase):
    @patch("workflow._run_attached_assets", return_value=0)
    def test_no_arguments_runs_manually_attached_assets(self, run_attached_assets):
        self.assertEqual(0, main([]))
        run_attached_assets.assert_called_once_with()

    @patch("workflow.analyze_session")
    def test_one_attached_pair_writes_package_at_results_root(self, analyze_session):
        analyze_session.return_value = {"streams": []}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_root = root / "data"
            results_root = root / "results"
            clipped = (
                data_root
                / f"ecephys_{SESSION_ID}"
                / "ecephys"
                / "ecephys_clipped"
            )
            clipped.mkdir(parents=True)
            sorted_path = data_root / f"ecephys_{SESSION_ID}_sorted_2026-07-16_16-50-35"
            sorted_path.mkdir()

            exit_code = _run_attached_assets(data_root, results_root)

            self.assertEqual(0, exit_code)
            self.assertEqual(
                results_root, analyze_session.call_args.kwargs["output_dir"]
            )
            self.assertTrue((results_root / "batch_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()