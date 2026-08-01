from pathlib import Path
import tempfile
import unittest

from codeocean.components import EveryoneRole

from codeocean_assets import AssetRecord, SessionAssetPair
from dispatch import (
    DispatchConfig,
    build_analysis_run_params,
    build_monitor_settings,
    dispatch_subject,
)


SESSION_ID = "853781_2026-07-01_12-20-31"


def asset_pair():
    return SessionAssetPair(
        session_id=SESSION_ID,
        raw=AssetRecord(
            id="raw-id",
            name=f"ecephys_{SESSION_ID}",
            data_level="raw",
            subject_id="853781",
            state="ready",
        ),
        sorted=AssetRecord(
            id="sorted-id",
            name=f"ecephys_{SESSION_ID}_sorted_2026-07-16_16-50-35",
            data_level="derived",
            subject_id="853781",
            source_data=(f"ecephys_{SESSION_ID}",),
            state="ready",
        ),
        pairing_method="source_data",
    )


def config(output_dir=Path("/results"), **overrides):
    values = {
        "subject_id": "853781",
        "analysis_capsule_id": "analysis-capsule",
        "monitor_capsule_id": "monitor-capsule",
        "output_dir": output_dir,
    }
    values.update(overrides)
    return DispatchConfig(**values)


class DispatchModelTests(unittest.TestCase):
    def test_child_run_has_stable_mounts_and_source_arguments(self):
        run_params = build_analysis_run_params(
            asset_pair(), config(streams=("ProbeA-1",))
        )

        self.assertEqual("analysis-capsule", run_params.capsule_id)
        self.assertEqual(["raw_ephys", "sorted_ephys"], [a.mount for a in run_params.data_assets])
        self.assertIn("raw-id", run_params.parameters)
        self.assertIn("sorted-id", run_params.parameters)
        self.assertIn("ProbeA-1", run_params.parameters)

    def test_capture_is_private_by_default(self):
        settings = build_monitor_settings(asset_pair(), config())

        self.assertEqual(
            EveryoneRole.None_, settings.capture_settings.permissions.everyone
        )
        self.assertIsNone(settings.capture_settings.target)
        settings.model_validate_json(settings.model_dump_json())

    def test_publish_targets_open_data_and_public_viewer(self):
        settings = build_monitor_settings(
            asset_pair(), config(publish=True, open_data_bucket="aind-open-data")
        )

        self.assertEqual(
            EveryoneRole.Viewer, settings.capture_settings.permissions.everyone
        )
        self.assertEqual(
            "aind-open-data", settings.capture_settings.target.aws.bucket
        )


class FakeCatalog:
    def __init__(self):
        pair = asset_pair()
        self.assets = (pair.raw, pair.sorted)

    def search_subject_assets(self, subject_id):
        return self.assets, ()

    def find_existing_optotagging_asset(self, pair, analysis_capsule_id):
        return None


class FailingComputations:
    def run_capsule(self, params):
        raise AssertionError("dry-run must not submit a computation")


class FakeClient:
    computations = FailingComputations()


class DispatchSubjectTests(unittest.TestCase):
    def test_dry_run_writes_manifest_without_submission(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            manifest, exit_code = dispatch_subject(
                config(output_dir=output_dir, dry_run=True),
                FakeCatalog(),
                FakeClient(),
            )

            self.assertEqual(0, exit_code)
            self.assertEqual("dry_run", manifest["status"])
            self.assertEqual("planned", manifest["sessions"][0]["status"])
            self.assertTrue(
                (output_dir / "subject_853781_dispatch_manifest.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()