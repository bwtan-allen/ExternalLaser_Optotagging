from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.processing import Processing

from result_package import build_result_package


SESSION_ID = "853781_2026-07-01_12-20-31"
RAW_NAME = f"ecephys_{SESSION_ID}"
SORTED_NAME = f"{RAW_NAME}_sorted_2026-07-16_16-50-35"


SOURCE_DESCRIPTION = {
    "creation_time": "2026-07-16T16:50:35.848431Z",
    "data_level": "derived",
    "data_summary": "Optotagging source ecephys session",
    "funding_source": [
        {
            "fundee": [{"name": "unknown"}],
            "funder": {
                "abbreviation": "AI",
                "name": "Allen Institute",
                "registry": "Research Organization Registry (ROR)",
                "registry_identifier": "03cpe7c52",
            },
        }
    ],
    "group": "ephys",
    "institution": {
        "abbreviation": "AIND",
        "name": "Allen Institute for Neural Dynamics",
        "registry": "Research Organization Registry (ROR)",
        "registry_identifier": "04szwah67",
    },
    "investigators": [{"name": "Bowen Tan"}],
    "license": "CC-BY-4.0",
    "modalities": [
        {
            "abbreviation": "ecephys",
            "name": "Extracellular electrophysiology",
        }
    ],
    "name": SORTED_NAME,
    "project_name": "GPP",
    "subject_id": "853781",
    "source_data": [RAW_NAME],
}


class ResultPackageTests(unittest.TestCase):
    def test_builds_schema_valid_metadata_and_source_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw = root / RAW_NAME
            sorted_path = root / SORTED_NAME
            output = root / "results"
            raw.mkdir()
            sorted_path.mkdir()
            (raw / "data_description.json").write_text(
                json.dumps({**SOURCE_DESCRIPTION, "name": RAW_NAME, "data_level": "raw", "source_data": None}),
                encoding="utf-8",
            )
            (sorted_path / "data_description.json").write_text(
                json.dumps(SOURCE_DESCRIPTION), encoding="utf-8"
            )
            for filename in ("subject.json", "session.json", "procedures.json"):
                (sorted_path / filename).write_text("{}", encoding="utf-8")

            manifest = {
                "subject_id": "853781",
                "session_id": SESSION_ID,
                "started_at": "2026-08-01T10:00:00Z",
                "completed_at": "2026-08-01T10:05:00Z",
                "settings": {"opto_recording": 0},
                "source_assets": {
                    "raw": {"id": "raw-id", "name": RAW_NAME},
                    "sorted": {"id": "sorted-id", "name": SORTED_NAME},
                },
                "code_ocean": {"capsule_id": "capsule-id"},
            }

            package = build_result_package(output, raw, sorted_path, manifest)

            description = DataDescription.model_validate_json(
                (output / "data_description.json").read_text(encoding="utf-8")
            )
            processing = Processing.model_validate_json(
                (output / "processing.json").read_text(encoding="utf-8")
            )
            self.assertEqual("derived", str(description.data_level))
            self.assertEqual([RAW_NAME, SORTED_NAME], description.source_data)
            self.assertIn("optotagging", description.tags)
            self.assertEqual("Optotagging analysis", processing.data_processes[0].name)
            self.assertEqual(description.name, package["asset_name"])
            self.assertTrue((output / "subject.json").is_file())
            self.assertTrue(
                (output / "source_metadata" / "sorted" / "data_description.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()