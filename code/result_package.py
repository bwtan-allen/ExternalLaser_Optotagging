"""Build an AIND metadata package around optotagging analysis outputs."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from aind_data_schema.core.data_description import DataDescription
from aind_data_schema.core.processing import (
    Code,
    DataProcess,
    ProcessName,
    ProcessStage,
    Processing,
)
from aind_data_schema.utils.inheritance import derive_data_description_from_derived


CANONICAL_METADATA_FILES = ("subject.json", "session.json", "procedures.json")
OPTIONAL_METADATA_FILES = ("rig.json",)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _metadata_root(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    for directory in (candidate, *candidate.parents):
        if (directory / "data_description.json").is_file():
            return directory
    raise FileNotFoundError(f"Could not locate data_description.json above {path}")


def _write_model(path: Path, model) -> None:
    path.write_text(model.model_dump_json(indent=3) + "\n", encoding="utf-8")


def _copy_source_metadata(source_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source_root.glob("*.json")):
        shutil.copy2(source_path, destination / source_path.name)


def _code_url(manifest: Mapping[str, Any]) -> str:
    domain = os.getenv(
        "CODEOCEAN_DOMAIN", "https://codeocean.allenneuraldynamics.org"
    ).rstrip("/")
    capsule_id = manifest.get("code_ocean", {}).get("capsule_id")
    return f"{domain}/capsule/{capsule_id}" if capsule_id else domain


def build_result_package(
    output_dir: str | Path,
    raw_path: str | Path,
    sorted_path: str | Path,
    analysis_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Write validated AIND metadata and provenance for one successful run."""
    output_dir = Path(output_dir)
    raw_root = _metadata_root(raw_path)
    sorted_root = _metadata_root(sorted_path)

    source_description = DataDescription.model_validate_json(
        (sorted_root / "data_description.json").read_text(encoding="utf-8")
    )
    raw_name = (
        analysis_manifest["source_assets"]["raw"].get("name") or raw_root.name
    )
    sorted_name = (
        analysis_manifest["source_assets"]["sorted"].get("name")
        or sorted_root.name
    )
    completed_at = _parse_datetime(analysis_manifest["completed_at"])
    tags = sorted(
        set(source_description.tags or ())
        | {
            "derived",
            "ecephys",
            "optotagging",
            str(analysis_manifest["subject_id"]),
        }
    )
    result_description = derive_data_description_from_derived(
        source_description,
        process_name="optotagging",
        source_data=[raw_name, sorted_name],
        creation_time=completed_at,
        tags=tags,
    )

    code_version = (
        os.getenv("CO_CAPSULE_VERSION")
        or os.getenv("CODE_VERSION")
        or "development"
    )
    code = Code(
        url=_code_url(analysis_manifest),
        name="Optotagging analysis capsule",
        version=code_version,
        run_script=Path("code/run"),
        language="Python",
        parameters={
            **analysis_manifest.get("settings", {}),
            "raw_asset_id": analysis_manifest["source_assets"]["raw"].get("id"),
            "sorted_asset_id": analysis_manifest["source_assets"]["sorted"].get(
                "id"
            ),
        },
    )
    process = DataProcess(
        process_type=ProcessName.ANALYSIS,
        name="Optotagging analysis",
        stage=ProcessStage.ANALYSIS,
        code=code,
        experimenters=["AIND Scientific Computing"],
        start_date_time=_parse_datetime(analysis_manifest["started_at"]),
        end_date_time=completed_at,
        output_path="optotagging",
        notes="Laser response metrics and responsive-unit summary figures.",
    )
    processing = Processing.create_with_sequential_process_graph([process])

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_model(output_dir / "data_description.json", result_description)
    _write_model(output_dir / "processing.json", processing)

    for filename in CANONICAL_METADATA_FILES:
        source_file = sorted_root / filename
        if not source_file.is_file():
            raise FileNotFoundError(
                f"Sorted asset is missing required metadata file {filename}"
            )
        shutil.copy2(source_file, output_dir / filename)
    for filename in OPTIONAL_METADATA_FILES:
        source_file = sorted_root / filename
        if source_file.is_file():
            shutil.copy2(source_file, output_dir / filename)

    _copy_source_metadata(raw_root, output_dir / "source_metadata" / "raw")
    _copy_source_metadata(sorted_root, output_dir / "source_metadata" / "sorted")

    custom_metadata = {
        "data level": "derived",
        "modality": "ecephys",
        "analysis": "optotagging",
        "subject id": str(analysis_manifest["subject_id"]),
        "session id": str(analysis_manifest["session_id"]),
        "raw asset id": analysis_manifest["source_assets"]["raw"].get("id"),
        "sorted asset id": analysis_manifest["source_assets"]["sorted"].get("id"),
        "analysis capsule version": code_version,
    }
    package_manifest = {
        "schema_version": 1,
        "asset_name": result_description.name,
        "source_assets": analysis_manifest["source_assets"],
        "tags": tags,
        "custom_metadata": custom_metadata,
        "metadata_files": [
            "data_description.json",
            "processing.json",
            *CANONICAL_METADATA_FILES,
        ],
    }
    (output_dir / "package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return package_manifest