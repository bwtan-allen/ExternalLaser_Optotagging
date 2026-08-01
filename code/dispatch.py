"""Resolve a subject's ecephys assets and dispatch optotagging analyses."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Sequence

from aind_codeocean_pipeline_monitor.models import (
    CaptureSettings,
    PipelineMonitorSettings,
)
from codeocean.components import EveryoneRole, Permissions
from codeocean.computation import DataAssetsRunParam, RunParams
from codeocean.data_asset import AWSS3Target, ResultsInfo, Target

from codeocean_assets import (
    CodeOceanCatalog,
    SessionAssetPair,
    resolve_asset_pairs,
)


@dataclass(frozen=True)
class DispatchConfig:
    subject_id: str
    analysis_capsule_id: str
    monitor_capsule_id: str
    output_dir: Path = Path("/results")
    analysis_version: int | None = None
    session_ids: tuple[str, ...] = ()
    streams: tuple[str, ...] = ()
    all_sortings: bool = False
    dry_run: bool = False
    force: bool = False
    publish: bool = False
    open_data_bucket: str = "aind-open-data"
    computation_timeout: float = 48 * 60 * 60


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def build_analysis_run_params(
    pair: SessionAssetPair, config: DispatchConfig
) -> RunParams:
    """Build the isolated child analysis computation."""
    parameters = [
        "analyze",
        "--raw-path",
        "/data/raw_ephys",
        "--sorted-path",
        "/data/sorted_ephys",
        "--session-id",
        pair.session_id,
        "--raw-asset-id",
        pair.raw.id,
        "--raw-asset-name",
        pair.raw.name,
        "--sorted-asset-id",
        pair.sorted.id,
        "--sorted-asset-name",
        pair.sorted.name,
    ]
    for stream in config.streams:
        parameters.extend(("--stream", stream))
    return RunParams(
        capsule_id=config.analysis_capsule_id,
        version=config.analysis_version,
        data_assets=[
            DataAssetsRunParam(id=pair.raw.id, mount="raw_ephys"),
            DataAssetsRunParam(id=pair.sorted.id, mount="sorted_ephys"),
        ],
        parameters=parameters,
    )


def build_monitor_settings(
    pair: SessionAssetPair, config: DispatchConfig
) -> PipelineMonitorSettings:
    """Wrap one child run with capture and permission settings."""
    permissions = Permissions(
        everyone=(EveryoneRole.Viewer if config.publish else EveryoneRole.None_)
    )
    target = (
        Target(aws=AWSS3Target(bucket=config.open_data_bucket))
        if config.publish
        else None
    )
    capture_settings = CaptureSettings(
        tags=["derived", "ecephys", "optotagging", config.subject_id],
        description=f"Optotagging analysis for ecephys session {pair.session_id}",
        custom_metadata={
            "data level": "derived",
            "experiment type": "optotagging",
            "subject id": config.subject_id,
        },
        process_name_suffix="optotagging",
        permissions=permissions,
        target=target,
        results_info=ResultsInfo(
            capsule_id=config.analysis_capsule_id,
            version=config.analysis_version,
            run_script="code/run",
            data_assets=[pair.raw.id, pair.sorted.id],
        ),
    )
    return PipelineMonitorSettings(
        computation_timeout=config.computation_timeout,
        run_params=build_analysis_run_params(pair, config),
        capture_settings=capture_settings,
    )


def build_monitor_run_params(
    pair: SessionAssetPair, config: DispatchConfig
) -> RunParams:
    settings = build_monitor_settings(pair, config)
    return RunParams(
        capsule_id=config.monitor_capsule_id,
        parameters=[settings.model_dump_json(exclude_none=True)],
    )


def _asset_record_dict(asset) -> dict[str, Any]:
    value = asdict(asset)
    value["metadata"] = dict(asset.metadata)
    return value


def _pair_dict(pair: SessionAssetPair) -> dict[str, Any]:
    return {
        "session_id": pair.session_id,
        "raw_asset_id": pair.raw.id,
        "raw_asset_name": pair.raw.name,
        "sorted_asset_id": pair.sorted.id,
        "sorted_asset_name": pair.sorted.name,
        "pairing_method": pair.pairing_method,
    }


def dispatch_subject(
    config: DispatchConfig,
    catalog: CodeOceanCatalog,
    client,
) -> tuple[dict[str, Any], int]:
    """Resolve, de-duplicate, and submit all requested session pairs."""
    assets, catalog_warnings = catalog.search_subject_assets(config.subject_id)
    resolution = resolve_asset_pairs(
        assets,
        config.subject_id,
        all_sortings=config.all_sortings,
        session_ids=config.session_ids,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "subject_id": config.subject_id,
        "created_at": _utc_now(),
        "dry_run": config.dry_run,
        "publish": config.publish,
        "analysis_capsule_id": config.analysis_capsule_id,
        "analysis_version": config.analysis_version,
        "monitor_capsule_id": config.monitor_capsule_id,
        "candidates": [_asset_record_dict(asset) for asset in assets],
        "warnings": [*catalog_warnings, *resolution.warnings],
        "resolution_errors": list(resolution.errors),
        "sessions": [],
    }
    failed = bool(resolution.errors) or not resolution.pairs

    for pair in resolution.pairs:
        entry = {
            **_pair_dict(pair),
            "intended_asset_name": (
                f"ecephys_{pair.session_id}_optotagging_<completion-time>"
            ),
        }
        if not config.force:
            try:
                existing = catalog.find_existing_optotagging_asset(
                    pair, config.analysis_capsule_id
                )
            except Exception as exc:
                entry.update(
                    status="idempotency_check_failed", error=str(exc)
                )
                manifest["sessions"].append(entry)
                failed = True
                continue
            if existing is not None:
                entry.update(
                    status="skipped_existing", existing_asset_id=existing.id
                )
                manifest["sessions"].append(entry)
                continue

        monitor_params = build_monitor_run_params(pair, config)
        entry["monitor_run_params"] = monitor_params.to_dict()
        if config.dry_run:
            entry["status"] = "planned"
            manifest["sessions"].append(entry)
            continue
        try:
            computation = client.computations.run_capsule(monitor_params)
            entry.update(
                status="submitted", monitor_computation_id=computation.id
            )
        except Exception as exc:
            entry.update(status="submission_failed", error=str(exc))
            failed = True
        manifest["sessions"].append(entry)

    manifest["status"] = "failed" if failed else (
        "dry_run" if config.dry_run else "submitted"
    )
    manifest_path = (
        config.output_dir / f"subject_{config.subject_id}_dispatch_manifest.json"
    )
    _write_json(manifest_path, manifest)
    return manifest, int(failed)


def add_dispatch_parser(subparsers) -> None:
    analysis_capsule_id = os.getenv("ANALYSIS_CAPSULE_ID") or os.getenv(
        "CO_CAPSULE_ID"
    )
    monitor_capsule_id = os.getenv("PIPELINE_MONITOR_CAPSULE_ID")
    parser = subparsers.add_parser(
        "dispatch", help="Resolve and submit all ecephys sessions for a subject"
    )
    parser.add_argument("--subject-id", required=True)
    parser.add_argument(
        "--analysis-capsule-id",
        default=analysis_capsule_id,
        required=analysis_capsule_id is None,
    )
    parser.add_argument(
        "--monitor-capsule-id",
        default=monitor_capsule_id,
        required=monitor_capsule_id is None,
    )
    parser.add_argument("--analysis-version", type=int)
    parser.add_argument("--session-id", action="append")
    parser.add_argument("--stream", action="append")
    parser.add_argument("--all-sortings", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--open-data-bucket", default="aind-open-data")
    parser.add_argument("--output-dir", default="/results")
    parser.set_defaults(handler=run_dispatch_command)


def run_dispatch_command(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    try:
        catalog = CodeOceanCatalog.from_environment()
        config = DispatchConfig(
            subject_id=str(args.subject_id),
            analysis_capsule_id=args.analysis_capsule_id,
            monitor_capsule_id=args.monitor_capsule_id,
            output_dir=output_dir,
            analysis_version=args.analysis_version,
            session_ids=tuple(args.session_id or ()),
            streams=tuple(args.stream or ()),
            all_sortings=args.all_sortings,
            dry_run=args.dry_run,
            force=args.force,
            publish=args.publish,
            open_data_bucket=args.open_data_bucket,
        )
        _, exit_code = dispatch_subject(config, catalog, catalog.client)
        return exit_code
    except Exception as exc:
        _write_json(
            output_dir / f"subject_{args.subject_id}_dispatch_manifest.json",
            {
                "schema_version": 1,
                "subject_id": str(args.subject_id),
                "status": "failed",
                "created_at": _utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        print(f"Dispatch failed: {exc}")
        return 1