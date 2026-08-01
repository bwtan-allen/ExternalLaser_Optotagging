"""Command-line workflow for mounted optotagging analysis assets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import os
from pathlib import Path
import re
import sys
import traceback
from typing import Any, Sequence

import numpy as np

import optotagging_analysis as oa
import plotting_funcs as pf
from result_package import build_result_package


SESSION_PATTERN = re.compile(r"\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")
REQUIRED_TRIAL_COLUMNS = {
    "duration",
    "emission_location",
    "interval",
    "num_pulses",
    "param_group",
    "power",
    "pulse_interval",
    "site",
    "type",
    "wavelength",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def extract_session_id(value: str | Path) -> str:
    match = SESSION_PATTERN.search(str(value))
    if not match:
        raise ValueError(f"No ecephys acquisition session ID found in {value}")
    return match.group(0)


def get_streams_to_analyze(ap_streams, emission_locations):
    streams_to_analyze = []
    seen = set()
    for emission_location in emission_locations:
        probe_names = {
            f"Probe{probe_letter.upper()}"
            for probe_letter in re.findall(r"Probe\s*([A-Za-z])", emission_location)
        }
        for stream_name in ap_streams:
            stream_probe = re.sub(r"(?:-AP)?-?\d*$", "", stream_name)
            stream_info = (stream_name, emission_location)
            if stream_probe in probe_names and stream_info not in seen:
                streams_to_analyze.append(stream_info)
                seen.add(stream_info)
    return streams_to_analyze


def resolve_session_inputs(
    raw_path: str | Path,
    sorted_path: str | Path,
    trials_csv: str | Path | None = None,
) -> tuple[Path, Path, Path, str]:
    """Resolve and validate the mounted folders needed by one analysis."""
    raw_path = Path(raw_path)
    sorted_path = Path(sorted_path)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw ephys path does not exist: {raw_path}")
    if not sorted_path.is_dir():
        raise FileNotFoundError(f"Sorted ephys path does not exist: {sorted_path}")

    if raw_path.name == "ecephys_clipped":
        clipped_path = raw_path
    else:
        clipped_candidates = sorted(
            path for path in raw_path.rglob("ecephys_clipped") if path.is_dir()
        )
        if len(clipped_candidates) != 1:
            raise ValueError(
                f"Expected exactly one ecephys_clipped folder under {raw_path}; "
                f"found {len(clipped_candidates)}"
            )
        clipped_path = clipped_candidates[0]

    session_id = extract_session_id(clipped_path)
    sorted_session_id = extract_session_id(sorted_path)
    if sorted_session_id != session_id:
        raise ValueError(
            f"Raw session {session_id} does not match sorted session "
            f"{sorted_session_id}"
        )

    settings_files = sorted(clipped_path.glob("Record Node ???/settings.xml"))
    if not settings_files:
        raise FileNotFoundError(f"No Record Node settings.xml found under {clipped_path}")

    if trials_csv is None:
        trials_candidates = sorted(clipped_path.glob("*opto.csv"))
        if len(trials_candidates) != 1:
            raise ValueError(
                f"Expected exactly one opto CSV under {clipped_path}; "
                f"found {len(trials_candidates)}"
            )
        trials_path = trials_candidates[0]
    else:
        trials_path = Path(trials_csv)
        if not trials_path.is_file():
            raise FileNotFoundError(f"Opto trials CSV does not exist: {trials_path}")

    curated_directories = sorted(
        path for path in sorted_path.glob("*curated") if path.is_dir()
    )
    if not curated_directories:
        raise FileNotFoundError(f"No curated sorting folder found under {sorted_path}")
    if not (sorted_path / "postprocessed").is_dir():
        raise FileNotFoundError(
            f"No postprocessed waveform folder found under {sorted_path}"
        )

    return clipped_path, sorted_path, trials_path, session_id


def _add_best_power_columns(laser_response_metrics, trial_types) -> None:
    for group_name in list(laser_response_metrics.keys()):
        metrics = laser_response_metrics[group_name]
        columns = list(metrics.columns)
        for trial_type in trial_types:
            response_columns = [
                item
                for item in columns
                if trial_type in item and "num_sig_pulses" in item
            ]
            for unit_index in metrics.index.tolist():
                best_response = 0
                best_column = None
                for column in response_columns:
                    value = metrics.at[unit_index, column]
                    if value > best_response:
                        best_response = value
                        best_column = column
                if best_column is None:
                    continue
                parts = best_column.split("_")
                best_power = next(part for part in parts if "mW" in part)
                prefix = f"{trial_type}_train"
                source_prefix = f"{prefix}_{best_power}"
                metrics.at[unit_index, f"{prefix}_best_power"] = float(
                    best_power[:-2]
                )
                metrics.at[unit_index, f"{prefix}_max_num_sig_pulses"] = metrics.at[
                    unit_index, f"{source_prefix}_num_sig_pulses"
                ]
                for metric_name in (
                    "mean_latency",
                    "latency_range",
                    "mean_time_to_first_spike",
                    "mean_jitter",
                    "mean_reliability",
                ):
                    metrics.at[
                        unit_index, f"{prefix}_best_{metric_name}"
                    ] = metrics.at[unit_index, f"{source_prefix}_{metric_name}"]


def _unit_values(frame, mask) -> list[Any]:
    values = np.array(frame.unit_id.tolist())[mask]
    return [value.item() if hasattr(value, "item") else value for value in values]


def analyze_session(
    raw_path: str | Path,
    sorted_path: str | Path,
    *,
    output_dir: str | Path = "/results",
    trials_csv: str | Path | None = None,
    requested_streams: Sequence[str] = (),
    opto_recording: int = 0,
    flip_nidaq: bool = False,
    ignore_onset_offset: bool = True,
    expected_session_id: str | None = None,
    raw_asset_id: str | None = None,
    raw_asset_name: str | None = None,
    sorted_asset_id: str | None = None,
    sorted_asset_name: str | None = None,
) -> dict[str, Any]:
    """Analyze one mounted raw/sorted asset pair and write a result manifest."""
    started_at = utc_now()
    clipped_path, sorted_path, trials_path, session_id = resolve_session_inputs(
        raw_path, sorted_path, trials_csv
    )
    if expected_session_id and session_id != expected_session_id:
        raise ValueError(
            f"Mounted session {session_id} does not match requested session "
            f"{expected_session_id}"
        )

    output_dir = Path(output_dir)
    metrics_dir = output_dir / "optotagging" / "metrics"
    figures_dir = output_dir / "optotagging" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Constructing analyzer for {session_id}")
    analyzer = oa.OptotaggingAnalysis(
        str(clipped_path),
        str(sorted_path),
        trials_csv=str(trials_path),
        opto_recording=opto_recording,
        flip_NIDAQ=flip_nidaq,
    )
    missing_columns = REQUIRED_TRIAL_COLUMNS - set(analyzer.trial_ids.columns)
    if missing_columns:
        raise ValueError(
            "Opto trials CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if len(analyzer.laser_onset_times) == 0:
        raise ValueError("No laser onset events were found")

    trial_types = np.unique(analyzer.trial_ids.type)
    trials_query = {
        "type": trial_types,
        "param_group": ["train"],
        "power": np.unique(analyzer.trial_ids.power),
    }
    suffixes = [None, None, "mW"]

    ap_streams = analyzer.get_stream_names()
    opto_locations = np.unique(analyzer.trial_ids.emission_location)
    streams_to_analyze = get_streams_to_analyze(ap_streams, opto_locations)
    requested_stream_set = set(requested_streams)
    if requested_stream_set:
        streams_to_analyze = [
            stream_info
            for stream_info in streams_to_analyze
            if stream_info[0] in requested_stream_set
        ]
    if not streams_to_analyze:
        raise ValueError("No AP streams matched the optogenetic emission locations")

    print(
        "Streams with opto stimulation to analyze: "
        f"{[stream for stream, _ in streams_to_analyze]}"
    )
    stream_summaries: list[dict[str, Any]] = []

    for stream_name, probe_name in streams_to_analyze:
        print(f"Loading data for session {session_id}, {stream_name}")
        timestamps, sorting_data = analyzer.get_sorting_output(stream_name)
        if not sorting_data:
            raise FileNotFoundError(
                f"No curated sorting outputs matched stream {stream_name}"
            )

        stimulus_start = analyzer.laser_onset_times[0] - 1
        session_start = timestamps[0]
        pre_stim_duration = stimulus_start - session_start
        if pre_stim_duration <= 0:
            raise ValueError(
                f"Session {session_id} has no positive pre-stimulus interval"
            )

        laser_response_metrics = analyzer.one_probe_laser_responses(
            timestamps,
            sorting_data,
            trials_query,
            probe_name,
            suffixes=suffixes,
            ignore_onset_offset=ignore_onset_offset,
            pre_opto_duration=pre_stim_duration,
        )
        _add_best_power_columns(laser_response_metrics, trial_types)

        stream_summary: dict[str, Any] = {
            "stream": stream_name,
            "emission_location": probe_name,
            "groups": [],
        }
        for group_name, metrics in laser_response_metrics.items():
            metrics_path = (
                metrics_dir
                / f"{session_id}_{stream_name}_{group_name}_laser_response_metrics.csv"
            )
            metrics.to_csv(metrics_path)

            sorting_output = sorting_data[group_name]["sorting_output"]
            decoder_label = sorting_output.get_property("decoder_label")
            if decoder_label is None:
                raise ValueError(
                    f"Sorting group {group_name} for {stream_name} has no "
                    "decoder_label property"
                )

            print(f"Channel group {group_name}:")
            output_stem = figures_dir / f"{session_id}_{stream_name}_{group_name}"
            for response_color in ("red", "blue"):
                for plot_suffix in ("", "_pulse_plot"):
                    Path(
                        f"{output_stem}_{response_color}_responsive"
                        f"{plot_suffix}.png"
                    ).unlink(missing_ok=True)

            if "external_red_train_max_num_sig_pulses" in metrics.columns:
                red_responsive = metrics.query(
                    "external_red_train_max_num_sig_pulses >= 4 and "
                    "external_red_train_best_mean_jitter < 0.006 and "
                    "pre_stim_isi_ratio < 0.5"
                )
                good_red_mask = ~(
                    decoder_label[red_responsive.index.tolist()] == "noise"
                )
                red_units = _unit_values(red_responsive, good_red_mask)
                print(f"{len(red_units)} externally tagged red units: {red_units}")
            else:
                red_units = []

            if "external_blue_train_max_num_sig_pulses" in metrics.columns:
                blue_responsive = metrics.query(
                    "external_blue_train_max_num_sig_pulses == 5 and "
                    "external_blue_train_best_mean_jitter < 0.006 and "
                    "pre_stim_isi_ratio < 0.5"
                )
                if "external_red_train_max_num_sig_pulses" in metrics.columns:
                    blue_responsive = blue_responsive[
                        ~blue_responsive.index.isin(red_responsive.index)
                    ]
                good_blue_mask = ~(
                    decoder_label[blue_responsive.index.tolist()] == "noise"
                )
                blue_units = _unit_values(blue_responsive, good_blue_mask)
                print(f"{len(blue_units)} externally tagged blue units: {blue_units}")
            else:
                blue_units = []

            for color, units in (("red", red_units), ("blue", blue_units)):
                if not units:
                    continue
                figure_title = (
                    f"{session_id}_{stream_name}_{group_name}_{color}_responsive"
                )
                pf.multi_unit_raster_plot(
                    np.asarray(units),
                    sorting_output,
                    timestamps,
                    sorting_data[group_name]["waveform_extractor"],
                    analyzer.trial_ids,
                    analyzer.laser_onset_times,
                    trial_types,
                    probe_name,
                    figure_title,
                    output_dir=figures_dir,
                )
                pf.multi_unit_pulse_plot(
                    np.asarray(units),
                    sorting_output,
                    timestamps,
                    metrics,
                    analyzer.laser_onset_times,
                    analyzer.trial_ids,
                    trial_types,
                    probe_name,
                    figure_title + "_pulse_plot",
                    output_dir=figures_dir,
                )

            stream_summary["groups"].append(
                {
                    "name": group_name,
                    "metric_rows": len(metrics),
                    "metrics_file": str(metrics_path.relative_to(output_dir)),
                    "red_responsive_unit_ids": red_units,
                    "blue_responsive_unit_ids": blue_units,
                }
            )
        stream_summaries.append(stream_summary)

    manifest = {
        "schema_version": 1,
        "status": "success",
        "subject_id": session_id.split("_", 1)[0],
        "session_id": session_id,
        "started_at": started_at,
        "completed_at": utc_now(),
        "source_assets": {
            "raw": {
                "id": raw_asset_id,
                "name": raw_asset_name or f"ecephys_{session_id}",
            },
            "sorted": {
                "id": sorted_asset_id,
                "name": sorted_asset_name or Path(sorted_path).name,
            },
        },
        "inputs": {
            "recording_clipped_folder": str(clipped_path),
            "recording_sorted_folder": str(sorted_path),
            "trials_csv": str(trials_path),
        },
        "settings": {
            "opto_recording": opto_recording,
            "flip_nidaq": flip_nidaq,
            "ignore_onset_offset": ignore_onset_offset,
            "requested_streams": sorted(requested_stream_set),
        },
        "streams": stream_summaries,
        "output_files": [],
        "code_ocean": {
            "capsule_id": os.getenv("CO_CAPSULE_ID"),
            "computation_id": os.getenv("CO_COMPUTATION_ID"),
        },
    }
    manifest["result_package"] = build_result_package(
        output_dir, raw_path, sorted_path, manifest
    )
    manifest["output_files"] = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "analysis_manifest.json"
    )
    write_json(output_dir / "analysis_manifest.json", manifest)
    return manifest


def _parse_streams(cli_streams: Sequence[str] | None) -> tuple[str, ...]:
    values = list(cli_streams or ())
    values.extend(os.getenv("STREAM_NAMES", "").split(","))
    return tuple(sorted({value.strip() for value in values if value.strip()}))


def _run_analyze(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    try:
        analyze_session(
            args.raw_path,
            args.sorted_path,
            output_dir=output_dir,
            trials_csv=args.trials_csv,
            requested_streams=_parse_streams(args.stream),
            opto_recording=args.opto_recording,
            flip_nidaq=args.flip_nidaq,
            ignore_onset_offset=not args.include_onset_offset,
            expected_session_id=args.session_id,
            raw_asset_id=args.raw_asset_id,
            raw_asset_name=args.raw_asset_name,
            sorted_asset_id=args.sorted_asset_id,
            sorted_asset_name=args.sorted_asset_name,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "completed_at": utc_now(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "failure.json", failure)
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_attached_assets(
    data_root: Path = Path("/data"), results_root: Path = Path("/results")
) -> int:
    session_filters = {
        value.strip()
        for value in os.getenv("SESSION_IDS", "").split(",")
        if value.strip()
    }
    clipped_folders = sorted(
        Path(path)
        for path in glob.glob(
            str(data_root / "ecephys_*" / "**" / "ecephys_clipped"),
            recursive=True,
        )
    )
    clipped_folders = [
        path
        for path in clipped_folders
        if not session_filters
        or any(value in extract_session_id(path) for value in session_filters)
    ]
    batch_entries = []
    failed = not clipped_folders
    if not clipped_folders:
        batch_entries.append(
            {"status": "failed", "error": f"No raw ecephys assets found under {data_root}"}
        )
    for clipped_path in clipped_folders:
        session_id = extract_session_id(clipped_path)
        sorted_candidates = sorted(
            data_root.glob(f"ecephys_{session_id}_sorted*")
        )
        session_output = (
            results_root if len(clipped_folders) == 1 else results_root / session_id
        )
        if not sorted_candidates:
            message = f"No sorted asset for session {session_id}"
            print(message)
            batch_entries.append({"session_id": session_id, "error": message})
            failed = True
            continue
        try:
            manifest = analyze_session(
                clipped_path,
                sorted_candidates[-1],
                output_dir=session_output,
                requested_streams=_parse_streams(None),
            )
            batch_entries.append(
                {
                    "session_id": session_id,
                    "status": "success",
                    "manifest": str(session_output / "analysis_manifest.json"),
                    "streams": len(manifest["streams"]),
                }
            )
        except Exception as exc:
            failed = True
            write_json(
                session_output / "failure.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "completed_at": utc_now(),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            batch_entries.append(
                {"session_id": session_id, "status": "failed", "error": str(exc)}
            )
    write_json(
        results_root / "batch_manifest.json",
        {"completed_at": utc_now(), "sessions": batch_entries},
    )
    return int(failed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze one mounted raw/sorted ecephys pair"
    )
    analyze_parser.add_argument("--raw-path", default="/data/raw_ephys")
    analyze_parser.add_argument("--sorted-path", default="/data/sorted_ephys")
    analyze_parser.add_argument("--output-dir", default="/results")
    analyze_parser.add_argument("--trials-csv")
    analyze_parser.add_argument("--session-id")
    analyze_parser.add_argument("--stream", action="append")
    analyze_parser.add_argument("--opto-recording", type=int, default=0)
    analyze_parser.add_argument("--flip-nidaq", action="store_true")
    analyze_parser.add_argument("--include-onset-offset", action="store_true")
    analyze_parser.add_argument("--raw-asset-id")
    analyze_parser.add_argument("--raw-asset-name")
    analyze_parser.add_argument("--sorted-asset-id")
    analyze_parser.add_argument("--sorted-asset-name")
    analyze_parser.set_defaults(handler=_run_analyze)
    from dispatch import add_dispatch_parser

    add_dispatch_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        return _run_attached_assets()
    args = build_parser().parse_args(arguments)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())