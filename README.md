# Subject-wide optotagging analysis

This capsule analyzes paired raw and sorted ecephys assets and packages each
session as an AIND-derived result suitable for Code Ocean capture.

## Manual Reproducible Run

The Reproducible Run button remains supported. Attach raw and sorted assets with
their normal Code Ocean mount names under `/data`, then run the capsule without
parameters. The entry point scans for:

```text
/data/ecephys_<session>/**/ecephys_clipped
/data/ecephys_<session>_sorted_*
```

Every matching pair is analyzed. With one attached pair, the export-ready package
is written directly to `/results`:

```text
/results/analysis_manifest.json
/results/data_description.json
/results/processing.json
/results/optotagging/metrics/
/results/optotagging/figures/
/results/batch_manifest.json
```

When multiple pairs are attached, each package is placed under
`/results/<session_id>/` to prevent collisions.

`SESSION_IDS` and `STREAM_NAMES` remain available as comma-separated filters.

## Explicit Single Session

The dispatcher uses stable mount names, but the command can also be run directly:

```bash
./run analyze \
  --raw-path /data/raw_ephys \
  --sorted-path /data/sorted_ephys \
  --session-id 853781_2026-07-01_12-20-31
```

Use repeated `--stream` arguments to restrict probe shanks.

## Subject Dispatch

Subject dispatch searches the Code Ocean catalog, selects the newest valid
sorting for each acquisition session, and starts one monitored child computation
per pair. Start with a dry run:

```bash
./run dispatch \
  --subject-id 853781 \
  --analysis-capsule-id "$ANALYSIS_CAPSULE_ID" \
  --monitor-capsule-id "$PIPELINE_MONITOR_CAPSULE_ID" \
  --dry-run
```

The runtime requires `CODEOCEAN_TOKEN` or `API_SECRET` with catalog and
computation permissions. The token entered for the VS Code MCP server is scoped
to that editor process and is not automatically exposed to a reproducible run.

By default captured assets are private. `--publish` explicitly grants public
viewer access and targets the `aind-open-data` bucket. Use it only after reviewing
the one-session package and confirming data-release approval.