"""Normalize and pair Code Ocean ecephys data assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import re
from typing import Any, Iterable, Mapping, Sequence

import requests


SESSION_PATTERN = re.compile(
    r"(?P<session>\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
)
REJECTED_STATES = {"archived", "failed", "pending", "processing", "uploading"}
SORTING_TAGS = {"curated", "sorted", "spikesorted", "spike-sorted"}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Cannot normalize asset record of type {type(value).__name__}")


def _first_value(mappings: Iterable[Mapping[str, Any]], *keys: str) -> Any:
    for mapping in mappings:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    return None


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _parse_creation_time(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_creation_time(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    return str(value)


def _infer_data_level(name: str) -> str | None:
    session_id = extract_session_id(name)
    if not session_id or not name.startswith("ecephys_"):
        return None
    if "_sorted_" in name or "_optotagging_" in name:
        return "derived"
    if name == f"ecephys_{session_id}":
        return "raw"
    return None


def extract_session_id(value: str) -> str | None:
    """Return the acquisition session token embedded in a name."""
    match = SESSION_PATTERN.search(value)
    return match.group("session") if match else None


@dataclass(frozen=True)
class AssetRecord:
    """The subset of Code Ocean and AIND metadata needed for resolution."""

    id: str
    name: str
    data_level: str | None = None
    subject_id: str | None = None
    creation_time: str | None = None
    source_data: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    state: str | None = None
    archived: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_value(cls, value: Any) -> "AssetRecord":
        """Normalize an SDK model or API response mapping."""
        record = _as_mapping(value)
        metadata = _as_mapping(record.get("metadata", {}))
        custom_metadata = _as_mapping(record.get("custom_metadata", {}))
        data_description = _as_mapping(record.get("data_description", {}))
        sources = (record, data_description, metadata, custom_metadata)

        asset_id = _first_value(sources, "id", "asset_id")
        name = _first_value(sources, "name")
        if not asset_id or not name:
            raise ValueError("Code Ocean asset records require non-empty id and name")

        archived = bool(_first_value(sources, "archived", "is_archived") or False)
        data_level = _normalize_text(
            _first_value(sources, "data_level", "data level")
        ) or _infer_data_level(str(name))
        return cls(
            id=str(asset_id),
            name=str(name),
            data_level=data_level,
            subject_id=_normalize_text(
                _first_value(sources, "subject_id", "subject id")
            ),
            creation_time=_normalize_creation_time(
                _first_value(sources, "creation_time", "created")
            ),
            source_data=_as_tuple(
                _first_value(sources, "source_data", "source data")
            ),
            tags=_as_tuple(_first_value(sources, "tags")),
            state=_normalize_text(_first_value(sources, "state", "status")),
            archived=archived,
            metadata=data_description or metadata or custom_metadata,
        )

    @property
    def session_id(self) -> str | None:
        return extract_session_id(self.name)

    @property
    def inferred_subject_id(self) -> str | None:
        if self.subject_id:
            return self.subject_id
        session_id = self.session_id
        return session_id.split("_", 1)[0] if session_id else None

    @property
    def is_ready(self) -> bool:
        return not self.archived and (self.state or "").lower() not in REJECTED_STATES

    @property
    def is_raw_ecephys(self) -> bool:
        return self.data_level == "raw" and self.name.startswith("ecephys_")

    @property
    def is_sorted_ecephys(self) -> bool:
        normalized_tags = {tag.lower() for tag in self.tags}
        has_sorting_role = "_sorted_" in self.name or bool(
            normalized_tags & SORTING_TAGS
        )
        return self.data_level == "derived" and has_sorting_role

    @property
    def creation_datetime(self) -> datetime:
        return _parse_creation_time(self.creation_time)


def _normalize_text(value: Any) -> str | None:
    return str(value) if value is not None else None


@dataclass(frozen=True)
class SessionAssetPair:
    """One raw acquisition and one compatible sorted ecephys asset."""

    session_id: str
    raw: AssetRecord
    sorted: AssetRecord
    pairing_method: str


@dataclass(frozen=True)
class PairResolution:
    """Selected pairs plus actionable catalog diagnostics."""

    pairs: tuple[SessionAssetPair, ...]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def _lineage_session(asset: AssetRecord) -> str | None:
    lineage_sessions = {
        session
        for source in asset.source_data
        if (session := extract_session_id(source)) is not None
    }
    if len(lineage_sessions) == 1:
        return lineage_sessions.pop()
    return None


def resolve_asset_pairs(
    values: Iterable[Any],
    subject_id: str | int,
    *,
    all_sortings: bool = False,
    session_ids: Sequence[str] | None = None,
) -> PairResolution:
    """Resolve ready raw/sorted pairs for one subject.

    AIND ``source_data`` lineage is authoritative. The acquisition session token
    in the sorted asset name is used only when lineage is unavailable.
    """
    requested_subject = str(subject_id)
    requested_sessions = set(session_ids or ())
    assets = tuple(AssetRecord.from_value(value) for value in values)
    candidates = tuple(
        asset
        for asset in assets
        if asset.inferred_subject_id == requested_subject and asset.is_ready
    )

    raw_by_session: dict[str, list[AssetRecord]] = {}
    for asset in candidates:
        if asset.is_raw_ecephys and asset.session_id:
            raw_by_session.setdefault(asset.session_id, []).append(asset)

    warnings: list[str] = []
    errors: list[str] = []
    unambiguous_raw: dict[str, AssetRecord] = {}
    for session_id, raw_assets in sorted(raw_by_session.items()):
        if requested_sessions and session_id not in requested_sessions:
            continue
        if len(raw_assets) != 1:
            errors.append(
                f"Session {session_id} has {len(raw_assets)} ready raw assets; "
                "refusing to choose one"
            )
            continue
        unambiguous_raw[session_id] = raw_assets[0]

    sortings_by_session: dict[str, list[tuple[AssetRecord, str]]] = {}
    for asset in candidates:
        if not asset.is_sorted_ecephys:
            continue
        lineage_session = _lineage_session(asset)
        if lineage_session:
            session_id = lineage_session
            method = "source_data"
        else:
            session_id = asset.session_id
            method = "name_fallback"
        if not session_id or (requested_sessions and session_id not in requested_sessions):
            continue
        if session_id not in unambiguous_raw:
            errors.append(
                f"Sorted asset {asset.id} references session {session_id}, "
                "but no unambiguous ready raw asset was found"
            )
            continue
        if method == "name_fallback":
            warnings.append(
                f"Sorted asset {asset.id} has no usable source_data lineage; "
                f"paired to {session_id} by name"
            )
        sortings_by_session.setdefault(session_id, []).append((asset, method))

    pairs: list[SessionAssetPair] = []
    for session_id, raw_asset in sorted(unambiguous_raw.items()):
        sorting_candidates = sortings_by_session.get(session_id, [])
        if not sorting_candidates:
            errors.append(f"Session {session_id} has no ready sorted ecephys asset")
            continue
        sorting_candidates.sort(
            key=lambda item: (item[0].creation_datetime, item[0].id), reverse=True
        )
        selected = sorting_candidates if all_sortings else sorting_candidates[:1]
        if not all_sortings and len(sorting_candidates) > 1:
            skipped_ids = ", ".join(item[0].id for item in sorting_candidates[1:])
            warnings.append(
                f"Session {session_id}: selected latest sorting "
                f"{sorting_candidates[0][0].id}; skipped {skipped_ids}"
            )
        pairs.extend(
            SessionAssetPair(
                session_id=session_id,
                raw=raw_asset,
                sorted=sorting_asset,
                pairing_method=method,
            )
            for sorting_asset, method in selected
        )

    if requested_sessions:
        unresolved = requested_sessions - {pair.session_id for pair in pairs}
        for session_id in sorted(unresolved):
            if not any(session_id in error for error in errors):
                errors.append(f"Requested session {session_id} could not be resolved")

    return PairResolution(
        pairs=tuple(pairs), warnings=tuple(warnings), errors=tuple(errors)
    )


class CodeOceanCatalog:
    """Authenticated catalog access with AIND metadata enrichment."""

    def __init__(self, client, *, request_timeout: float = 60):
        self.client = client
        self.request_timeout = request_timeout

    @classmethod
    def from_environment(cls) -> "CodeOceanCatalog":
        from codeocean import CodeOcean

        domain = os.getenv(
            "CODEOCEAN_DOMAIN", "https://codeocean.allenneuraldynamics.org"
        )
        token = os.getenv("CODEOCEAN_TOKEN") or os.getenv("API_SECRET")
        if not token:
            raise RuntimeError(
                "Set CODEOCEAN_TOKEN or API_SECRET to a Code Ocean API token "
                "with data-asset search permission"
            )
        return cls(CodeOcean(domain=domain, token=token, retries=3))

    def _read_data_description(self, asset_id: str) -> Mapping[str, Any] | None:
        files = self.client.data_assets.list_data_asset_files(asset_id)
        description_item = next(
            (
                item
                for item in files.items
                if item.type == "file" and item.name == "data_description.json"
            ),
            None,
        )
        if description_item is None:
            return None
        urls = self.client.data_assets.get_data_asset_file_urls(
            asset_id, description_item.path
        )
        response = requests.get(urls.download_url, timeout=self.request_timeout)
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Asset {asset_id} data_description.json is not a JSON object"
            )
        return value

    def search_subject_assets(
        self, subject_id: str | int
    ) -> tuple[tuple[AssetRecord, ...], tuple[str, ...]]:
        from codeocean.components import SearchFilter, SortOrder
        from codeocean.data_asset import DataAssetSearchParams, DataAssetSortBy

        search_params = DataAssetSearchParams(
            limit=1000,
            archived=False,
            sort_field=DataAssetSortBy.Created,
            sort_order=SortOrder.Descending,
            filters=[
                SearchFilter(key="name", value=f"ecephys_{subject_id}_")
            ],
        )
        assets: list[AssetRecord] = []
        warnings: list[str] = []
        iterator = self.client.data_assets.search_data_assets_iterator(search_params)
        for asset in iterator:
            value = _as_mapping(asset)
            try:
                description = self._read_data_description(str(value["id"]))
            except Exception as exc:
                description = None
                warnings.append(
                    f"Could not read data_description.json for asset "
                    f"{value.get('id')}: {exc}"
                )
            if description is None:
                warnings.append(
                    f"Asset {value.get('id')} has no readable "
                    "data_description.json; using name metadata fallback"
                )
            enriched = dict(value)
            enriched["data_description"] = description or {}
            assets.append(AssetRecord.from_value(enriched))
        return tuple(assets), tuple(warnings)

    def find_existing_optotagging_asset(
        self,
        pair: SessionAssetPair,
        analysis_capsule_id: str,
    ):
        """Return an existing ready result with matching input provenance."""
        from codeocean.components import SearchFilter
        from codeocean.data_asset import DataAssetSearchParams

        search_params = DataAssetSearchParams(
            limit=1000,
            archived=False,
            filters=[
                SearchFilter(key="name", value=f"ecephys_{pair.session_id}_optotagging"),
                SearchFilter(key="tags", value="optotagging"),
            ],
        )
        source_ids = {pair.raw.id, pair.sorted.id}
        for asset in self.client.data_assets.search_data_assets_iterator(search_params):
            provenance = getattr(asset, "provenance", None)
            provenance_ids = set(getattr(provenance, "data_assets", None) or ())
            if (
                str(getattr(asset, "state", "")) == "ready"
                and source_ids.issubset(provenance_ids)
                and getattr(provenance, "capsule", None) == analysis_capsule_id
            ):
                return asset
        return None