"""Cluster assignment helpers for document-level generative QA."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from src.pipeline.data.contracts import CanonicalQARecord


VALID_CLUSTER_STRATEGIES = frozenset({"metadata_field"})


def cfg_get(cfg, key: str, default=None):
    """Read a value from a DictConfig, mapping, or namespace-like object."""

    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass(frozen=True)
class ClusterStrategySpec:
    """Resolved document-clustering strategy."""

    strategy: str
    metadata_field: str | None = None

    def folder_label(self) -> str:
        safe_field = str(self.metadata_field).replace(".", "_")
        return f"metadata_field_{safe_field}"


def resolve_cluster_strategy(
    data_cfg,
    *,
    default_strategy: str = "metadata_field",
) -> ClusterStrategySpec:
    """Resolve a supported document-clustering strategy from configuration."""

    strategy_value = cfg_get(data_cfg, "cluster_strategy", None) or default_strategy
    strategy = str(strategy_value)
    if strategy not in VALID_CLUSTER_STRATEGIES:
        raise ValueError(
            f"Unknown cluster strategy {strategy!r}; "
            f"valid values: {sorted(VALID_CLUSTER_STRATEGIES)}"
        )
    metadata_field = cfg_get(data_cfg, "cluster_metadata_field", None)
    if not metadata_field:
        raise ValueError(
            "cluster_strategy=metadata_field requires data.cluster_metadata_field"
        )
    return ClusterStrategySpec(strategy=strategy, metadata_field=metadata_field)


def metadata_value_for_record(record: CanonicalQARecord, metadata_field: str) -> str:
    """Look up a clustering field from canonical record metadata."""

    value = record.metadata.get(metadata_field)
    if value is None and "." in metadata_field:
        parts = metadata_field.split(".")
        if parts[0] == "metadata":
            value = record.metadata
            for part in parts[1:]:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(part)
    if value is None:
        raise ValueError(
            f"Record {record.record_id!r} is missing metadata field "
            f"{metadata_field!r} required by cluster_strategy=metadata_field"
        )
    return str(value)


def assign_clusters_by_metadata_field(
    records: Iterable[CanonicalQARecord],
    *,
    metadata_field: str,
    expected_num_clusters: int | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Assign each distinct metadata value to one document cluster."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[metadata_value_for_record(record, metadata_field)].append(record.record_id)

    ordered_values = sorted(grouped)
    value_to_cluster = {value: idx for idx, value in enumerate(ordered_values)}
    if expected_num_clusters is not None and expected_num_clusters > 0:
        if len(value_to_cluster) != int(expected_num_clusters):
            raise ValueError(
                f"cluster_strategy=metadata_field for {metadata_field!r} produced "
                f"{len(value_to_cluster)} clusters, but data.num_clusters="
                f"{expected_num_clusters}"
            )

    record_to_cluster = {
        record_id: value_to_cluster[value]
        for value, record_ids in grouped.items()
        for record_id in record_ids
    }
    return record_to_cluster, value_to_cluster
