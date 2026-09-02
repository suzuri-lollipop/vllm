# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit layer selection for weight offloading."""


def parse_layer_spec(spec: str) -> frozenset[int] | None:
    """Parse an explicit layer index selection.

    The spec is a comma-separated list of decoder layer indices and inclusive
    ranges, e.g. `"0,2,24-47"`. Indices are global (they are not affected by
    pipeline parallel sharding). An empty spec selects nothing and is
    represented as `None` so callers can distinguish "no explicit selection"
    from "an explicit empty selection".

    Args:
        spec: The layer selection string.

    Returns:
        The selected layer indices, or None if `spec` is empty.

    Raises:
        ValueError: If the spec is malformed or contains a negative index or a
            descending range.
    """
    spec = spec.strip()
    if not spec:
        return None

    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        start_str, sep, end_str = part.partition("-")
        try:
            start = int(start_str)
            end = int(end_str) if sep else start
        except ValueError:
            raise ValueError(
                f"Invalid layer selection {part!r} in {spec!r}: expected an "
                "integer or an inclusive 'start-end' range"
            ) from None
        if start < 0 or end < 0:
            raise ValueError(
                f"Invalid layer selection {part!r} in {spec!r}: "
                "layer indices must be non-negative"
            )
        if end < start:
            raise ValueError(
                f"Invalid layer range {part!r} in {spec!r}: "
                f"end ({end}) is before start ({start})"
            )
        layers.update(range(start, end + 1))

    if not layers:
        raise ValueError(f"Layer selection {spec!r} does not select any layer")
    return frozenset(layers)


def matches_param(name: str, param_segments: set[str]) -> bool:
    """Check whether a parameter name contains one of the given name segments.

    Matching is on full dot-separated segments, so `"experts.w2_weight"`
    matches `"mlp.experts.w2_weight"` while `"w2"` does not.
    """
    padded = f".{name}."
    return any(f".{segment}." in padded for segment in param_segments)
