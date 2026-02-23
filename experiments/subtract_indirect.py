"""Post-processing: filter circuit to short-range edges with optional subtraction.

For distance-1 edges (adjacent layers): kept as-is.
For distance-2 edges: subtract the single indirect path (exact, no overcounting).
For distance 3+: dropped (overcounting makes subtraction unreliable).

The max_distance parameter controls which edges to keep (default: 2).

Usage:
    poetry run python -m experiments.subtract_indirect results/circuit_XXXX.json [max_distance]
"""

import json
import sys
from pathlib import Path

import numpy as np

from feature_circuit_discovery.export import export_circuit_html


def filter_to_direct_edges(data: dict, max_distance: int = 2) -> dict:
    """Keep only short-range edges, with exact subtraction for distance-2.

    - Distance 1: kept unchanged (always direct).
    - Distance 2: exact subtraction of the single indirect path through
      the one intermediate layer (no overcounting possible).
    - Distance 3+: dropped entirely.
    """
    # Build lookup: (up_layer, down_layer) -> gradient matrix
    matrices = {}
    for pair in data["layer_pairs"]:
        key = (pair["upstream_layer"], pair["downstream_layer"])
        matrices[key] = np.array(pair["gradient_matrix"], dtype=np.float64)

    layers = sorted(set(l["layer_idx"] for l in data["layers"]))

    direct = {}
    for (i, j), mat in matrices.items():
        dist = j - i
        if dist > max_distance:
            continue  # drop long-range edges

        if dist == 1:
            # Adjacent: always direct
            direct[(i, j)] = mat
        elif dist == 2:
            # Exactly one intermediate layer k = i+1 — subtraction is exact
            k = i + 1
            d = mat.copy()
            if (i, k) in matrices and (k, j) in matrices:
                d -= matrices[(k, j)] @ matrices[(i, k)]
            direct[(i, j)] = d
        else:
            # Distance 3+: keep total (subtraction overcounts)
            direct[(i, j)] = mat

    # Build new data keeping only the retained pairs
    new_data = json.loads(json.dumps(data))  # deep copy
    new_pairs = []
    for pair in new_data["layer_pairs"]:
        key = (pair["upstream_layer"], pair["downstream_layer"])
        if key not in direct:
            continue
        mat = direct[key]
        pair["gradient_matrix"] = [
            [round(float(mat[r, c]), 6) for c in range(mat.shape[1])]
            for r in range(mat.shape[0])
        ]
        abs_mat = np.abs(mat)
        pair["stats"] = {
            "max_abs": round(float(abs_mat.max()), 6),
            "mean_abs": round(float(abs_mat.mean()), 6),
            "nonzero_count": int(np.count_nonzero(mat)),
            "total_elements": int(mat.size),
        }
        new_pairs.append(pair)

    new_data["layer_pairs"] = new_pairs
    new_data["metadata"]["post_processing"] = (
        f"direct_edges_max_dist_{max_distance}"
        + (" + dist2_subtraction" if max_distance >= 2 else "")
    )
    return new_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: poetry run python -m experiments.subtract_indirect <circuit.json> [max_distance]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    max_distance = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    print(f"Loading {input_path}...")
    with open(input_path) as f:
        data = json.load(f)

    n_pairs = len(data["layer_pairs"])
    n_layers = len(data["layers"])
    print(f"Found {n_pairs} layer pairs across {n_layers} layers")
    print(f"Max distance: {max_distance} (keeping dist 1-{max_distance}, dropping rest)")
    if max_distance >= 2:
        print("Distance-2 edges: subtracting exact single-hop indirect path")

    new_data = filter_to_direct_edges(data, max_distance=max_distance)

    kept = len(new_data["layer_pairs"])
    dropped = n_pairs - kept
    print(f"\nKept {kept}/{n_pairs} layer pairs (dropped {dropped})")

    # Save JSON
    suffix = f"_direct_d{max_distance}"
    output_path = input_path.with_stem(input_path.stem + suffix)
    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=2)
    print(f"Saved JSON: {output_path}")

    # Generate HTML
    html_path = export_circuit_html(output_path)
    print(f"Saved HTML: {html_path}")

    # Print stats per distance
    print("\n--- Edge stats by distance ---")
    by_dist = {}
    for pair in new_data["layer_pairs"]:
        dist = pair["downstream_layer"] - pair["upstream_layer"]
        by_dist.setdefault(dist, []).append(pair["stats"]["max_abs"])

    for dist in sorted(by_dist):
        vals = by_dist[dist]
        label = "(adjacent)" if dist == 1 else "(subtracted)" if dist == 2 else ""
        print(f"  Dist {dist} {label}: {len(vals)} pairs, "
              f"max |grad| = {max(vals):.4f}, "
              f"mean max = {sum(vals)/len(vals):.4f}")
