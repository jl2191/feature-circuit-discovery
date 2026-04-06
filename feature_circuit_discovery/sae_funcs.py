"""Neuronpedia API utilities for fetching SAE feature descriptions and metadata."""

from feature_circuit_discovery.data import (
    NEURONPEDIA_MODEL_ID,
    NEURONPEDIA_SAE_SET,
)


def describe_feature(layer: int, idx: int) -> str | None:
    """Fetch the auto-interp description of an SAE feature from Neuronpedia.

    Args:
        layer: Layer index.
        idx: Feature index in the 16k-width SAE.

    Returns:
        Description string, or None if unavailable.
    """
    import requests

    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL_ID}/{layer}-{NEURONPEDIA_SAE_SET}/{idx}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        explanations = data.get("explanations", [])
        if explanations:
            return explanations[0].get("description")
        return None
    except Exception:
        return None


def describe_features(
    features: list[tuple[int, int]],
    max_concurrent: int = 10,
) -> dict[tuple[int, int], str]:
    """Fetch descriptions for multiple features from Neuronpedia.

    Args:
        features: List of (layer, feature_idx) tuples.
        max_concurrent: Max concurrent requests.

    Returns:
        Dict mapping (layer, feature_idx) -> description string.
    """
    import concurrent.futures
    import requests

    results: dict[tuple[int, int], str] = {}

    def fetch_one(layer: int, idx: int) -> tuple[int, int, str | None]:
        url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL_ID}/{layer}-{NEURONPEDIA_SAE_SET}/{idx}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return (layer, idx, None)
            data = resp.json()
            explanations = data.get("explanations", [])
            if explanations:
                return (layer, idx, explanations[0].get("description"))
            return (layer, idx, None)
        except Exception:
            return (layer, idx, None)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(fetch_one, layer, idx) for layer, idx in features]
        for future in concurrent.futures.as_completed(futures):
            layer, idx, desc = future.result()
            if desc:
                results[(layer, idx)] = desc

    return results


def fetch_feature_detail(layer: int, idx: int) -> dict | None:
    """Fetch rich feature data from Neuronpedia.

    Returns a dict with keys: desc, pos_str, pos_values, neg_str, neg_values,
    frac_nonzero, examples — or None if unavailable.
    """
    import requests

    url = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL_ID}/{layer}-{NEURONPEDIA_SAE_SET}/{idx}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        explanations = data.get("explanations", [])
        examples = []
        for act in (data.get("activations") or [])[:5]:
            examples.append({
                "tokens": act.get("tokens", []),
                "values": act.get("values", []),
                "max_token_index": act.get("maxValueTokenIndex", 0),
                "max_value": act.get("maxValue", 0),
            })
        return {
            "desc": explanations[0].get("description") if explanations else None,
            "pos_str": data.get("pos_str", [])[:10],
            "pos_values": data.get("pos_values", [])[:10],
            "neg_str": data.get("neg_str", [])[:10],
            "neg_values": data.get("neg_values", [])[:10],
            "frac_nonzero": data.get("frac_nonzero"),
            "examples": examples,
        }
    except Exception:
        return None


def fetch_feature_details(
    features: list[tuple[int, int]],
    max_concurrent: int = 10,
) -> dict[tuple[int, int], dict]:
    """Fetch rich feature data for multiple features from Neuronpedia.

    Args:
        features: List of (layer, feature_idx) tuples.
        max_concurrent: Max concurrent requests.

    Returns:
        Dict mapping (layer, feature_idx) -> detail dict.
    """
    import concurrent.futures

    results: dict[tuple[int, int], dict] = {}

    def _fetch_one(layer: int, idx: int) -> tuple[int, int, dict | None]:
        return (layer, idx, fetch_feature_detail(layer, idx))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = [executor.submit(_fetch_one, layer, idx) for layer, idx in features]
        for future in concurrent.futures.as_completed(futures):
            layer, idx, detail = future.result()
            if detail:
                results[(layer, idx)] = detail

    return results
