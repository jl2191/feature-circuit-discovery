"""Export circuit data to JSON and generate interactive HTML visualizer."""

import json
from datetime import datetime
from pathlib import Path

import torch


def export_circuit_json(
    metadata: dict,
    activated_features: list[torch.Tensor],
    gradient_matrices: list[torch.Tensor],
    layer_pairs: list[tuple[int, int]],
    output_path: str | Path,
    activation_frequencies: dict[int, dict[int, float]] | None = None,
    logit_gradients: dict[int, torch.Tensor] | None = None,
    logit_token_labels: list[str] | None = None,
    logit_token_ids: list[int] | None = None,
    token_gradients: dict[int, torch.Tensor] | None = None,
    token_labels: list[str] | None = None,
    token_ids: list[int] | None = None,
) -> Path:
    """Serialize circuit discovery results to JSON.

    Args:
        metadata: Dict with keys like model, prompts, activation_threshold, dtype.
        activated_features: List of 1D tensors of SAE feature indices per layer.
        gradient_matrices: List of (n_down, n_up) gradient tensors, one per layer pair.
        layer_pairs: List of (upstream_layer, downstream_layer) tuples.
        output_path: Where to write the JSON file.
        activation_frequencies: Optional dict mapping layer_idx -> {feature_id: frequency}.
        logit_gradients: Optional dict mapping upstream_layer -> (n_tokens, n_feats) tensor.
        logit_token_labels: Optional list of token label strings (e.g. [" Yes", " No"]).
        logit_token_ids: Optional list of token IDs corresponding to labels.
        token_gradients: Optional dict mapping layer_idx -> (n_features, seq_len) tensor.
        token_labels: Optional list of input token strings.
        token_ids: Optional list of input token IDs.

    Returns:
        Path to the written JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build layers array
    layers = []
    seen_layers = set()
    for up, down in layer_pairs:
        for idx in (up, down):
            if idx not in seen_layers and idx < len(activated_features):
                seen_layers.add(idx)
                feat_ids = activated_features[idx].cpu().tolist()
                layer_data = {
                    "layer_idx": idx,
                    "active_feature_ids": feat_ids,
                    "num_active": len(feat_ids),
                }
                if activation_frequencies and idx in activation_frequencies:
                    layer_data["activation_frequencies"] = {
                        str(k): round(v, 4)
                        for k, v in activation_frequencies[idx].items()
                    }
                layers.append(layer_data)
    layers.sort(key=lambda x: x["layer_idx"])

    # Build layer_pairs array
    pairs_data = []
    for (up_layer, down_layer), grad_mat in zip(layer_pairs, gradient_matrices):
        mat = grad_mat.cpu()
        n_down, n_up = mat.shape

        up_feats = activated_features[up_layer][:n_up].cpu().tolist()
        down_feats = activated_features[down_layer][:n_down].cpu().tolist()

        # Round to 6 decimal places to keep file size reasonable
        matrix_list = [[round(float(mat[i, j]), 6) for j in range(n_up)] for i in range(n_down)]

        abs_mat = mat.abs()
        pairs_data.append({
            "upstream_layer": up_layer,
            "downstream_layer": down_layer,
            "upstream_feature_ids": up_feats,
            "downstream_feature_ids": down_feats,
            "gradient_matrix": matrix_list,
            "stats": {
                "max_abs": round(float(abs_mat.max()), 6),
                "mean_abs": round(float(abs_mat.mean()), 6),
                "nonzero_count": int(mat.count_nonzero()),
                "total_elements": int(mat.numel()),
            },
        })

    # Build logit_nodes section if logit gradients were computed
    logit_nodes_data = None
    if logit_gradients and logit_token_labels and logit_token_ids:
        gradients_by_layer = {}
        for layer_idx, grad_tensor in logit_gradients.items():
            mat = grad_tensor.cpu()
            n_up = mat.shape[1]
            up_feats = activated_features[layer_idx][:n_up].cpu().tolist()
            gradients_by_layer[str(layer_idx)] = {
                "upstream_feature_ids": up_feats,
                "gradients": [
                    [round(float(mat[t, j]), 6) for j in range(n_up)]
                    for t in range(mat.shape[0])
                ],
            }
        logit_nodes_data = {
            "token_labels": logit_token_labels,
            "token_ids": logit_token_ids,
            "gradients_by_layer": gradients_by_layer,
        }

    # Build token_nodes section if token gradients were computed
    token_nodes_data = None
    if token_gradients and token_labels and token_ids:
        tok_gradients_by_layer = {}
        for layer_idx, grad_tensor in token_gradients.items():
            mat = grad_tensor.cpu()
            n_feats = mat.shape[0]
            down_feats = activated_features[layer_idx][:n_feats].cpu().tolist()
            tok_gradients_by_layer[str(layer_idx)] = {
                "downstream_feature_ids": down_feats,
                "gradients": [
                    [round(float(mat[f, t]), 6) for t in range(mat.shape[1])]
                    for f in range(n_feats)
                ],
            }
        token_nodes_data = {
            "tokens": token_labels,
            "token_ids": token_ids,
            "gradients_by_layer": tok_gradients_by_layer,
        }

    meta_out = {
            "model": metadata.get("model", "google/gemma-3-1b-pt"),
            "sae_repo": metadata.get("sae_repo", "google/gemma-scope-2-1b-pt"),
            "sae_width": metadata.get("sae_width", 16384),
            "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
            "prompts": metadata.get("prompts", []),
            "activation_threshold": metadata.get("activation_threshold", 0.9),
            "device": metadata.get("device", "cpu"),
            "dtype": metadata.get("dtype", "bfloat16"),
    }
    if "n_features_per_layer" in metadata:
        meta_out["n_features_per_layer"] = metadata["n_features_per_layer"]
    if "feature_selection" in metadata:
        meta_out["feature_selection"] = metadata["feature_selection"]

    data = {
        "metadata": meta_out,
        "layers": layers,
        "layer_pairs": pairs_data,
    }

    if logit_nodes_data:
        data["logit_nodes"] = logit_nodes_data

    if token_nodes_data:
        data["token_nodes"] = token_nodes_data

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def export_circuit_html(
    json_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a self-contained HTML visualizer with embedded circuit data.

    Args:
        json_path: Path to the circuit JSON file.
        output_path: Where to write the HTML. Defaults to same name as JSON with .html.

    Returns:
        Path to the written HTML file.
    """
    json_path = Path(json_path)
    if output_path is None:
        output_path = json_path.with_suffix(".html")
    output_path = Path(output_path)

    with open(json_path) as f:
        json_data = f.read()

    html = _HTML_TEMPLATE.replace("__CIRCUIT_DATA_PLACEHOLDER__", json_data)

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def export_standalone_visualizer(output_path: str | Path = "results/visualizer.html") -> Path:
    """Generate a standalone HTML visualizer with no embedded data.

    The user must load a JSON file via the file picker to visualize results.
    This is useful for comparing multiple experiment results.

    Returns:
        Path to the written HTML file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use empty placeholder that triggers file-picker-only mode
    empty_data = '{"metadata":{},"layers":[],"layer_pairs":[]}'
    html = _HTML_TEMPLATE.replace("__CIRCUIT_DATA_PLACEHOLDER__", empty_data)

    with open(output_path, "w") as f:
        f.write(html)

    return output_path


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Feature Circuit Visualizer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    display: flex;
    height: 100vh;
    overflow: hidden;
  }
  #sidebar {
    width: 310px;
    min-width: 310px;
    background: #16213e;
    padding: 20px;
    overflow-y: auto;
    border-right: 1px solid #0f3460;
  }
  #sidebar h1 { font-size: 18px; margin-bottom: 4px; color: #e94560; }
  #sidebar .subtitle { font-size: 12px; color: #888; margin-bottom: 16px; }
  .control-group { margin-bottom: 14px; }
  .control-group label {
    display: block; font-size: 13px; font-weight: 600; margin-bottom: 5px; color: #a0c4ff;
  }
  .control-group select, .control-group input[type="range"] { width: 100%; }
  select {
    background: #1a1a2e; color: #e0e0e0; border: 1px solid #0f3460;
    padding: 6px 8px; border-radius: 4px; font-size: 13px;
  }
  input[type="range"] {
    -webkit-appearance: none; height: 6px; background: #0f3460;
    border-radius: 3px; outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 16px; height: 16px;
    background: #e94560; border-radius: 50%; cursor: pointer;
  }
  #stats {
    background: #1a1a2e; border: 1px solid #0f3460; border-radius: 6px;
    padding: 10px; font-size: 12px; line-height: 1.7;
  }
  #stats .sl { color: #888; }
  #stats .sv { color: #e94560; font-weight: 600; }
  #graph-container { flex: 1; overflow: hidden; position: relative; }
  #graph-container svg { position: absolute; top: 0; left: 0; }
  #graph-3d { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: none; }
  .edge { pointer-events: stroke; cursor: pointer; }
  .edge:hover { stroke-opacity: 1 !important; stroke-width: 4px !important; }
  .node circle { cursor: grab; stroke: #fff; stroke-width: 1.5px; }
  .node circle:active { cursor: grabbing; }
  .node rect { cursor: grab; stroke: #fff; stroke-width: 1.5px; }
  .node rect:active { cursor: grabbing; }
  .node text { font-size: 9px; fill: #ccc; pointer-events: none; }
  .node.dimmed circle { opacity: 0.15; }
  .node.dimmed text { opacity: 0.15; }
  .edge.dimmed { stroke-opacity: 0.02 !important; }
  .layer-label { fill: #555; font-size: 11px; font-weight: 600; }
  .layer-line { stroke: #222; stroke-width: 1; stroke-dasharray: 4,4; }
  #tooltip {
    position: fixed; background: #16213e; border: 1px solid #0f3460;
    border-radius: 6px; padding: 8px 12px; font-size: 12px;
    pointer-events: auto; display: none; z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4); max-width: 350px;
  }
  #tooltip a { pointer-events: auto; }
  #tooltip .tl { color: #888; }
  #tooltip .tv { color: #e94560; font-weight: 600; }
  .legend { margin-top: 12px; font-size: 11px; }
  .legend-item { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
  .legend-swatch { width: 18px; height: 3px; border-radius: 2px; }
  #tooltip a { color: #4dabf7; text-decoration: none; }
  #tooltip a:hover { text-decoration: underline; }
  #load-json-btn {
    width: 100%; padding: 8px; margin-bottom: 6px;
    background: #0f3460; color: #a0c4ff; border: 1px solid #4dabf7;
    border-radius: 4px; font-size: 12px; cursor: pointer; font-weight: 600;
  }
  #load-json-btn:hover { background: #1a4a8a; }
  #loaded-file { font-size: 10px; color: #666; margin-bottom: 10px; word-break: break-all; }
  .section-label { font-size: 11px; color: #666; font-weight: 600; margin-top: 10px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
  #metadata { font-size: 10px; color: #555; margin-top: 12px; line-height: 1.5; }
</style>
</head>
<body>

<div id="sidebar">
  <h1>Feature Circuit Visualizer</h1>
  <div class="subtitle">Multi-Layer SAE Gradient Circuit</div>

  <div class="control-group">
    <button id="view-mode-btn" style="width:100%;padding:8px;margin-bottom:6px;background:#0f3460;color:#a0c4ff;border:1px solid #4dabf7;border-radius:4px;font-size:12px;cursor:pointer;font-weight:600;">Switch to 3D View</button>
    <button id="load-json-btn" onclick="document.getElementById('json-file-input').click()">Load JSON File...</button>
    <input type="file" id="json-file-input" accept=".json" style="display:none">
    <div id="loaded-file"></div>
  </div>

  <div class="section-label">Feature Selection</div>

  <div class="control-group">
    <label>Node Selection Mode</label>
    <select id="selection-mode">
      <option value="topn">Top N Features</option>
      <option value="edge-threshold">Edge Threshold</option>
    </select>
  </div>

  <div class="control-group" id="topn-group">
    <label>Global Top N Features: <span id="topn-value"></span></label>
    <input type="range" id="topn-slider" min="5" max="200" value="30">
  </div>

  <div class="control-group" id="rank-group">
    <label>Ranking Mode</label>
    <select id="rank-mode">
      <option value="gradient">Total |Gradient| Influence</option>
      <option value="logit">Logit |Gradient| (Yes+No)</option>
      <option value="frequency">Activation Frequency</option>
    </select>
  </div>

  <div class="control-group">
    <label>Max Layer Distance</label>
    <select id="max-dist">
      <option value="0">All pairs</option>
      <option value="1" selected>Adjacent only (d=1)</option>
      <option value="2">d &le; 2</option>
      <option value="3">d &le; 3</option>
      <option value="5">d &le; 5</option>
    </select>
  </div>

  <div class="control-group">
    <label>Min |Gradient| Threshold: <span id="threshold-value"></span></label>
    <input type="range" id="threshold-slider" min="0" max="100" value="10" step="1">
  </div>

  <div class="control-group" id="token-threshold-group" style="display:none;">
    <label>Token Edge Threshold: <span id="token-threshold-value"></span></label>
    <input type="range" id="token-threshold-slider" min="0" max="100" value="10" step="1">
  </div>

  <div class="section-label">Layout Forces</div>

  <div class="control-group">
    <label>Repulsion Strength: <span id="charge-value">-150</span></label>
    <input type="range" id="charge-slider" min="-500" max="-20" value="-150" step="10">
  </div>

  <div class="control-group">
    <label>Link Distance: <span id="link-dist-value">80</span></label>
    <input type="range" id="link-dist-slider" min="10" max="300" value="80" step="10">
  </div>

  <div class="control-group">
    <label>Edge Opacity: <span id="opacity-value"></span></label>
    <input type="range" id="opacity-slider" min="5" max="100" value="50" step="5">
  </div>

  <div class="control-group">
    <label>Stats</label>
    <div id="stats"></div>
  </div>

  <div class="section-label">Export</div>

  <div class="control-group">
    <button id="export-graph-btn" style="width:100%;padding:8px;background:#0f3460;color:#a0c4ff;border:1px solid #4dabf7;border-radius:4px;font-size:12px;cursor:pointer;font-weight:600;">Export Graph JSON for LLM...</button>
    <div id="export-progress" style="display:none;font-size:11px;color:#a0c4ff;margin-top:4px;"></div>
  </div>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch" style="background: #4dabf7;"></div>
      <span>Positive gradient (excitatory)</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background: #ff6b6b;"></div>
      <span>Negative gradient (inhibitory)</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background: #e94560; height: 10px; width: 10px; border-radius: 50%;"></div>
      <span>Logit attribution node</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background: #51cf66; height: 10px; width: 10px; border-radius: 2px;"></div>
      <span>Input token node</span>
    </div>
    <div class="legend-item">
      <span style="color:#888;">Node size = ranking score &middot; Click node to trace &middot; Drag to reposition</span>
    </div>
  </div>

  <div id="metadata"></div>
</div>

<div id="graph-container">
  <svg id="graph"></svg>
  <div id="graph-3d"></div>
</div>

<div id="tooltip"></div>

<script>
let DATA = __CIRCUIT_DATA_PLACEHOLDER__;

// --- Mutable state recomputed on data load ---
let featureGradScore = {};
let featureFreq = {};
let allFeatures = [];
let globalMaxAbs = 0;

function initData(data) {
  DATA = data;
  featureGradScore = {};
  featureFreq = {};
  allFeatures = [];

  // Set Neuronpedia config based on model
  const modelId = (DATA.metadata && DATA.metadata.model) || "";
  npConfig = NEURONPEDIA_CONFIG_MAP[modelId] || null;

  DATA.layers.forEach(layer => {
    const li = layer.layer_idx;
    layer.active_feature_ids.forEach(fid => {
      const key = li + ":" + fid;
      featureGradScore[key] = 0;
      if (layer.activation_frequencies && layer.activation_frequencies[fid] !== undefined) {
        featureFreq[key] = layer.activation_frequencies[fid];
      } else {
        featureFreq[key] = 1.0;
      }
    });
  });

  DATA.layer_pairs.forEach(pair => {
    const mat = pair.gradient_matrix;
    pair.upstream_feature_ids.forEach((uid, j) => {
      const key = pair.upstream_layer + ":" + uid;
      let sum = 0;
      for (let i = 0; i < mat.length; i++) sum += Math.abs(mat[i][j]);
      featureGradScore[key] = (featureGradScore[key] || 0) + sum;
    });
    pair.downstream_feature_ids.forEach((did, i) => {
      const key = pair.downstream_layer + ":" + did;
      let sum = 0;
      for (let j = 0; j < mat[i].length; j++) sum += Math.abs(mat[i][j]);
      featureGradScore[key] = (featureGradScore[key] || 0) + sum;
    });
  });

  DATA.layers.forEach(layer => {
    layer.active_feature_ids.forEach(fid => {
      const key = layer.layer_idx + ":" + fid;
      allFeatures.push({
        layer: layer.layer_idx,
        id: fid,
        key: key,
        gradScore: featureGradScore[key] || 0,
        freq: featureFreq[key] || 0,
      });
    });
  });

  globalMaxAbs = DATA.layer_pairs.length > 0
    ? Math.max(...DATA.layer_pairs.map(p => p.stats.max_abs))
    : 0;
  topnSlider.max = Math.min(allFeatures.length, 500);

  // Parse logit nodes
  DATA._logitNodes = [];
  DATA._logitEdges = [];
  if (DATA.logit_nodes) {
    const ln = DATA.logit_nodes;
    ln.token_labels.forEach((label, tIdx) => {
      DATA._logitNodes.push({
        key: "logit:" + tIdx,
        layer: "logit",
        id: ln.token_ids[tIdx],
        label: label,
        isLogitNode: true,
        gradScore: 0,
        freq: 0,
      });
    });
    // Build logit edges from all layers' features to logit nodes
    Object.keys(ln.gradients_by_layer).forEach(layerStr => {
      const layerData = ln.gradients_by_layer[layerStr];
      const upLayer = parseInt(layerStr);
      layerData.upstream_feature_ids.forEach((uid, j) => {
        ln.token_labels.forEach((label, tIdx) => {
          const val = layerData.gradients[tIdx][j];
          DATA._logitEdges.push({
            uKey: upLayer + ":" + uid,
            dKey: "logit:" + tIdx,
            uLayer: upLayer,
            uId: uid,
            dLayer: "logit",
            dId: ln.token_ids[tIdx],
            dLabel: label,
            value: val,
            isLogitEdge: true,
          });
        });
      });
    });
    // Compute total |gradient| for each logit node (for display)
    DATA._logitNodes.forEach(node => {
      let total = 0;
      DATA._logitEdges.forEach(e => {
        if (e.dKey === node.key) total += Math.abs(e.value);
      });
      node.gradScore = total;
    });
    // Compute per-feature logit grad score (sum of |grad| across all logit tokens)
    const featureLogitScore = {};
    DATA._logitEdges.forEach(e => {
      featureLogitScore[e.uKey] = (featureLogitScore[e.uKey] || 0) + Math.abs(e.value);
    });
    allFeatures.forEach(f => { f.logitScore = featureLogitScore[f.key] || 0; });
  }

  // Parse token nodes (input token attribution)
  DATA._tokenNodes = [];
  DATA._tokenEdges = [];
  DATA._tokenMaxAbs = 0;
  if (DATA.token_nodes) {
    const tn = DATA.token_nodes;
    tn.tokens.forEach((label, tIdx) => {
      DATA._tokenNodes.push({
        key: "token:" + tIdx,
        layer: "token",
        id: tn.token_ids[tIdx],
        label: label,
        tIdx: tIdx,
        isTokenNode: true,
        gradScore: 0,
        freq: 0,
      });
    });
    // Build token edges: from token nodes to downstream feature nodes
    Object.keys(tn.gradients_by_layer).forEach(layerStr => {
      const layerData = tn.gradients_by_layer[layerStr];
      const downLayer = parseInt(layerStr);
      layerData.downstream_feature_ids.forEach((did, fIdx) => {
        tn.tokens.forEach((label, tIdx) => {
          const val = layerData.gradients[fIdx][tIdx];
          DATA._tokenEdges.push({
            uKey: "token:" + tIdx,
            dKey: downLayer + ":" + did,
            uLayer: "token",
            uId: tn.token_ids[tIdx],
            uLabel: label,
            dLayer: downLayer,
            dId: did,
            value: val,
            isTokenEdge: true,
          });
          if (Math.abs(val) > DATA._tokenMaxAbs) DATA._tokenMaxAbs = Math.abs(val);
        });
      });
    });
    // Compute total |gradient| for each token node
    DATA._tokenNodes.forEach(node => {
      let total = 0;
      DATA._tokenEdges.forEach(e => {
        if (e.uKey === node.key) total += Math.abs(e.value);
      });
      node.gradScore = total;
    });
    // Compute per-feature token grad score
    DATA._tokenEdges.forEach(e => {
      const fKey = e.dKey;
      const feat = allFeatures.find(f => f.key === fKey);
      if (feat) feat.tokenScore = (feat.tokenScore || 0) + Math.abs(e.value);
    });
    // Show token threshold slider
    document.getElementById("token-threshold-group").style.display = "";
  } else {
    document.getElementById("token-threshold-group").style.display = "none";
  }

  const meta = DATA.metadata || {};
  document.getElementById("metadata").innerHTML =
    `<b>Model:</b> ${meta.model || "?"}<br>` +
    `<b>Time:</b> ${meta.timestamp || "?"}<br>` +
    `<b>Prompts:</b> ${(meta.prompts || []).length} (thresh=${meta.activation_threshold || "?"})<br>` +
    `<b>Device:</b> ${meta.device || "?"} (${meta.dtype || "?"})<br>` +
    `<b>Layers:</b> ${DATA.layers.length} | <b>Pairs:</b> ${DATA.layer_pairs.length}`;

  selectedNode = null;
  rebuildGraph();
}

// --- JSON file loader ---
document.getElementById("json-file-input").addEventListener("change", function(e) {
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById("loaded-file").textContent = file.name;
  const reader = new FileReader();
  reader.onload = function(ev) {
    try { initData(JSON.parse(ev.target.result)); }
    catch (err) { alert("Failed to parse JSON: " + err.message); }
  };
  reader.readAsText(file);
});

// --- Neuronpedia config: derived from DATA.metadata.model in initData() ---
const descCache = {};
let hoveredNodeKey = null;  // track which node tooltip is showing

// Model ID -> Neuronpedia config mapping
const NEURONPEDIA_CONFIG_MAP = {
  "google/gemma-3-1b-pt": { npModel: "gemma-3-1b", scope: "gemmascope-2-res-16k", layers: new Set([7, 13, 17, 22]) },
  "google/gemma-2-2b":    { npModel: "gemma-2-2b", scope: "gemmascope-res-16k",   layers: null },  // null = all layers
};
let npConfig = NEURONPEDIA_CONFIG_MAP["google/gemma-3-1b-pt"];  // default

function npHasLayer(layer) {
  return npConfig && (npConfig.layers === null || npConfig.layers.has(layer));
}

function npFeatureUrl(layer, featureId) {
  return `https://www.neuronpedia.org/${npConfig.npModel}/${layer}-${npConfig.scope}/${featureId}`;
}

function npApiUrl(layer, featureId) {
  return `https://www.neuronpedia.org/api/feature/${npConfig.npModel}/${layer}-${npConfig.scope}/${featureId}`;
}

function buildNodeTooltipHTML(d, description) {
  const score = nodeScore(d);
  const scoreLabel = currentMode === "logit" ? "Logit |grad|" : currentMode === "gradient" ? "Total |grad|" : "Activation freq";
  let html =
    `<span class="tl">Layer ${d.layer} Feature</span> <span class="tv">#${d.id}</span><br>` +
    `<span class="tl">${scoreLabel}:</span> <span class="tv">${score.toFixed(4)}</span>`;
  if (npHasLayer(d.layer)) {
    if (description === undefined) {
      html += `<br><span class="tl" style="font-style:italic;">Loading description...</span>`;
    } else if (description) {
      html += `<br><span class="tl">Description:</span> <span style="color:#a0c4ff;font-style:italic;">"${description}"</span>`;
    }
    html += `<br><a href="${npFeatureUrl(d.layer, d.id)}" target="_blank" style="color:#4dabf7;text-decoration:none;font-size:11px;">Neuronpedia &#8599;</a>`;
  }
  return html;
}

function fetchAndShowDescription(d) {
  const key = d.layer + ":" + d.id;
  if (key in descCache) return;  // already cached, tooltip was built with it
  if (!npHasLayer(d.layer)) { descCache[key] = null; return; }
  descCache[key] = undefined;  // mark as loading
  fetch(npApiUrl(d.layer, d.id))
    .then(r => r.ok ? r.json() : null)
    .then(data => {
      descCache[key] = (data && data.explanations && data.explanations.length > 0)
        ? (data.explanations[0].description || null) : null;
      // Update tooltip if still hovering this node
      if (hoveredNodeKey === key) {
        tooltip.innerHTML = buildNodeTooltipHTML(d, descCache[key]);
      }
    })
    .catch(() => {
      descCache[key] = null;
      if (hoveredNodeKey === key) {
        tooltip.innerHTML = buildNodeTooltipHTML(d, null);
      }
    });
}

// --- State ---
let selectedNode = null;
let pinnedTooltip = null;  // key of node whose tooltip is pinned
let simulation = null;
let currentNodes = [];
let currentEdges = [];
let currentActiveLayers = [];
let layerXPositions = {};
let maxScore = 1;
let currentMode = "gradient";

// --- Controls ---
const topnSlider = document.getElementById("topn-slider");
const rankMode = document.getElementById("rank-mode");
const maxDistSelect = document.getElementById("max-dist");
const thresholdSlider = document.getElementById("threshold-slider");
const chargeSlider = document.getElementById("charge-slider");
const linkDistSlider = document.getElementById("link-dist-slider");
const opacitySlider = document.getElementById("opacity-slider");
const tokenThresholdSlider = document.getElementById("token-threshold-slider");
const tooltip = document.getElementById("tooltip");

// --- SVG setup with zoom ---
const container = document.getElementById("graph-container");
const svgEl = d3.select("#graph");
let gRoot, gEdges, gNodes, gLabels;

function setupSVG() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  svgEl.attr("width", w).attr("height", h);
  svgEl.selectAll("*").remove();

  const zoom = d3.zoom()
    .scaleExtent([0.1, 5])
    .on("zoom", (event) => { gRoot.attr("transform", event.transform); });
  svgEl.call(zoom);

  gRoot = svgEl.append("g");
  gLabels = gRoot.append("g");
  gEdges = gRoot.append("g");
  gNodes = gRoot.append("g");
}
setupSVG();
window.addEventListener("resize", () => {
  svgEl.attr("width", container.clientWidth).attr("height", container.clientHeight);
});

// --- Event listeners ---
const selectionModeSelect = document.getElementById("selection-mode");
selectionModeSelect.addEventListener("change", () => { selectedNode = null; rebuildGraph(); });
topnSlider.addEventListener("input", () => { selectedNode = null; rebuildGraph(); });
rankMode.addEventListener("change", () => { selectedNode = null; rebuildGraph(); });
maxDistSelect.addEventListener("change", () => { selectedNode = null; rebuildGraph(); });
thresholdSlider.addEventListener("input", rebuildGraph);
tokenThresholdSlider.addEventListener("input", rebuildGraph);

chargeSlider.addEventListener("input", function() {
  document.getElementById("charge-value").textContent = this.value;
  if (simulation) {
    simulation.force("charge").strength(+this.value);
    simulation.alpha(0.5).restart();
  }
});

linkDistSlider.addEventListener("input", function() {
  document.getElementById("link-dist-value").textContent = this.value;
  if (simulation) {
    simulation.force("link").distance(+this.value);
    simulation.alpha(0.5).restart();
  }
});

opacitySlider.addEventListener("input", function() {
  const opacity = +this.value / 100;
  document.getElementById("opacity-value").textContent = opacity.toFixed(2);
  gEdges.selectAll("line").attr("stroke-opacity", d => {
    // Keep dimmed edges dim
    if (selectedNode) {
      if (d.uKey !== selectedNode && d.dKey !== selectedNode) return 0.02;
    }
    return opacity;
  });
});

function nodeScore(d) {
  if (currentMode === "logit") return d.logitScore || 0;
  if (currentMode === "frequency") return d.freq;
  return d.gradScore;
}

// --- Main rebuild: recompute nodes/edges, restart simulation ---
function rebuildGraph() {
  const selectionMode = selectionModeSelect.value;
  const topN = +topnSlider.value;
  const mode = rankMode.value;
  currentMode = mode;
  const threshPct = +thresholdSlider.value;
  // Exponential mapping: slider 0-100 → threshold 0 to globalMaxAbs
  // Using cube: fine control at low end, coarse at high end
  const threshold = threshPct === 0 ? 0 : globalMaxAbs * Math.pow(threshPct / 100, 3);
  const maxDist = +maxDistSelect.value;  // 0 = all
  const opacity = +opacitySlider.value / 100;
  const chargeStrength = +chargeSlider.value;
  const linkDist = +linkDistSlider.value;

  document.getElementById("topn-value").textContent = topN;
  document.getElementById("threshold-value").textContent = threshold.toFixed(4);
  document.getElementById("opacity-value").textContent = opacity.toFixed(2);

  // Toggle visibility of top-N-only controls
  document.getElementById("topn-group").style.display = selectionMode === "topn" ? "" : "none";
  document.getElementById("rank-group").style.display = selectionMode === "topn" ? "" : "none";

  const hasToken = DATA._tokenNodes && DATA._tokenNodes.length > 0;
  const tokenThreshPct = +tokenThresholdSlider.value;
  const tokenThreshold = (hasToken && tokenThreshPct > 0)
    ? DATA._tokenMaxAbs * Math.pow(tokenThreshPct / 100, 3) : 0;
  if (hasToken) {
    document.getElementById("token-threshold-value").textContent = tokenThreshold.toFixed(4);
  }

  let selected, selectedKeys;
  const edges = [];
  let globalMaxEdge = 0;
  const hasLogit = DATA._logitNodes && DATA._logitNodes.length > 0;

  if (selectionMode === "topn") {
    // --- Top N mode: rank features, take top N, filter edges by threshold ---
    const sorted = [...allFeatures].sort((a, b) => nodeScore(b) - nodeScore(a));
    selected = sorted.slice(0, topN);
    selectedKeys = new Set(selected.map(f => f.key));

    DATA.layer_pairs.forEach(pair => {
      if (maxDist > 0 && (pair.downstream_layer - pair.upstream_layer) > maxDist) return;
      const mat = pair.gradient_matrix;
      pair.upstream_feature_ids.forEach((uid, j) => {
        const uKey = pair.upstream_layer + ":" + uid;
        if (!selectedKeys.has(uKey)) return;
        pair.downstream_feature_ids.forEach((did, i) => {
          const dKey = pair.downstream_layer + ":" + did;
          if (!selectedKeys.has(dKey)) return;
          const val = mat[i][j];
          if (Math.abs(val) >= threshold) {
            edges.push({ uKey, dKey, uLayer: pair.upstream_layer, uId: uid,
                          dLayer: pair.downstream_layer, dId: did, value: val });
            if (Math.abs(val) > globalMaxEdge) globalMaxEdge = Math.abs(val);
          }
        });
      });
    });
    if (hasLogit) {
      DATA._logitEdges.forEach(e => {
        if (!selectedKeys.has(e.uKey)) return;
        if (Math.abs(e.value) >= threshold) {
          edges.push(e);
          if (Math.abs(e.value) > globalMaxEdge) globalMaxEdge = Math.abs(e.value);
        }
      });
    }
    if (hasToken) {
      DATA._tokenEdges.forEach(e => {
        if (!selectedKeys.has(e.dKey)) return;
        if (Math.abs(e.value) >= tokenThreshold) {
          edges.push(e);
          if (Math.abs(e.value) > globalMaxEdge) globalMaxEdge = Math.abs(e.value);
        }
      });
    }
  } else {
    // --- Edge threshold mode: find all edges above threshold, derive nodes ---
    selectedKeys = new Set();

    DATA.layer_pairs.forEach(pair => {
      if (maxDist > 0 && (pair.downstream_layer - pair.upstream_layer) > maxDist) return;
      const mat = pair.gradient_matrix;
      pair.upstream_feature_ids.forEach((uid, j) => {
        const uKey = pair.upstream_layer + ":" + uid;
        pair.downstream_feature_ids.forEach((did, i) => {
          const dKey = pair.downstream_layer + ":" + did;
          const val = mat[i][j];
          if (Math.abs(val) >= threshold) {
            edges.push({ uKey, dKey, uLayer: pair.upstream_layer, uId: uid,
                          dLayer: pair.downstream_layer, dId: did, value: val });
            selectedKeys.add(uKey);
            selectedKeys.add(dKey);
            if (Math.abs(val) > globalMaxEdge) globalMaxEdge = Math.abs(val);
          }
        });
      });
    });
    if (hasLogit) {
      DATA._logitEdges.forEach(e => {
        if (Math.abs(e.value) >= threshold) {
          edges.push(e);
          selectedKeys.add(e.uKey);
          if (Math.abs(e.value) > globalMaxEdge) globalMaxEdge = Math.abs(e.value);
        }
      });
    }
    if (hasToken) {
      DATA._tokenEdges.forEach(e => {
        if (Math.abs(e.value) >= tokenThreshold) {
          edges.push(e);
          selectedKeys.add(e.dKey);
          if (Math.abs(e.value) > globalMaxEdge) globalMaxEdge = Math.abs(e.value);
        }
      });
    }
    // Build selected feature list from edge endpoints
    const featureByKey = {};
    allFeatures.forEach(f => { featureByKey[f.key] = f; });
    selected = [];
    selectedKeys.forEach(key => {
      if (featureByKey[key]) selected.push(featureByKey[key]);
    });
  }
  currentEdges = edges;

  // Build per-layer groupings from selected features
  const layerFeatures = {};
  selected.forEach(f => {
    if (!layerFeatures[f.layer]) layerFeatures[f.layer] = [];
    layerFeatures[f.layer].push(f);
  });
  const activeLayers = Object.keys(layerFeatures).map(Number).sort((a, b) => a - b);
  currentActiveLayers = activeLayers;

  // Stats
  const layerCounts = activeLayers.map(l => `L${l}:${layerFeatures[l].length}`).join(", ");
  document.getElementById("stats").innerHTML =
    `<span class="sl">Features:</span> <span class="sv">${selected.length}</span> across <span class="sv">${activeLayers.length}</span> layers<br>` +
    `<span class="sl">Visible edges:</span> <span class="sv">${edges.length}</span><br>` +
    `<span class="sl">Per layer:</span> <span style="color:#aaa;font-size:11px;">${layerCounts}</span>`;

  // Compute fixed x positions per layer column (token + layers + logit)
  const w = container.clientWidth;
  const h = container.clientHeight;
  const marginX = 60;
  const tokenCols = hasToken ? 1 : 0;
  const totalCols = tokenCols + activeLayers.length + (hasLogit ? 1 : 0);
  const colSpacing = totalCols > 1
    ? (w - 2 * marginX) / (totalCols - 1)
    : 0;

  layerXPositions = {};
  if (hasToken) {
    layerXPositions["token"] = marginX;
  }
  activeLayers.forEach((layerIdx, colIdx) => {
    layerXPositions[layerIdx] = marginX + (tokenCols + colIdx) * colSpacing;
  });
  if (hasLogit) {
    layerXPositions["logit"] = marginX + (tokenCols + activeLayers.length) * colSpacing;
  }

  maxScore = selected.length > 0
    ? Math.max(...selected.map(f => nodeScore(f)))
    : 1;

  // Build simulation node data — preserve positions if keys match
  const oldPosMap = {};
  currentNodes.forEach(n => { oldPosMap[n.key] = { x: n.x, y: n.y }; });

  currentNodes = selected.map(f => {
    const targetX = layerXPositions[f.layer] || w / 2;
    const old = oldPosMap[f.key];
    return {
      ...f,
      targetX,
      fx: targetX,
      x: targetX,
      y: old ? old.y : h / 2 + (Math.random() - 0.5) * h * 0.6,
    };
  });

  // Add token nodes (always visible, leftmost column)
  if (hasToken) {
    const tokenX = layerXPositions["token"];
    DATA._tokenNodes.forEach((tn, idx) => {
      const old = oldPosMap[tn.key];
      currentNodes.push({
        ...tn,
        targetX: tokenX,
        fx: tokenX,
        x: tokenX,
        y: old ? old.y : h / 2 + (idx - (DATA._tokenNodes.length - 1) / 2) * 40,
      });
    });
  }

  // Add logit nodes (always visible, not subject to top-N)
  if (hasLogit) {
    const logitX = layerXPositions["logit"];
    DATA._logitNodes.forEach((ln, idx) => {
      const old = oldPosMap[ln.key];
      currentNodes.push({
        ...ln,
        targetX: logitX,
        fx: logitX,
        x: logitX,
        y: old ? old.y : h / 2 + (idx - (DATA._logitNodes.length - 1) / 2) * 60,
      });
    });
  }

  // Build link data for d3-force (reference by index)
  const nodeIndex = {};
  currentNodes.forEach((n, i) => { nodeIndex[n.key] = i; });
  const links = edges
    .filter(e => nodeIndex[e.uKey] !== undefined && nodeIndex[e.dKey] !== undefined)
    .map(e => ({
      source: nodeIndex[e.uKey],
      target: nodeIndex[e.dKey],
      value: e.value
    }));

  // Stop old simulation
  if (simulation) simulation.stop();

  // Create force simulation
  simulation = d3.forceSimulation(currentNodes)
    .force("link", d3.forceLink(links).distance(linkDist).strength(0.1))
    .force("charge", d3.forceManyBody().strength(chargeStrength))
    .force("collide", d3.forceCollide().radius(d => {
      if (d.isTokenNode) return 18;
      if (d.isLogitNode) return 16;
      return 6 + 8 * (nodeScore(d) / (maxScore || 1));
    }))
    .force("y", d3.forceY(h / 2).strength(0.03))
    .alphaDecay(0.01)
    .on("tick", ticked);

  // --- Draw layer guide lines ---
  gLabels.selectAll("*").remove();
  if (hasToken) {
    const x = layerXPositions["token"];
    gLabels.append("line")
      .attr("class", "layer-line")
      .attr("x1", x).attr("x2", x)
      .attr("y1", -2000).attr("y2", 4000);
    gLabels.append("text")
      .attr("class", "layer-label")
      .attr("x", x).attr("y", 20)
      .attr("text-anchor", "middle")
      .text("Input");
  }
  activeLayers.forEach(layerIdx => {
    const x = layerXPositions[layerIdx];
    gLabels.append("line")
      .attr("class", "layer-line")
      .attr("x1", x).attr("x2", x)
      .attr("y1", -2000).attr("y2", 4000);
    gLabels.append("text")
      .attr("class", "layer-label")
      .attr("x", x).attr("y", 20)
      .attr("text-anchor", "middle")
      .text(`L${layerIdx}`);
  });
  if (hasLogit) {
    const x = layerXPositions["logit"];
    gLabels.append("line")
      .attr("class", "layer-line")
      .attr("x1", x).attr("x2", x)
      .attr("y1", -2000).attr("y2", 4000);
    gLabels.append("text")
      .attr("class", "layer-label")
      .attr("x", x).attr("y", 20)
      .attr("text-anchor", "middle")
      .text("Logit");
  }

  // --- Draw edges ---
  const edgeSelection = gEdges.selectAll("line")
    .data(edges, d => d.uKey + "-" + d.dKey);
  edgeSelection.exit().remove();
  const edgeEnter = edgeSelection.enter().append("line").attr("class", "edge");
  const edgeMerge = edgeEnter.merge(edgeSelection)
    .attr("stroke", d => d.value >= 0 ? "#4dabf7" : "#ff6b6b")
    .attr("stroke-width", d => 0.4 + 3 * (Math.abs(d.value) / (globalMaxEdge || 1)))
    .attr("stroke-opacity", opacity)
    .attr("data-ukey", d => d.uKey)
    .attr("data-dkey", d => d.dKey)
    .on("mouseover", function(event, d) {
      if (pinnedTooltip) return;
      tooltip.style.display = "block";
      const fromLabel = d.isTokenEdge
        ? `Token "${d.uLabel}"`
        : `L${d.uLayer} #${d.uId}`;
      const toLabel = d.isLogitEdge
        ? `Logit "${d.dLabel}"`
        : `L${d.dLayer} #${d.dId}`;
      tooltip.innerHTML =
        `<span class="tl">From:</span> <span class="tv">${fromLabel}</span><br>` +
        `<span class="tl">To:</span> <span class="tv">${toLabel}</span><br>` +
        `<span class="tl">Gradient:</span> <span class="tv">${d.value.toFixed(6)}</span>`;
      tooltip.style.left = (event.clientX + 12) + "px";
      tooltip.style.top = (event.clientY - 10) + "px";
    })
    .on("mousemove", function(event) {
      if (pinnedTooltip) return;
      tooltip.style.left = (event.clientX + 12) + "px";
      tooltip.style.top = (event.clientY - 10) + "px";
    })
    .on("mouseout", () => {
      if (pinnedTooltip) return;
      tooltip.style.display = "none";
    });

  // --- Draw nodes ---
  const nodeSelection = gNodes.selectAll("g.node")
    .data(currentNodes, d => d.key);
  nodeSelection.exit().remove();

  const nodeEnter = nodeSelection.enter().append("g").attr("class", "node");

  // Token nodes get squares, others get circles
  nodeEnter.each(function(d) {
    const g = d3.select(this);
    if (d.isTokenNode) {
      g.append("rect")
        .attr("x", -12).attr("y", -12)
        .attr("width", 24).attr("height", 24)
        .attr("rx", 3).attr("ry", 3)
        .attr("fill", "#51cf66");
    } else {
      g.append("circle")
        .attr("r", d.isLogitNode ? 12 : 4 + 6 * (nodeScore(d) / (maxScore || 1)))
        .attr("fill", () => {
          if (d.isLogitNode) return "#e94560";
          const t = activeLayers.length > 1
            ? activeLayers.indexOf(d.layer) / (activeLayers.length - 1) : 0.5;
          return d3.interpolateRdYlBu(1 - t);
        });
    }
  });

  nodeEnter.append("text")
    .attr("dy", d => {
      if (d.isTokenNode) return -16;
      if (d.isLogitNode) return -16;
      return -(6 + 6 * (nodeScore(d) / (maxScore || 1))) - 2;
    })
    .attr("text-anchor", "middle")
    .attr("font-weight", d => (d.isLogitNode || d.isTokenNode) ? "bold" : "normal")
    .attr("font-size", d => (d.isLogitNode || d.isTokenNode) ? "12px" : "9px")
    .attr("fill", d => d.isTokenNode ? "#51cf66" : d.isLogitNode ? "#e94560" : "#ccc")
    .text(d => d.isTokenNode ? d.label : d.isLogitNode ? d.label : d.id);

  const nodeMerge = nodeEnter.merge(nodeSelection);

  // Update sizes and colors on merge (for existing nodes that changed)
  nodeMerge.select("circle")
    .attr("r", d => {
      if (d.isLogitNode) return 12;
      return 4 + 6 * (nodeScore(d) / (maxScore || 1));
    })
    .attr("fill", d => {
      if (d.isLogitNode) return "#e94560";
      const t = activeLayers.length > 1
        ? activeLayers.indexOf(d.layer) / (activeLayers.length - 1) : 0.5;
      return d3.interpolateRdYlBu(1 - t);
    });

  nodeMerge
    .on("mouseover", function(event, d) {
      if (pinnedTooltip) return;  // don't override pinned tooltip
      tooltip.style.display = "block";
      if (d.isTokenNode) {
        hoveredNodeKey = null;
        tooltip.innerHTML =
          `<span class="tl">Input Token</span> <span class="tv">"${d.label}"</span><br>` +
          `<span class="tl">Position:</span> <span class="tv">${d.tIdx}</span><br>` +
          `<span class="tl">Token ID:</span> <span class="tv">${d.id}</span><br>` +
          `<span class="tl">Total |gradient|:</span> <span class="tv">${d.gradScore.toFixed(4)}</span>`;
      } else if (d.isLogitNode) {
        hoveredNodeKey = null;
        tooltip.innerHTML =
          `<span class="tl">Logit Node</span> <span class="tv">${d.label}</span><br>` +
          `<span class="tl">Token ID:</span> <span class="tv">${d.id}</span><br>` +
          `<span class="tl">Total |gradient|:</span> <span class="tv">${d.gradScore.toFixed(4)}</span>`;
      } else {
        hoveredNodeKey = d.layer + ":" + d.id;
        const cached = descCache[hoveredNodeKey];
        tooltip.innerHTML = buildNodeTooltipHTML(d, cached);
        if (!(hoveredNodeKey in descCache)) fetchAndShowDescription(d);
      }
      tooltip.style.left = (event.clientX + 12) + "px";
      tooltip.style.top = (event.clientY - 10) + "px";
    })
    .on("mousemove", function(event) {
      if (pinnedTooltip) return;
      tooltip.style.left = (event.clientX + 12) + "px";
      tooltip.style.top = (event.clientY - 10) + "px";
    })
    .on("mouseout", () => {
      if (pinnedTooltip) return;  // keep pinned tooltip visible
      hoveredNodeKey = null;
      tooltip.style.display = "none";
    })
    .on("click", function(event, d) {
      event.stopPropagation();
      // Toggle pinned tooltip
      if (pinnedTooltip === d.key) {
        pinnedTooltip = null;
        tooltip.style.display = "none";
      } else {
        pinnedTooltip = d.key;
        // Show tooltip content for clicked node
        if (d.isTokenNode) {
          hoveredNodeKey = null;
          tooltip.innerHTML =
            `<span class="tl">Input Token</span> <span class="tv">"${d.label}"</span><br>` +
            `<span class="tl">Position:</span> <span class="tv">${d.tIdx}</span><br>` +
            `<span class="tl">Token ID:</span> <span class="tv">${d.id}</span><br>` +
            `<span class="tl">Total |gradient|:</span> <span class="tv">${d.gradScore.toFixed(4)}</span>`;
        } else if (d.isLogitNode) {
          hoveredNodeKey = null;
          tooltip.innerHTML =
            `<span class="tl">Logit Node</span> <span class="tv">${d.label}</span><br>` +
            `<span class="tl">Token ID:</span> <span class="tv">${d.id}</span><br>` +
            `<span class="tl">Total |gradient|:</span> <span class="tv">${d.gradScore.toFixed(4)}</span>`;
        } else {
          hoveredNodeKey = d.layer + ":" + d.id;
          const cached = descCache[hoveredNodeKey];
          tooltip.innerHTML = buildNodeTooltipHTML(d, cached);
          if (!(hoveredNodeKey in descCache)) fetchAndShowDescription(d);
        }
        tooltip.style.display = "block";
        tooltip.style.left = (event.clientX + 12) + "px";
        tooltip.style.top = (event.clientY - 10) + "px";
      }
      selectedNode = selectedNode === d.key ? null : d.key;
      applySelection();
    })
    .call(d3.drag()
      .container(function() { return gRoot.node(); })
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended));

  applySelection();
}

// --- Simulation tick: update positions ---
function ticked() {
  const opacity = +opacitySlider.value / 100;

  // x is pinned via fx on each node — no manual override needed

  // Build a quick lookup from key -> node for edge positioning
  const nodeMap = {};
  currentNodes.forEach(n => { nodeMap[n.key] = n; });

  gEdges.selectAll("line")
    .attr("x1", d => nodeMap[d.uKey] ? nodeMap[d.uKey].x : 0)
    .attr("y1", d => nodeMap[d.uKey] ? nodeMap[d.uKey].y : 0)
    .attr("x2", d => nodeMap[d.dKey] ? nodeMap[d.dKey].x : 0)
    .attr("y2", d => nodeMap[d.dKey] ? nodeMap[d.dKey].y : 0);

  gNodes.selectAll("g.node")
    .attr("transform", d => `translate(${d.x},${d.y})`);
}

// --- Drag handlers ---
function dragstarted(event, d) {
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}

function dragged(event, d) {
  d.fx = event.x;
  d.fy = event.y;
}

function dragended(event, d) {
  if (!event.active) simulation.alphaTarget(0);
  d.fx = d.targetX;  // snap x back to layer column
  d.fy = null;
}

// --- Selection highlighting ---
function applySelection() {
  const opacity = +opacitySlider.value / 100;
  if (!selectedNode) {
    d3.selectAll(".node").classed("dimmed", false);
    gEdges.selectAll("line").attr("stroke-opacity", opacity);
    return;
  }

  const connectedKeys = new Set();
  connectedKeys.add(selectedNode);

  gEdges.selectAll("line").attr("stroke-opacity", function(d) {
    if (d.uKey === selectedNode || d.dKey === selectedNode) {
      connectedKeys.add(d.uKey);
      connectedKeys.add(d.dKey);
      return opacity;
    }
    return 0.02;
  });

  d3.selectAll(".node").classed("dimmed", d => !connectedKeys.has(d.key));
}

// Click on background to deselect
svgEl.on("click", function(event) {
  if (event.target === svgEl.node()) {
    selectedNode = null;
    pinnedTooltip = null;
    tooltip.style.display = "none";
    applySelection();
  }
});

// ---------------------------------------------------------------------------
// 3D Visualization
// ---------------------------------------------------------------------------
let viewMode = '2d';
let scene3d, camera3d, renderer3d, controls3d;
let node3dMeshes = [];
let edge3dLines = [];
let raycaster3d, mouse3d;
let threeDInitialized = false;

function init3D() {
  if (threeDInitialized) return;
  threeDInitialized = true;

  const container3d = document.getElementById('graph-3d');
  const w = container3d.clientWidth || container.clientWidth;
  const h = container3d.clientHeight || container.clientHeight;

  scene3d = new THREE.Scene();
  scene3d.background = new THREE.Color(0x1a1a2e);

  camera3d = new THREE.PerspectiveCamera(75, w / h, 0.1, 10000);
  camera3d.position.set(0, 200, 800);

  renderer3d = new THREE.WebGLRenderer({ antialias: true });
  renderer3d.setSize(w, h);
  container3d.appendChild(renderer3d.domElement);

  const ambientLight = new THREE.AmbientLight(0x404040, 2);
  scene3d.add(ambientLight);
  const dirLight = new THREE.DirectionalLight(0xffffff, 1);
  dirLight.position.set(1, 1, 1);
  scene3d.add(dirLight);

  controls3d = new THREE.OrbitControls(camera3d, renderer3d.domElement);
  controls3d.enableDamping = true;
  controls3d.dampingFactor = 0.05;
  controls3d.autoRotate = true;
  controls3d.autoRotateSpeed = 0.5;

  raycaster3d = new THREE.Raycaster();
  mouse3d = new THREE.Vector2();

  renderer3d.domElement.addEventListener('mousemove', function(event) {
    const rect = renderer3d.domElement.getBoundingClientRect();
    mouse3d.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse3d.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  });

  renderer3d.domElement.addEventListener('click', function(event) {
    raycaster3d.setFromCamera(mouse3d, camera3d);
    const intersects = raycaster3d.intersectObjects(node3dMeshes);
    if (intersects.length > 0) {
      const d = intersects[0].object.userData.node;
      // Toggle selection
      selectedNode = selectedNode === d.key ? null : d.key;
      pinnedTooltip = selectedNode ? d.key : null;
      if (selectedNode) {
        tooltip.style.display = "block";
        if (d.isTokenNode) {
          tooltip.innerHTML =
            '<span class="tl">Input Token</span> <span class="tv">"' + d.label + '"</span><br>' +
            '<span class="tl">Position:</span> <span class="tv">' + d.tIdx + '</span><br>' +
            '<span class="tl">Token ID:</span> <span class="tv">' + d.id + '</span><br>' +
            '<span class="tl">Total |gradient|:</span> <span class="tv">' + d.gradScore.toFixed(4) + '</span>';
        } else if (d.isLogitNode) {
          tooltip.innerHTML =
            '<span class="tl">Logit Node</span> <span class="tv">' + d.label + '</span><br>' +
            '<span class="tl">Token ID:</span> <span class="tv">' + d.id + '</span><br>' +
            '<span class="tl">Total |gradient|:</span> <span class="tv">' + d.gradScore.toFixed(4) + '</span>';
        } else {
          hoveredNodeKey = d.layer + ":" + d.id;
          const cached = descCache[hoveredNodeKey];
          tooltip.innerHTML = buildNodeTooltipHTML(d, cached);
          if (!(hoveredNodeKey in descCache)) fetchAndShowDescription(d);
        }
        tooltip.style.left = (event.clientX + 12) + "px";
        tooltip.style.top = (event.clientY - 10) + "px";
      } else {
        tooltip.style.display = "none";
      }
      update3DHighlighting();
    } else {
      selectedNode = null;
      pinnedTooltip = null;
      tooltip.style.display = "none";
      update3DHighlighting();
    }
  });

  animate3D();

  window.addEventListener("resize", function() {
    if (viewMode !== '3d') return;
    const w2 = container.clientWidth;
    const h2 = container.clientHeight;
    camera3d.aspect = w2 / h2;
    camera3d.updateProjectionMatrix();
    renderer3d.setSize(w2, h2);
  });
}

function animate3D() {
  requestAnimationFrame(animate3D);
  if (viewMode === '3d' && controls3d) {
    controls3d.update();
    renderer3d.render(scene3d, camera3d);
  }
}

function rebuild3DGraph() {
  if (!threeDInitialized) return;

  // Clear old meshes/lines
  node3dMeshes.forEach(m => scene3d.remove(m));
  edge3dLines.forEach(l => scene3d.remove(l));
  node3dMeshes = [];
  edge3dLines = [];
  const topN = +topnSlider.value;
  const mode = rankMode.value;
  currentMode = mode;
  const threshPct = +thresholdSlider.value;
  const threshold3d = threshPct === 0 ? 0 : globalMaxAbs * Math.pow(threshPct / 100, 3);
  const maxDist = +maxDistSelect.value;

  // Select features (same logic as 2D)
  const sorted = [...allFeatures].sort((a, b) => nodeScore(b) - nodeScore(a));
  const selected = sorted.slice(0, topN);
  const selectedKeys = new Set(selected.map(f => f.key));

  const layerFeatures = {};
  selected.forEach(f => {
    if (!layerFeatures[f.layer]) layerFeatures[f.layer] = [];
    layerFeatures[f.layer].push(f);
  });
  const activeLayers = Object.keys(layerFeatures).map(Number).sort((a, b) => a - b);
  const ms = selected.length > 0 ? Math.max(...selected.map(f => nodeScore(f))) : 1;

  // Build edges
  const edges3d = [];
  const hasLogit = DATA._logitNodes && DATA._logitNodes.length > 0;
  DATA.layer_pairs.forEach(pair => {
    if (maxDist > 0 && (pair.downstream_layer - pair.upstream_layer) > maxDist) return;
    const mat = pair.gradient_matrix;
    pair.upstream_feature_ids.forEach((uid, j) => {
      const uKey = pair.upstream_layer + ":" + uid;
      if (!selectedKeys.has(uKey)) return;
      pair.downstream_feature_ids.forEach((did, i) => {
        const dKey = pair.downstream_layer + ":" + did;
        if (!selectedKeys.has(dKey)) return;
        const val = mat[i][j];
        if (Math.abs(val) >= threshold3d) {
          edges3d.push({ source: uKey, target: dKey, value: val });
        }
      });
    });
  });
  if (hasLogit) {
    DATA._logitEdges.forEach(e => {
      if (!selectedKeys.has(e.uKey)) return;
      if (Math.abs(e.value) >= threshold3d) {
        edges3d.push({ source: e.uKey, target: e.dKey, value: e.value, isLogitEdge: true });
      }
    });
  }
  const hasToken3d = DATA._tokenNodes && DATA._tokenNodes.length > 0;
  if (hasToken3d) {
    const tokThreshPct = +tokenThresholdSlider.value;
    const tokThreshold3d = tokThreshPct === 0 ? 0 : DATA._tokenMaxAbs * Math.pow(tokThreshPct / 100, 3);
    DATA._tokenEdges.forEach(e => {
      if (!selectedKeys.has(e.dKey)) return;
      if (Math.abs(e.value) >= tokThreshold3d) {
        edges3d.push({ source: e.uKey, target: e.dKey, value: e.value, isTokenEdge: true });
      }
    });
  }

  // Create node data — initial random Y/Z, fixed X per layer
  const layerSpacing = 60;
  const tokenOffset3d = hasToken3d ? 1 : 0;
  const nodeData = [];

  // Token nodes at leftmost position
  if (hasToken3d) {
    DATA._tokenNodes.forEach((tn, idx) => {
      selectedKeys.add(tn.key);
      nodeData.push({
        key: tn.key, layer: "token", id: tn.id, label: tn.label, tIdx: tn.tIdx,
        gradScore: tn.gradScore, freq: 0, logitScore: 0,
        isTokenNode: true,
        px: (-1 - (activeLayers.length - 1) / 2) * layerSpacing,
        py: (idx - (DATA._tokenNodes.length - 1) / 2) * 20,
        pz: 0,
      });
    });
  }

  selected.forEach(f => {
    nodeData.push({
      key: f.key, layer: f.layer, id: f.id,
      gradScore: f.gradScore, freq: f.freq, logitScore: f.logitScore || 0,
      isLogitNode: false,
      px: (activeLayers.indexOf(f.layer) - (activeLayers.length - 1) / 2) * layerSpacing,
      py: (Math.random() - 0.5) * 100,
      pz: (Math.random() - 0.5) * 100,
    });
  });

  if (hasLogit) {
    DATA._logitNodes.forEach((ln, idx) => {
      selectedKeys.add(ln.key);
      nodeData.push({
        key: ln.key, layer: "logit", id: ln.id, label: ln.label,
        gradScore: ln.gradScore, freq: 0, logitScore: 0,
        isLogitNode: true,
        px: ((activeLayers.length) - (activeLayers.length - 1) / 2) * layerSpacing,
        py: (idx - (DATA._logitNodes.length - 1) / 2) * 20,
        pz: 0,
      });
    });
  }

  // Simple 3D repulsion simulation (Y/Z only, X stays fixed per layer)
  // Run synchronously — fast enough for ~30-200 nodes
  const repStrength = 500;
  const minDist = 5;
  for (let iter = 0; iter < 200; iter++) {
    for (let i = 0; i < nodeData.length; i++) {
      const a = nodeData[i];
      if (a.isLogitNode || a.isTokenNode) continue;
      let fy = 0, fz = 0;
      for (let j = 0; j < nodeData.length; j++) {
        if (i === j) continue;
        const b = nodeData[j];
        const dy = a.py - b.py;
        const dz = a.pz - b.pz;
        const dx = a.px - b.px;
        const dist2 = dx * dx + dy * dy + dz * dz;
        const dist = Math.sqrt(dist2) || minDist;
        // Repulsion — stronger for same-layer nodes
        const strength = a.layer === b.layer ? repStrength : repStrength * 0.3;
        const f = strength / (dist2 || minDist * minDist);
        fy += (dy / dist) * f;
        fz += (dz / dist) * f;
      }
      // Weak centering to prevent drift
      fy -= a.py * 0.01;
      fz -= a.pz * 0.01;
      a.py += fy * 0.5;
      a.pz += fz * 0.5;
    }
  }
  const nodeMap3d = {};
  nodeData.forEach((n, i) => { nodeMap3d[n.key] = i; });

  // Create Three.js node meshes at their positions
  nodeData.forEach(n => {
    let geometry;
    if (n.isTokenNode) {
      geometry = new THREE.BoxGeometry(8, 8, 8);
    } else {
      const r = n.isLogitNode ? 6 : 2 + 4 * (nodeScore(n) / (ms || 1));
      geometry = new THREE.SphereGeometry(r, 16, 16);
    }
    let color;
    if (n.isTokenNode) {
      color = new THREE.Color(0x51cf66);
    } else if (n.isLogitNode) {
      color = new THREE.Color(0xe94560);
    } else {
      const t = activeLayers.length > 1
        ? activeLayers.indexOf(n.layer) / (activeLayers.length - 1) : 0.5;
      color = new THREE.Color(d3.interpolateRdYlBu(1 - t));
    }
    const material = new THREE.MeshPhongMaterial({
      color: color,
      emissive: new THREE.Color(0x000000),
      emissiveIntensity: 0,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(n.px, n.py, n.pz);
    mesh.userData.node = n;
    scene3d.add(mesh);
    node3dMeshes.push(mesh);
  });

  // Create Three.js edge lines
  edges3d.forEach(e => {
    const si = nodeMap3d[e.source];
    const ti = nodeMap3d[e.target];
    if (si === undefined || ti === undefined) return;
    const s = nodeData[si];
    const t = nodeData[ti];
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array([s.px, s.py, s.pz, t.px, t.py, t.pz]);
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const edgeColor = e.value >= 0 ? 0x4dabf7 : 0xff6b6b;
    const material = new THREE.LineBasicMaterial({
      color: edgeColor,
      opacity: 0.3,
      transparent: true,
    });
    const line = new THREE.Line(geometry, material);
    line.userData.link = e;
    scene3d.add(line);
    edge3dLines.push(line);
  });

  update3DHighlighting();
}

function update3DHighlighting() {
  const connectedKeys = new Set();
  if (selectedNode) {
    connectedKeys.add(selectedNode);
    edge3dLines.forEach(l => {
      const link = l.userData.link;
      const sKey = link.source.key || link.source;
      const tKey = link.target.key || link.target;
      if (sKey === selectedNode || tKey === selectedNode) {
        connectedKeys.add(sKey);
        connectedKeys.add(tKey);
      }
    });
  }

  node3dMeshes.forEach(mesh => {
    const n = mesh.userData.node;
    if (!selectedNode) {
      mesh.material.opacity = 1;
      mesh.material.transparent = false;
      mesh.material.emissiveIntensity = 0;
      mesh.scale.set(1, 1, 1);
    } else if (n.key === selectedNode) {
      mesh.material.emissive = new THREE.Color(0xff4444);
      mesh.material.emissiveIntensity = 0.6;
      mesh.scale.set(1.5, 1.5, 1.5);
      mesh.material.opacity = 1;
      mesh.material.transparent = false;
    } else if (connectedKeys.has(n.key)) {
      mesh.material.emissive = new THREE.Color(0xffa500);
      mesh.material.emissiveIntensity = 0.3;
      mesh.scale.set(1.2, 1.2, 1.2);
      mesh.material.opacity = 1;
      mesh.material.transparent = false;
    } else {
      mesh.material.emissive = new THREE.Color(0x000000);
      mesh.material.emissiveIntensity = 0;
      mesh.material.opacity = 0.15;
      mesh.material.transparent = true;
      mesh.scale.set(0.7, 0.7, 0.7);
    }
  });

  edge3dLines.forEach(l => {
    const link = l.userData.link;
    const sKey = link.source.key || link.source;
    const tKey = link.target.key || link.target;
    if (!selectedNode) {
      l.material.opacity = 0.4;
    } else if (sKey === selectedNode || tKey === selectedNode) {
      l.material.opacity = 0.8;
    } else {
      l.material.opacity = 0.03;
    }
  });
}

// Toggle 2D/3D
document.getElementById('view-mode-btn').addEventListener('click', function() {
  if (viewMode === '2d') {
    viewMode = '3d';
    this.textContent = 'Switch to 2D View';
    document.getElementById('graph').style.display = 'none';
    document.getElementById('graph-3d').style.display = 'block';
    init3D();
    rebuild3DGraph();
  } else {
    viewMode = '2d';
    this.textContent = 'Switch to 3D View';
    document.getElementById('graph').style.display = 'block';
    document.getElementById('graph-3d').style.display = 'none';
    selectedNode = null;
    pinnedTooltip = null;
    tooltip.style.display = "none";
    rebuildGraph();
  }
});

// Hook controls to also rebuild 3D when changed
const orig_rebuildGraph = rebuildGraph;
rebuildGraph = function() {
  orig_rebuildGraph();
  if (viewMode === '3d') rebuild3DGraph();
};

// --- Export graph JSON for LLM ---
document.getElementById("export-graph-btn").addEventListener("click", exportGraphJSON);

async function exportGraphJSON() {
  const btn = document.getElementById("export-graph-btn");
  const progressEl = document.getElementById("export-progress");
  btn.disabled = true;
  btn.style.opacity = "0.5";
  progressEl.style.display = "block";
  progressEl.textContent = "Collecting nodes...";

  // Collect visible nodes and edges
  const featureNodes = currentNodes.filter(n => !n.isLogitNode);
  const logitNodesVisible = currentNodes.filter(n => n.isLogitNode);

  // Fetch descriptions for nodes on supported layers (skip already cached)
  const toFetch = featureNodes.filter(n =>
    npHasLayer(n.layer) && !((n.layer + ":" + n.id) in descCache)
  );

  if (toFetch.length > 0) {
    for (let i = 0; i < toFetch.length; i++) {
      const n = toFetch[i];
      const key = n.layer + ":" + n.id;
      progressEl.textContent = `Fetching descriptions... ${i + 1}/${toFetch.length}`;
      try {
        const url = npApiUrl(n.layer, n.id);
        const resp = await fetch(url);
        if (resp.ok) {
          const data = await resp.json();
          descCache[key] = (data && data.explanations && data.explanations.length > 0)
            ? (data.explanations[0].description || null) : null;
        } else {
          descCache[key] = null;
        }
      } catch (e) {
        descCache[key] = null;
      }
      // Small delay to avoid rate limiting
      if (i < toFetch.length - 1) await new Promise(r => setTimeout(r, 50));
    }
  }

  progressEl.textContent = "Building export...";

  // Build nodes grouped by layer
  const nodesByLayer = {};
  featureNodes.forEach(n => {
    const lk = "L" + n.layer;
    if (!nodesByLayer[lk]) nodesByLayer[lk] = [];
    const key = n.layer + ":" + n.id;
    const entry = { id: n.id };
    if (descCache[key]) entry.desc = descCache[key];
    entry.score = +nodeScore(n).toFixed(4);
    nodesByLayer[lk].push(entry);
  });
  // Sort each layer's features by score descending
  Object.values(nodesByLayer).forEach(arr => arr.sort((a, b) => b.score - a.score));

  if (logitNodesVisible.length > 0) {
    nodesByLayer["Logit"] = logitNodesVisible.map(n => ({
      token: n.label,
      token_id: n.id,
      total_grad: +n.gradScore.toFixed(4),
    }));
  }

  const tokenNodesVisible = currentNodes.filter(n => n.isTokenNode);
  if (tokenNodesVisible.length > 0) {
    nodesByLayer["Input"] = tokenNodesVisible.map(n => ({
      token: n.label,
      token_id: n.id,
      position: n.tIdx,
      total_grad: +n.gradScore.toFixed(4),
    }));
  }

  // Build edges sorted by |gradient| descending
  const edges = currentEdges.map(e => ({
    src: e.isTokenEdge ? ("Input:" + e.uLabel) : ("L" + e.uLayer + ":" + e.uId),
    dst: e.isLogitEdge ? ("Logit:" + e.dLabel) : ("L" + e.dLayer + ":" + e.dId),
    grad: +e.value.toFixed(4),
  }));
  edges.sort((a, b) => Math.abs(b.grad) - Math.abs(a.grad));

  // Build context paragraph for LLM interpretation
  const threshPct = +thresholdSlider.value;
  const threshold = threshPct === 0 ? 0 : globalMaxAbs * Math.pow(threshPct / 100, 3);
  const maxDist = +maxDistSelect.value;
  const layerList = Object.keys(nodesByLayer).filter(k => k !== "Logit").sort((a, b) => parseInt(a.slice(1)) - parseInt(b.slice(1)));
  const npLayerNote = npConfig
    ? (npConfig.layers ? " Neuronpedia auto-interp descriptions are only available for layers " + [...npConfig.layers].join(", ") + "; other layers lack descriptions." : "")
    : " Neuronpedia descriptions are not available for this model.";

  const context = "This is a feature circuit extracted from the transformer language model " + (DATA.metadata.model || "unknown") + ". "
    + "Each node represents a feature learned by a Sparse Autoencoder (SAE) trained on the model's residual stream at a specific layer. "
    + "SAE features are interpretable directions in activation space; their 'desc' field (when available) is an auto-generated natural language description of what inputs cause that feature to activate. "
    + "Each edge represents the gradient of a downstream SAE feature's activation with respect to an upstream SAE feature's activation, computed across the prompts listed in metadata. "
    + "A positive gradient (excitatory) means increasing the upstream feature's activation tends to increase the downstream feature's activation; a negative gradient (inhibitory) means the opposite. "
    + "Larger absolute gradient values indicate stronger functional connections. "
    + "The 'score' on each node is the ranking metric used to select which features to display (total gradient influence, logit gradient, or activation frequency depending on the mode)."
    + "\n\n"
    + "The prompts used to compute this circuit are listed in metadata.prompts. "
    + "Features that are consistently active across these prompts and strongly connected to each other likely form a functional circuit relevant to the shared task structure of those prompts. "
    + "Logit nodes (if present) represent specific output tokens; edges to logit nodes show how strongly each feature influences the probability of that token being predicted. "
    + "Input token nodes (if present) represent the input tokens; edges from input tokens to features show how much each token position contributes to each feature's activation via embedding perturbation gradients."
    + npLayerNote
    + "\n\nFiltering criteria applied to this export: "
    + "Features shown are the top " + topnSlider.value + " per layer "
    + "ranked by " + currentMode + ". "
    + "Only edges with |gradient| >= " + threshold.toFixed(6) + " are included. "
    + "Max layer distance: " + (maxDist > 0 ? "d <= " + maxDist : "all pairs") + ". "
    + (DATA.metadata.n_features_per_layer
      ? "The full circuit was computed with " + DATA.metadata.n_features_per_layer + " features per layer"
        + (DATA.metadata.feature_selection ? " selected by " + DATA.metadata.feature_selection : "") + "."
      : "");

  const exportData = {
    metadata: DATA.metadata,
    context: context,
    nodes_by_layer: nodesByLayer,
    edges: edges,
  };

  // Trigger download
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
  const dlUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = dlUrl;
  const ts = new Date().toISOString().slice(0, 19).replace(/[T:]/g, "_");
  a.download = "circuit_export_" + ts + ".json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(dlUrl);

  progressEl.textContent = "Done!";
  btn.disabled = false;
  btn.style.opacity = "1";
  setTimeout(() => { progressEl.style.display = "none"; }, 2000);
}

// Initial load
initData(DATA);
</script>
</body>
</html>
"""
