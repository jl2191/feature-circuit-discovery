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

    data = {
        "metadata": {
            "model": metadata.get("model", "google/gemma-3-1b-pt"),
            "sae_repo": metadata.get("sae_repo", "google/gemma-scope-2-1b-pt"),
            "sae_width": metadata.get("sae_width", 16384),
            "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
            "prompts": metadata.get("prompts", []),
            "activation_threshold": metadata.get("activation_threshold", 0.9),
            "device": metadata.get("device", "cpu"),
            "dtype": metadata.get("dtype", "bfloat16"),
        },
        "layers": layers,
        "layer_pairs": pairs_data,
    }

    if logit_nodes_data:
        data["logit_nodes"] = logit_nodes_data

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
    <label>Global Top N Features: <span id="topn-value"></span></label>
    <input type="range" id="topn-slider" min="5" max="200" value="30">
  </div>

  <div class="control-group">
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
        key: "logit:" + ln.token_ids[tIdx],
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
            dKey: "logit:" + ln.token_ids[tIdx],
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

// --- Neuronpedia description cache & tooltip integration ---
const descCache = {};
let hoveredNodeKey = null;  // track which node tooltip is showing

function buildNodeTooltipHTML(d, description) {
  const score = nodeScore(d);
  const scoreLabel = currentMode === "logit" ? "Logit |grad|" : currentMode === "gradient" ? "Total |grad|" : "Activation freq";
  const npUrl = `https://www.neuronpedia.org/gemma-3-1b/${d.layer}-gemmascope-2-res-16k/${d.id}`;
  let html =
    `<span class="tl">Layer ${d.layer} Feature</span> <span class="tv">#${d.id}</span><br>` +
    `<span class="tl">${scoreLabel}:</span> <span class="tv">${score.toFixed(4)}</span>`;
  if (description === undefined) {
    html += `<br><span class="tl" style="font-style:italic;">Loading description...</span>`;
  } else if (description) {
    html += `<br><span class="tl">Description:</span> <span style="color:#a0c4ff;font-style:italic;">"${description}"</span>`;
  }
  html += `<br><a href="${npUrl}" target="_blank" style="color:#4dabf7;text-decoration:none;font-size:11px;">Neuronpedia &#8599;</a>`;
  return html;
}

function fetchAndShowDescription(d) {
  const key = d.layer + ":" + d.id;
  if (key in descCache) return;  // already cached, tooltip was built with it
  descCache[key] = undefined;  // mark as loading
  const url = `https://www.neuronpedia.org/api/feature/gemma-3-1b/${d.layer}-gemmascope-2-res-16k/${d.id}`;
  fetch(url)
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
topnSlider.addEventListener("input", () => { selectedNode = null; rebuildGraph(); });
rankMode.addEventListener("change", () => { selectedNode = null; rebuildGraph(); });
maxDistSelect.addEventListener("change", () => { selectedNode = null; rebuildGraph(); });
thresholdSlider.addEventListener("input", rebuildGraph);

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

  // Rank and select top N features globally
  const sorted = [...allFeatures].sort((a, b) => nodeScore(b) - nodeScore(a));
  const selected = sorted.slice(0, topN);
  const selectedKeys = new Set(selected.map(f => f.key));

  const layerFeatures = {};
  selected.forEach(f => {
    if (!layerFeatures[f.layer]) layerFeatures[f.layer] = [];
    layerFeatures[f.layer].push(f);
  });
  const activeLayers = Object.keys(layerFeatures).map(Number).sort((a, b) => a - b);
  currentActiveLayers = activeLayers;

  // Build edges
  const edges = [];
  let globalMaxEdge = 0;
  DATA.layer_pairs.forEach(pair => {
    // Layer distance filter
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
  // Add logit edges (subject to threshold filter)
  const hasLogit = DATA._logitNodes && DATA._logitNodes.length > 0;
  if (hasLogit) {
    DATA._logitEdges.forEach(e => {
      if (!selectedKeys.has(e.uKey)) return;
      if (Math.abs(e.value) >= threshold) {
        edges.push(e);
        if (Math.abs(e.value) > globalMaxEdge) globalMaxEdge = Math.abs(e.value);
      }
    });
  }
  currentEdges = edges;

  // Stats
  const layerCounts = activeLayers.map(l => `L${l}:${layerFeatures[l].length}`).join(", ");
  document.getElementById("stats").innerHTML =
    `<span class="sl">Features:</span> <span class="sv">${selected.length}</span> across <span class="sv">${activeLayers.length}</span> layers<br>` +
    `<span class="sl">Visible edges:</span> <span class="sv">${edges.length}</span><br>` +
    `<span class="sl">Per layer:</span> <span style="color:#aaa;font-size:11px;">${layerCounts}</span>`;

  // Compute fixed x positions per layer column (+ logit column)
  const w = container.clientWidth;
  const h = container.clientHeight;
  const marginX = 60;
  const totalCols = activeLayers.length + (hasLogit ? 1 : 0);
  const colSpacing = totalCols > 1
    ? (w - 2 * marginX) / (totalCols - 1)
    : 0;

  layerXPositions = {};
  activeLayers.forEach((layerIdx, colIdx) => {
    layerXPositions[layerIdx] = marginX + colIdx * colSpacing;
  });
  if (hasLogit) {
    layerXPositions["logit"] = marginX + activeLayers.length * colSpacing;
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
      x: old ? old.x : targetX + (Math.random() - 0.5) * 10,
      y: old ? old.y : h / 2 + (Math.random() - 0.5) * h * 0.6,
    };
  });

  // Add logit nodes (always visible, not subject to top-N)
  if (hasLogit) {
    const logitX = layerXPositions["logit"];
    DATA._logitNodes.forEach((ln, idx) => {
      const old = oldPosMap[ln.key];
      currentNodes.push({
        ...ln,
        targetX: logitX,
        x: old ? old.x : logitX,
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
      if (d.isLogitNode) return 16;
      return 6 + 8 * (nodeScore(d) / (maxScore || 1));
    }))
    .force("x", d3.forceX(d => d.targetX).strength(0.8))
    .force("y", d3.forceY(h / 2).strength(0.02))
    .alphaDecay(0.01)
    .on("tick", ticked);

  // --- Draw layer guide lines ---
  gLabels.selectAll("*").remove();
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
      tooltip.style.display = "block";
      const toLabel = d.isLogitEdge
        ? `Logit "${d.dLabel}"`
        : `L${d.dLayer} #${d.dId}`;
      tooltip.innerHTML =
        `<span class="tl">From:</span> <span class="tv">L${d.uLayer} #${d.uId}</span><br>` +
        `<span class="tl">To:</span> <span class="tv">${toLabel}</span><br>` +
        `<span class="tl">Gradient:</span> <span class="tv">${d.value.toFixed(6)}</span>`;
      tooltip.style.left = (event.clientX + 12) + "px";
      tooltip.style.top = (event.clientY - 10) + "px";
    })
    .on("mousemove", function(event) {
      tooltip.style.left = (event.clientX + 12) + "px";
      tooltip.style.top = (event.clientY - 10) + "px";
    })
    .on("mouseout", () => { tooltip.style.display = "none"; });

  // --- Draw nodes ---
  const nodeSelection = gNodes.selectAll("g.node")
    .data(currentNodes, d => d.key);
  nodeSelection.exit().remove();

  const nodeEnter = nodeSelection.enter().append("g").attr("class", "node");

  nodeEnter.append("circle")
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

  nodeEnter.append("text")
    .attr("dy", d => {
      if (d.isLogitNode) return -16;
      return -(6 + 6 * (nodeScore(d) / (maxScore || 1))) - 2;
    })
    .attr("text-anchor", "middle")
    .attr("font-weight", d => d.isLogitNode ? "bold" : "normal")
    .attr("font-size", d => d.isLogitNode ? "12px" : "9px")
    .attr("fill", d => d.isLogitNode ? "#e94560" : "#ccc")
    .text(d => d.isLogitNode ? d.label : d.id);

  const nodeMerge = nodeEnter.merge(nodeSelection);

  // Update circle sizes and colors on merge (for existing nodes that changed)
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
      if (d.isLogitNode) {
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
        if (d.isLogitNode) {
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
  d.fx = null;
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
        if (d.isLogitNode) {
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

  // Create node data — initial random Y/Z, fixed X per layer
  const layerSpacing = 60;
  const nodeData = [];
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
      if (!selectedKeys.has(ln.key)) selectedKeys.add(ln.key);
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
      if (a.isLogitNode) continue;
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
    const r = n.isLogitNode ? 6 : 2 + 4 * (nodeScore(n) / (ms || 1));
    const geometry = new THREE.SphereGeometry(r, 16, 16);
    let color;
    if (n.isLogitNode) {
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

// Initial load
initData(DATA);
</script>
</body>
</html>
"""
