// viewer.js

// ============================================================================
// SVG + base groups
// ============================================================================
const svg = d3.select("#mainSVG");
const width  = svg.node().clientWidth  || (window.innerWidth  - 280);
const height = svg.node().clientHeight || (window.innerHeight - 120);

const rootG   = svg.append("g");
const linksG  = rootG.append("g").attr("class", "links");
const nodesG  = rootG.append("g").attr("class", "nodes");
const labelsG = rootG.append("g").attr("class", "cluster-labels");

// zoom/pan
const zoom = d3.zoom()
  .scaleExtent([0.1, 5])
  .on("zoom", (event) => rootG.attr("transform", event.transform));
svg.call(zoom).on("dblclick.zoom", null);

// ============================================================================
// Controls & panels
// ============================================================================
const lensSelect      = document.getElementById("lensSelect");
const colorSelect     = document.getElementById("colorSelect");
const nodeSizeSlider  = document.getElementById("nodeSizeSlider");
const edgeWidthSlider = document.getElementById("edgeWidthSlider");
const stepSlider      = document.getElementById("stepSlider");
const playBtn         = document.getElementById("playBtn");
const graphStatsDiv   = document.getElementById("graphStats");
const legendBar       = document.getElementById("legendBar");
const timelineSvg     = d3.select("#timelineSVG");
const timelineLabel   = document.getElementById("timelineLabel");
const traceSvg        = d3.select("#traceSVG");
const modelTabs       = document.querySelectorAll(".model-tab");

const selectedSummaryDiv     = document.getElementById("selectedSummary");
const selectedCompositionDiv = document.getElementById("selectedComposition");

// ============================================================================
// State
// ============================================================================
let nodes = [];
let links = [];
let simulation = null;

let currentLens      = "pca";
let currentColorMode = "anomaly_fraction";
let nodeSizeScale    = parseFloat(nodeSizeSlider.value)  || 1.4;
let edgeWidthScale   = parseFloat(edgeWidthSlider.value) || 2.4;
let maxBirthStep     = 0;
let currentStep      = 0;
let playing          = false;

let degreeMap        = {};
let selectedNodeId   = null;

// ============================================================================
// Colour scales (BRIGHT, no dark/navy inside)
// ============================================================================
// ANOMALY FRACTION
const anomalyColor = d3.scaleSequential(
  t => d3.interpolateRgbBasis(["#fee2e2", "#fdba74", "#f97316", "#b91c1c"])(t)
).domain([0, 1]);

// Type colours (pies + legend)
const typeColors = d3.scaleOrdinal()
  .domain(["0","1","2","3","4","5"])
  .range([
    "#e5e7eb", // Normal (light grey)
    "#fb923c", // Mean shift (bright orange)
    "#f97373", // Spike (coral red)
    "#c4b5fd", // Variance (soft violet)
    "#22d3ee", // Dropout (cyan)
    "#4ade80"  // Regime (light green)
  ]);

// Dominant channel colours 
const channelColors = d3.scaleOrdinal(d3.schemeSet3);

// time trace state
let traceX           = null;
let traceCursor      = null;
let traceData        = [];
let traceTypeByStep  = [];
let traceSegments    = [];

// node selections
let nodeGroupSel  = null;
let nodeCircleSel = null;

// ============================================================================
// Helper functions
// ============================================================================
function normalizeColorMode(value) {
  if (!value) return "";
  const v = value.toString().toLowerCase().replace(/\s+/g, "_");
  if (v === "anomaly_type" || v === "type" || v === "dominant_type") {
    return "anomaly_type";
  }
  if (v === "anomaly_fraction" || v === "fraction") {
    return "anomaly_fraction";
  }
  if (v === "dominant_channel" || v === "channel_dominance" || v === "channel") {
    return "dominant_channel";
  }
  if (v === "reconstruction_error" || v === "mean_score" || v === "recon_error") {
    return "reconstruction_error";
  }
  return v;
}

// Build anomaly-type segments for pies (1..5, ignoring normal except pure-normal nodes)
function getTypeSegments(d) {
  let hist = Array.isArray(d.type_hist) && d.type_hist.length === 6
    ? d.type_hist.slice()
    : null;

  if (!hist) {
    const frac = d.anomaly_fraction ?? 0;
    const t    = d.dominant_type ?? 0;
    hist = [0, 0, 0, 0, 0, 0];
    if (t > 0) hist[t] = frac;
  }

  const segments = [];
  for (let k = 1; k <= 5; k++) {
    const v = hist[k] || 0;
    if (v > 0) segments.push({ type: k, value: v });
  }

  if (segments.length === 0) {
    const normalVal = (hist[0] != null ? hist[0] : 1);
    segments.push({ type: 0, value: normalVal });
  }

  return segments;
}

const typeNames = ["Normal", "Mean shift", "Spike", "Variance", "Dropout", "Regime"];

// ============================================================================
// Model tabs
// ============================================================================
modelTabs.forEach(tab => {
  tab.addEventListener("click", () => {
    modelTabs.forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    const path = tab.getAttribute("data-json");
    loadModel(path);
  });
});

// default model
loadModel("synthetic_mapper_pca_umap.json");

// ============================================================================
// Load + initialise
// ============================================================================
function loadModel(path) {
  fetch(path)
    .then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then(data => initWithData(data, path))
    .catch(err => {
      console.error("Error loading model", path, err);
      graphStatsDiv.innerHTML = `<span style="color:#f97373;">Could not load <b>${path}</b>.</span>`;
    });
}

function initWithData(data, modelName) {
  nodesG.selectAll("*").remove();
  linksG.selectAll("*").remove();
  labelsG.selectAll("*").remove();
  traceSvg.selectAll("*").remove();
  if (simulation) simulation.stop();

  nodes = data.nodes || [];
  links = data.links || [];
  const meta = data.meta || {};

  degreeMap = {};
  links.forEach(l => {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    degreeMap[s] = (degreeMap[s] || 0) + 1;
    degreeMap[t] = (degreeMap[t] || 0) + 1;
  });

  const maxBirthFromNodes = d3.max(nodes, d => d.birth_step || 0) || 0;
  maxBirthStep = (meta.max_birth_step != null)
    ? meta.max_birth_step
    : maxBirthFromNodes;

  currentStep      = 0;
  selectedNodeId   = null;

  stepSlider.min   = 0;
  stepSlider.max   = maxBirthStep;
  stepSlider.value = 0;

  graphStatsDiv.innerHTML = `
    <div>File: <b>${modelName}</b></div>
    <div>Nodes: <b>${nodes.length}</b></div>
    <div>Links: <b>${links.length}</b></div>
    <div>Window: <b>${meta.window_size}</b>, stride <b>${meta.stride}</b></div>
    <div>Channels: <b>${(meta.channels || []).join(", ")}</b></div>
  `;

  buildLegend();

  normalizeCoords("pca");
  normalizeCoords("umap");

  createSimulation();
  computeTimeSeriesTrace();
  drawTimeSeriesTrace();
  updateTraceCursor();
  updateVisibility();
  currentColorMode = normalizeColorMode(colorSelect.value);
  updateColors();
  updateNodeShapes();
  updateEdgeWidth();

  lensSelect.onchange = () => {
    currentLens = lensSelect.value.toLowerCase();
    repositionFromLens();
  };

  colorSelect.onchange = () => {
    currentColorMode = normalizeColorMode(colorSelect.value);
    updateColors();
  };

  nodeSizeSlider.oninput = () => {
    nodeSizeScale = parseFloat(nodeSizeSlider.value);
    updateNodeShapes();
  };

  edgeWidthSlider.oninput = () => {
    edgeWidthScale = parseFloat(edgeWidthSlider.value);
    updateEdgeWidth();
  };

  stepSlider.oninput = () => {
    currentStep = parseInt(stepSlider.value, 10) || 0;
    updateVisibility();
    updateTimelineMarker();
  };

  playBtn.onclick = togglePlay;
}

// ============================================================================
// Lens normalisation
// ============================================================================
function normalizeCoords(key) {
  const xs = nodes.map(d => d[key][0]);
  const ys = nodes.map(d => d[key][1]);
  const xScale = d3.scaleLinear().domain(d3.extent(xs)).range([40, width - 40]);
  const yScale = d3.scaleLinear().domain(d3.extent(ys)).range([40, height - 40]);
  nodes.forEach(d => {
    d[key + "Norm"] = [xScale(d[key][0]), yScale(d[key][1])];
  });
}

function repositionFromLens() {
  const k = currentLens === "umap" ? "umap" : "pca";
  nodes.forEach(d => {
    const norm = d[k + "Norm"] || d["pcaNorm"];
    d.x = norm[0];
    d.y = norm[1];
  });
  if (simulation) simulation.alpha(0.8).restart();
}

// ============================================================================
// Disjoint force layout + labels + pies
// ============================================================================
function createSimulation() {
  const lensKey = currentLens === "umap" ? "umap" : "pca";
  nodes.forEach(d => {
    const norm = d[lensKey + "Norm"] || d["pcaNorm"];
    d.x = norm[0];
    d.y = norm[1];
  });

  const compIndex   = computeComponents(nodes, links);
  const nComponents = d3.max(compIndex) + 1 || 1;
  nodes.forEach((d, i) => d.component = compIndex[i]);

  const R = Math.min(width, height) * 0.32;
  const componentCenters = [];
  for (let k = 0; k < nComponents; k++) {
    const angle = (2 * Math.PI * k) / nComponents;
    componentCenters.push([
      width  / 2 + R * Math.cos(angle),
      height / 2 + R * Math.sin(angle),
    ]);
  }

  simulation = d3.forceSimulation(nodes)
    .force(
      "link",
      d3.forceLink(links)
        .id(d => d.id)
        .distance(42)
        .strength(0.9)
    )
    .force("charge", d3.forceManyBody().strength(-55))
    .force(
      "collide",
      d3.forceCollide()
        .radius(d => nodeRadius(d) + 5)
        .iterations(2)
    )
    .force(
      "x",
      d3.forceX(d => componentCenters[d.component][0]).strength(0.20)
    )
    .force(
      "y",
      d3.forceY(d => componentCenters[d.component][1]).strength(0.20)
    )
    .alpha(1)
    .alphaDecay(0.03)
    .on("tick", ticked);

  const linkSel = linksG.selectAll("line")
    .data(links)
    .enter()
    .append("line")
    .attr("stroke", "#f9fafb")
    .attr("stroke-linecap", "round");

  nodeGroupSel = nodesG.selectAll("g.node")
    .data(nodes)
    .enter()
    .append("g")
    .attr("class", "node")
    .style("cursor", "pointer")
    .on("click", (event, d) => {
      event.stopPropagation();
      selectNode(d);
    })
    .call(
      d3.drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded)
    );

  nodeCircleSel = nodeGroupSel.append("circle")
    .attr("stroke", "#020617")
    .attr("stroke-width", 0.9)
    .attr("fill", "#e5e7eb"); 

  nodeGroupSel.append("g").attr("class", "node-pie");

  simulation._linkSel   = linkSel;
  simulation._nodeGroup = nodeGroupSel;

  const labelData = d3.range(nComponents).map(k => ({ component: k }));
  const labelSel = labelsG.selectAll("text")
    .data(labelData)
    .enter()
    .append("text")
    .attr("text-anchor", "middle")
    .attr("dy", "-6")
    .attr("font-size", 11)
    .attr("fill", "#e5e7eb")
    .text(d => `C${d.component}`);
  simulation._labelSel = labelSel;

  svg.on("click", (event) => {
    if (event.target === svg.node()) {
      clearSelection();
    }
  });

  ticked();
}

function nodeRadius(d) {
  return (2 + Math.log(1 + d.size)) * nodeSizeScale;
}

function updateNodeShapes() {
  if (!nodeGroupSel) return;

  const arcGen = d3.arc();
  const pieGen = d3.pie().value(d => d.value);

  nodeGroupSel.each(function (d) {
    const r = nodeRadius(d);
    const g = d3.select(this);

    g.select("circle").attr("r", r);

    const segments = getTypeSegments(d);
    const pieData  = pieGen(segments);

    const pieG = g.select("g.node-pie");
    const paths = pieG.selectAll("path").data(pieData, p => p.data.type);

    paths.enter()
      .append("path")
      .merge(paths)
      .attr("d", arcGen.innerRadius(0).outerRadius(r))
      .attr("fill", p => typeColors(String(p.data.type)))
      .attr("stroke", "none")
      .attr("opacity", currentColorMode === "anomaly_type" ? 0.97 : 0);

    paths.exit().remove();
  });

  nodeGroupSel.classed("selected", d => d.id === selectedNodeId);
}

function ticked() {
  if (!simulation) return;

  simulation._linkSel
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);

  if (nodeGroupSel) {
    nodeGroupSel.attr("transform", d => `translate(${d.x},${d.y})`);
  }

  const compToPoints = new Map();
  nodes.forEach(d => {
    if (!compToPoints.has(d.component)) compToPoints.set(d.component, []);
    compToPoints.get(d.component).push([d.x, d.y]);
  });

  simulation._labelSel
    .attr("x", d => {
      const pts = compToPoints.get(d.component) || [];
      return pts.length ? d3.mean(pts, p => p[0]) : width / 2;
    })
    .attr("y", d => {
      const pts = compToPoints.get(d.component) || [];
      return pts.length ? d3.mean(pts, p => p[1]) : height / 2;
    });
}

function computeComponents(nodes, links) {
  const idToIdx = new Map(nodes.map((d, i) => [d.id, i]));
  const adj = nodes.map(() => []);

  links.forEach(l => {
    const src = typeof l.source === "object" ? l.source.id : l.source;
    const tgt = typeof l.target === "object" ? l.target.id : l.target;
    const a = idToIdx.get(src);
    const b = idToIdx.get(tgt);
    if (a == null || b == null) return;
    adj[a].push(b);
    adj[b].push(a);
  });

  const comp = new Array(nodes.length).fill(-1);
  let cid = 0;
  for (let i = 0; i < nodes.length; i++) {
    if (comp[i] !== -1) continue;
    const stack = [i];
    comp[i] = cid;
    while (stack.length) {
      const v = stack.pop();
      for (const nb of adj[v]) {
        if (comp[nb] === -1) {
          comp[nb] = cid;
          stack.push(nb);
        }
      }
    }
    cid++;
  }
  return comp;
}

function dragStarted(event, d) {
  if (!event.active && simulation) simulation.alphaTarget(0.3).restart();
  d.fx = d.x;
  d.fy = d.y;
}
function dragged(event, d) {
  d.fx = event.x;
  d.fy = event.y;
}
function dragEnded(event, d) {
  if (!event.active && simulation) simulation.alphaTarget(0);
  d.fx = null;
  d.fy = null;
}

// ============================================================================
// Selection + composition panel
// ============================================================================
function clearSelectedPanel() {
  selectedSummaryDiv.textContent = "Click a node in the graph to inspect its anomaly composition.";
  selectedCompositionDiv.innerHTML = "";
}

function clearSelection() {
  selectedNodeId = null;
  if (nodeGroupSel) {
    nodeGroupSel.classed("selected", false);
  }
  clearSelectedPanel();
}

function selectNode(d) {
  selectedNodeId = d.id;
  if (nodeGroupSel) {
    nodeGroupSel.classed("selected", n => n.id === selectedNodeId);
  }

  let hist = Array.isArray(d.type_hist) && d.type_hist.length === 6
    ? d.type_hist.slice()
    : null;

  if (!hist) {
    hist = [0,0,0,0,0,0];
    const t = d.dominant_type ?? 0;
    if (t >= 0 && t < 6) hist[t] = 1;
  }

  const total = hist.reduce((a,b) => a + b, 0) || 1;
  const perc = hist.map(c => (c / total) * 100);
  const maxIdx = perc.reduce((best, v, i) => (v > perc[best] ? i : best), 0);
  const purity = perc[maxIdx] / 100;
  const degree = degreeMap[d.id] || 0;
  const dominantLabel = typeNames[maxIdx] || `Type ${maxIdx}`;

  selectedSummaryDiv.innerHTML = `
    <div>ID: <b>${d.id}</b></div>
    <div>Size: <b>${d.size}</b></div>
    <div>Purity: <b>${(purity * 100).toFixed(1)}%</b></div>
    <div>Dominant: <b>${dominantLabel}</b> (${perc[maxIdx].toFixed(1)}%)</div>
    <div>Degree: <b>${degree}</b></div>
  `;

  const rows = [];
  for (let k = 0; k < 6; k++) {
    if (hist[k] <= 0) continue;
    rows.push({ type: k, name: typeNames[k], pct: perc[k] });
  }
  rows.sort((a,b) => b.pct - a.pct);

  selectedCompositionDiv.innerHTML = rows.map(r => `
    <div class="comp-row">
      <div class="comp-label">
        <div class="comp-dot" style="background:${typeColors(String(r.type))};"></div>
        <span>${r.name}</span>
      </div>
      <div class="comp-val">${r.pct.toFixed(1)}%</div>
    </div>
  `).join("");
}

// ============================================================================
// Visibility / colours / edges
// ============================================================================
function isVisible(d) {
  return (d.birth_step || 0) <= currentStep;
}

function updateVisibility() {
  if (!simulation || !nodeGroupSel) return;
  nodeGroupSel.style("opacity", d => (isVisible(d) ? 1 : 0.18));
  simulation._linkSel.style("opacity", d =>
    (isVisible(d.source) && isVisible(d.target)) ? 0.35 : 0.05
  );
  updateColors();
}

function updateColors() {
  if (!nodeGroupSel) return;

  if (currentColorMode === "anomaly_type") {
    nodeGroupSel.selectAll("g.node-pie path").attr("opacity", 0.97);
    nodeCircleSel.attr("fill", "#020617");

  } else if (currentColorMode === "anomaly_fraction") {
    nodeGroupSel.selectAll("g.node-pie path").attr("opacity", 0);
    nodeCircleSel.attr("fill", d =>
      isVisible(d) ? anomalyColor(d.anomaly_fraction ?? 0) : "#4b5563"
    );

  } else if (currentColorMode === "dominant_channel") {
    nodeGroupSel.selectAll("g.node-pie path").attr("opacity", 0);
    nodeCircleSel.attr("fill", d =>
      isVisible(d) ? channelColors(String(d.dominant_channel)) : "#4b5563"
    );

  } else if (currentColorMode === "reconstruction_error") {
    nodeGroupSel.selectAll("g.node-pie path").attr("opacity", 0);

    const scores = nodes
      .map(d => d.mean_score)
      .filter(v => v != null && !Number.isNaN(v));
    let minS = 0, maxS = 1;
    if (scores.length > 0) {
      minS = d3.min(scores);
      maxS = d3.max(scores);
      if (minS === maxS) {
        minS -= 1e-6;
        maxS += 1e-6;
      }
    }

    const reconColor = d3.scaleSequential(
      t => d3.interpolateRgbBasis(["#ddf9ff", "#60a5fa", "#6366f1", "#c4b5fd", "#facc15"])(t)
    ).domain([minS, maxS]);

    nodeCircleSel.attr("fill", d => {
      if (!isVisible(d)) return "#4b5563";
      const s = d.mean_score;
      if (s == null || Number.isNaN(s) || scores.length === 0) {
        return "#9ca3af";
      }
      return reconColor(s);
    });

  } else {
    nodeGroupSel.selectAll("g.node-pie path").attr("opacity", 0);
    nodeCircleSel.attr("fill", "#93c5fd");
  }

  nodeGroupSel.classed("selected", d => d.id === selectedNodeId);
}

function updateEdgeWidth() {
  if (!simulation) return;
  simulation._linkSel.attr("stroke-width", edgeWidthScale);
}

// ============================================================================
// Legend
// ============================================================================
function buildLegend() {
  const items = [
    ["0", "Normal"],
    ["1", "Mean shift"],
    ["2", "Spike"],
    ["3", "Variance"],
    ["4", "Dropout"],
    ["5", "Regime"],
  ];
  legendBar.innerHTML = "";
  items.forEach(([k, label]) => {
    const div = document.createElement("div");
    div.className = "legendItem";
    const sw = document.createElement("div");
    sw.className = "legendSwatch";
    sw.style.background = typeColors(k);
    const span = document.createElement("span");
    span.textContent = label;
    div.appendChild(sw);
    div.appendChild(span);
    legendBar.appendChild(div);
  });
}

// ============================================================================
// Timeline + trace
// ============================================================================
//Stub
function updateTimelineMarker() {
  if (timelineLabel) {
    timelineLabel.textContent = `birth_step ≤ ${currentStep}`;
  }
  updateTraceCursor();
}

function computeTimeSeriesTrace() {
  const T = maxBirthStep;
  traceData       = new Array(T + 1).fill(0);
  traceTypeByStep = new Array(T + 1).fill(0);
  const typeScore = new Array(T + 1).fill(0);

  nodes.forEach(n => {
    const t = n.birth_step;
    if (t < 0 || t > T) return;
    const frac = n.anomaly_fraction ?? 0;
    const typ  = n.dominant_type ?? 0;

    traceData[t] += frac;
    if (frac > typeScore[t]) {
      typeScore[t] = frac;
      traceTypeByStep[t] = typ;
    }
  });

  const maxVal = d3.max(traceData) || 1;
  traceData = traceData.map(v => v / maxVal);

  traceSegments = [];
  if (T >= 0) {
    let curType  = traceTypeByStep[0];
    let curStart = 0;
    for (let t = 1; t <= T; t++) {
      const tp = traceTypeByStep[t];
      if (tp !== curType) {
        traceSegments.push({ type: curType, start: curStart, end: t - 1 });
        curType  = tp;
        curStart = t;
      }
    }
    traceSegments.push({ type: curType, start: curStart, end: T });
  }
}

function drawTimeSeriesTrace() {
  traceSvg.selectAll("*").remove();
  const W = +traceSvg.attr("width");
  const H = +traceSvg.attr("height");

  traceX = d3.scaleLinear().domain([0, maxBirthStep]).range([20, W - 20]);
  const y = d3.scaleLinear().domain([0, 1]).range([H - 8, 6]);

  const barY = H - 6;
  const barH = 4;
  traceSvg.append("g")
    .selectAll("rect")
    .data(traceSegments.filter(seg => seg.type !== 0))
    .enter()
    .append("rect")
    .attr("x", seg => traceX(seg.start))
    .attr("width", seg => {
      const w = traceX(seg.end) - traceX(seg.start);
      return w <= 0 ? 1 : w;
    })
    .attr("y", barY)
    .attr("height", barH)
    .attr("fill", seg => typeColors(String(seg.type)))
    .attr("opacity", 0.95);

  const line = d3.line()
    .x((d, i) => traceX(i))
    .y(d => y(d))
    .curve(d3.curveMonotoneX);

  traceSvg.append("path")
    .datum(traceData)
    .attr("d", line)
    .attr("stroke", "#60a5fa")
    .attr("stroke-width", 2)
    .attr("fill", "none");

  traceCursor = traceSvg.append("line")
    .attr("stroke", "#f97316")
    .attr("stroke-width", 2)
    .attr("y1", 4)
    .attr("y2", H - 4);
}

function updateTraceCursor() {
  if (!traceCursor || !traceX) return;
  const x = traceX(currentStep);
  traceCursor.attr("x1", x).attr("x2", x);
}

// ============================================================================
// Animation
// ============================================================================
function togglePlay() {
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
  if (playing) playLoop();
}

function playLoop() {
  if (!playing) return;
  const delta = Math.max(1, Math.round(maxBirthStep / 160));
  currentStep += delta;
  if (currentStep >= maxBirthStep) {
    currentStep = maxBirthStep;
    playing = false;
    playBtn.textContent = "Play";
  }
  stepSlider.value = currentStep;
  updateVisibility();
  updateTimelineMarker();
  if (playing) setTimeout(playLoop, 80);
}

