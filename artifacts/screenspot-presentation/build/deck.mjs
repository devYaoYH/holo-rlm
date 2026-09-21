import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/yaoyiheng/Documents/ChatGPT/GUI VLM Fine Tuning";
const SKILL_DIR = "/Users/yaoyiheng/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations";
const TMP_DIR = path.join(workspaceDir, "artifacts/screenspot-presentation/build");
const FINAL_PPTX = path.join(workspaceDir, "artifacts/screenspot-presentation/holo-attribution-research-v22.pptx");
const RUNTIME_PYTHON = "/Users/yaoyiheng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
const family = resolvePresentationFont();
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

const C = {
  ink: "#10151C",
  paper: "#F5F2EB",
  white: "#FFFFFF",
  muted: "#66717D",
  line: "#D9D4C8",
  orange: "#F05A3C",
  orangeSoft: "#FCE2D9",
  blue: "#2D84D3",
  blueSoft: "#DDECF9",
  green: "#18A875",
  gold: "#E5B249",
  deep: "#1C2733",
};

const balancedLayer19 = JSON.parse(
  await fs.readFile(
    path.join(workspaceDir, "artifacts/screenspot-presentation/screenspot-balanced-layer19-summary.json"),
    "utf8",
  ),
);
const benchmarkBreakdown = JSON.parse(
  await fs.readFile(
    path.join(workspaceDir, "artifacts/screenspot-presentation/screenspot-benchmark-breakdown.json"),
    "utf8",
  ),
);
const categoryExamples = JSON.parse(
  await fs.readFile(
    path.join(workspaceDir, "artifacts/screenspot-presentation/screenspot-category-examples.json"),
    "utf8",
  ),
);
const targetSizeAnalysis = JSON.parse(
  await fs.readFile(
    path.join(workspaceDir, "artifacts/screenspot-presentation/screenspot-target-size-analysis.json"),
    "utf8",
  ),
);

const successRaw = path.join(workspaceDir, "data/attributions/screenspot-powerpoint_windows_59/preview-raw.png");
const successDiff = path.join(workspaceDir, "data/attributions/screenspot-powerpoint_windows_59/preview-target-minus-prompt-baseline.png");
const failureDiff = path.join(workspaceDir, "data/attributions/screenspot-powerpoint_windows_48/preview-target-minus-prompt-baseline.png");
const hotelContrast = path.join(
  workspaceDir,
  "artifacts/screenspot-presentation/live/hotel-cheapest-multiframe",
);
const hotelDiffMaps = [0, 1, 2, 3].map((frame) =>
  path.join(
    hotelContrast,
    `preview-frame-${String(frame).padStart(3, "0")}-target-minus-prompt-baseline.png`,
  ),
);
const causalDir = path.join(workspaceDir, "artifacts/causal-intervention/powerpoint_windows_59_swap");
const causalClean = path.join(causalDir, "clean-focus.png");
const causalCorrupted = path.join(causalDir, "corrupted-focus.png");
const causal = JSON.parse(await fs.readFile(path.join(causalDir, "results.json"), "utf8"));
const deltaLensPath = path.join(
  workspaceDir,
  "data/remote-results/delta-lens-qwen35-vs-holo31-20260921-dcd8c5e5/run",
  "delta-lens-qwen35-vs-holo31-20260921-dcd8c5e5/delta-lens.json",
);
const deltaLens = JSON.parse(await fs.readFile(deltaLensPath, "utf8"));
const deltaCases = deltaLens.comparison.cases;
const screenDeltaCases = deltaCases.filter((row) => row.protocol === "hcompany_element_localization_v1");
const screenFinalLayers = screenDeltaCases.map((row) => row.layers.at(-1));
const screenMeanFinalDelta = screenFinalLayers.reduce((sum, row) => sum + row.delta_margin_nats, 0) / screenFinalLayers.length;
const screenPositiveFinalCount = screenFinalLayers.filter((row) => row.delta_margin_nats > 0).length;
const screenBoundaryFlips = screenFinalLayers.filter((row) => row.base_margin_nats < 0 && row.tuned_margin_nats > 0).length;
const hotelFinalDeltas = deltaCases
  .filter((row) => row.protocol === "holo_desktop_action_v1")
  .map((row) => row.layers.at(-1).delta_margin_nats);
const screenLayerRows = screenDeltaCases[0].layers.filter((row) => row.layer >= 0);
const deltaLayerLabels = screenLayerRows.map((row) => `L${row.layer}`);
const screenMeanLayerDeltas = screenLayerRows.map((row) => {
  const layer = row.layer;
  return Number((screenDeltaCases.reduce(
    (sum, item) => sum + item.layers.find((candidate) => candidate.layer === layer).delta_margin_nats,
    0,
  ) / screenDeltaCases.length).toFixed(3));
});
const hotelFinalCase = deltaCases.find((row) => row.id === "hotel_test_0035_large_ui_step_2");
const hotelFinalLayerDeltas = hotelFinalCase.layers
  .filter((row) => row.layer >= 0)
  .map((row) => Number(row.delta_margin_nats.toFixed(3)));
const attentionDeltaDir = path.join(workspaceDir, "artifacts/screenspot-presentation/delta-lens-v22");
const attentionDeltaSummary = JSON.parse(
  await fs.readFile(path.join(attentionDeltaDir, "attention-delta-summary.json"), "utf8"),
);
const qwenAttention = path.join(attentionDeltaDir, "attention-qwen-direct.png");
const holoAttention = path.join(attentionDeltaDir, "attention-holo-direct.png");
const attentionDifference = path.join(attentionDeltaDir, "attention-holo-minus-qwen-direct.png");
const directTargetDelta = attentionDeltaSummary.screen_target_delta.direct_attention;
const valueTargetDelta = attentionDeltaSummary.screen_target_delta.value_norm_attention;
const hotelValueFrames = attentionDeltaSummary.hotel_frame_allocation.value_norm_attention;

function rect(slide, left, top, width, height, fill, radius = false, line = "none") {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill,
    line: line === "none" ? { fill: "none", width: 0 } : { fill: line, width: 1 },
  });
}

function textBox(slide, text, left, top, width, height, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: options.fill ?? "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: family,
    fontSize: options.fontSize ?? 22,
    bold: options.bold ?? false,
    color: options.color ?? C.ink,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: options.autoFit ?? "shrinkText",
    wrap: "square",
    insets: options.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return shape;
}

function richText(slide, paragraphs, left, top, width, height, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: options.fill ?? "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = paragraphs;
  shape.text.style = {
    typeface: family,
    fontSize: options.fontSize ?? 22,
    color: options.color ?? C.ink,
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: "shrinkText",
    wrap: "square",
    insets: options.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return shape;
}

function pill(slide, label, left, top, width, fill, color = C.ink) {
  rect(slide, left, top, width, 30, fill, true);
  textBox(slide, label, left + 10, top + 2, width - 20, 26, {
    fontSize: 13,
    bold: true,
    color,
    alignment: "center",
    verticalAlignment: "middle",
  });
}

function clickMarker(slide, x, y, size = 18) {
  const halo = 8;
  slide.shapes.add({
    geometry: "ellipse",
    position: { left: x - size, top: y - size, width: size * 2, height: size * 2 },
    fill: "none",
    line: { fill: C.ink, width: halo },
  });
  slide.shapes.add({
    geometry: "ellipse",
    position: { left: x - size, top: y - size, width: size * 2, height: size * 2 },
    fill: "none",
    line: { fill: C.white, width: 3 },
  });
  rect(slide, x - size - 6, y - 4, size * 2 + 12, 8, C.ink, true);
  rect(slide, x - 4, y - size - 6, 8, size * 2 + 12, C.ink, true);
  rect(slide, x - size - 6, y - 1.5, size * 2 + 12, 3, C.white, true);
  rect(slide, x - 1.5, y - size - 6, 3, size * 2 + 12, C.white, true);
}

function slideTitle(slide, kicker, title, index, dark = false) {
  const fg = dark ? C.white : C.ink;
  textBox(slide, kicker.toUpperCase(), 64, 38, 820, 20, { fontSize: 12, bold: true, color: dark ? "#9FC8F0" : C.blue });
  textBox(slide, title, 64, 62, 1050, 58, { fontSize: 38, bold: true, color: fg });
  textBox(slide, String(index).padStart(2, "0"), 1176, 45, 40, 24, { fontSize: 13, bold: true, color: dark ? "#96A2AE" : C.muted, alignment: "right" });
}

function footer(slide, value, dark = false) {
  rect(slide, 64, 680, 1152, 1, dark ? "#3A4652" : C.line);
  textBox(slide, value, 64, 688, 1152, 16, { fontSize: 10, color: dark ? "#8794A2" : C.muted });
}

async function image(slide, source, left, top, width, height, options = {}) {
  const bytes = await fs.readFile(source);
  const element = slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt: options.alt ?? "ScreenSpot-Pro attribution overlay",
    fit: options.fit ?? "cover",
    position: { left, top, width, height },
    ...(options.crop ? { crop: options.crop } : {}),
    geometry: options.geometry ?? "roundRect",
    borderRadius: options.borderRadius ?? 12,
  });
  rect(slide, left, top, width, height, "none", true, options.line ?? C.line);
  return element;
}

function metricCard(slide, left, top, width, label, value, accent = C.orange, caption = "") {
  rect(slide, left, top, width, 108, C.white, true, C.line);
  rect(slide, left, top, 7, 108, accent, true);
  textBox(slide, label.toUpperCase(), left + 20, top + 14, width - 32, 18, { fontSize: 11, bold: true, color: C.muted });
  textBox(slide, value, left + 20, top + 34, width - 32, 38, { fontSize: 31, bold: true, color: C.ink });
  if (caption) textBox(slide, caption, left + 20, top + 77, width - 32, 20, { fontSize: 11, color: C.muted });
}

function mixHex(start, end, fraction) {
  const parse = (value) => [1, 3, 5].map((index) => Number.parseInt(value.slice(index, index + 2), 16));
  const [sr, sg, sb] = parse(start);
  const [er, eg, eb] = parse(end);
  const mix = (a, b) => Math.round(a + (b - a) * fraction).toString(16).padStart(2, "0");
  return `#${mix(sr, er)}${mix(sg, eg)}${mix(sb, eb)}`;
}

function addHeadMatrix(slide, headCase, left, top, width) {
  const layers = [3, 7, 11, 15, 19, 23, 27, 31];
  const heads = Array.from({ length: 16 }, (_, index) => index);
  const rows = headCase.analysis.layer_head_statistics.heads;
  const lookup = new Map(
    rows.map((row) => [`${row.transformer_layer}:${row.head}`, row.prompt_difference.target_lift]),
  );
  const peak = rows.reduce((best, row) =>
    row.prompt_difference.target_lift > best.prompt_difference.target_lift ? row : best,
  );
  pill(slide, headCase.label, left, top, 166, headCase.accent, C.white);
  textBox(
    slide,
    `peak L${peak.transformer_layer}/H${peak.head} · ${peak.prompt_difference.target_lift.toFixed(2)}×`,
    left + 174,
    top + 4,
    width - 174,
    22,
    { fontSize: 11, bold: true, color: C.muted, alignment: "right" },
  );
  const values = [
    ["", ...heads.map((head) => `H${head}`)],
    ...layers.map((layer) => [
      `L${layer}`,
      ...heads.map((head) => Math.round(lookup.get(`${layer}:${head}`) ?? 0)),
    ]),
  ];
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left,
    top: top + 38,
    width,
    height: 246,
    values,
    columnWidths: [42, ...heads.map(() => (width - 42) / 16)],
  });
  table.cells.block({ row: 0, column: 0, rowCount: values.length, columnCount: values[0].length }).assign({
    margins: { left: 0, right: 0, top: 0, bottom: 0 },
  });
  table.borders.assign({ style: "solid", fill: C.white, width: 0.6 });
  for (let rowIndex = 0; rowIndex < values.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < values[0].length; columnIndex += 1) {
      const cell = table.getCell(rowIndex, columnIndex);
      if (rowIndex === 0 || columnIndex === 0) {
        cell.fill = C.deep;
        cell.text.style = {
          typeface: family,
          fontSize: rowIndex === 0 ? 5.5 : 7,
          bold: true,
          color: C.white,
          alignment: "center",
          verticalAlignment: "middle",
          autoFit: "shrinkText",
        };
        continue;
      }
      const value = Number(values[rowIndex][columnIndex]);
      const intensity = Math.sqrt(Math.min(value, 100) / 100);
      cell.fill = mixHex(C.paper, C.orange, intensity);
      cell.text.style = {
        typeface: family,
        fontSize: 5.5,
        bold: value >= 40,
        color: intensity > 0.58 ? C.white : C.ink,
        alignment: "center",
        verticalAlignment: "middle",
        autoFit: "shrinkText",
      };
    }
  }
}

// 1 — cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  await image(slide, successDiff, 704, 0, 576, 720, { alt: "Instruction-specific saliency on the Psychedelic vibrant template", borderRadius: 0, geometry: "rect", line: "none", crop: { left: 0.28, top: 0.08, right: 0.28, bottom: 0.08 } });
  rect(slide, 640, 0, 126, 720, { type: "gradient", gradientKind: "linear", angleDeg: 0, stops: [{ offset: 0, color: C.ink }, { offset: 100000, color: "#10151C00" }] });
  pill(slide, "MECHANISTIC GUI ANALYSIS", 64, 62, 224, C.orange, C.white);
  textBox(slide, "When attention\nstops being generic", 64, 142, 590, 168, { fontSize: 55, bold: true, color: C.white });
  textBox(slide, "Action-token attribution for static GUI grounding and multi-turn visual memory", 68, 340, 535, 104, { fontSize: 24, color: "#CBD3DB" });
  textBox(slide, "Holo 3.1 4B · local Metal traces · 21 Sep 2026", 68, 605, 535, 28, { fontSize: 14, bold: true, color: "#97A4B2" });
  slide.speakerNotes.textFrame.setText("0:00–0:35 — Frame the question: what is specific to the current instruction after removing generic image saliency, and can the same machinery follow evidence across a multi-turn trajectory? The visual is the final prompt-differential map for the ScreenSpot visual hit. Source: data/attributions/screenspot-powerpoint_windows_59/analysis.json.");
}

// 2 — two evaluation settings
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Evaluation", "Two case-study settings reveal different failure modes", 2);
  rect(slide, 64, 144, 548, 460, C.white, true, C.line);
  pill(slide, "STATIC", 90, 170, 86, C.blue, C.white);
  textBox(slide, "ScreenSpot-Pro", 90, 216, 300, 34, { fontSize: 27, bold: true });
  textBox(slide, "One screenshot, one instruction, one click", 90, 258, 420, 28, { fontSize: 18, color: C.muted });
  textBox(slide, "Visual hit", 90, 330, 150, 22, { fontSize: 14, bold: true, color: C.green });
  textBox(slide, "Psychedelic vibrant", 90, 358, 210, 30, { fontSize: 20, bold: true });
  textBox(slide, "Grounding miss", 326, 330, 160, 22, { fontSize: 14, bold: true, color: C.orange });
  textBox(slide, "Create new slide", 326, 358, 190, 30, { fontSize: 20, bold: true });
  rect(slide, 90, 420, 462, 1, C.line);
  textBox(slide, "These examples share PowerPoint on Windows, but use different images and prompts. They illustrate failure modes rather than form a matched causal pair.", 90, 448, 462, 96, { fontSize: 16, color: C.muted });

  rect(slide, 650, 144, 566, 460, C.deep, true);
  pill(slide, "MULTI-TURN", 680, 170, 120, C.orange, C.white);
  textBox(slide, "Synthetic hotel search", 680, 216, 420, 34, { fontSize: 27, bold: true, color: C.white });
  textBox(slide, "Four model calls retain every earlier screenshot", 680, 258, 454, 28, { fontSize: 18, color: "#B9C3CD" });
  const sequence = [
    ["01", "SCROLL", "500 px"],
    ["02", "SCROLL", "500 px"],
    ["03", "SCROLL", "500 px"],
    ["04", "CLICK", "miss"],
  ];
  sequence.forEach(([num, action, detail], i) => {
    const y = 316 + i * 56;
    pill(slide, num, 680, y, 46, i === 3 ? C.orange : "#344252", C.white);
    textBox(slide, action, 744, y + 2, 128, 22, { fontSize: 14, bold: true, color: C.white });
    textBox(slide, detail, 1010, y + 2, 130, 22, { fontSize: 15, bold: true, color: i === 3 ? "#F2BDAF" : "#B9C3CD", alignment: "right" });
  });
  textBox(slide, "Target: Signal Quay Rooms at $172", 680, 555, 458, 24, { fontSize: 15, bold: true, color: "#F0C7BC" });
  footer(slide, "ScreenSpot-Pro supplies real GUI cases · the deterministic hotel fixture supplies replayable trajectories");
  slide.speakerNotes.textFrame.setText("0:35–1:15 — Separate the two settings. ScreenSpot gives real high-resolution grounding examples. The synthetic hotel task gives exact replay and multi-frame memory. The two PowerPoint examples are not a same-image success/failure pair, so use them to motivate hypotheses rather than causal claims. The hotel trace is a Holo failure after three scrolls and one click. Sources: docs/screenspot-case-study.md and data/attributions/traj-20260919T164806Z-ba81a14a16/trajectory.json.");
}

// 3 - full ScreenSpot-Pro replication
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Full benchmark replication", "65.9% overall; target size explains part of the icon gap", 3);
  textBox(slide, "Holo3.1-4B, official element-localization harness, all 1,581 ScreenSpot-Pro items", 64, 118, 1152, 24, { fontSize: 14, color: C.muted });

  textBox(slide, "65.9%", 64, 162, 270, 68, { fontSize: 56, bold: true, color: C.ink });
  textBox(slide, "1,042 correct out of 1,581", 68, 232, 280, 24, { fontSize: 17, bold: true, color: C.muted });
  textBox(slide, "Reported 4B reference ≈ 66.5%", 68, 275, 280, 22, { fontSize: 15, bold: true, color: C.blue });
  textBox(slide, "Difference: −0.6 percentage point", 68, 303, 280, 22, { fontSize: 14, color: C.muted });
  textBox(slide, "100% valid JSON responses", 68, 341, 280, 22, { fontSize: 15, bold: true, color: C.green });

  rect(slide, 374, 158, 1, 216, C.line);
  textBox(slide, "ACCURACY BY TARGET UI", 404, 158, 260, 20, { fontSize: 11, bold: true, color: C.blue });
  const uiChart = slide.charts.add("bar", {
    position: { left: 392, top: 184, width: 506, height: 188 },
    categories: ["Text targets  n=977", "Icon targets  n=604"],
    series: [{
      name: "Accuracy",
      values: [benchmarkBreakdown.by_ui_type.text.accuracy, benchmarkBreakdown.by_ui_type.icon.accuracy]
        .map((value) => Number(value.toFixed(4))),
      fill: C.blue,
    }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 50 },
    hasLegend: false,
    xAxis: { visible: false, minimumScale: 0, maximumScale: 1, majorGridlines: null },
    yAxis: { textStyle: { fill: C.ink, fontSize: 13, bold: true }, line: { fill: "none", width: 0 } },
    dataLabels: { showValue: false },
    chartFill: C.paper,
    chartLine: { fill: "none", width: 0 },
    plotAreaFill: C.paper,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(uiChart, { fontFamily: family });
  textBox(slide, `${(benchmarkBreakdown.by_ui_type.icon.accuracy * 100).toFixed(1)}%`, 694, 218, 84, 24, { fontSize: 13, bold: true });
  textBox(slide, `${(benchmarkBreakdown.by_ui_type.text.accuracy * 100).toFixed(1)}%`, 819, 288, 84, 24, { fontSize: 13, bold: true });

  rect(slide, 930, 158, 286, 216, C.deep, true);
  textBox(slide, "TARGET-SIZE CONFOUND", 954, 182, 240, 20, { fontSize: 11, bold: true, color: C.gold });
  textBox(slide, `${targetSizeAnalysis.median_area_ratio_text_over_icon.toFixed(2)}×`, 954, 214, 112, 42, { fontSize: 34, bold: true, color: C.orange });
  textBox(slide, "larger median text-target area", 954, 258, 236, 28, { fontSize: 14, color: "#F2BDAF" });
  textBox(slide, `${(targetSizeAnalysis.raw_accuracy_gap_text_minus_icon * 100).toFixed(1)} pp raw gap`, 954, 300, 236, 24, { fontSize: 18, bold: true, color: C.white });
  textBox(slide, `${(targetSizeAnalysis.size_standardization.gap_text_minus_icon * 100).toFixed(1)} pp after area standardization`, 954, 330, 236, 28, { fontSize: 14, color: "#C4CED7" });

  textBox(slide, "UI TYPE × INSTRUCTION ACTION FAMILY", 64, 408, 420, 20, { fontSize: 11, bold: true, color: C.blue });
  textBox(slide, "Weakest labeled cell: icon × file, 36.1% (n=36)", 792, 408, 424, 20, { fontSize: 12, color: C.muted, alignment: "right" });
  const familyLabels = benchmarkBreakdown.action_family_order;
  const shortFamily = {
    "Navigate / reveal": "NAVIGATE\nREVEAL",
    "Select / activate": "SELECT\nACTIVATE",
    "Create / insert": "CREATE\nINSERT",
    "Edit / format": "EDIT\nFORMAT",
    "Remove / close": "REMOVE\nCLOSE",
    "File / transfer": "FILE\nTRANSFER",
    Other: "OTHER",
  };
  const matrixLookup = new Map(
    benchmarkBreakdown.ui_action_matrix.map((row) => [`${row.ui_type}:${row.action_family}`, row]),
  );
  const matrixValues = [
    ["UI TYPE", ...familyLabels.map((family) => shortFamily[family])],
    ...["icon", "text"].map((uiType) => [
      uiType.toUpperCase(),
      ...familyLabels.map((family) => {
        const row = matrixLookup.get(`${uiType}:${family}`);
        return `${(row.accuracy * 100).toFixed(1)}%\nn=${row.count}`;
      }),
    ]),
  ];
  const matrixTable = slide.tables.add({
    rows: matrixValues.length,
    columns: matrixValues[0].length,
    left: 64,
    top: 438,
    width: 1152,
    height: 142,
    values: matrixValues,
    columnWidths: [116, ...familyLabels.map(() => 148)],
    rowHeights: [42, 50, 50],
  });
  matrixTable.cells.block({ row: 0, column: 0, rowCount: matrixValues.length, columnCount: matrixValues[0].length }).assign({
    margins: { left: 5, right: 5, top: 2, bottom: 2 },
  });
  matrixTable.borders.assign({ style: "solid", fill: C.paper, width: 1 });
  for (let rowIndex = 0; rowIndex < matrixValues.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < matrixValues[0].length; columnIndex += 1) {
      const cell = matrixTable.getCell(rowIndex, columnIndex);
      if (rowIndex === 0 || columnIndex === 0) {
        cell.fill = C.deep;
        cell.text.style = {
          typeface: family,
          fontSize: rowIndex === 0 ? 9 : 12,
          bold: true,
          color: C.white,
          alignment: "center",
          verticalAlignment: "middle",
          autoFit: "shrinkText",
        };
        continue;
      }
      const uiType = rowIndex === 1 ? "icon" : "text";
      const actionFamily = familyLabels[columnIndex - 1];
      const value = matrixLookup.get(`${uiType}:${actionFamily}`).accuracy;
      const fraction = Math.max(0, Math.min(1, (value - 0.30) / 0.60));
      cell.fill = mixHex(C.orangeSoft, "#DCEFE7", fraction);
      cell.text.style = {
        typeface: family,
        fontSize: 12,
        bold: true,
        color: value < 0.45 ? "#A33D29" : C.ink,
        alignment: "center",
        verticalAlignment: "middle",
        autoFit: "shrinkText",
      };
    }
  }
  footer(slide, "Raw point-in-box accuracy. Area standardization uses 10 pooled target-area bins; application, action, density, aspect ratio, and label noise remain uncontrolled.");
  slide.speakerNotes.textFrame.setText("1:15–2:10 — This is the complete local replication: 1,581 items, 1,042 strict point-in-box successes, no request failures, and 100% valid JSON. The 65.9% result is 0.6 percentage point below the approximately 66.5% Holo3.1-4B reference supplied by the project owner. ScreenSpot-Pro's UI type describes the annotated click target: icon means no text hint is present; targets with text labels belong to the text category even when an icon is also present. Raw accuracy is 79.4% for text targets and 44.0% for icon targets, but the comparison is confounded by target geometry. The median normalized text-target area is 4.77 times the median icon-target area. Median heights are similar at 25 versus 24 pixels, while median widths are 111 versus 26 pixels. Direct standardization over ten pooled target-area rank bins reduces the text-minus-icon gap from 35.4 to 17.4 percentage points. This coarse adjustment does not control application, platform, action family, aspect ratio, surrounding visual density, or annotation noise. The bottom matrix remains a diagnostic raw breakdown. Its action columns use our first-recognized-verb dictionary rather than an official ScreenSpot-Pro label. Sources: data/remote-results/run/screenspot-full-official/summary.json, artifacts/screenspot-presentation/screenspot-target-size-analysis.json, scripts/analyze_screenspot_target_size.py, and https://openreview.net/pdf?id=BS75DmIsRL.");
}

// 4 — method
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Calculation", "From generated coordinate tokens back to image patches", 4);
  textBox(slide, "A attention weights · V value vectors · R rolled-out token influence · M image-patch map for generated x/y value tokens", 64, 118, 1152, 22, { fontSize: 13, color: C.muted });
  const blocks = [
    ["1", "VALUE-NORM", "A′ = normalize(A ⊙ ‖V‖₂)", "Suppress high attention paths carrying little value signal", C.orangeSoft, C.orange],
    ["2", "CROSS-LAYER ROLLOUT", "Rₗ = ½I + ½A′ₗ\nR = Rₗ ··· R₁", "Compose all 8 captured full-attention blocks with residual flow", C.blueSoft, C.blue],
    ["3", "TOKEN → PARAMETER", "M(x,y) = meanₛ Rₛ", "Average x/y value-token maps; token length cannot inflate a parameter", "#E4F4EE", C.green],
  ];
  blocks.forEach(([num, label, equation, desc, fill, accent], i) => {
    const x = 64 + i * 395;
    rect(slide, x, 154, 356, 230, C.white, true, C.line);
    pill(slide, num, x + 22, 175, 46, accent, C.white);
    textBox(slide, label, x + 84, 179, 230, 22, { fontSize: 13, bold: true, color: accent });
    rect(slide, x + 22, 220, 312, 72, fill, true);
    textBox(slide, equation, x + 36, 232, 284, 48, { fontSize: 20, bold: true, alignment: "center", verticalAlignment: "middle" });
    textBox(slide, desc, x + 25, 316, 304, 52, { fontSize: 15, color: C.muted, alignment: "center" });
  });
  textBox(slide, "Two baselines ask different questions", 64, 422, 500, 30, { fontSize: 22, bold: true });
  rect(slide, 64, 468, 548, 156, C.deep, true);
  textBox(slide, "CAUSAL PREVIOUS-TOKEN", 88, 490, 260, 20, { fontSize: 13, bold: true, color: "#9FC8F0" });
  textBox(slide, "Dtoken = M(x,y) − mean(M(s < start(x)))", 88, 526, 472, 28, { fontSize: 19, bold: true, color: C.white });
  textBox(slide, "What is above the generation state already present before the coordinate sequence?", 88, 567, 470, 38, { fontSize: 14, color: "#B9C3CD" });
  rect(slide, 650, 468, 566, 156, C.white, true, C.line);
  textBox(slide, "SAME-IMAGE INSTRUCTION ENSEMBLE", 676, 490, 330, 20, { fontSize: 13, bold: true, color: C.orange });
  textBox(slide, "Dprompt = Mtarget − mean(Mcontrol,k)", 676, 526, 470, 28, { fontSize: 19, bold: true });
  textBox(slide, "What is specific to this instruction after subtracting inherent image saliency?", 676, 567, 470, 38, { fontSize: 14, color: C.muted });
  footer(slide, "Target lift = positive target-minus-control mass inside the target box ÷ target-box area fraction. 1× means spatially uniform residual mass.");
  slide.speakerNotes.textFrame.setText("2:10–3:20 — Define the symbols once. A is the attention-weight matrix. V contains the value vectors. R is the residual-aware rollout across the eight captured full-attention blocks. M is the resulting image-patch map traced from the generated x/y coordinate-value tokens. Value-norm changes the routing matrix before rollout. Rollout mixes attention and identity 50/50 and multiplies the captured layers in model order. Then average only the generated x/y value-token maps. Target lift is the positive target-minus-control mass inside the target box divided by the target-box area fraction; 1× means spatially uniform positive residual mass. Hybrid linear-attention blocks are omitted because they expose no equivalent square softmax matrix.");
}

// 5 — success raw vs prompt difference
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Descriptive attribution", "Same-image controls reveal the selected template", 5);
  textBox(slide, "“Create a Psychedelic vibrant presentation”", 64, 122, 780, 30, { fontSize: 19, bold: true, color: C.muted });
  pill(slide, "WHITE RING = ISSUED CLICK", 940, 119, 276, C.deep, C.white);
  await image(slide, successRaw, 64, 174, 548, 342, { alt: "Raw value-norm rollout on success case", crop: { left: 0, top: 0.02, right: 0.30, bottom: 0.31 } });
  await image(slide, successDiff, 668, 174, 548, 342, { alt: "Target minus diverse-instruction baseline on success case", crop: { left: 0, top: 0.02, right: 0.30, bottom: 0.31 } });
  pill(slide, "RAW ROLLOUT", 84, 192, 122, C.deep, C.white);
  pill(slide, "TARGET − PROMPT BASELINE", 688, 192, 228, C.orange, C.white);
  metricCard(slide, 64, 540, 252, "Target lift", "3.85×", C.muted, "raw rollout");
  metricCard(slide, 330, 540, 282, "Peak", "outside", C.muted, "generic top-left saliency");
  metricCard(slide, 668, 540, 252, "Target lift", "16.08×", C.orange, "instruction differential");
  metricCard(slide, 934, 540, 282, "Peak", "inside", C.green, "3.2% of frame diagonal away");
  footer(slide, "Local trace: powerpoint_windows_59 · 4 same-image controls · LOO cosine 0.925 mean / 0.828 minimum");
  slide.speakerNotes.textFrame.setText("3:20–4:20 — This slide establishes the measurement we use from here onward. The raw map is dominated by stable top-left saliency. With four diverse same-image prompts averaged after per-request L1 normalization, the peak moves inside the annotated template and lift rises from 3.85× to 16.08×. Leave-one-control-out cosine is 0.925 mean and 0.828 minimum. The repaired click is visually correct at (1394, 632), but the original tool syntax is malformed, so official correctness remains false. Source: data/attributions/screenspot-powerpoint_windows_59/analysis.json.");
}

// 6 — failure
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Descriptive attribution", "The map reaches the task region but misses the New Slide control", 6);
  textBox(slide, "Instruction: Create new slide", 80, 126, 500, 28, { fontSize: 19, bold: true, color: C.muted });
  pill(slide, "WHITE RING = ISSUED CLICK", 558, 121, 276, C.deep, C.white);
  await image(slide, failureDiff, 64, 164, 770, 438, { alt: "Prompt differential saliency for the Create new slide failure", crop: { left: 0, top: 0, right: 0.34, bottom: 0.43 } });
  rect(slide, 872, 164, 344, 438, C.deep, true);
  pill(slide, "GROUNDING MISS", 900, 190, 150, C.orange, C.white);
  textBox(slide, "The differential map concentrates in the correct top-left task region…", 900, 246, 270, 84, { fontSize: 23, bold: true, color: C.white });
  textBox(slide, "…but its peak lands on the slide thumbnail, not the tiny New Slide button.", 900, 346, 270, 74, { fontSize: 18, color: "#C5CFD8" });
  rect(slide, 900, 452, 270, 1, "#465563");
  textBox(slide, "Predicted click", 900, 474, 132, 20, { fontSize: 12, bold: true, color: "#91A0AE" });
  textBox(slide, "(331, 270)", 1034, 471, 136, 26, { fontSize: 20, bold: true, color: C.white, alignment: "right" });
  textBox(slide, "Prompt-diff peak", 900, 514, 142, 20, { fontSize: 12, bold: true, color: "#91A0AE" });
  textBox(slide, "9.9% diag away", 1032, 511, 138, 26, { fontSize: 18, bold: true, color: C.orange, alignment: "right" });
  textBox(slide, "Attention reaches the task region, while precise target binding remains off-control.", 900, 554, 270, 34, { fontSize: 13, color: "#F1C7BC" });
  footer(slide, "Local trace: powerpoint_windows_48 · prompt-baseline stability 0.970 mean / 0.953 minimum");
  slide.speakerNotes.textFrame.setText("4:20–5:15 — From this point, headline the same-image instruction differential rather than raw attention. The differential favors the top-left slide and ribbon region, yet its peak is on the slide thumbnail and the click lands there. The target itself is only 0.074% of the frame. This is compatible with fine-grained binding or localization failure after coarse semantic routing. Source: data/attributions/screenspot-powerpoint_windows_48/analysis.json.");
}

// 7 - multi-turn trajectory
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Descriptive attribution", "Target-card saliency persists, but the click misses", 7);
  textBox(slide, "Target minus same-history instruction ensemble for the final x/y coordinate tokens", 64, 122, 930, 25, { fontSize: 15, color: C.muted });
  pill(slide, "WHITE RING = CLICK", 1000, 119, 216, C.deep, C.white);
  const labels = ["FRAME 0 · START", "FRAME 1 · SCROLL 500", "FRAME 2 · TARGET APPEARS", "FRAME 3 · CLICK"];
  for (let i = 0; i < hotelDiffMaps.length; i += 1) {
    const x = 64 + i * 286;
    await image(slide, hotelDiffMaps[i], x, 166, 270, 169, { alt: `Hotel trajectory prompt-differential frame ${i}` });
    pill(slide, labels[i], x + 10, 346, 220, i === 3 ? C.orange : C.deep, C.white);
  }
  clickMarker(slide, 922 + (960 / 1280) * 270, 166 + (120 / 800) * 169, 16);
  metricCard(slide, 64, 398, 350, "Frame 2 target-card lift", "2.78×", C.blue, "target button at viewport edge");
  metricCard(slide, 432, 398, 350, "Frame 3 target-card lift", "8.59×", C.orange, "current-frame prompt differential");
  metricCard(slide, 800, 398, 416, "Issued click", "MISS", C.orange, "(960,120): correct card, wrong affordance");
  rect(slide, 64, 530, 1152, 108, C.deep, true);
  textBox(slide, "Interpretation", 90, 551, 150, 22, { fontSize: 13, bold: true, color: C.gold });
  textBox(slide, "Target-card evidence persists across frames. Final control localization still fails.", 90, 579, 680, 34, { fontSize: 23, bold: true, color: C.white });
  textBox(slide, "Shifted-layout follow-up: £122 gets 4.12× causal lift vs 0.81× for £135, yet the malformed click still misses.", 800, 548, 374, 66, { fontSize: 14, color: "#C4CED7" });
  footer(slide, "Resolution caveat: 1280×800 source → 640×384 vision raster → 20×12 merged-token map · button ≈ 1.6×1.0 token");
  slide.speakerNotes.textFrame.setText("5:15–6:25 — Every probe receives the identical four screenshots and identical three-scroll action history. Only the requested click target changes. We normalize each request across every patch in all four frames, average three usable control maps, then subtract. The cheapest hotel button has 2.78× lift when it first appears at the bottom of frame 2 and 8.59× in the current frame. Yet the peak stays outside the button and Holo clicks (960,120), on the Signal Quay card rather than View details. Resolution is an important confound: after patch merging, this 105×65-pixel control spans only about 1.6×1.0 visual tokens, so coarse retrieval can survive while precise binding degrades. The four-frame history and longer prompt also co-vary, so this case does not isolate resolution causally. In a separate single-frame diagnostic, frozen item test-0035 moves the price column from roughly 80% to 58% of viewport width and enlarges the UI. The £122 target price receives 4.12× causal previous-token lift for x versus 0.81× on the £135 runner-up; y is 4.01× versus 0.91×. Holo still emits malformed, off-target coordinates. This supports position-robust price selection but is not a same-image prompt-baseline proof. Three controls are included in the multi-frame map; one is excluded because generation omitted a y span. LOO cosine is 0.898 mean / 0.769 minimum. Sources: data/attributions/hotel-cheapest-multiframe-contrast/analysis.json and data/attributions/hotel-test-0035-large-ui-final/target-viewer/viewer.html.");
}

// 8 - balanced multi-item layer/head aggregate
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Pilot aggregate · n=4", "Layer 19 has the highest mean lift, narrowly ahead of Layer 15", 8);
  textBox(slide, "Target lift uses four visible alternate click tasks on the same screenshot. Official harness: 2 hits and 2 misses.", 64, 118, 1152, 25, { fontSize: 13, color: C.muted });

  const layerRows = [...balancedLayer19.layers].sort((a, b) => a.transformer_layer - b.transformer_layer);
  const layerMax = Math.max(...layerRows.map((row) => row.mean));
  textBox(slide, "MEAN TARGET LIFT ACROSS 8 CAPTURED FULL-ATTENTION LAYERS", 64, 157, 626, 20, { fontSize: 11, bold: true, color: C.blue });
  layerRows.forEach((row, index) => {
    const y = 188 + index * 32;
    const focused = row.transformer_layer === 19;
    textBox(slide, `L${row.transformer_layer}`, 64, y + 2, 40, 20, { fontSize: 12, bold: true, color: focused ? C.orange : C.ink });
    rect(slide, 110, y, 486, 22, "#E6E1D7", true);
    rect(slide, 110, y, Math.max(4, 486 * row.mean / layerMax), 22, focused ? C.orange : C.deep, true);
    textBox(slide, `${row.mean.toFixed(1)}×`, 608, y + 1, 70, 20, { fontSize: 12, bold: true, color: focused ? C.orange : C.ink, alignment: "right" });
  });

  rect(slide, 714, 156, 502, 286, C.deep, true);
  textBox(slide, "WHAT REPLICATED", 740, 178, 220, 20, { fontSize: 11, bold: true, color: C.gold });
  textBox(slide, "L19 ranks #1 of 8", 740, 207, 290, 34, { fontSize: 27, bold: true, color: C.white });
  textBox(slide, "158.4× mean vs 151.4× at L15", 740, 245, 430, 24, { fontSize: 14, color: "#C4CED7" });
  textBox(slide, "L15 and L19 are both full-attention layers.", 740, 267, 430, 20, { fontSize: 12, bold: true, color: "#9FC8F0" });
  rect(slide, 740, 294, 450, 1, "#465563");
  textBox(slide, "TOP HEADS FORM AN L19 CLUSTER", 740, 307, 300, 20, { fontSize: 11, bold: true, color: C.gold });
  textBox(slide, "H11 312.1×   H14 293.8×", 740, 333, 420, 24, { fontSize: 18, bold: true, color: C.white });
  textBox(slide, "H10 264.7×   H2 245.6×", 740, 362, 420, 24, { fontSize: 18, bold: true, color: C.white });
  textBox(slide, "H10 ranks #3 among 16 heads within L19 and never ranks first on an individual item.", 740, 397, 430, 32, { fontSize: 12, color: "#C4CED7" });

  const itemValues = [
    ["ITEM", "OUTCOME", "L19 LIFT", "L19 RANK", "L19/H10", "H10 RANK"],
    ...balancedLayer19.per_item.map((row) => [
      row.label.replace(" success", "").replace(" failure", ""),
      row.benchmark_correct ? "HIT" : "MISS",
      `${row.focus_layer_target_lift.toFixed(1)}×`,
      `#${row.focus_layer_rank}`,
      `${row.focus_head_target_lift.toFixed(1)}×`,
      `#${row.focus_head_rank_within_layer}`,
    ]),
  ];
  const table = slide.tables.add({
    rows: itemValues.length,
    columns: itemValues[0].length,
    left: 64,
    top: 468,
    width: 1152,
    height: 142,
    values: itemValues,
    columnWidths: [284, 130, 180, 170, 190, 198],
  });
  table.cells.block({ row: 0, column: 0, rowCount: itemValues.length, columnCount: itemValues[0].length }).assign({
    margins: { left: 8, right: 8, top: 2, bottom: 2 },
  });
  table.borders.assign({ style: "solid", fill: C.paper, width: 1 });
  for (let rowIndex = 0; rowIndex < itemValues.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < itemValues[0].length; columnIndex += 1) {
      const cell = table.getCell(rowIndex, columnIndex);
      cell.fill = rowIndex === 0 ? C.deep : rowIndex % 2 === 0 ? "#ECE7DD" : C.white;
      if (rowIndex > 0 && columnIndex === 1) cell.fill = itemValues[rowIndex][1] === "HIT" ? "#DCEFE7" : C.orangeSoft;
      cell.text.style = {
        typeface: family,
        fontSize: rowIndex === 0 ? 10 : 12,
        bold: rowIndex === 0 || columnIndex === 1,
        color: rowIndex === 0 ? C.white : C.ink,
        alignment: columnIndex === 0 ? "left" : "center",
        verticalAlignment: "middle",
        autoFit: "shrinkText",
      };
    }
  }
  footer(slide, "Captured full-attention layers: L3, L7, L11, L15, L19, L23, L27, L31. Intervening Gated DeltaNet layers are outside this rollout.");
  slide.speakerNotes.textFrame.setText("6:25–7:30 — This slide replaces the earlier three selected matrices with the balanced official-protocol pilot. Each item uses four visible, task-valid alternate click instructions on the same screenshot and the same 2,097,152-pixel eager-attention profile. Layer rank means rank among the eight captured full-attention layers. H10 rank means rank among the 16 heads within Layer 19. Holo3.1-4B uses full attention at zero-based layers 3, 7, 11, 15, 19, 23, 27, and 31, so both Layer 15 and Layer 19 are full-attention layers. The intervening layers use Gated DeltaNet and are outside this rollout. Across two hits and two misses, Layer 19 has the highest mean at 158.4×, narrowly ahead of Layer 15 at 151.4×. Layer 19 is top on two items and has median item rank 2. H11, H14, H10, and H2 form the strongest aggregate cluster. H10 ranks third within Layer 19 and never ranks first on an individual item. This is descriptive routing evidence from a four-item pilot, not a stable Layer 19 mechanism. Sources: data/remote-results/attention-layer19-balanced-v1/aggregate/aggregate.json and https://huggingface.co/Hcompany/Holo-3.1-4B/blob/main/config.json.");
}

// 9 - matched causal setup
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Causal intervention · one matched case", "Matched tile swap reverses the teacher-forced coordinate margin", 9);
  textBox(slide, "Instruction: Create a Psychedelic vibrant presentation", 64, 120, 820, 26, { fontSize: 17, bold: true, color: C.muted });
  await image(slide, causalClean, 64, 164, 548, 258, { alt: "Clean PowerPoint template gallery focused on Woven fibers and Psychedelic vibrant" });
  await image(slide, causalCorrupted, 668, 164, 548, 258, { alt: "Focused PowerPoint gallery after swapping the target and Woven fibers tiles" });
  pill(slide, "CLEAN", 84, 184, 78, C.deep, C.white);
  pill(slide, "0.91% OF PIXELS SWAPPED", 688, 184, 198, C.orange, C.white);
  textBox(slide, "Requested tile at (482,351)", 64, 430, 548, 22, { fontSize: 15, bold: true, color: C.ink, alignment: "center" });
  textBox(slide, "Requested tile moves to (406,351)", 668, 430, 548, 22, { fontSize: 15, bold: true, color: C.ink, alignment: "center" });
  rect(slide, 64, 462, 1152, 154, C.deep, true);
  textBox(slide, "TEACHER-FORCED COORDINATE MARGIN", 90, 481, 390, 20, { fontSize: 13, bold: true, color: C.gold });
  textBox(slide, "margin = log p(correct coordinate tokens) − log p(distractor coordinate tokens)", 90, 512, 770, 30, { fontSize: 20, bold: true, color: C.white });
  textBox(slide, "+1 nat multiplies the target-to-distractor probability ratio by e ≈ 2.72", 90, 550, 770, 22, { fontSize: 15, bold: true, color: C.gold });
  textBox(slide, "Six coordinate tokens per candidate; syntax excluded; later tokens use each candidate’s own prefix", 90, 580, 770, 20, { fontSize: 12, color: "#BBC6D0" });
  metricCard(slide, 888, 485, 134, "Clean", "+1.70", C.green, "5.47× target");
  metricCard(slide, 1038, 485, 150, "Corrupted", "−2.64", C.orange, "14.07× distractor");
  footer(slide, "Lossless equal-size tile swap · 0.91% of pixels changed · localized perturbation may alter recognition and coordinate binding");
  slide.speakerNotes.textFrame.setText("7:30–8:20 — Introduce the matched causal pair. The pixel corruption crops both equal-size tiles from the original before either paste, then exchanges them without resizing or resampling; 0.91% of pixels change. The swap creates a localized visual counterfactual while preserving task wording and layout. It may alter recognition as well as coordinate binding. The teacher-forced coordinate margin is the log probability of the correct six x/y value tokens minus the log probability of the distractor coordinate tokens. Syntax tokens are excluded. A nat is a natural-log unit, so exp(margin) is the target-to-distractor sequence-probability ratio. +1.70 nats means 5.47× target preference. −2.64 nats means 14.07× distractor preference. Source: artifacts/causal-intervention/powerpoint_windows_59_swap/results.json.");
}

// 10 - causal results
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Causal intervention · one matched case", "Target-region residuals recover one-third of the coordinate margin", 10);
  const cleanSource = rect(slide, 64, 122, 202, 34, C.orangeSoft, true, C.orange);
  const corruptDestination = rect(slide, 300, 122, 216, 34, C.blueSoft, true, C.blue);
  const marginEffect = rect(slide, 550, 122, 258, 34, "#E4F4EE", true, C.green);
  textBox(slide, "Clean activation source", 76, 128, 178, 20, { fontSize: 13, bold: true, color: C.orange, alignment: "center" });
  textBox(slide, "Corrupted run destination", 312, 128, 192, 20, { fontSize: 13, bold: true, color: C.blue, alignment: "center" });
  textBox(slide, "Change in coordinate margin", 562, 128, 234, 20, { fontSize: 13, bold: true, color: C.green, alignment: "center" });
  // artifact_tool places `head` at the source end, so connect in reverse to render
  // the intended left-to-right activation-patching flow.
  slide.shapes.connect(corruptDestination, cleanSource, { kind: "straight", fromSide: "left", toSide: "right", line: { style: "solid", fill: C.muted, width: 2 }, head: { type: "arrow", width: "med", length: "med" } });
  slide.shapes.connect(marginEffect, corruptDestination, { kind: "straight", fromSide: "left", toSide: "right", line: { style: "solid", fill: C.muted, width: 2 }, head: { type: "arrow", width: "med", length: "med" } });
  const ordered = [
    "layer_19_visual_token_residuals",
    "layer_19_target_patch_residuals",
    "layer_19_head_10_attention_output",
  ].map((name) => causal.interventions.find((row) => row.component === name));
  const chart = slide.charts.add("bar", {
    position: { left: 64, top: 180, width: 744, height: 390 },
    categories: ["All visual residuals", "Target-patch residuals", "Isolated L19 / H10"],
    series: [
      { name: "Clean activation restoration", values: ordered.map((row) => Number(row.restoration_nats.toFixed(3))), fill: C.orange },
      { name: "Clean-run ablation drop", values: ordered.map((row) => Number(row.ablation_drop_nats.toFixed(3))), fill: C.blue },
    ],
    barOptions: { direction: "column", grouping: "clustered" },
    hasLegend: true,
    legend: { position: "bottom" },
    yAxis: { title: "Change in coordinate margin (nats)", minimumScale: -0.2, maximumScale: 2.0, majorUnit: 0.5 },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.00" },
    chartFill: C.white,
    chartLine: { fill: C.line, width: 1 },
    plotAreaFill: C.white,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(chart, { fontFamily: family });
  textBox(slide, "Other restoration probes: distractor −0.13 · MLP15 +0.03 · MLP19 +0.12 · MLP23 +0.20 nats", 64, 585, 744, 20, { fontSize: 12, color: C.muted, alignment: "center" });
  rect(slide, 840, 180, 376, 410, C.deep, true);
  pill(slide, "TARGET REGION", 870, 206, 146, C.green, C.white);
  textBox(slide, "33.2%", 870, 252, 150, 44, { fontSize: 38, bold: true, color: C.white });
  textBox(slide, "of the 4.34-nat gap recovered by four target patches", 870, 296, 300, 45, { fontSize: 17, color: "#BDE7D8" });
  textBox(slide, "+0.024 nats", 870, 358, 210, 35, { fontSize: 28, bold: true, color: C.white });
  textBox(slide, "restored by isolated L19/H10", 870, 396, 300, 24, { fontSize: 16, color: "#F2BDAF" });
  rect(slide, 870, 438, 316, 1, "#465563");
  pill(slide, "CAUSAL READING", 870, 456, 150, C.gold, C.white);
  textBox(slide, "Regional, not head-local", 870, 500, 306, 32, { fontSize: 24, bold: true, color: C.white });
  textBox(slide, "Target residuals pass restoration and ablation; the selected head alone does not.", 870, 540, 306, 42, { fontSize: 14, color: "#C4CED7" });
  footer(slide, "Causal bars: one matched tile swap · activation patching replaces one corrupted-run tensor with its clean counterpart");
  slide.speakerNotes.textFrame.setText("8:20–9:15 — The schematic defines activation patching: capture one component from the clean run, replace the corresponding component in the corrupted run, then measure the change in teacher-forced coordinate margin. Patching all clean layer-19 visual-token residuals restores 1.85 nats, or 42.7% of the clean–corrupt gap. Restricting the patch to four target-overlapping tokens restores 1.44 nats, or 33.2%. Mean-replacing those same clean tokens removes 1.48 nats. The isolated layer-19/head-10 attention output restores only 0.024 nats, so the causal claim stays regional and residual-stream level for this one matched case. The chart leads with those three contrasts; the distractor and MLP probes remain in the secondary line. Source: artifacts/causal-intervention/powerpoint_windows_59_swap/results.json.");
}

// 11 - standalone Qwen-to-Holo delta lens
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Fine-tuning delta lens", "Holo changes both action preference and visual routing", 11);
  textBox(slide, "Identical Holo processor, official prompts, images, and teacher-forced action tokens for Qwen3.5-4B and Holo3.1-4B", 64, 114, 1152, 22, { fontSize: 13, color: C.muted });

  const panelCrop = { left: 0, top: 0.04, right: 0.29, bottom: 0.38 };
  const panelWidth = 360;
  const panelTop = 154;
  const panelHeight = 190;
  const panelLefts = [64, 460, 856];
  const panelLabels = [
    `QWEN · ${(directTargetDelta.base_target_mass * 100).toFixed(1)}% TARGET MASS`,
    `HOLO · ${(directTargetDelta.tuned_target_mass * 100).toFixed(1)}% · PEAK ENTERS TARGET`,
    "HOLO − QWEN · ORANGE GAIN / BLUE LOSS",
  ];
  await image(slide, qwenAttention, panelLefts[0], panelTop, panelWidth, panelHeight, { alt: "Qwen direct teacher-forced attention on the Psychedelic vibrant template", crop: panelCrop });
  await image(slide, holoAttention, panelLefts[1], panelTop, panelWidth, panelHeight, { alt: "Holo direct teacher-forced attention on the Psychedelic vibrant template", crop: panelCrop });
  await image(slide, attentionDifference, panelLefts[2], panelTop, panelWidth, panelHeight, { alt: "Signed Holo minus Qwen direct-attention redistribution", crop: panelCrop });
  panelLabels.forEach((label, index) => pill(slide, label, panelLefts[index] + 10, panelTop + 10, panelWidth - 20, index === 2 ? C.deep : index === 1 ? C.orange : C.blue, C.white));

  textBox(slide, "ACTION-MARGIN DELTA BY LAYER", 64, 370, 520, 18, { fontSize: 11, bold: true, color: C.blue });
  const layerChart = slide.charts.add("line", {
    position: { left: 64, top: 392, width: 720, height: 214 },
    categories: deltaLayerLabels,
    series: [
      { name: "ScreenSpot mean", values: screenMeanLayerDeltas, line: { fill: C.orange, width: 3 }, marker: { style: "none" } },
      { name: "Hotel final click", values: hotelFinalLayerDeltas, line: { fill: C.blue, width: 2 }, marker: { style: "none" } },
    ],
    hasLegend: true,
    legend: { position: "bottom" },
    yAxis: { title: "Holo − Qwen margin (nats/token)", minimumScale: -3.5, maximumScale: 2.5, majorUnit: 1 },
    xAxis: { title: "Transformer layer" },
    chartFill: C.white,
    chartLine: { fill: C.line, width: 1 },
    plotAreaFill: C.white,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(layerChart, { fontFamily: family });

  textBox(slide, "HOTEL FINAL STEP · VALUE-NORM ATTENTION", 816, 370, 400, 18, { fontSize: 11, bold: true, color: C.blue });
  const hotelChart = slide.charts.add("bar", {
    position: { left: 816, top: 392, width: 400, height: 214 },
    categories: ["Earliest", "Middle", "Current"],
    series: [
      { name: "Qwen", values: hotelValueFrames.base_frame_mass.map((value) => Number((value * 100).toFixed(1))), fill: C.blue },
      { name: "Holo", values: hotelValueFrames.tuned_frame_mass.map((value) => Number((value * 100).toFixed(1))), fill: C.orange },
    ],
    barOptions: { direction: "column", grouping: "clustered" },
    hasLegend: true,
    legend: { position: "bottom" },
    yAxis: { title: "% of image attention", minimumScale: 0, maximumScale: 70, majorUnit: 10 },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0" },
    chartFill: C.white,
    chartLine: { fill: C.line, width: 1 },
    plotAreaFill: C.white,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(hotelChart, { fontFamily: family });

  rect(slide, 64, 620, 1152, 38, C.deep, true);
  textBox(slide, `${screenPositiveFinalCount}/${screenFinalLayers.length} ScreenSpot final shifts positive · mean +${screenMeanFinalDelta.toFixed(2)} nats/token · ${screenBoundaryFlips} cross zero`, 82, 628, 540, 22, { fontSize: 13, bold: true, color: C.white });
  textBox(slide, `Method check: direct target lift +${directTargetDelta.delta_target_lift.toFixed(2)}×; value-norm target lift ${valueTargetDelta.delta_target_lift.toFixed(2)}×`, 642, 628, 552, 22, { fontSize: 13, bold: true, color: "#F5E2B8", alignment: "right" });
  footer(slide, "Native input ceiling: ScreenSpot retains 99.6% area; hotel frames retain 97.8% after patch alignment · attention is descriptive, not causal");
  slide.speakerNotes.textFrame.setText("9:15–10:20 — This is the standalone fine-tuning comparison. The layer chart teacher-forces identical oracle and distractor action tokens through the official Qwen3.5-4B base and Holo3.1-4B, using the same Holo processor, prompts, image order, and action suffixes. All five ScreenSpot probes end with a positive Holo-minus-Qwen oracle-margin shift, averaging 1.14 nats per scored token; three cross from a negative base margin to a positive Holo margin. The hotel final click ends at +0.37 nats per token, while earlier hotel steps are mixed. The top triptych is a separate teacher-forced attention diagnostic on powerpoint_windows_59. It averages the six oracle coordinate-token query rows across 16 heads and all eight full-attention layers, then normalizes over image patches. Direct target mass rises from 15.8% to 17.2%, and the direct-attention peak moves into the annotated target. Value-norm weighting is not directionally identical: target lift falls by 1.07, so the defensible conclusion is redistribution, not uniform concentration. In the three-frame hotel prompt, Holo moves value-norm image attention away from the earliest frame and toward the middle and current frames. The ScreenSpot processor grid is 56 by 90 from a 2880 by 1800 source, retaining 99.6% of source area; each hotel frame uses a 22 by 32 grid from 1024 by 720, retaining 97.8%. Attention is descriptive; the layerwise logit lens also includes each checkpoint's own final norm and unembedding. Sources: data/remote-results/delta-lens-qwen35-vs-holo31-20260921-dcd8c5e5, data/local-results/attention-delta-ppt59-native-v2, and data/local-results/attention-delta-hotel-step2-native-v1.");
}

// 12 - research directions
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Research directions", "From one intervention to a causal evaluation set", 12);
  const directions = [
    ["01", "MATCHED INTERVENTION SPLIT", "Freeze a diverse set of target–distractor swaps, pre-register coordinate margins, and report effect distributions with bootstrap intervals.", C.orangeSoft, C.orange],
    ["02", "CAUSAL SUBSPACE SEARCH", "Test multi-head and low-rank residual subspaces. A single salient head was neither necessary nor sufficient in this case.", C.blueSoft, C.blue],
    ["03", "LONG-HORIZON VISUAL MEMORY", "Patch target evidence across historical frames to identify when retrieval succeeds but precise action binding fails.", "#E4F4EE", C.green],
    ["04", "DELTA-LENS DECOMPOSITION", "Separate residual changes from final norm and unembedding changes. Scale beyond five ScreenSpot probes and one hotel trace.", "#F3EBDD", C.gold],
  ];
  directions.forEach(([num, title, desc, fill, accent], i) => {
    const left = i % 2 === 0 ? 64 : 650;
    const top = i < 2 ? 148 : 350;
    rect(slide, left, top, 566, 176, C.white, true, C.line);
    pill(slide, num, left + 22, top + 22, 48, accent, C.white);
    rect(slide, left + 88, top + 22, 446, 31, fill, true);
    textBox(slide, title, left + 102, top + 28, 418, 20, { fontSize: 13, bold: true, color: accent });
    textBox(slide, desc, left + 26, top + 76, 512, 78, { fontSize: 16, color: C.ink });
  });
  rect(slide, 64, 550, 1152, 88, C.deep, true);
  textBox(slide, "TRAINING LOOP", 90, 570, 156, 20, { fontSize: 13, bold: true, color: C.gold });
  textBox(slide, "Oracle SFT, followed by measured causal recovery and GRPO + LoRA with verifiable reward", 264, 566, 902, 30, { fontSize: 21, bold: true, color: C.white, alignment: "center" });
  textBox(slide, "Use attribution to choose interventions, not as the reward target.", 264, 602, 902, 20, { fontSize: 14, color: "#B9C3CD", alignment: "center" });
  footer(slide, "Scale the matched protocol before making a population-level mechanism claim");
  slide.speakerNotes.textFrame.setText("10:20–11:10 — The single-case intervention and eight-case delta lens change the agenda. Scale the matched protocol across a frozen split and report effect distributions. Search causal subspaces because the salient head was not sufficient by itself. Extend patching across historical frames to separate memory retrieval from action binding. For the delta lens, separate residual-stream changes from changes in each checkpoint’s final norm and unembedding, then expand beyond five ScreenSpot probes and one hotel trace. The training program remains oracle SFT, measured causal effects, then GRPO plus LoRA with verifiable reward. Do not optimize the heatmap itself.");
}

// 13 - takeaways and demo
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  slideTitle(slide, "Takeaways", "Attribution narrows the search and intervention tests the mechanism", 13, true);
  const takeaways = [
    ["01", "Remove general awareness", "Same-image instruction controls make target-specific routing visible."],
    ["02", "Separate routing from mechanism", "The strongest attention head did not carry the coordinate effect by itself."],
    ["03", "Patch the target evidence", "Four target residual tokens recover one-third of the matched corruption gap."],
  ];
  takeaways.forEach(([num, title, desc], i) => {
    const y = 150 + i * 122;
    pill(slide, num, 64, y, 52, i === 0 ? C.orange : C.blue, C.white);
    textBox(slide, title, 138, y + 1, 430, 26, { fontSize: 20, bold: true, color: C.white });
    textBox(slide, desc, 138, y + 35, 430, 48, { fontSize: 15, color: "#B7C1CB" });
  });
  rect(slide, 646, 146, 570, 414, C.white, true);
  textBox(slide, "LIVE DEMO · 90 SECONDS", 676, 174, 320, 22, { fontSize: 13, bold: true, color: C.orange });
  const demo = [
    ["1", "Static hit: raw versus prompt difference", "Peak enters the selected template."],
    ["2", "Static miss: inspect click binding", "Correct task region, wrong tiny affordance."],
    ["3", "Hotel trace: step through four frames", "Evidence appears before the failed final click."],
    ["4", "Matched tile swap", "Compare target-patch recovery with the head-level null."],
  ];
  demo.forEach(([num, title, desc], i) => {
    const y = 218 + i * 78;
    pill(slide, num, 676, y, 38, i === 3 ? C.orange : C.deep, C.white);
    textBox(slide, title, 730, y + 1, 430, 24, { fontSize: 17, bold: true });
    textBox(slide, desc, 730, y + 29, 430, 22, { fontSize: 14, color: C.muted });
  });
  rect(slide, 64, 588, 1152, 50, "#202C38", true);
  textBox(slide, "Next decisive result: repeat the matched intervention across a frozen evaluation split", 90, 598, 1098, 28, { fontSize: 20, bold: true, color: C.white, alignment: "center", verticalAlignment: "middle" });
  footer(slide, "Viewers: ScreenSpot hit + miss · causal result: artifacts/causal-intervention/powerpoint_windows_59_swap/results.json", true);
  slide.speakerNotes.textFrame.setText("11:10–12:00 — Close on three defensible conclusions. The prompt ensemble exposes target-specific routing. Correct and incorrect cases can share coarse routing while differing in precise binding. In the matched intervention, regional layer-19 residuals pass both restoration and ablation tests, while the strongest direct-attention head does not. Fine-tuning moves every ScreenSpot diagnostic margin toward the oracle, but attention weighting choices and the hotel trajectory show that internal redistribution is selective. In the live demo, show raw versus prompt differential on the hit, the static miss, the hotel frames, then the clean/corrupted pair and intervention chart. The result is one matched case, so the next claim requires a frozen evaluation split.");
}

// 14 - appendix: examples for the 2 x 6 diagnostic taxonomy
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Appendix", "Examples for the 12 labeled UI × action cells", 14);
  textBox(slide, "Rows use the benchmark's target-element label; columns use our instruction-verb taxonomy", 64, 118, 1152, 24, { fontSize: 14, color: C.muted });

  const labeledFamilies = benchmarkBreakdown.action_family_order.filter((name) => name !== "Other");
  const appendixFamilyLabels = {
    "Navigate / reveal": "NAVIGATE\nREVEAL",
    "Select / activate": "SELECT\nACTIVATE",
    "Create / insert": "CREATE\nINSERT",
    "Edit / format": "EDIT\nFORMAT",
    "Remove / close": "REMOVE\nCLOSE",
    "File / transfer": "FILE\nTRANSFER",
  };
  const exampleLookup = new Map(
    categoryExamples.examples.map((row) => [`${row.ui_type}:${row.action_family}`, row]),
  );
  const appendixMatrixLookup = new Map(
    benchmarkBreakdown.ui_action_matrix.map((row) => [`${row.ui_type}:${row.action_family}`, row]),
  );
  const exampleValues = [
    ["OFFICIAL\nTARGET UI", ...labeledFamilies.map((name) => appendixFamilyLabels[name])],
    ...["icon", "text"].map((uiType) => [
      uiType.toUpperCase(),
      ...labeledFamilies.map((actionFamily) => {
        const metric = appendixMatrixLookup.get(`${uiType}:${actionFamily}`);
        const example = exampleLookup.get(`${uiType}:${actionFamily}`);
        if (!metric || !example) throw new Error(`Missing appendix example for ${uiType}:${actionFamily}`);
        return `${(metric.accuracy * 100).toFixed(1)}%   n=${metric.count}\n\n“${example.instruction}”\n${example.application}`;
      }),
    ]),
  ];
  const examplesTable = slide.tables.add({
    rows: exampleValues.length,
    columns: exampleValues[0].length,
    left: 64,
    top: 166,
    width: 1152,
    height: 450,
    values: exampleValues,
    columnWidths: [94, ...labeledFamilies.map(() => 1058 / 6)],
    rowHeights: [54, 198, 198],
  });
  examplesTable.cells.block({ row: 0, column: 0, rowCount: exampleValues.length, columnCount: exampleValues[0].length }).assign({
    margins: { left: 9, right: 9, top: 8, bottom: 8 },
  });
  examplesTable.borders.assign({ style: "solid", fill: C.paper, width: 2 });
  for (let rowIndex = 0; rowIndex < exampleValues.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < exampleValues[0].length; columnIndex += 1) {
      const cell = examplesTable.getCell(rowIndex, columnIndex);
      if (rowIndex === 0 || columnIndex === 0) {
        cell.fill = C.deep;
        cell.text.style = {
          typeface: family,
          fontSize: rowIndex === 0 ? 10 : 13,
          bold: true,
          color: C.white,
          alignment: "center",
          verticalAlignment: "middle",
          autoFit: "shrinkText",
        };
        continue;
      }
      const uiType = rowIndex === 1 ? "icon" : "text";
      const actionFamily = labeledFamilies[columnIndex - 1];
      const value = appendixMatrixLookup.get(`${uiType}:${actionFamily}`).accuracy;
      const fraction = Math.max(0, Math.min(1, (value - 0.30) / 0.60));
      cell.fill = mixHex(C.orangeSoft, "#DCEFE7", fraction);
      cell.text.style = {
        typeface: family,
        fontSize: 12,
        bold: false,
        color: value < 0.45 ? "#A33D29" : C.ink,
        alignment: "left",
        verticalAlignment: "middle",
        autoFit: "shrinkText",
      };
    }
  }
  footer(slide, "Official UI rule: icon only when no text hint appears; text otherwise. Examples show categories, not outcomes. Other omitted: 373 items, 64.1%.");
  slide.speakerNotes.textFrame.setText("Appendix — Read across by action family and down by the benchmark's official target-element label. ScreenSpot-Pro defines icon targets as elements with no text hint; any target with a text label belongs to the text category even when it also includes an icon. The action-family columns are our deterministic instruction-verb taxonomy, not official benchmark labels. Each cell shows cohort accuracy and one real completed instruction selected for clear category membership. We replaced two visually ambiguous Blender examples after review because their released ui_type annotations appear inconsistent with the paper's stated rule. The Other bucket is omitted from the 2 × 6 grid; it contains 373 items at 64.1% accuracy. Sources: https://openreview.net/pdf?id=BS75DmIsRL, https://huggingface.co/datasets/yyyang/UI-Grounding-Benchmarks/blob/main/ScreenSpot-Pro/annotations/blender_windows.json, data/remote-results/run/screenspot-full-official/summary.json, and artifacts/screenspot-presentation/screenspot-category-examples.json.");
}

const requirements = {
  explicitTotalSlideCount: 14,
  requiredNativeTableOwnerSlides: [3, 8, 14],
  requiredNativeChartOwnerSlides: [3, 10, 11],
  materializeLiteralChartWorkbooks: true,
};
const fontPolicy = { basis: "design", families: [family] };
const stagingDir = path.join(workspaceDir, ".codex-finalizer-screenspot");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate-v22.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "3",
    "--require-native-table-slide", "8",
    "--require-native-table-slide", "14",
  ],
  requiredNativeTableOwnerSlides: [3, 8, 14],
  requiredNativeChartOwnerSlides: [3, 10, 11],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${path.basename(FINAL_PPTX)}.validation.json`),
});
console.log(JSON.stringify({ final: FINAL_PPTX, font: family, result }, null, 2));
