import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/yaoyiheng/Documents/ChatGPT/GUI VLM Fine Tuning";
const SKILL_DIR = "/Users/yaoyiheng/.codex/plugins/cache/openai-primary-runtime/presentations/26.921.11914/skills/presentations";
const TMP_DIR = path.join(workspaceDir, "artifacts/screenspot-presentation/build");
const FINAL_PPTX = path.join(workspaceDir, "artifacts/screenspot-presentation/holo-attribution-research-v55.pptx");
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
  greenSoft: "#DDEFE7",
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
const resolutionAblation = JSON.parse(
  await fs.readFile(
    path.join(workspaceDir, "artifacts/screenspot-presentation/screenspot-resolution-ablation-v1.json"),
    "utf8",
  ),
);
const resolutionSaliencyDir = path.join(
  workspaceDir,
  "artifacts/screenspot-presentation/resolution-saliency-pilot-v1",
);
const resolutionSaliency = JSON.parse(
  await fs.readFile(path.join(resolutionSaliencyDir, "summary.json"), "utf8"),
);
const resolutionSaliencyCohort = JSON.parse(
  await fs.readFile(
    path.join(workspaceDir, "artifacts/screenspot-presentation/resolution-saliency-cohort18-v1/summary.json"),
    "utf8",
  ),
);
const causalPanelStats = JSON.parse(
  await fs.readFile(
    path.join(workspaceDir, "artifacts/screenspot-presentation/causal-panel-statistics-v1.json"),
    "utf8",
  ),
);
const resolutionSaliencyMaps = [100, 75, 50, 25].map((scale) =>
  path.join(resolutionSaliencyDir, `scale-${scale}.png`),
);

const successRaw = path.join(workspaceDir, "data/attributions/screenspot-powerpoint_windows_59/preview-raw.png");
const successDiff = path.join(workspaceDir, "data/attributions/screenspot-powerpoint_windows_59/preview-target-minus-prompt-baseline.png");
const freegenNativeDir = path.join(
  workspaceDir,
  "artifacts/screenspot-presentation/native-freegen-v1",
);
const freegenNative = JSON.parse(
  await fs.readFile(path.join(freegenNativeDir, "native-freegen-attribution-summary.json"), "utf8"),
);
const freegenHit = freegenNative.samples.powerpoint_windows_63;
const freegenMiss = freegenNative.samples.powerpoint_windows_54;
const freegenHitX = path.join(freegenNativeDir, "powerpoint_windows_63/holo-value-norm-x-token.png");
const freegenHitY = path.join(freegenNativeDir, "powerpoint_windows_63/holo-value-norm-y-token.png");
const freegenHitJoint = path.join(freegenNativeDir, "powerpoint_windows_63/holo-value-norm-joint-xy.png");
const freegenMissJoint = path.join(freegenNativeDir, "powerpoint_windows_54/holo-value-norm-joint-xy.png");
const slide67NativeDir = path.join(
  workspaceDir,
  "artifacts/screenspot-presentation/slide67-native-contrast-v1",
);
const slide67Native = JSON.parse(
  await fs.readFile(path.join(slide67NativeDir, "slide67-native-contrast-summary.json"), "utf8"),
);
const slide6Hit = slide67Native.cases.powerpoint_windows_63;
const slide7Miss = slide67Native.cases.powerpoint_windows_54;
const slide6Raw = path.join(slide67NativeDir, "powerpoint_windows_63/raw-context.png");
const slide6RawMetrics = slide6Hit.metrics.raw;
const slide6PromptDifference = path.join(
  slide67NativeDir,
  "powerpoint_windows_63/prompt_difference-context.png",
);
const slide7PromptDifference = path.join(
  slide67NativeDir,
  "powerpoint_windows_54/prompt_difference-context.png",
);
const hotelPairedDir = path.join(
  workspaceDir,
  "artifacts/screenspot-presentation/hotel-freegen-paired-v1-attribution",
);
const hotelPairedTrajectory = JSON.parse(
  await fs.readFile(path.join(hotelPairedDir, "trajectory.json"), "utf8"),
);
const hotelPairedTraceDir = path.join(hotelPairedDir, "trace-20260922T132737Z-f6a7faa36844");
const hotelPairedMaps = [0, 1, 2].map((frame) => path.join(
  hotelPairedTraceDir,
  `saliency-value-norm-rollout-frame-${String(frame).padStart(3, "0")}-step-063.png`,
));
const hotelMatchedBundleDir = path.join(
  workspaceDir,
  "data/remote-results/hotel-freegen-paired-v1-20260922/hotel-freegen-paired-v1",
);
const readNormalizedActions = async (checkpoint, count) => Promise.all(
  Array.from({ length: count }, async (_, index) => JSON.parse(
    await fs.readFile(
      path.join(hotelMatchedBundleDir, checkpoint, "bundles/test-0010/actions", `${String(index).padStart(4, "0")}.json`),
      "utf8",
    ),
  ).normalized),
);
const hotelMatchedHoloActions = await readNormalizedActions("holo", 3);
const hotelMatchedQwenActions = await readNormalizedActions("qwen", 4);
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
  .filter((row) => row.protocol.startsWith("holo_desktop_"))
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
const attentionDeltaDir = path.join(workspaceDir, "artifacts/screenspot-presentation/delta-lens-v23");
const attentionDeltaSummary = JSON.parse(
  await fs.readFile(path.join(attentionDeltaDir, "attention-delta-summary.json"), "utf8"),
);
const pairedControlDir = path.join(workspaceDir, "artifacts/screenspot-presentation/delta-lens-paired-controls-v1");
const pairedControlSummary = JSON.parse(
  await fs.readFile(path.join(pairedControlDir, "paired-control-attention-summary.json"), "utf8"),
);
const qwenAttention = path.join(pairedControlDir, "prompt-difference-qwen-value-norm.png");
const holoAttention = path.join(pairedControlDir, "prompt-difference-holo-value-norm.png");
const attentionDifference = path.join(pairedControlDir, "prompt-difference-holo-minus-qwen-value-norm.png");
const pairedValue = pairedControlSummary.methods.value_norm_attention;
const pairedDirect = pairedControlSummary.methods.direct_attention;
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

function clickMarker(slide, x, y, size = 18, color = C.orange) {
  const halo = 7;
  slide.shapes.add({
    geometry: "ellipse",
    position: { left: x - size, top: y - size, width: size * 2, height: size * 2 },
    fill: "none",
    line: { fill: color, width: halo },
  });
  slide.shapes.add({
    geometry: "ellipse",
    position: { left: x - size, top: y - size, width: size * 2, height: size * 2 },
    fill: "none",
    line: { fill: C.white, width: 3 },
  });
  rect(slide, x - size - 6, y - 3.5, size * 2 + 12, 7, color, true);
  rect(slide, x - 3.5, y - size - 6, 7, size * 2 + 12, color, true);
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
  textBox(slide, "Visual evidence and\nGUI action formation", 64, 142, 590, 168, { fontSize: 55, bold: true, color: C.white });
  textBox(slide, "Action-token attribution for static GUI grounding and multi-turn visual memory", 68, 340, 535, 104, { fontSize: 24, color: "#CBD3DB" });
  textBox(slide, "Holo 3.1 4B · local and NVIDIA traces · 22 Sep 2026", 68, 605, 535, 28, { fontSize: 14, bold: true, color: "#97A4B2" });
  slide.speakerNotes.textFrame.setText("0:00–0:35 — Frame the subject: how visual evidence reaches GUI actions, where resolution disrupts that route, and how fine-tuning changes the internal coordinate state. The visual is the final prompt-differential map for the ScreenSpot visual hit. Source: data/attributions/screenspot-powerpoint_windows_59/analysis.json.");
}

// 2 — two evaluation settings
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Evaluation", "Two settings test complementary GUI capabilities", 2);
  rect(slide, 64, 144, 548, 460, C.white, true, C.line);
  pill(slide, "STATIC", 90, 170, 86, C.blue, C.white);
  textBox(slide, "ScreenSpot-Pro", 90, 216, 300, 34, { fontSize: 27, bold: true });
  textBox(slide, "One screenshot, one instruction, one click", 90, 258, 420, 28, { fontSize: 18, color: C.muted });
  textBox(slide, "CAPABILITY", 90, 330, 150, 22, { fontSize: 12, bold: true, color: C.blue });
  textBox(slide, "UI localization", 90, 358, 300, 30, { fontSize: 23, bold: true });
  rect(slide, 90, 420, 462, 1, C.line);
  textBox(slide, "Can the model map one instruction to the correct clickable region in a dense, high-resolution interface?", 90, 448, 462, 80, { fontSize: 17, color: C.muted });

  rect(slide, 650, 144, 566, 460, C.deep, true);
  pill(slide, "MULTI-TURN", 680, 170, 120, C.orange, C.white);
  textBox(slide, "Synthetic hotel search", 680, 216, 420, 34, { fontSize: 27, bold: true, color: C.white });
  textBox(slide, "One matched case, native screenshots, four-action cap", 680, 258, 454, 28, { fontSize: 18, color: "#B9C3CD" });
  const sequence = [
    ["01", "OBSERVE", "native frame"],
    ["02", "SEARCH", "≤3 scrolls"],
    ["03", "RETAIN", "3 frames"],
    ["04", "ACT", "click or stop"],
  ];
  sequence.forEach(([num, action, detail], i) => {
    const y = 316 + i * 56;
    pill(slide, num, 680, y, 46, i === 3 ? C.green : "#344252", C.white);
    textBox(slide, action, 744, y + 2, 128, 22, { fontSize: 14, bold: true, color: C.white });
    textBox(slide, detail, 1010, y + 2, 130, 22, { fontSize: 15, bold: true, color: i === 3 ? C.green : "#B9C3CD", alignment: "right" });
  });
  textBox(slide, "CAPABILITY", 680, 555, 120, 18, { fontSize: 11, bold: true, color: C.gold });
  textBox(slide, "Context retrieval across frames", 808, 551, 370, 26, { fontSize: 18, bold: true, color: "#8FE0BF" });
  footer(slide, "Static evaluation isolates localization · the hotel environment tests whether earlier visual evidence reaches a later action");
  slide.speakerNotes.textFrame.setText("0:35–1:15 — Introduce two complementary capabilities. ScreenSpot-Pro isolates static UI localization from one screenshot and instruction. The synthetic hotel environment tests context retrieval and action formation across a sequence of screenshots. The matched test-0010 case runs Holo and its Qwen base independently from the same frozen fixture at native resolution, using the official HoloDesktop 0.1.10 tools, normalized coordinates, a four-action cap, and the three-screenshot retention policy. The incomplete multi-case batch is excluded from the finalized evidence set because a 384-token output cap truncated four Holo responses. Sources: docs/screenspot-case-study.md and data/remote-results/hotel-freegen-paired-v1-20260922/hotel-freegen-paired-v1/{holo,qwen}/bundles/test-0010.");
}

// 3 - full ScreenSpot-Pro replication
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Full benchmark replication", "65.9% overall, with icon file actions at 36.1%", 3);
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
  textBox(slide, "WEAKEST LABELED CELL", 954, 182, 240, 20, { fontSize: 11, bold: true, color: C.gold });
  textBox(slide, "Icon + file actions", 954, 215, 240, 30, { fontSize: 22, bold: true, color: C.white });
  textBox(slide, "36.1%", 954, 256, 150, 44, { fontSize: 36, bold: true, color: C.orange });
  textBox(slide, "13 / 36 correct", 1078, 268, 112, 24, { fontSize: 14, bold: true, color: "#F2BDAF", alignment: "right" });
  textBox(slide, "The larger icon + navigation cell reaches 40.3% on 149 items.", 954, 316, 236, 42, { fontSize: 13, color: "#C4CED7" });

  textBox(slide, "UI TYPE × INSTRUCTION ACTION FAMILY", 64, 408, 420, 20, { fontSize: 11, bold: true, color: C.blue });
  textBox(slide, "Accuracy and item count in each cell", 792, 408, 424, 20, { fontSize: 12, color: C.muted, alignment: "right" });
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
  footer(slide, "Raw point-in-box accuracy. UI type is the official target label; action families are diagnostic. Target geometry follows on the next slide.");
  slide.speakerNotes.textFrame.setText("1:15–2:10 — This is the complete local replication: 1,581 items, 1,042 strict point-in-box successes, no request failures, and 100% valid JSON. The 65.9% result is 0.6 percentage point below the approximately 66.5% Holo3.1-4B reference supplied by the project owner. ScreenSpot-Pro's UI type describes the annotated click target: icon means no text hint is present; targets with text labels belong to the text category even when an icon is also present. Raw accuracy is 79.4% for text targets and 44.0% for icon targets. The bottom matrix remains a diagnostic raw breakdown. Its action columns use our first-recognized-verb dictionary rather than an official ScreenSpot-Pro label. Icon plus file or transfer actions are lowest at 36.1% on 36 items. Icon plus navigation or reveal is the larger weak cell at 40.3% on 149 items. Do not interpret the raw UI split before examining target geometry on the next slide. Sources: data/remote-results/run/screenspot-full-official/summary.json, artifacts/screenspot-presentation/screenspot-benchmark-breakdown.json, and https://openreview.net/pdf?id=BS75DmIsRL.");
}

// 4 - target-area distribution
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Target geometry", "Text target boxes are 4.77× larger at the median", 4);
  textBox(slide, "Normalized area is the annotated click-box area divided by screenshot area. Each series sums to 100%.", 64, 118, 1152, 24, { fontSize: 14, color: C.muted });

  const areaBuckets = targetSizeAnalysis.target_area_distribution.buckets;
  textBox(slide, "SHARE OF TARGETS BY NORMALIZED AREA BUCKET", 64, 164, 720, 20, { fontSize: 11, bold: true, color: C.blue });
  const areaChart = slide.charts.add("bar", {
    position: { left: 64, top: 190, width: 820, height: 390 },
    categories: areaBuckets.map((row) => row.label),
    series: [
      { name: "Icon", values: areaBuckets.map((row) => Number((row.icon.share * 100).toFixed(1))), fill: C.orange },
      { name: "Text", values: areaBuckets.map((row) => Number((row.text.share * 100).toFixed(1))), fill: C.blue },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 45 },
    hasLegend: true,
    legend: { position: "bottom" },
    yAxis: { title: "Share within UI type (%)", minimumScale: 0, maximumScale: 45, majorUnit: 10 },
    xAxis: { title: "Target area as share of screenshot" },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0" },
    chartFill: C.white,
    chartLine: { fill: C.line, width: 1 },
    plotAreaFill: C.white,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(areaChart, { fontFamily: family });

  rect(slide, 924, 164, 292, 416, C.deep, true);
  textBox(slide, "MEDIAN TARGET AREA", 952, 194, 236, 20, { fontSize: 11, bold: true, color: C.gold });
  textBox(slide, `${targetSizeAnalysis.median_area_ratio_text_over_icon.toFixed(2)}×`, 952, 230, 236, 62, { fontSize: 52, bold: true, color: C.orange });
  textBox(slide, "text versus icon", 952, 294, 236, 24, { fontSize: 17, color: "#F2BDAF" });
  rect(slide, 952, 342, 236, 1, "#43515F");
  textBox(slide, "MEDIAN CLICK BOX", 952, 370, 236, 18, { fontSize: 11, bold: true, color: C.gold });
  textBox(slide, "Text   111 × 25 px", 952, 402, 236, 28, { fontSize: 20, bold: true, color: C.white });
  textBox(slide, "Icon     26 × 24 px", 952, 440, 236, 28, { fontSize: 20, bold: true, color: C.white });
  textBox(slide, "The difference is mostly width.", 952, 492, 236, 42, { fontSize: 14, color: "#C4CED7" });

  rect(slide, 64, 610, 1152, 44, C.deep, true);
  textBox(slide, "Point-in-box scoring gives wider text regions more click tolerance, so the raw UI gap mixes recognition with target geometry.", 84, 620, 1112, 24, { fontSize: 17, bold: true, color: C.white, alignment: "center" });
  footer(slide, "All 1,581 official annotations. Buckets show observed within-label distributions, not accuracy or a causal adjustment.");
  slide.speakerNotes.textFrame.setText("2:10–2:55 — Show the observed geometry before interpreting the raw icon-versus-text accuracy gap. Each item's target area is its annotated bounding-box area divided by screenshot area. The histogram uses fixed area buckets and normalizes separately within the 604 icon targets and 977 text targets, so each series sums to 100%. The median normalized text-target area is 4.77 times the icon median. Median heights are nearly identical at 25 versus 24 pixels, while median widths are 111 versus 26 pixels. Strict ScreenSpot accuracy is point-in-box, so wider text boxes provide more click tolerance. This distribution is more transparent than a single model-adjusted gap. It does not establish how much of the performance gap target size causes. Sources: artifacts/screenspot-presentation/screenspot-target-size-analysis.json, scripts/analyze_screenspot_target_size.py, and the official annotations at revision 210e78d3844251110bff86c95835ebd37a6930fa.");
}

// 5 - paired resolution ablation
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Resolution ablation", "Only 11% of native-success clicks survive at quarter resolution", 5);
  textBox(slide, "Paired retention across 36 ScreenSpot-Pro items that Holo localizes correctly at native resolution", 64, 118, 1152, 24, { fontSize: 14, color: C.muted });

  const resolutionRows = [...resolutionAblation.scales].sort((a, b) => b.linear_scale - a.linear_scale);
  textBox(slide, "SUCCESS RETENTION UNDER CONTROLLED DOWNSAMPLING", 64, 164, 760, 20, { fontSize: 11, bold: true, color: C.blue });
  const resolutionChart = slide.charts.add("bar", {
    position: { left: 64, top: 190, width: 820, height: 390 },
    categories: resolutionRows.map((row) => `${Math.round(row.linear_scale * 100)}%\n${(row.area_fraction * 100).toFixed(row.area_fraction < 0.1 ? 2 : 0)}% area`),
    series: [{
      name: "Success retention",
      values: resolutionRows.map((row) => Number((row.retention * 100).toFixed(1))),
      fill: C.blue,
    }],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 55 },
    hasLegend: false,
    yAxis: { title: "Previously correct clicks retained (%)", minimumScale: 0, maximumScale: 100, majorUnit: 20 },
    xAxis: { title: "Linear resolution and retained source area" },
    dataLabels: { showValue: true, position: "outEnd", numberFormatCode: "0.0" },
    chartFill: C.white,
    chartLine: { fill: C.line, width: 1 },
    plotAreaFill: C.white,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(resolutionChart, { fontFamily: family });

  const half = resolutionRows.find((row) => row.linear_scale === 0.5);
  const quarter = resolutionRows.find((row) => row.linear_scale === 0.25);
  rect(slide, 924, 164, 292, 416, C.deep, true);
  textBox(slide, "QUARTER RESOLUTION", 952, 194, 236, 20, { fontSize: 11, bold: true, color: C.gold });
  textBox(slide, `${(quarter.retention * 100).toFixed(1)}%`, 952, 226, 236, 62, { fontSize: 52, bold: true, color: C.orange });
  textBox(slide, `${quarter.strict_correct}/${quarter.count} clicks survive`, 952, 292, 236, 24, { fontSize: 17, color: "#F2BDAF" });
  textBox(slide, "95% CI 4.4–25.3%", 952, 322, 236, 20, { fontSize: 13, color: "#C4CED7" });
  rect(slide, 952, 354, 236, 1, "#43515F");
  textBox(slide, "AT HALF RESOLUTION", 952, 378, 236, 18, { fontSize: 11, bold: true, color: C.gold });
  textBox(slide, `Text   ${(half.by_ui_type.text.retention * 100).toFixed(1)}%`, 952, 410, 236, 28, { fontSize: 20, bold: true, color: C.white });
  textBox(slide, `Icon    ${(half.by_ui_type.icon.retention * 100).toFixed(1)}%`, 952, 448, 236, 28, { fontSize: 20, bold: true, color: C.orange });
  textBox(slide, "Smaller icon targets lose more of their successful clicks.", 952, 498, 236, 48, { fontSize: 14, color: "#C4CED7" });

  rect(slide, 64, 610, 1152, 44, C.deep, true);
  textBox(slide, "Reducing each dimension to 25% cuts retained area to 6.25% and success by 88.9 percentage points.", 84, 620, 1112, 24, { fontSize: 17, bold: true, color: C.white, alignment: "center" });
  footer(slide, "36 selected native successes · 144 deterministic requests · official prompt and normalized 0–1000 coordinates · all outputs valid JSON");
  slide.speakerNotes.textFrame.setText("2:55–3:50 — This is a paired success-retention experiment, not a new unconditional ScreenSpot-Pro accuracy estimate. We froze 36 items that were strict hits in the full native-resolution run: six size-spread targets in each application-by-UI-type stratum across Photoshop, PowerPoint, and VS Code. Every item was rerun in one Holo model process at 100%, 75%, 50%, and 25% of its original width and height using Lanczos downsampling. The official localization prompt, VisualLocalizerOutput schema, temperature zero, thinking-disabled decoding, and normalized 0–1000 coordinates remain fixed. The server stays at image_max_pixels 16777216, so it cannot add a second hidden resize. The same-run native rerun is 36 of 36. Retention is 25 of 36 at 75%, 22 of 36 at 50%, and 4 of 36 at 25% linear resolution. Wilson 95% intervals are 53.1 to 82.0%, 44.9 to 75.2%, and 4.4 to 25.3%. At half resolution, text targets retain 15 of 18 clicks while icon targets retain 7 of 18. Retained targets have about 4.1 times the median normalized area of failures. All 144 completions are valid JSON. The cohort is selected on native success, so use this as evidence of resolution sensitivity, not as the benchmark's full accuracy curve. Sources: artifacts/screenspot-presentation/screenspot-resolution-ablation-v1.json, docs/screenspot-resolution-ablation.md, and benchmarks/resolution_ablation/screenspot_success_retention_v1.json.");
}

// 6 — resolution and attention cohort
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Resolution and saliency", "Target-specific attribution weakens 24× at quarter resolution", 6);
  textBox(slide, "18 balanced native-success cases · value-norm rollout minus four same-image instruction controls", 64, 118, 1152, 24, { fontSize: 15, color: C.muted });

  const scaleRows = [...resolutionSaliencyCohort.scales].sort((a, b) => b.linear_scale - a.linear_scale);
  const exampleRows = resolutionSaliency.results.filter((row) => row.sample_id === "photoshop_windows_10");
  const scaleLabels = ["100%", "75%", "50%", "25%"];
  const panelWidth = 270;
  for (const [index, row] of scaleRows.entries()) {
    const left = 64 + index * 288;
    const example = exampleRows.find((item) => item.linear_scale === row.linear_scale);
    const accent = example.strict_correct ? C.green : C.orange;
    textBox(slide, scaleLabels[index], left, 158, 70, 24, { fontSize: 18, bold: true, color: accent });
    textBox(slide, example.strict_correct ? "HIT" : "MISS", left + 190, 160, 80, 18, { fontSize: 11, bold: true, color: accent, alignment: "right" });
    await image(slide, resolutionSaliencyMaps[index], left, 188, panelWidth, 170, {
      alt: `Prompt-differential value-norm attribution at ${scaleLabels[index]} linear resolution`,
    });
    textBox(slide, `${(example.metrics.prompt_difference.target_mass * 100).toFixed(example.metrics.prompt_difference.target_mass < 0.01 ? 2 : 1)}% target mass`, left, 366, panelWidth, 20, { fontSize: 12, bold: true, color: accent, alignment: "center" });
  }

  textBox(slide, "BALANCED 18-CASE COHORT", 64, 408, 240, 20, { fontSize: 11, bold: true, color: C.blue });
  const headers = ["", ...scaleLabels];
  const values = [
    headers,
    ["Strict hits", ...scaleRows.map((row) => `${row.groups.all.strict_correct_count}/18`)],
    ["Median target mass", ...scaleRows.map((row) => `${(row.groups.all.median_target_mass * 100).toFixed(2)}%`)],
    ["Median target lift", ...scaleRows.map((row) => `${row.groups.all.median_target_lift.toFixed(row.linear_scale === 0.25 ? 1 : 0)}×`)],
    ["Median best-patch rank", ...scaleRows.map((row) => `${row.groups.all.median_best_target_patch_rank}`)],
  ];
  const table = slide.tables.add({
    rows: values.length,
    columns: values[0].length,
    left: 64,
    top: 436,
    width: 694,
    height: 196,
    values,
    columnWidths: [190, 126, 126, 126, 126],
    rowHeights: [34, 40, 40, 40, 40],
  });
  table.cells.block({ row: 0, column: 0, rowCount: values.length, columnCount: values[0].length }).assign({
    margins: { left: 10, right: 10, top: 6, bottom: 6 },
  });
  table.borders.assign({ style: "solid", fill: C.paper, width: 2 });
  for (let rowIndex = 0; rowIndex < values.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < values[0].length; columnIndex += 1) {
      const cell = table.getCell(rowIndex, columnIndex);
      cell.fill = rowIndex === 0 || columnIndex === 0 ? C.deep : columnIndex === 4 ? C.orangeSoft : C.white;
      cell.text.style = {
        typeface: family,
        fontSize: rowIndex === 0 ? 11 : 13,
        bold: rowIndex === 0 || columnIndex === 0,
        color: rowIndex === 0 || columnIndex === 0 ? C.white : columnIndex === 4 ? "#A33D29" : C.ink,
        alignment: columnIndex === 0 ? "left" : "center",
        verticalAlignment: "middle",
        autoFit: "shrinkText",
      };
    }
  }

  rect(slide, 790, 408, 426, 224, C.deep, true);
  textBox(slide, "NO FIXED-GRID PADDING CONFOUND", 818, 432, 310, 18, { fontSize: 11, bold: true, color: C.gold });
  textBox(slide, "90×160 → 22×40", 818, 464, 350, 38, { fontSize: 28, bold: true, color: C.white });
  textBox(slide, "raw patch grid · ≈3,600 → 220 merged visual tokens", 818, 504, 350, 32, { fontSize: 14, color: "#C4CED7" });
  textBox(slide, "The grid shrinks with the image; it is not padded back to a fixed ≈1k-token budget.", 818, 544, 350, 42, { fontSize: 14, bold: true, color: "#8FE0BF" });
  textBox(slide, "Remaining confounds: fewer spatial samples and downsampling/aliasing.", 818, 590, 350, 28, { fontSize: 13, color: "#C4CED7" });
  footer(slide, "360 deterministic requests · official localization prompt and 0–1000 contract · eager attention · n=18");
  slide.speakerNotes.textFrame.setText("3:50–4:30 — This expands the six-case pilot to 18 native-resolution successes, balanced at three cases in each application-by-UI-type stratum across Photoshop, PowerPoint, and VS Code. Each case runs at 100, 75, 50, and 25 percent linear resolution. At every scale, Holo receives the official localization prompt and VisualLocalizerOutput contract, then the eager-attention harness traces the freely generated coordinate tokens. The primary attribution is value-norm rollout after subtracting the mean of four visible same-image alternate instructions. Median target-specific mass falls from 5.22 percent at native resolution to 2.18, 1.16, and 0.21 percent as resolution decreases, a 24-fold cohort-level reduction at quarter resolution. The paired within-case median retains 6.23 percent of native target mass at quarter resolution, with a bootstrap 95 percent interval of 0.18 to 14.47 percent. Median target lift falls from 168.7 to 5.4, and median best-target-patch rank falls from first to 6.5. Strict hits fall from 18 of 18 to 12, 10, and 3 of 18. In the exact Photoshop trace shown here, the processor records a 90 by 160 raw visual patch grid at native resolution and 22 by 40 at quarter resolution. With the model's two-by-two spatial merge, that is about 3,600 versus 220 visual tokens. The lower-resolution input is therefore not padded back to a fixed approximately one-thousand-token budget; the remaining resolution confounds are reduced spatial sampling and downsampling or aliasing. Localization loss explains most failures, but not all: 7 of 29 non-native misses still place the highest prompt-differential patch inside the oracle, including 3 of 8 misses at half resolution and 3 of 15 at quarter resolution. This repeatable dissociation motivates a separate coordinate-readout hypothesis. The cohort is conditioned on native success and remains a paired sensitivity diagnostic rather than an unconditional benchmark estimate. Sources: artifacts/screenspot-presentation/resolution-saliency-cohort18-v1/summary.json, data/remote-results/screenspot-resolution-saliency-pilot-v1, and docs/screenspot-resolution-saliency-cohort18.md.");
}

// 7 — native-resolution baseline comparison on a free-generation success
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Attribution baseline matters", "Instruction controls suppress generic edge saliency", 7);
  textBox(slide, "Instruction: Fill color · free Holo click (209,243)", 64, 116, 660, 24, { fontSize: 17, bold: true, color: C.muted });
  pill(slide, "GREEN = ORACLE · WHITE = CLICK", 838, 113, 378, C.deep, C.white);

  const panelWidth = 544;
  const panelHeight = 400;
  textBox(slide, "VALUE-NORM · NO INSTRUCTION BASELINE", 64, 153, panelWidth, 20, { fontSize: 11, bold: true, color: C.orange });
  textBox(slide, "VALUE-NORM · MINUS 4 SAME-IMAGE INSTRUCTIONS", 672, 153, panelWidth, 20, { fontSize: 11, bold: true, color: C.green });
  await image(slide, slide6Raw, 64, 180, panelWidth, panelHeight, { alt: "Native-resolution value-norm rollout for the generated x and y coordinate tokens" });
  await image(slide, slide6PromptDifference, 672, 180, panelWidth, panelHeight, { alt: "Native-resolution value-norm rollout after subtracting four same-image diverse instructions" });

  rect(slide, 64, 594, panelWidth, 58, C.deep, true);
  textBox(slide, `${slide6RawMetrics.target_lift.toFixed(0)}× target lift`, 84, 603, 200, 26, { fontSize: 21, bold: true, color: C.orange });
  textBox(slide, "shared edge + canvas structure remains", 292, 607, 292, 20, { fontSize: 12, color: "#D5DEE6", alignment: "right" });
  rect(slide, 672, 594, panelWidth, 58, C.deep, true);
  textBox(slide, `${slide6Hit.metrics.prompt_difference.target_lift.toFixed(0)}× target lift`, 692, 603, 220, 26, { fontSize: 21, bold: true, color: C.green });
  textBox(slide, `${(slide6Hit.metrics.prompt_difference.target_mass * 100).toFixed(1)}% target mass · LOO ≥${slide6Hit.stability.minimum_leave_one_out_cosine.toFixed(3)}`, 918, 607, 274, 20, { fontSize: 12, color: "#D5DEE6", alignment: "right" });
  footer(slide, "Resting-state analogy: same image and action; subtract the mean saliency shared across four visible alternate instructions");
  slide.speakerNotes.textFrame.setText("4:30–5:30 — Pull back to the larger screenshot context. The left panel is value-norm rollout for Holo's freely generated x/y coordinate tokens without an instruction baseline. It puts 4.43% of positive mass in the tiny oracle, a 110.7x lift, but also retains saliency shared across the image edge, upper-left chrome, and broad canvas structure. The right panel subtracts the mean of four separately measured same-image instructions: Cut, Copy, Send backward, and Add comment. This resting-state-like control suppresses those instruction-invariant false positives and raises target-specific concentration to 18.58% and 464.4x lift; the minimum leave-one-control-out cosine is 0.991. Both panels use the same native 2880x1800 screenshot, official localization prompt, Holo output (209,243), and x/y token aggregation. The subtraction isolates instruction-conditioned saliency; neither map by itself proves a causal mechanism. Source: data/remote-results/slide67-native-controls-20260922 and artifacts/screenspot-presentation/slide67-native-contrast-v1.");
}

// 8 — native-resolution free-generation failure after diverse-instruction subtraction
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Free generation · strict miss", "Target-specific attention survives a 7.4-pixel click miss", 8);
  textBox(slide, "Instruction: Choose the language for proofing tools", 64, 120, 700, 24, { fontSize: 17, bold: true, color: C.muted });
  pill(slide, "TARGET − 4 SAME-IMAGE INSTRUCTIONS", 834, 116, 382, C.deep, C.white);
  textBox(slide, "DIVERSE-INSTRUCTION SUBTRACTION", 64, 154, 480, 18, { fontSize: 11, bold: true, color: C.green });
  await image(slide, slide7PromptDifference, 64, 178, 790, 298, { alt: "Readable native-resolution crop of prompt-differential value-norm attribution for the proofing-language failure" });

  rect(slide, 890, 178, 326, 438, C.deep, true);
  pill(slide, "STRICT MISS", 916, 202, 126, C.orange, C.white);
  textBox(slide, "(124, 73)", 916, 250, 250, 40, { fontSize: 30, bold: true, color: C.white });
  textBox(slide, "7.4 px below the annotated box", 916, 300, 250, 52, { fontSize: 18, bold: true, color: "#F2BDAF" });
  rect(slide, 916, 372, 250, 1, "#465563");
  textBox(slide, "positive mass in oracle", 916, 394, 175, 20, { fontSize: 12, color: "#9EABB7" });
  textBox(slide, `${(slide7Miss.metrics.prompt_difference.target_mass * 100).toFixed(1)}%`, 1092, 390, 74, 26, { fontSize: 20, bold: true, color: C.green, alignment: "right" });
  textBox(slide, "target lift", 916, 438, 175, 20, { fontSize: 12, color: "#9EABB7" });
  textBox(slide, `${slide7Miss.metrics.prompt_difference.target_lift.toFixed(0)}×`, 1092, 434, 74, 26, { fontSize: 20, bold: true, color: C.green, alignment: "right" });
  textBox(slide, "peak patch", 916, 482, 175, 20, { fontSize: 12, color: "#9EABB7" });
  textBox(slide, "inside", 1092, 478, 74, 26, { fontSize: 18, bold: true, color: C.green, alignment: "right" });
  textBox(slide, `control stability ≥${slide7Miss.stability.minimum_leave_one_out_cosine.toFixed(3)}`, 916, 532, 250, 20, { fontSize: 13, bold: true, color: C.gold });

  rect(slide, 64, 500, 790, 116, C.white, true, C.line);
  textBox(slide, "ACTION-PRECISION FAILURE", 88, 520, 240, 18, { fontSize: 11, bold: true, color: C.orange });
  textBox(slide, "The target-specific peak is inside the proofing-language control, but the generated y coordinate crosses the strict point-in-box boundary.", 88, 548, 734, 48, { fontSize: 18, bold: true, color: C.ink });
  footer(slide, "Official free generation · normalized 0–1000 output · native 2880×1800 input · no downsampling");
  slide.speakerNotes.textFrame.setText("5:30–6:20 — This failure uses the same full-resolution official protocol and a four-instruction same-image baseline. Holo freely emits (124,73), projecting to (357.1,131.4) pixels. The x coordinate falls inside the annotated proofing-language control, while y lands 7.4 pixels below its bottom edge. After value-norm rollout and subtraction of Check accessibility, Translate, Add comment, and Show comments, the peak patch remains inside the oracle. The tiny box receives 17.15% of positive residual mass, a 278.6x lift, with a 0.951 leave-one-control-out floor. This pattern is consistent with localized visual evidence followed by imprecise coordinate extraction. Attention remains descriptive rather than causal. Source: data/remote-results/slide67-native-controls-20260922 and artifacts/screenspot-presentation/slide67-native-contrast-v1.");
}

// 9 - native-resolution free multi-turn trajectory
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Matched hotel trajectory", "Holo clicks while Qwen continues searching", 9);
  textBox(slide, "Exploratory n=1 · same frozen fixture · native 1280×800 screenshots · official tools", 64, 118, 900, 24, { fontSize: 15, color: C.muted });

  textBox(slide, "SAME TARGET · FJORD COMPASS LODGE · €166", 64, 156, 500, 18, { fontSize: 11, bold: true, color: C.blue });
  const matchedActions = [
    ["MODEL", "TURN 1", "TURN 2", "TURN 3", "TURN 4", "OUTCOME"],
    [
      "Holo",
      `scroll ${hotelMatchedHoloActions[0].delta_y}`,
      `scroll ${hotelMatchedHoloActions[1].delta_y}`,
      `click\n(${hotelMatchedHoloActions[2].x},${hotelMatchedHoloActions[2].y})`,
      "—",
      "correct click",
    ],
    [
      "Qwen",
      `scroll ${hotelMatchedQwenActions[0].delta_y}`,
      `scroll ${hotelMatchedQwenActions[1].delta_y}`,
      `scroll ${hotelMatchedQwenActions[2].delta_y}`,
      `scroll ${hotelMatchedQwenActions[3].delta_y}`,
      "cap reached",
    ],
  ];
  const matchedTable = slide.tables.add({
    rows: matchedActions.length,
    columns: matchedActions[0].length,
    left: 64,
    top: 186,
    width: 500,
    height: 166,
    values: matchedActions,
    columnWidths: [72, 76, 76, 92, 76, 108],
    rowHeights: [36, 64, 64],
  });
  matchedTable.cells.block({ row: 0, column: 0, rowCount: matchedActions.length, columnCount: matchedActions[0].length }).assign({
    margins: { left: 6, right: 6, top: 5, bottom: 5 },
  });
  matchedTable.borders.assign({ style: "solid", fill: C.paper, width: 2 });
  for (let rowIndex = 0; rowIndex < matchedActions.length; rowIndex += 1) {
    for (let columnIndex = 0; columnIndex < matchedActions[0].length; columnIndex += 1) {
      const cell = matchedTable.getCell(rowIndex, columnIndex);
      cell.fill = rowIndex === 0 ? C.deep : rowIndex === 1 ? C.greenSoft : C.blueSoft;
      cell.text.style = {
        typeface: family,
        fontSize: rowIndex === 0 ? 10 : 11,
        bold: rowIndex === 0 || columnIndex === 0 || columnIndex === 5,
        color: rowIndex === 0 ? C.white : rowIndex === 1 && columnIndex === 5 ? C.green : rowIndex === 2 && columnIndex === 5 ? C.blue : C.ink,
        alignment: "center",
        verticalAlignment: "middle",
        autoFit: "shrinkText",
      };
    }
  }

  textBox(slide, "BOTH TRACK €166 BY TURN 3", 64, 382, 300, 18, { fontSize: 11, bold: true, color: C.orange });
  textBox(slide, "Holo sees all six prices and clicks. Qwen identifies €166 as the lowest observed price, then spends its fourth action scrolling.", 64, 410, 500, 62, { fontSize: 18, bold: true, color: C.ink });
  rect(slide, 64, 506, 500, 116, C.white, true, C.line);
  textBox(slide, "CLAIM BOUNDARY", 88, 526, 190, 18, { fontSize: 11, bold: true, color: C.orange });
  textBox(slide, "This matched case motivates a stopping and action-readout hypothesis. It does not estimate checkpoint accuracy.", 88, 553, 452, 48, { fontSize: 16, bold: true, color: C.ink });

  textBox(slide, "TRACED HOLO REPLICATE · FINAL VALUE-NORM ROLLOUT", 596, 156, 620, 18, { fontSize: 11, bold: true, color: C.blue });
  textBox(slide, "HISTORY 1", 596, 181, 132, 15, { fontSize: 9, bold: true, color: C.muted, alignment: "center" });
  await image(slide, hotelPairedMaps[0], 596, 201, 132, 83, { alt: "First retained frame in the successful Holo hotel trajectory", fit: "contain", geometry: "rect", borderRadius: 0 });
  textBox(slide, "HISTORY 2", 596, 297, 132, 15, { fontSize: 9, bold: true, color: C.muted, alignment: "center" });
  await image(slide, hotelPairedMaps[1], 596, 317, 132, 83, { alt: "Second retained frame in the successful Holo hotel trajectory", fit: "contain", geometry: "rect", borderRadius: 0 });
  textBox(slide, "CURRENT FRAME", 744, 181, 472, 15, { fontSize: 9, bold: true, color: C.green, alignment: "center" });
  const currentLeft = 744;
  const currentTop = 201;
  const currentWidth = 472;
  const currentHeight = 295;
  await image(slide, hotelPairedMaps[2], currentLeft, currentTop, currentWidth, currentHeight, { alt: "Current frame for Holo's correct Fjord Compass Lodge click with value-norm rollout", fit: "contain", geometry: "rect", borderRadius: 0 });
  const finalClick = hotelPairedTrajectory.actions.at(-1);
  const oracleBox = [977, 191, 1105, 241];
  rect(
    slide,
    currentLeft + oracleBox[0] * currentWidth / 1280,
    currentTop + oracleBox[1] * currentHeight / 800,
    (oracleBox[2] - oracleBox[0]) * currentWidth / 1280,
    (oracleBox[3] - oracleBox[1]) * currentHeight / 800,
    "none",
    true,
    C.green,
  );
  clickMarker(
    slide,
    currentLeft + finalClick.x * currentWidth / 1280,
    currentTop + finalClick.y * currentHeight / 800,
    4,
    C.orange,
  );

  rect(slide, 596, 510, 620, 112, C.deep, true);
  textBox(slide, "TRACED REPLICATE", 620, 529, 150, 18, { fontSize: 10, bold: true, color: C.green });
  textBox(slide, "Fjord Compass Lodge · €166", 620, 550, 300, 25, { fontSize: 19, bold: true, color: C.white });
  textBox(slide, "Two scrolls, then click (1023,216) inside the oracle button", 620, 579, 360, 20, { fontSize: 13, bold: true, color: C.orange });
  textBox(slide, "Attribution appears across all three retained frames. These maps are descriptive and do not yet subtract diverse-instruction controls.", 970, 531, 220, 70, { fontSize: 12, color: "#CBD3DB" });
  footer(slide, "Independent free generation · HoloDesktop 0.1.10 tools · normalized 0–1000 coordinates · native resolution");
  slide.speakerNotes.textFrame.setText("6:20–7:20 — This slide reports one complete matched free-generation case and excludes the incomplete multi-case batch. Holo and Qwen start from the same frozen test-0010 fixture, use native 1280 by 800 screenshots, the official HoloDesktop 0.1.10 tool schema, normalized coordinates, and a four-action cap. Holo scrolls 500 pixels twice, inventories all six prices, then clicks Fjord Compass Lodge at normalized (1023,208). Qwen scrolls 250 pixels four times. By its third response, Qwen records €166 as the lowest observed price, but it spends the fourth action scrolling again and reaches the cap without clicking. This supports a stopping and action-readout hypothesis within one case, not a checkpoint accuracy claim. The visual uses a separately traced Holo replicate on the same fixture: two scrolls followed by click (1023,216). Its final value-norm rollout appears across the two retained history frames and the current frame. These maps remain descriptive and do not include same-image diverse-instruction subtraction. The larger batch is absent from the finalized evidence because a 384-token cap truncated four Holo responses. Sources: data/remote-results/hotel-freegen-paired-v1-20260922/hotel-freegen-paired-v1/holo/bundles/test-0010; data/remote-results/hotel-freegen-paired-v1-20260922/hotel-freegen-paired-v1/qwen/bundles/test-0010; artifacts/screenspot-presentation/hotel-freegen-paired-v1-attribution/trajectory.json.");
}

// 10 - causal protocol across eight mirrored prompts
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Causal experiment", "Where visual evidence becomes a coordinate action", 10);
  textBox(slide, "Eight mirrored prompts hold the official request, native image, and teacher-forced coordinate strings fixed", 64, 118, 1090, 24, { fontSize: 14, color: C.muted });

  pill(slide, "CLEAN", 64, 160, 84, C.green, C.white);
  await image(slide, causalClean, 64, 198, 250, 166, { alt: "Original PowerPoint template row with the target and distractor tiles" });
  textBox(slide, "Original tile positions", 64, 374, 250, 22, { fontSize: 15, bold: true });
  pill(slide, "CORRUPTED", 64, 430, 116, C.orange, C.white);
  await image(slide, causalCorrupted, 64, 468, 250, 166, { alt: "PowerPoint row after an equal-size target and distractor tile swap" });
  textBox(slide, "Two equal-size tiles swap", 64, 644, 250, 18, { fontSize: 13, bold: true });

  rect(slide, 344, 160, 872, 474, C.white, true, C.line);
  textBox(slide, "TEACHER-FORCED ACTION SEQUENCE", 372, 186, 330, 18, { fontSize: 11, bold: true, color: C.blue });
  textBox(slide, "Image evidence and instruction tokens precede the scored x-coordinate digits", 372, 210, 730, 24, { fontSize: 15, color: C.muted });
  const stages = [
    ["IMAGE", "patch residuals", C.blueSoft, C.blue, 150],
    ["INSTRUCTION", "target name", "#E9EDF1", C.deep, 170],
    ["JSON", "{ x:", C.orangeSoft, C.orange, 90],
    ["X DIGITS", "scored action", C.orangeSoft, C.orange, 150],
    ["Y DIGITS", "held matched", C.greenSoft, C.green, 140],
  ];
  let stageLeft = 372;
  stages.forEach(([heading, body, fill, accent, width], index) => {
    rect(slide, stageLeft, 258, width, 82, fill, true, accent);
    textBox(slide, heading, stageLeft + 10, 271, width - 20, 18, { fontSize: 11, bold: true, color: accent, alignment: "center" });
    textBox(slide, body, stageLeft + 10, 298, width - 20, 24, { fontSize: 13, bold: heading === "X DIGITS", alignment: "center" });
    stageLeft += width + (index < stages.length - 1 ? 14 : 0);
  });

  textBox(slide, "Replace one internal state in the corrupted run with its clean-run counterpart", 372, 380, 760, 24, { fontSize: 17, bold: true });
  rect(slide, 372, 422, 346, 154, C.blueSoft, true, C.blue);
  pill(slide, "1 · VISUAL REGION", 392, 442, 168, C.blue, C.white);
  textBox(slide, "Patch target or distractor image-region residuals to test whether localized visual evidence restores the coordinate preference.", 392, 486, 304, 68, { fontSize: 14 });
  rect(slide, 746, 422, 414, 154, C.orangeSoft, true, C.orange);
  pill(slide, "2 · COORDINATE STATE", 766, 442, 194, C.orange, C.white);
  textBox(slide, "Patch coordinate-token residuals across layers. Then test the selected Layer 15 state and each attention head at x-token positions.", 766, 486, 370, 68, { fontSize: 14 });

  textBox(slide, "Metric: log p(target x digits) − log p(distractor x digits)", 372, 594, 788, 24, { fontSize: 16, bold: true, color: C.deep, alignment: "center" });
  footer(slide, "Four equal-tile pairs × two mirrored instructions · native 2880×1800 · official VisualLocalizerOutput contract");
  slide.speakerNotes.textFrame.setText("7:20–8:20 — Define the causal panel without phase labels. Four equal-size tile pairs produce eight mirrored instructions. The corruption swaps the target and distractor tiles without resizing or changing the prompt. Both checkpoints receive the Holo processor and official VisualLocalizerOutput request. The model is teacher-forced through equal-length coordinate candidates. The metric is the x-digit log-probability margin because the paired y coordinate stays fixed in this horizontal panel. First screen visual-region residuals and coordinate states across layers. Then test the selected Layer 15 coordinate state and all sixteen attention heads. Sources: benchmarks/activation_patching/qwen_holo_action_panel_v1 and benchmarks/activation_patching/qwen_holo_action_panel_v1_phase_b_layer15.");
}

// 11 - causal result across eight mirrored prompts
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  const pairedStats = causalPanelStats.paired_holo_minus_qwen;
  slideTitle(slide, "Causal result · Layer 15", "Fine-tuning doubles recovery of the coordinate state", 11);
  textBox(slide, "Eight mirrored prompts · x-coordinate teacher forcing · same processor, prompt, image bytes, and candidate syntax", 64, 118, 1120, 24, { fontSize: 14, color: C.muted });

  rect(slide, 64, 160, 550, 166, C.greenSoft, true, C.green);
  textBox(slide, "PATCH CLEAN STATE INTO THE CORRUPTED RUN", 86, 184, 470, 20, { fontSize: 11, bold: true, color: C.green });
  textBox(slide, "Holo  +1.01 nats", 86, 222, 230, 32, { fontSize: 25, bold: true, color: C.green });
  textBox(slide, "Qwen  +0.10 nats", 332, 222, 220, 32, { fontSize: 25, bold: true, color: C.deep });
  textBox(slide, "Positive restoration: Holo 7/8 · Qwen 5/8 prompts", 86, 272, 470, 28, { fontSize: 14, color: C.muted });

  rect(slide, 638, 160, 578, 166, C.white, true, C.line);
  textBox(slide, "PAIRED HOLO − QWEN EFFECT", 660, 184, 480, 20, { fontSize: 11, bold: true, color: C.blue });
  textBox(slide, `${pairedStats.likelihood_ratio_factor.toFixed(2)}×`, 660, 216, 170, 48, { fontSize: 38, bold: true, color: C.blue });
  textBox(slide, "+105% likelihood-ratio recovery", 826, 224, 340, 28, { fontSize: 20, bold: true, color: C.deep });
  textBox(slide, `bootstrap 95% CI ${pairedStats.likelihood_ratio_factor_95_ci[0].toFixed(2)}×–${pairedStats.likelihood_ratio_factor_95_ci[1].toFixed(2)}× · exact paired p=${pairedStats.exact_one_sided_sign_flip_p_mean.toFixed(4)}`, 660, 278, 510, 28, { fontSize: 13, color: C.muted });

  textBox(slide, "WHAT THE INTERVENTIONS SHOW", 64, 356, 520, 18, { fontSize: 11, bold: true, color: C.blue });
  const findings = [
    ["1", "Paired margin", `The median Holo-minus-Qwen restoration is +${pairedStats.median_nats.toFixed(2)} nats; every prompt uses identical clean/corrupt images and candidate strings.`, C.greenSoft, C.green],
    ["2", "Shared necessity", "Setting the Layer 15 coordinate state to zero in a clean run costs 4.51 nats for Holo and 6.27 for Qwen: both checkpoints rely on it.", "#E9EDF1", C.deep],
    ["3", "Distributed mechanism", "All 16 heads were tested. No single head explains the checkpoint difference; the recoverable effect is distributed in the residual state.", C.orangeSoft, C.orange],
  ];
  findings.forEach(([num, heading, body, fill, accent], index) => {
    const top = 384 + index * 72;
    rect(slide, 64, top, 1152, 56, fill, true, accent);
    pill(slide, num, 82, top + 13, 30, accent, C.white);
    textBox(slide, heading, 132, top + 10, 250, 22, { fontSize: 15, bold: true, color: accent });
    textBox(slide, body, 394, top + 8, 796, 40, { fontSize: 14, verticalAlignment: "middle" });
  });

  rect(slide, 64, 612, 1152, 42, C.deep, true);
  textBox(slide, "Fine-tuning strengthens recoverable action state; it does not create a Holo-only coordinate circuit.", 84, 622, 1112, 20, { fontSize: 16, bold: true, color: C.white, alignment: "center" });
  footer(slide, "Exploratory n=8 within-image panel · paired bootstrap and all 256 sign flips · not an independent-image benchmark");
  slide.speakerNotes.textFrame.setText("8:20–9:20 — Lead with the directly interpretable intervention. Replacing the corrupted-run Layer 15 x-coordinate residual with the clean-run state raises Holo's target-over-distractor margin by a median 1.01 nats in seven of eight prompts. Qwen's median is 0.10 nats and is positive in five of eight. The paired Holo-minus-Qwen median is 0.718 nats. Because the margin is a log target-versus-distractor sequence likelihood ratio, exponentiating the paired difference gives a 2.05-times likelihood-ratio recovery, or 105 percent larger. A paired nonparametric bootstrap gives a 95 percent interval of 1.74 to 3.15 times. An exact one-sided paired sign-flip test over all 256 assignments gives p=0.0078; with only eight mirrored prompts from four synthetic image pairs, treat this as exploratory within-panel evidence rather than population-level benchmark significance. Zero-ablation means setting that same Layer 15 coordinate residual to zero in the clean run, then measuring the clean-margin loss: 4.51 nats for Holo and 6.27 for Qwen. Both checkpoints rely on the state, so fine-tuning increases recoverability rather than creating a Holo-exclusive circuit. The all-head sweep also rejects a single-head explanation. Sources: artifacts/screenspot-presentation/causal-panel-statistics-v1.json, data/remote-results/qwen-holo-action-panel-v1-20260922, and data/remote-results/qwen-holo-action-panel-v1-20260922-phase-b-layer15.");
}

// Legacy exploratory slides omitted from the presentation flow.
if (false) {
// 10 - balanced multi-item layer/head aggregate
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Pilot aggregate · n=4", "Layer 19 has the highest mean lift, narrowly ahead of Layer 15", 10);
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
  slide.speakerNotes.textFrame.setText("8:05–9:10 — This slide replaces the earlier three selected matrices with the balanced official-protocol pilot. Each item uses four visible, task-valid alternate click instructions on the same screenshot and the same 2,097,152-pixel eager-attention profile. Layer rank means rank among the eight captured full-attention layers. H10 rank means rank among the 16 heads within Layer 19. Holo3.1-4B uses full attention at zero-based layers 3, 7, 11, 15, 19, 23, 27, and 31, so both Layer 15 and Layer 19 are full-attention layers. The intervening layers use Gated DeltaNet and are outside this rollout. Across two hits and two misses, Layer 19 has the highest mean at 158.4×, narrowly ahead of Layer 15 at 151.4×. Layer 19 is top on two items and has median item rank 2. H11, H14, H10, and H2 form the strongest aggregate cluster. H10 ranks third within Layer 19 and never ranks first on an individual item. This is descriptive routing evidence from a four-item pilot, not a stable Layer 19 mechanism. Sources: data/remote-results/attention-layer19-balanced-v1/aggregate/aggregate.json and https://huggingface.co/Hcompany/Holo-3.1-4B/blob/main/config.json.");
}

// 11 - matched causal setup
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Causal intervention · one matched case", "Matched tile swap reverses the teacher-forced coordinate margin", 11);
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
  slide.speakerNotes.textFrame.setText("9:10–10:00 — Introduce the matched causal pair. The pixel corruption crops both equal-size tiles from the original before either paste, then exchanges them without resizing or resampling; 0.91% of pixels change. The swap creates a localized visual counterfactual while preserving task wording and layout. It may alter recognition as well as coordinate binding. The teacher-forced coordinate margin is the log probability of the correct six x/y value tokens minus the log probability of the distractor coordinate tokens. Syntax tokens are excluded. A nat is a natural-log unit, so exp(margin) is the target-to-distractor sequence-probability ratio. +1.70 nats means 5.47× target preference. −2.64 nats means 14.07× distractor preference. Source: artifacts/causal-intervention/powerpoint_windows_59_swap/results.json.");
}

// 12 - causal results
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Causal intervention · one matched case", "Target-region residuals recover one-third of the coordinate margin", 12);
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
  slide.speakerNotes.textFrame.setText("10:00–10:55 — The schematic defines activation patching: capture one component from the clean run, replace the corresponding component in the corrupted run, then measure the change in teacher-forced coordinate margin. Patching all clean layer-19 visual-token residuals restores 1.85 nats, or 42.7% of the clean–corrupt gap. Restricting the patch to four target-overlapping tokens restores 1.44 nats, or 33.2%. Mean-replacing those same clean tokens removes 1.48 nats. The isolated layer-19/head-10 attention output restores only 0.024 nats, so the causal claim stays regional and residual-stream level for this one matched case. The chart leads with those three contrasts; the distractor and MLP probes remain in the secondary line. Source: artifacts/causal-intervention/powerpoint_windows_59_swap/results.json.");
}

// 13 - standalone Qwen-to-Holo delta lens
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Fine-tuning delta lens", "Action margins improve without stronger value-weighted target grounding", 13);
  textBox(slide, "Same native screenshot and official prompt, with independent oracle coordinates for four alternate instructions per checkpoint", 64, 114, 1152, 22, { fontSize: 13, color: C.muted });

  const panelCrop = { left: 0, top: 0.04, right: 0.29, bottom: 0.38 };
  const panelWidth = 360;
  const panelTop = 154;
  const panelHeight = 190;
  const panelLefts = [64, 460, 856];
  const panelLabels = [
    `QWEN · TARGET − CONTROL +${(pairedValue.base.prompt_difference_target_region_mass * 100).toFixed(1)} PP`,
    `HOLO · TARGET − CONTROL +${(pairedValue.tuned.prompt_difference_target_region_mass * 100).toFixed(1)} PP`,
    `HOLO − QWEN · ${(pairedValue.delta_prompt_difference_target_region_mass * 100).toFixed(1)} PP AT TARGET`,
  ];
  await image(slide, qwenAttention, panelLefts[0], panelTop, panelWidth, panelHeight, { alt: "Qwen value-norm target minus same-image instruction-control attribution", crop: panelCrop });
  await image(slide, holoAttention, panelLefts[1], panelTop, panelWidth, panelHeight, { alt: "Holo value-norm target minus same-image instruction-control attribution", crop: panelCrop });
  await image(slide, attentionDifference, panelLefts[2], panelTop, panelWidth, panelHeight, { alt: "Signed Holo minus Qwen value-norm prompt-difference attribution", crop: panelCrop });
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
  textBox(slide, `Paired value-norm ${(pairedValue.delta_prompt_difference_target_region_mass * 100).toFixed(2)} pp · LOO ≥${Math.min(pairedValue.base.minimum_leave_one_out_cosine, pairedValue.tuned.minimum_leave_one_out_cosine).toFixed(2)} · raw +${(pairedDirect.delta_prompt_difference_target_region_mass * 100).toFixed(2)} pp`, 642, 628, 552, 22, { fontSize: 13, bold: true, color: "#F5E2B8", alignment: "right" });
  footer(slide, "Native input ceiling: ScreenSpot retains 99.6% area; hotel frames retain 97.8% after patch alignment · attention is descriptive, not causal");
  slide.speakerNotes.textFrame.setText("10:55–12:00 — This is the standalone fine-tuning comparison. The layer chart teacher-forces identical oracle and distractor action tokens through the official Qwen3.5-4B base and Holo3.1-4B, using the same Holo processor, prompts, image order, and action suffixes. All five ScreenSpot probes end with a positive Holo-minus-Qwen oracle-margin shift, averaging 1.14 nats per scored token; three cross from a negative base margin to a positive Holo margin. The hotel final click ends at +0.37 nats per token, while earlier hotel steps are mixed. The top triptych is a separate paired instruction-control diagnostic on powerpoint_windows_59. Each checkpoint sees the same native-resolution screenshot and official localization prompt for the target plus four visible alternate tasks: blank presentation, Scientific discovery, Animal magnetism, and the template search box. Every task uses an independently image-derived oracle coordinate, so neither model's rollout selects a control target. We separately L1-normalize the five value-norm maps, subtract the mean control map from the target map within each checkpoint, then subtract Qwen from Holo. The target-minus-control mass is 13.15 percentage points for Qwen and 12.37 for Holo, so the paired Holo-minus-Qwen change is −0.78 point. Leave-one-control-out cosine remains at least 0.949 for Qwen and 0.950 for Holo. Raw attention again moves in the opposite direction at +1.36 points. This one case supports a fine-tuning shift in action preference, but not stronger value-weighted target grounding. In the three-frame hotel prompt, Holo moves value-norm image attention away from the earliest frame and toward the middle and current frames. The ScreenSpot processor grid is 56 by 90 from a 2880 by 1800 source, retaining 99.6% of source area; each hotel frame uses a 22 by 32 grid from 1024 by 720, retaining 97.8%. Attention is descriptive; the layerwise logit lens also includes each checkpoint's own final norm and unembedding. Sources: data/remote-results/delta-lens-qwen35-vs-holo31-20260921-dcd8c5e5, data/local-results/attention-delta-ppt59-controls-v1, artifacts/screenspot-presentation/delta-lens-paired-controls-v1/paired-control-attention-summary.json, and data/local-results/attention-delta-hotel-step2-native-v1.");
}

}

// 12 - future research directions
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Next tests", "Completed evidence narrows the next experiments", 12);
  const directions = [
    ["01", "TOKEN GRID VS RESAMPLING", "Hold the visual-token grid fixed while varying blur and resampling, then vary grid size with matched image content.", C.orangeSoft, C.orange],
    ["02", "HELD-OUT CAUSAL GENERALIZATION", "Repeat the Layer 15 clean-state patch on independent screenshots, applications, and layouts with a preregistered paired test.", C.blueSoft, C.blue],
    ["03", "CONTROL-CORRECTED VISUAL MEMORY", "Add same-image alternate-instruction controls to the matched hotel history, then remove or patch retained frames at the final click.", C.greenSoft, C.green],
    ["04", "STOPPING AND ACTION ABLATION", "Cross checkpoint, tool-call contract, and action budget to separate coordinate policy from continued scrolling and premature clicks.", "#F3EBDD", C.gold],
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
  textBox(slide, "COMPLETED EVIDENCE", 90, 570, 190, 20, { fontSize: 13, bold: true, color: C.gold });
  textBox(slide, "Resolution attribution, one matched hotel trajectory, and the Layer 15 causal panel", 286, 564, 850, 30, { fontSize: 19, bold: true, color: C.white, alignment: "center" });
  textBox(slide, "The next experiments isolate image sampling, output contracts, visual memory, and held-out generalization.", 286, 601, 850, 20, { fontSize: 13, color: "#B9C3CD", alignment: "center" });
  footer(slide, "Mechanism claims now require controlled interventions beyond the completed cohorts");
  slide.speakerNotes.textFrame.setText("9:20–10:10 — The 18-case resolution-attribution cohort is complete, so it is no longer listed as future work. The next resolution experiment should separate reduced spatial sampling from resampling artifacts by independently controlling image blur and visual-grid size. The complete matched hotel case motivates two tests: add same-image alternate-instruction controls to the multi-frame trace, then intervene on retained frames; and repeat the matched comparison across new fixtures with a validated completion budget to test stopping and action readout. The Layer 15 panel has a paired effect across eight mirrored prompts, but it still needs independent screenshots and applications before a broad mechanism claim. Sources: artifacts/screenspot-presentation/resolution-saliency-cohort18-v1/summary.json; artifacts/screenspot-presentation/hotel-freegen-paired-v1-attribution/trajectory.json; artifacts/screenspot-presentation/causal-panel-statistics-v1.json.");
}

// 13 - takeaways
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  slideTitle(slide, "Key takeaways", "Resolution shapes grounding\nFine-tuning strengthens recoverable action state", 13, true);
  const takeaways = [
    ["01", "Resolution changes grounding", "Only 4 of 36 clicks survive; cohort target mass falls 24× at quarter resolution."],
    ["02", "Attention can outlive the click", "Across reduced-resolution runs, 7 of 29 misses still rank the oracle patch first."],
    ["03", "Stopping differs in one matched case", "Both models track €166. Holo clicks after two scrolls, while Qwen keeps searching until the action cap."],
  ];
  takeaways.forEach(([num, title, desc], i) => {
    const y = 150 + i * 122;
    pill(slide, num, 64, y, 52, i === 0 ? C.orange : C.blue, C.white);
    textBox(slide, title, 138, y + 1, 430, 26, { fontSize: 20, bold: true, color: C.white });
    textBox(slide, desc, 138, y + 35, 430, 48, { fontSize: 15, color: "#B7C1CB" });
  });
  rect(slide, 646, 146, 570, 414, C.white, true);
  textBox(slide, "WHAT THE CAUSAL PANEL ADDS", 676, 174, 360, 22, { fontSize: 13, bold: true, color: C.orange });
  const causalTakeaways = [
    ["1", "Recoverable Layer 15 state", "Clean Holo state restores +1.01 nats; Qwen restores +0.10."],
    ["2", "Shared coordinate machinery", "Zeroing the same state harms both checkpoints."],
    ["3", "Distributed mechanism", "No single attention head explains the restoration effect."],
  ];
  causalTakeaways.forEach(([num, title, desc], i) => {
    const y = 226 + i * 98;
    pill(slide, num, 676, y, 38, i === 0 ? C.orange : C.deep, C.white);
    textBox(slide, title, 730, y + 1, 430, 24, { fontSize: 17, bold: true });
    textBox(slide, desc, 730, y + 31, 430, 38, { fontSize: 14, color: C.muted });
  });
  rect(slide, 64, 588, 1152, 50, "#202C38", true);
  textBox(slide, "Working interpretation: fine-tuning improves coordinate readout more clearly than value-weighted target grounding", 90, 598, 1098, 28, { fontSize: 19, bold: true, color: C.white, alignment: "center", verticalAlignment: "middle" });
  footer(slide, "End of main deck. Attention is descriptive. The eight-prompt intervention is causal within one image.", true);
  slide.speakerNotes.textFrame.setText("10:10–11:10 — Close the main deck on the combined evidence. ScreenSpot replication shows that geometry and resolution strongly condition localization. Only four of 36 native-success clicks survive at quarter resolution, and the balanced 18-case attribution cohort finds a 24-fold drop in median target-specific mass. Most reduced-resolution misses also lose target localization, but 7 of 29 still place the highest task-specific patch inside the annotated target. In the complete matched hotel case, both checkpoints record €166 as the lowest observed price. Holo terminates with the correct click after two scrolls, while Qwen spends all four actions scrolling. The separately traced Holo replicate carries value-weighted attention across all three retained frames. This one case motivates a stopping and action-readout hypothesis but does not estimate checkpoint accuracy. The causal panel adds the strongest model-comparison evidence: Holo develops a more recoverable Layer 15 coordinate state than Qwen, while both models rely on that state and no single attention head explains the effect. The current hypothesis is that fine-tuning strengthens action-state formation and coordinate extraction more clearly than value-weighted target grounding. Slides A1 and A2 are backup material for methods and benchmark taxonomy.");
}

// 14 - methods appendix
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Appendix · definitions", "From generated coordinate tokens back to image patches", "A1");
  textBox(slide, "A attention weights · V value vectors · R rolled-out token influence · M image-patch map for generated x/y value tokens", 64, 118, 1152, 22, { fontSize: 13, color: C.muted });
  const blocks = [
    ["1", "VALUE-NORM", "A′ = normalize(A ⊙ ‖V‖₂)", "Down-weight attention paths whose value vectors carry little magnitude", C.orangeSoft, C.orange],
    ["2", "CROSS-LAYER ROLLOUT", "Rₗ = ½I + ½A′ₗ\nR = Rₗ ··· R₁", "Compose all eight captured full-attention blocks with residual flow", C.blueSoft, C.blue],
    ["3", "TOKEN TO PARAMETER", "M(x,y) = meanₛ Rₛ", "Average the x/y value-token maps so token length cannot inflate a parameter", C.greenSoft, C.green],
  ];
  blocks.forEach(([num, label, equation, desc, fill, accent], i) => {
    const left = 64 + i * 395;
    rect(slide, left, 154, 356, 230, C.white, true, C.line);
    pill(slide, num, left + 22, 175, 46, accent, C.white);
    textBox(slide, label, left + 84, 179, 230, 22, { fontSize: 13, bold: true, color: accent });
    rect(slide, left + 22, 220, 312, 72, fill, true);
    textBox(slide, equation, left + 36, 232, 284, 48, { fontSize: 20, bold: true, alignment: "center", verticalAlignment: "middle" });
    textBox(slide, desc, left + 25, 316, 304, 52, { fontSize: 15, color: C.muted, alignment: "center" });
  });
  textBox(slide, "The baseline determines the question", 64, 422, 500, 30, { fontSize: 22, bold: true });
  rect(slide, 64, 468, 548, 156, C.deep, true);
  textBox(slide, "RAW VALUE-NORM MAP", 88, 490, 260, 20, { fontSize: 13, bold: true, color: "#9FC8F0" });
  textBox(slide, "Mtarget", 88, 526, 472, 28, { fontSize: 21, bold: true, color: C.white });
  textBox(slide, "Where does the generated coordinate draw value-weighted attention on this image?", 88, 567, 470, 38, { fontSize: 14, color: "#B9C3CD" });
  rect(slide, 650, 468, 566, 156, C.white, true, C.line);
  textBox(slide, "DIVERSE-INSTRUCTION BASELINE", 676, 490, 330, 20, { fontSize: 13, bold: true, color: C.orange });
  textBox(slide, "Mtarget − mean(Mcontrol,k)", 676, 526, 470, 28, { fontSize: 19, bold: true });
  textBox(slide, "What remains specific to this instruction after shared image saliency is removed?", 676, 567, 470, 38, { fontSize: 14, color: C.muted });
  footer(slide, "Target lift = positive target-minus-control mass inside the target box ÷ target-box area fraction. 1× means a uniform residual map.");
  slide.speakerNotes.textFrame.setText("Backup methods slide — Value-norm multiplies attention weights by the corresponding value-vector magnitude before normalization. Residual-aware rollout mixes each captured full-attention matrix with identity and composes the eight matrices in model order. We average only the generated x/y value-token maps. The raw map asks where the coordinate route goes. The diverse-instruction baseline subtracts four same-image alternate tasks and asks what remains specific to the target instruction. Target lift compares positive target mass with the target's share of image area. Hybrid Gated DeltaNet layers do not expose an equivalent square attention matrix and therefore sit outside this rollout.");
}

// 15 - appendix: examples for the 2 x 6 diagnostic taxonomy
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Appendix · benchmark taxonomy", "Examples for the 12 labeled UI × action cells", "A2");
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
  explicitTotalSlideCount: 15,
  requiredNativeTableOwnerSlides: [3, 6, 9, 15],
  requiredNativeChartOwnerSlides: [3, 4, 5],
  materializeLiteralChartWorkbooks: true,
};
const fontPolicy = { basis: "design", families: [family] };
const stagingDir = path.join(workspaceDir, ".codex-finalizer-screenspot-v55");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate-v55.pptx");
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
    "--require-native-table-slide", "6",
    "--require-native-table-slide", "9",
    "--require-native-table-slide", "15",
  ],
  requiredNativeTableOwnerSlides: [3, 6, 9, 15],
  requiredNativeChartOwnerSlides: [3, 4, 5],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${path.basename(FINAL_PPTX)}.validation.json`),
});
console.log(JSON.stringify({ final: FINAL_PPTX, font: family, result }, null, 2));
