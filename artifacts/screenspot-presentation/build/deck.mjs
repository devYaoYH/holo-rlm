import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/yaoyiheng/Documents/ChatGPT/GUI VLM Fine Tuning";
const SKILL_DIR = "/Users/yaoyiheng/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations";
const TMP_DIR = path.join(workspaceDir, "artifacts/screenspot-presentation/build");
const FINAL_PPTX = path.join(workspaceDir, "artifacts/screenspot-presentation/holo-screenspot-attribution-v2.pptx");
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

const successRaw = path.join(workspaceDir, "data/attributions/screenspot-powerpoint_windows_59/preview-raw.png");
const successDiff = path.join(workspaceDir, "data/attributions/screenspot-powerpoint_windows_59/preview-target-minus-prompt-baseline.png");
const failureDiff = path.join(workspaceDir, "data/attributions/screenspot-powerpoint_windows_48/preview-target-minus-prompt-baseline.png");

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

// 1 — cover
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  await image(slide, successDiff, 704, 0, 576, 720, { alt: "Instruction-specific saliency on the Psychedelic vibrant template", borderRadius: 0, geometry: "rect", line: "none", crop: { left: 0.28, top: 0.08, right: 0.28, bottom: 0.08 } });
  rect(slide, 640, 0, 126, 720, { type: "gradient", gradientKind: "linear", angleDeg: 0, stops: [{ offset: 0, color: C.ink }, { offset: 100000, color: "#10151C00" }] });
  pill(slide, "MECHANISTIC GUI ANALYSIS", 64, 62, 224, C.orange, C.white);
  textBox(slide, "When attention\nstops being generic", 64, 142, 590, 168, { fontSize: 55, bold: true, color: C.white });
  textBox(slide, "A ScreenSpot-Pro case study of value-norm rollout, causal token baselines, and same-image instruction contrasts", 68, 340, 535, 104, { fontSize: 24, color: "#CBD3DB" });
  textBox(slide, "Holo 3.1 4B · local Metal traces · 20 Sep 2026", 68, 605, 535, 28, { fontSize: 14, bold: true, color: "#97A4B2" });
  slide.speakerNotes.textFrame.setText("0:00–0:45 — Frame the question: not merely where attention is high, but what is specific to the instruction after removing general image saliency. The visual is the final prompt-differential map for the success case. Source: local analysis at data/attributions/screenspot-powerpoint_windows_59/analysis.json.");
}

// 2 — benchmark protocol
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Question", "Can attention explain a GUI hit—and a miss?", 2);
  rect(slide, 64, 144, 548, 458, C.white, true, C.line);
  textBox(slide, "Why ScreenSpot-Pro", 92, 170, 250, 30, { fontSize: 22, bold: true });
  richText(slide, [
    [{ run: "1,581", textStyle: { bold: true, color: C.orange } }, " high-resolution screenshot/instruction pairs"],
    [{ run: "23 apps", textStyle: { bold: true } }, " across 3 operating systems and 5 industries"],
    ["Average target occupies only ", { run: "0.07%", textStyle: { bold: true, color: C.orange } }, " of the frame"],
  ], 92, 224, 468, 130, { fontSize: 20 });
  rect(slide, 92, 382, 468, 1, C.line);
  textBox(slide, "Controlled pair", 92, 405, 220, 25, { fontSize: 16, bold: true, color: C.blue });
  textBox(slide, "Same application + OS\nPowerPoint on Windows", 92, 438, 190, 72, { fontSize: 20, bold: true });
  textBox(slide, "Visual hit\nPsychedelic vibrant", 310, 438, 120, 72, { fontSize: 16, bold: true, color: C.green });
  textBox(slide, "Miss\nCreate new slide", 450, 438, 110, 72, { fontSize: 16, bold: true, color: C.orange });
  rect(slide, 650, 144, 566, 458, C.deep, true);
  textBox(slide, "Three outcomes—not one", 680, 170, 330, 32, { fontSize: 22, bold: true, color: C.white });
  const outcomes = [
    ["01", "FORMAT VALID", "Strict tool schema"],
    ["02", "VISUAL GROUNDING", "Point-in-box after logged integer repair"],
    ["03", "OFFICIAL RESULT", "Formatting and grounding must both pass"],
  ];
  outcomes.forEach(([num, title, desc], i) => {
    const y = 232 + i * 100;
    pill(slide, num, 680, y, 50, i === 1 ? C.blue : "#344252", C.white);
    textBox(slide, title, 748, y + 1, 360, 22, { fontSize: 14, bold: true, color: C.white });
    textBox(slide, desc, 748, y + 28, 390, 35, { fontSize: 15, color: "#B9C3CD" });
  });
  textBox(slide, "A deterministic first-integer repair diagnoses vision separately. It never upgrades the official score.", 680, 528, 474, 48, { fontSize: 14, color: "#F0C7BC" });
  footer(slide, "Sources: ScreenSpot-Pro paper (arXiv:2504.07981) · official repository · official dataset");
  slide.speakerNotes.textFrame.setText("0:45–1:45 — ScreenSpot-Pro contains 1,581 unique examples across 23 apps, 3 OSes and 5 industries; the paper reports an average target area of 0.07%. Explain the controlled pair and the score separation. Sources: https://arxiv.org/abs/2504.07981 ; https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding ; https://huggingface.co/datasets/likaixin/ScreenSpot-Pro .");
}

// 3 — method
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Calculation", "From generated coordinate tokens back to image patches", 3);
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
  footer(slide, "Implementation: docs/attention-attribution.md · tests/test_attribution.py · tests/test_contrast.py");
  slide.speakerNotes.textFrame.setText("1:45–3:10 — Walk left to right. Value-norm changes the routing matrix before rollout. Rollout mixes attention and identity 50/50 and multiplies the eight conventional full-attention layers in model order. Then average only the generated x/y value tokens. Emphasize strict causality for the token baseline and exact same image/settings for prompt controls. Hybrid linear-attention blocks are omitted because they expose no square softmax matrix.");
}

// 4 — success raw vs prompt difference
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Success case", "Subtracting image saliency reveals the selected template", 4);
  textBox(slide, "“Create a Psychedelic vibrant presentation”", 64, 122, 780, 30, { fontSize: 19, bold: true, color: C.muted });
  await image(slide, successRaw, 64, 174, 548, 342, { alt: "Raw value-norm rollout on success case", crop: { left: 0, top: 0.02, right: 0.30, bottom: 0.31 } });
  await image(slide, successDiff, 668, 174, 548, 342, { alt: "Target minus diverse-instruction baseline on success case", crop: { left: 0, top: 0.02, right: 0.30, bottom: 0.31 } });
  pill(slide, "RAW ROLLOUT", 84, 192, 122, C.deep, C.white);
  pill(slide, "TARGET − PROMPT BASELINE", 688, 192, 228, C.orange, C.white);
  metricCard(slide, 64, 540, 252, "Target lift", "3.85×", C.muted, "raw rollout");
  metricCard(slide, 330, 540, 282, "Peak", "outside", C.muted, "generic top-left saliency");
  metricCard(slide, 668, 540, 252, "Target lift", "16.08×", C.orange, "instruction differential");
  metricCard(slide, 934, 540, 282, "Peak", "inside", C.green, "3.2% of frame diagonal away");
  footer(slide, "Local trace: powerpoint_windows_59 · 4 same-image controls · LOO cosine 0.925 mean / 0.828 minimum");
  slide.speakerNotes.textFrame.setText("3:10–4:25 — The raw map is dominated by stable top-left saliency. With four diverse same-image prompts averaged after per-request L1 normalization, the peak moves inside the annotated template and lift rises from 3.85× to 16.08×. Leave-one-control-out cosine is 0.925 mean and 0.828 minimum. The repaired click is visually correct at (1394, 632), but the original tool syntax is malformed, so official correctness remains false. Source: data/attributions/screenspot-powerpoint_windows_59/analysis.json.");
}

// 5 — failure
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Failure case", "Semantic neighborhood found; precise affordance missed", 5);
  textBox(slide, "Instruction: Create new slide", 80, 126, 500, 28, { fontSize: 19, bold: true, color: C.muted });
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
  textBox(slide, "The model resolves the intent class, but not the target binding needed for action.", 900, 554, 270, 34, { fontSize: 13, color: "#F1C7BC" });
  footer(slide, "Local trace: powerpoint_windows_48 · prompt-baseline stability 0.970 mean / 0.953 minimum");
  slide.speakerNotes.textFrame.setText("4:25–5:35 — This miss is not absence of task-conditioned vision. The instruction differential clearly favors the top-left slide/ribbon region, yet the peak is on the slide thumbnail and the click lands there. The target itself is only 0.074% of the frame. This suggests a fine-grained binding/localization failure after coarse semantic routing. Source: data/attributions/screenspot-powerpoint_windows_48/analysis.json.");
}

// 6 — comparison table
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Baseline comparison", "Different controls change the diagnosis", 6);
  textBox(slide, "Positive-mass target lift; peak status shown separately", 64, 122, 650, 25, { fontSize: 15, color: C.muted });
  const values = [
    ["Case", "Raw rollout", "Causal token", "Prompt differential", "Peak in target?", "LOO stability"],
    ["Visual hit · theme", "3.85×", "4.82×", "16.08×", "YES", "0.925 / 0.828"],
    ["Miss · new slide", "13.85×", "14.68×", "13.50×", "NO", "0.970 / 0.953"],
  ];
  const table = slide.tables.add({ rows: 3, columns: 6, left: 64, top: 176, width: 1152, height: 214, values });
  table.styleOptions = { headerRow: true, bandedRows: false };
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 6 }).assign({ fill: C.deep, textStyle: { typeface: family, color: C.white, fontSize: 14, bold: true } });
  table.cells.block({ row: 1, column: 0, rowCount: 2, columnCount: 6 }).assign({ fill: C.white, textStyle: { typeface: family, color: C.ink, fontSize: 16 } });
  table.getCell(1, 3).fill = C.orangeSoft;
  table.getCell(1, 4).fill = "#DFF4EA";
  table.getCell(2, 4).fill = C.orangeSoft;
  rect(slide, 64, 430, 548, 188, C.white, true, C.line);
  textBox(slide, "What each baseline removes", 90, 454, 330, 28, { fontSize: 20, bold: true });
  textBox(slide, "Causal token", 90, 504, 135, 22, { fontSize: 14, bold: true, color: C.blue });
  textBox(slide, "Generation state before x begins", 232, 504, 330, 22, { fontSize: 15, color: C.muted });
  textBox(slide, "Prompt ensemble", 90, 548, 135, 22, { fontSize: 14, bold: true, color: C.orange });
  textBox(slide, "Image-inherent saliency under other tasks", 232, 548, 330, 42, { fontSize: 15, color: C.muted });
  rect(slide, 650, 430, 566, 188, C.deep, true);
  textBox(slide, "Key caution", 678, 454, 170, 24, { fontSize: 16, bold: true, color: C.gold });
  textBox(slide, "Target lift alone is insufficient.", 678, 497, 486, 34, { fontSize: 25, bold: true, color: C.white });
  textBox(slide, "The miss has high lift because its tiny box sits near a broadly salient corner. Peak location and the overlay reveal the binding error.", 678, 543, 466, 58, { fontSize: 16, color: "#C4CED7" });
  footer(slide, "All maps use value-norm cross-layer rollout; signed maps are scored on positive attribution mass");
  slide.speakerNotes.textFrame.setText("5:35–6:45 — Compare the three methods. The causal baseline modestly changes both cases; the prompt ensemble transforms the success diagnosis. Stress that lift normalizes by target area and can be high for tiny targets near broadly salient regions. Read lift with peak-inside, distance, and the map itself. LOO stability shows the four-prompt control mean is not driven by one prompt. Sources: both local analysis.json files.");
}

// 7 — layers and heads
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Layer + head analysis", "A mid–late specialization emerges only in the hit", 7);
  const categories = ["L3", "L7", "L11", "L15", "L19", "L23", "L27", "L31"];
  const chart = slide.charts.add("line", {
    position: { left: 64, top: 154, width: 760, height: 442 },
    categories,
    series: [
      { name: "Visual hit", values: [0.28, 0.69, 1.90, 14.26, 38.99, 25.62, 22.37, 12.16], line: { fill: C.orange, width: 4 }, marker: { symbol: "circle", size: 8 } },
      { name: "Grounding miss", values: [1.71, 7.60, 4.02, 3.27, 2.88, 2.73, 2.37, 4.26], line: { fill: C.blue, width: 4 }, marker: { symbol: "circle", size: 8 } },
    ],
    hasLegend: true,
    legend: { position: "bottom" },
    lineOptions: { grouping: "standard", smooth: false },
    xAxis: { title: "Captured full-attention transformer layer" },
    yAxis: { title: "Mean head target lift", minimumScale: 0, maximumScale: 45, majorUnit: 10 },
    chartFill: C.white,
    chartLine: { fill: C.line, width: 1 },
    plotAreaFill: C.white,
    plotAreaLine: { fill: "none", width: 0 },
  });
  applyPresentationChartFont(chart, { fontFamily: family });
  rect(slide, 858, 154, 358, 442, C.deep, true);
  pill(slide, "SUCCESS", 886, 184, 94, C.orange, C.white);
  textBox(slide, "Layer 19", 886, 236, 260, 42, { fontSize: 34, bold: true, color: C.white });
  textBox(slide, "38.99× mean head lift", 886, 282, 270, 28, { fontSize: 18, color: "#F2BDAF" });
  rect(slide, 886, 336, 302, 1, "#465563");
  textBox(slide, "Head 10 · L19", 886, 362, 250, 24, { fontSize: 16, bold: true, color: "#9FC8F0" });
  textBox(slide, "98.05×", 886, 396, 250, 48, { fontSize: 42, bold: true, color: C.white });
  textBox(slide, "prompt-differential target lift", 886, 446, 260, 22, { fontSize: 13, color: "#AAB6C1" });
  textBox(slide, "Per-head rows are direct value-norm attention; a multiplied rollout no longer has unique head identity.", 886, 506, 286, 62, { fontSize: 14, color: "#C4CED7" });
  footer(slide, "Exploratory N=2 finding: layer/head claims require replication across the frozen test set");
  slide.speakerNotes.textFrame.setText("6:45–8:05 — The success develops sharply from layer 15, peaks at layer 19, and remains elevated through layer 27. The miss stays relatively flat. Within the hit, layer 19 head 10 reaches 98.05× target lift. Be precise: these head statistics use direct value-norm attention because head identity is not preserved after cross-layer multiplication. This is an exploratory two-case result, suitable for a hypothesis, not a pruning or intervention claim. Source: layer_head_statistics in both local analysis.json files.");
}

// 8 — takeaways and demo
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  slideTitle(slide, "Takeaways", "What we can say—and what to test next", 8, true);
  const takeaways = [
    ["01", "Baseline choice is substantive", "Prompt controls exposed target-specific routing that raw rollout hid."],
    ["02", "The miss is a binding failure", "The task region is found; the precise toolbar affordance is not."],
    ["03", "Layer 19 is a testable hypothesis", "Intervene or ablate; do not infer causality from attention alone."],
  ];
  takeaways.forEach(([num, title, desc], i) => {
    const y = 154 + i * 116;
    pill(slide, num, 64, y, 52, i === 0 ? C.orange : C.blue, C.white);
    textBox(slide, title, 138, y + 1, 430, 26, { fontSize: 20, bold: true, color: C.white });
    textBox(slide, desc, 138, y + 35, 430, 48, { fontSize: 15, color: "#B7C1CB" });
  });
  rect(slide, 646, 146, 570, 396, C.white, true);
  textBox(slide, "LIVE DEMO · 90 SECONDS", 676, 174, 320, 22, { fontSize: 13, bold: true, color: C.orange });
  textBox(slide, "1. Start on raw rollout", 676, 220, 430, 28, { fontSize: 20, bold: true });
  textBox(slide, "Generic corner saliency dominates.", 702, 254, 430, 24, { fontSize: 15, color: C.muted });
  textBox(slide, "2. Switch to target − prompt baseline", 676, 300, 470, 28, { fontSize: 20, bold: true });
  textBox(slide, "Peak moves into the success target.", 702, 334, 430, 24, { fontSize: 15, color: C.muted });
  textBox(slide, "3. Sort heads by prompt difference", 676, 380, 470, 28, { fontSize: 20, bold: true });
  textBox(slide, "Layer 19 / head 10 rises to the top.", 702, 414, 430, 24, { fontSize: 15, color: C.muted });
  textBox(slide, "4. Open the miss", 676, 460, 430, 28, { fontSize: 20, bold: true });
  textBox(slide, "Show region-level success but click-level failure.", 702, 494, 430, 24, { fontSize: 15, color: C.muted });
  rect(slide, 64, 576, 1152, 62, "#202C38", true);
  textBox(slide, "Next: intervention-based validation across the frozen ScreenSpot-Pro split", 90, 591, 1098, 34, { fontSize: 21, bold: true, color: C.white, alignment: "center", verticalAlignment: "middle" });
  footer(slide, "Viewers: data/attributions/screenspot-powerpoint_windows_{59,48}/viewer.html · code + docs in repository", true);
  slide.speakerNotes.textFrame.setText("8:05–10:00 — Summarize three claims, then run the 90-second viewer flow if time permits. Close on the next experiment: causal validation by masking, patch swapping, or controlled head/layer interventions across the frozen test split. Q&A seed: Why attention? It is a routing diagnostic, not a faithfulness guarantee. Why four prompts? Fast demo with LOO stability; scale to a larger, stratified control bank for publication. Sources: local viewers and https://arxiv.org/abs/2504.07981 .");
}

const requirements = {
  explicitTotalSlideCount: 8,
  requiredNativeTableOwnerSlides: [6],
  requiredNativeChartOwnerSlides: [7],
  materializeLiteralChartWorkbooks: true,
};
const fontPolicy = { basis: "design", families: [family] };
const stagingDir = path.join(workspaceDir, ".codex-finalizer-screenspot");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate-v2.pptx");
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
    "--require-native-table-slide", "6",
  ],
  requiredNativeTableOwnerSlides: [6],
  requiredNativeChartOwnerSlides: [7],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${path.basename(FINAL_PPTX)}.validation.json`),
});
console.log(JSON.stringify({ final: FINAL_PPTX, font: family, result }, null, 2));
