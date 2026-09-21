import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/yaoyiheng/Documents/ChatGPT/GUI VLM Fine Tuning";
const SKILL_DIR = "/Users/yaoyiheng/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.61513/skills/presentations";
const TMP_DIR = path.join(workspaceDir, "artifacts/screenspot-presentation/build");
const FINAL_PPTX = path.join(workspaceDir, "artifacts/screenspot-presentation/holo-attribution-research-v3.pptx");
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
const hotelContrast = path.join(
  workspaceDir,
  "data/attributions/hotel-cheapest-multiframe-contrast",
);
const hotelDiffMaps = [0, 1, 2, 3].map((frame) =>
  path.join(
    hotelContrast,
    `preview-frame-${String(frame).padStart(3, "0")}-target-minus-prompt-baseline.png`,
  ),
);

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
  textBox(slide, "Action-token attribution for static GUI grounding and multi-turn visual memory", 68, 340, 535, 104, { fontSize: 24, color: "#CBD3DB" });
  textBox(slide, "Holo 3.1 4B · local Metal traces · 21 Sep 2026", 68, 605, 535, 28, { fontSize: 14, bold: true, color: "#97A4B2" });
  slide.speakerNotes.textFrame.setText("0:00–0:35 — Frame the question: what is specific to the current instruction after removing generic image saliency, and can the same machinery follow evidence across a multi-turn trajectory? The visual is the final prompt-differential map for the ScreenSpot visual hit. Source: data/attributions/screenspot-powerpoint_windows_59/analysis.json.");
}

// 2 — two evaluation settings
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Evaluation", "Static grounding and trajectory memory expose different failures", 2);
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
  slide.speakerNotes.textFrame.setText("0:35–1:25 — Separate the two settings. ScreenSpot gives real high-resolution grounding examples. The synthetic hotel task gives exact replay and multi-frame memory. The two PowerPoint examples are not a same-image success/failure pair, so use them to motivate hypotheses rather than causal claims. The hotel trace is a Holo failure after three scrolls and one click. Sources: docs/screenspot-case-study.md and data/attributions/traj-20260919T164806Z-ba81a14a16/trajectory.json.");
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
  slideTitle(slide, "Visual grounding hit", "Subtracting image saliency reveals the selected template", 4);
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
  slide.speakerNotes.textFrame.setText("2:35–3:45 — This slide establishes the measurement we use from here onward. The raw map is dominated by stable top-left saliency. With four diverse same-image prompts averaged after per-request L1 normalization, the peak moves inside the annotated template and lift rises from 3.85× to 16.08×. Leave-one-control-out cosine is 0.925 mean and 0.828 minimum. The repaired click is visually correct at (1394, 632), but the original tool syntax is malformed, so official correctness remains false. Source: data/attributions/screenspot-powerpoint_windows_59/analysis.json.");
}

// 5 — failure
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Grounding miss", "The right region wins, but the wrong affordance receives the click", 5);
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
  slide.speakerNotes.textFrame.setText("3:45–4:45 — From this point, headline the same-image instruction differential rather than raw attention. The differential favors the top-left slide and ribbon region, yet its peak is on the slide thumbnail and the click lands there. The target itself is only 0.074% of the frame. This is compatible with fine-grained binding or localization failure after coarse semantic routing. Source: data/attributions/screenspot-powerpoint_windows_48/analysis.json.");
}

// 6 - multi-turn trajectory
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Multi-turn prompt baseline", "The right hotel is retrieved, then its button is missed", 6);
  textBox(slide, "Target minus same-history instruction ensemble for the final x/y coordinate tokens", 64, 122, 930, 25, { fontSize: 15, color: C.muted });
  const labels = ["FRAME 0 · START", "FRAME 1 · SCROLL 500", "FRAME 2 · TARGET APPEARS", "FRAME 3 · CLICK"];
  for (let i = 0; i < hotelDiffMaps.length; i += 1) {
    const x = 64 + i * 286;
    await image(slide, hotelDiffMaps[i], x, 166, 270, 169, { alt: `Hotel trajectory prompt-differential frame ${i}` });
    pill(slide, labels[i], x + 10, 346, 220, i === 3 ? C.orange : C.deep, C.white);
  }
  metricCard(slide, 64, 398, 350, "Earlier-frame target lift", "2.78×", C.blue, "frame 2: target button at viewport edge");
  metricCard(slide, 432, 398, 350, "Current-frame target lift", "8.59×", C.orange, "frame 3: prompt differential");
  metricCard(slide, 800, 398, 416, "Issued click", "MISS", C.orange, "(960,120): correct card, wrong affordance");
  rect(slide, 64, 530, 1152, 108, C.deep, true);
  textBox(slide, "Interpretation", 90, 551, 150, 22, { fontSize: 13, bold: true, color: C.gold });
  textBox(slide, "Semantic retrieval survives across frames; precise action binding does not.", 90, 579, 680, 34, { fontSize: 24, bold: true, color: C.white });
  textBox(slide, "Peak remains outside the button. Attention is diagnostic evidence, not a causal explanation of the miss.", 800, 554, 374, 58, { fontSize: 15, color: "#C4CED7" });
  footer(slide, "3 included controls · 1 excluded without a y span · LOO cosine 0.898 mean / 0.769 minimum");
  slide.speakerNotes.textFrame.setText("4:45–6:05 — Every probe receives the identical four screenshots and identical three-scroll action history. Only the requested click target changes. We normalize each request across every patch in all four frames, average three usable control maps, then subtract. The cheapest hotel button has 2.78× lift when it first appears at the bottom of frame 2 and 8.59× in the current frame. Yet the peak stays outside the button and Holo clicks (960,120), on the Signal Quay card rather than View details. One fourth control was excluded because generation omitted a y parameter span. The cautious claim is semantic retrieval with failed affordance localization, not causal proof. Source: data/attributions/hotel-cheapest-multiframe-contrast/analysis.json.");
}

// 7 - layers and candidate confidence signals
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Layer lens", "Mid-late routing may be an early correctness feature", 7);
  const categories = ["L3", "L7", "L11", "L15", "L19", "L23", "L27", "L31"];
  const chart = slide.charts.add("line", {
    position: { left: 64, top: 154, width: 724, height: 438 },
    categories,
    series: [
      { name: "Static visual hit", values: [0.28, 0.69, 1.90, 14.26, 38.99, 25.62, 22.37, 12.16], line: { fill: C.orange, width: 4 }, marker: { symbol: "circle", size: 8 } },
      { name: "Static miss", values: [1.71, 7.60, 4.02, 3.27, 2.88, 2.73, 2.37, 4.26], line: { fill: C.blue, width: 4 }, marker: { symbol: "circle", size: 8 } },
      { name: "Hotel miss", values: [0.53, 0.29, 2.74, 7.65, 10.47, 4.36, 6.79, 5.71], line: { fill: C.green, width: 4 }, marker: { symbol: "circle", size: 8 } },
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
  rect(slide, 820, 154, 396, 438, C.deep, true);
  pill(slide, "LAYER 19", 850, 182, 108, C.orange, C.white);
  textBox(slide, "38.99×", 850, 232, 148, 43, { fontSize: 36, bold: true, color: C.white });
  textBox(slide, "static hit", 1000, 245, 150, 24, { fontSize: 15, color: "#F2BDAF" });
  textBox(slide, "10.47×", 850, 292, 148, 36, { fontSize: 27, bold: true, color: C.white });
  textBox(slide, "hotel miss", 1000, 301, 150, 24, { fontSize: 15, color: "#8DDBC1" });
  textBox(slide, "2.88×", 850, 340, 148, 36, { fontSize: 27, bold: true, color: C.white });
  textBox(slide, "static miss", 1000, 349, 150, 24, { fontSize: 15, color: "#9FC8F0" });
  rect(slide, 850, 397, 336, 1, "#465563");
  textBox(slide, "Prospective test", 850, 420, 230, 22, { fontSize: 14, bold: true, color: C.gold });
  textBox(slide, "Fit a tiny outcome or ambiguity readout on frozen layer features.", 850, 451, 318, 54, { fontSize: 19, bold: true, color: C.white });
  textBox(slide, "Evaluate held-out Brier score, ECE, and selective risk. N=3 examples cannot establish confidence.", 850, 520, 318, 53, { fontSize: 14, color: "#C4CED7" });
  footer(slide, "Per-head rows use direct value-norm attention; multiplied rollout has no unique head identity");
  slide.speakerNotes.textFrame.setText("6:05–7:20 — Add the hotel miss to the layer view. The visual hit separates sharply at layers 15 to 27, while the two misses remain lower, although the hotel miss has a moderate layer-19 signal. This suggests a candidate feature family, not a calibrated confidence score. A proper experiment freezes the backbone, fits a tiny readout on training cases, and evaluates Brier score, expected calibration error, and selective risk on a held-out split. Head-level examples remain direct value-norm attention because cross-layer rollout destroys unique head identity. Sources: layer_head_statistics in the three local analysis files.");
}

// 8 - research directions
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  slideTitle(slide, "Research directions", "Turn a diagnostic into testable model improvements", 8);
  const directions = [
    ["01", "CALIBRATED DECISION HEADS", "Predict action success, ambiguity, and safety-refusal risk. Train with proper scoring rules; trigger clarification or abstention by calibrated thresholds.", C.orangeSoft, C.orange],
    ["02", "CAUSAL VALIDATION", "Mask, patch-swap, ablate, or activation-patch the highlighted regions. Ask whether changing the routed evidence changes the action.", C.blueSoft, C.blue],
    ["03", "LONG-HORIZON VISUAL MEMORY", "Measure which historical frame supplies evidence, then learn retrieval or compression that preserves task-relevant patches without full-frame quadratic cost.", "#E4F4EE", C.green],
    ["04", "FINE-TUNING DELTA LENS", "Teacher-force identical action tokens through base Qwen and Holo. Compare where instruction-specific routing appears, disappears, or sharpens.", "#F3EBDD", C.gold],
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
  textBox(slide, "Oracle SFT  →  measured mid-training lift  →  GRPO + LoRA with verifiable environment reward", 264, 566, 902, 30, { fontSize: 21, bold: true, color: C.white, alignment: "center" });
  textBox(slide, "Use attribution for audit and diagnosis, not as the reward target.", 264, 602, 902, 20, { fontSize: 14, color: "#B9C3CD", alignment: "center" });
  footer(slide, "The highest-value next result is an intervention, not a prettier heatmap");
  slide.speakerNotes.textFrame.setText("7:20–8:35 — Four directions follow directly from the evidence. First, calibrated heads can support success prediction, ambiguity clarification, and safety refusal, but must be judged by proper scoring rules and held-out selective risk. Second, causal interventions test faithfulness. Third, the full-resolution four-frame probe exposed a quadratic memory bottleneck, motivating learned visual memory retrieval or compression. Fourth, a teacher-forced base-Qwen versus Holo comparison can localize what tool fine-tuning changed. The training program remains oracle SFT, measured lift, then GRPO plus LoRA with verifiable reward. Do not optimize the heatmap itself.");
}

// 9 - takeaways and demo
{
  const slide = presentation.slides.add();
  slide.background.fill = C.ink;
  slideTitle(slide, "Takeaways", "A baseline-first story, with falsifiable next steps", 9, true);
  const takeaways = [
    ["01", "Remove general awareness", "Same-image instruction controls make target-specific routing visible."],
    ["02", "Separate retrieval from binding", "Both misses find a relevant region but fail at precise affordance selection."],
    ["03", "Treat layers as candidate features", "Calibrate and intervene before calling them confidence or mechanism."],
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
    ["1", "Static hit: raw → prompt difference", "Peak enters the selected template."],
    ["2", "Static miss: inspect click binding", "Correct task region, wrong tiny affordance."],
    ["3", "Hotel trace: step through four frames", "Evidence appears before the failed final click."],
    ["4", "Sort layer and head statistics", "Turn observations into intervention hypotheses."],
  ];
  demo.forEach(([num, title, desc], i) => {
    const y = 218 + i * 78;
    pill(slide, num, 676, y, 38, i === 2 ? C.orange : C.deep, C.white);
    textBox(slide, title, 730, y + 1, 430, 24, { fontSize: 17, bold: true });
    textBox(slide, desc, 730, y + 29, 430, 22, { fontSize: 14, color: C.muted });
  });
  rect(slide, 64, 588, 1152, 50, "#202C38", true);
  textBox(slide, "Next decisive experiment: patch intervention across a frozen, matched evaluation split", 90, 598, 1098, 28, { fontSize: 20, bold: true, color: C.white, alignment: "center", verticalAlignment: "middle" });
  footer(slide, "Viewers: ScreenSpot hit + miss · data/attributions/hotel-cheapest-multiframe-contrast/viewer.html", true);
  slide.speakerNotes.textFrame.setText("8:35–10:00 — Close on three defensible conclusions. The prompt ensemble changes the measurement materially. Correct and incorrect cases can share coarse routing while differing in precise binding. Layer patterns are candidate readout features, not confidence by themselves. In the live demo, show raw versus prompt differential on the hit, the static miss, then step through all four hotel frames and point out the earlier-frame and final-frame target lift. Finish with the intervention experiment. Q&A: attention is a routing diagnostic, the controls are matched but small, and these three cases are hypothesis-generating rather than population estimates.");
}

const requirements = {
  explicitTotalSlideCount: 9,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [7],
  materializeLiteralChartWorkbooks: true,
};
const fontPolicy = { basis: "design", families: [family] };
const stagingDir = path.join(workspaceDir, ".codex-finalizer-screenspot");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "candidate-v3.pptx");
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
  ],
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [7],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${path.basename(FINAL_PPTX)}.validation.json`),
});
console.log(JSON.stringify({ final: FINAL_PPTX, font: family, result }, null, 2));
