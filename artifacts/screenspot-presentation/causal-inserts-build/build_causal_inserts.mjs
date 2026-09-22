import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/yaoyiheng/Documents/ChatGPT/GUI VLM Fine Tuning";
const SKILL_DIR = "/Users/yaoyiheng/.codex/plugins/cache/openai-primary-runtime/presentations/26.921.11914/skills/presentations";
const RUNTIME_PYTHON = "/Users/yaoyiheng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const TMP_DIR = path.join(workspaceDir, "artifacts/screenspot-presentation/causal-inserts-build");
const FINAL_PPTX = path.join(workspaceDir, "artifacts/screenspot-presentation/causal-inserts-output/causal-intervention-inserts-v6.pptx");
const STAGING_DIR = path.join(TMP_DIR, ".codex-finalizer");
const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

const family = resolvePresentationFont();
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = {
  ink: "#10151C", paper: "#F5F2EB", white: "#FFFFFF", muted: "#66717D", line: "#D9D4C8",
  orange: "#F05A3C", orangeSoft: "#FCE2D9", blue: "#2D84D3", blueSoft: "#DDECF9",
  green: "#18A875", greenSoft: "#DDEFE7", deep: "#1C2733", gold: "#E5B249",
};
const causalDir = path.join(workspaceDir, "artifacts/causal-intervention/powerpoint_windows_59_swap");
const cleanFocus = path.join(causalDir, "clean-focus.png");
const corruptFocus = path.join(causalDir, "corrupted-focus.png");

function rect(slide, left, top, width, height, fill, radius = false, line = "none") {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill,
    line: line === "none" ? { fill: "none", width: 0 } : { fill: line, width: 1 },
  });
}

function text(slide, value, left, top, width, height, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox", position: { left, top, width, height }, fill: options.fill ?? "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    typeface: family, fontSize: options.fontSize ?? 20, bold: options.bold ?? false,
    color: options.color ?? C.ink, alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top", autoFit: options.autoFit ?? "shrinkText",
    wrap: "square", insets: options.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return shape;
}

function title(slide, kicker, value, page) {
  text(slide, kicker.toUpperCase(), 64, 38, 820, 20, { fontSize: 12, bold: true, color: C.blue });
  text(slide, value, 64, 62, 1050, 58, { fontSize: 38, bold: true });
  text(slide, String(page).padStart(2, "0"), 1176, 45, 40, 24, { fontSize: 13, bold: true, color: C.muted, alignment: "right" });
}

function footer(slide, value) {
  rect(slide, 64, 680, 1152, 1, C.line);
  text(slide, value, 64, 688, 1152, 16, { fontSize: 10, color: C.muted });
}

function label(slide, value, left, top, width, fill, color = C.white) {
  rect(slide, left, top, width, 28, fill, true);
  text(slide, value, left + 8, top + 3, width - 16, 21, { fontSize: 11, bold: true, color, alignment: "center", verticalAlignment: "middle" });
}

async function image(slide, source, left, top, width, height, alt) {
  const bytes = await fs.readFile(source);
  slide.images.add({
    blob: bytes, contentType: "image/png", alt, fit: "cover",
    position: { left, top, width, height }, geometry: "roundRect", borderRadius: 12,
  });
  rect(slide, left, top, width, height, "none", true, C.line);
}

// Insert slide 1: protocol and causal graph
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  title(slide, "Pre-registered Qwen to Holo experiment", "Where visual evidence becomes a coordinate action", 13);
  text(slide, "Completed on eight mirrored prompts: same official request, native image resolution, and teacher-forced coordinate strings", 64, 120, 1050, 24, { fontSize: 14, color: C.muted });

  label(slide, "CLEAN", 64, 164, 84, C.green);
  await image(slide, cleanFocus, 64, 202, 250, 166, "Original template row with target and distractor tile");
  text(slide, "Original image\nPsychedelic tile at x=482", 64, 378, 250, 44, { fontSize: 14, bold: true });
  label(slide, "CORRUPT", 64, 450, 84, C.orange);
  await image(slide, corruptFocus, 64, 488, 250, 166, "Equal-size target and distractor template tiles swapped");
  text(slide, "Only two 200×172 tiles swap\nNo resize, no prompt change", 64, 660, 250, 18, { fontSize: 12, color: C.muted });

  rect(slide, 344, 162, 872, 492, C.white, true, C.line);
  text(slide, "ONE CAUSAL SEQUENCE", 372, 188, 300, 18, { fontSize: 11, bold: true, color: C.blue });
  text(slide, "Image tokens precede the instruction. The final blocks cover every teacher-forced coordinate digit.", 372, 210, 760, 24, { fontSize: 14, color: C.muted });

  rect(slide, 392, 304, 760, 3, C.line);
  const stages = [
    { x: 392, w: 172, fill: C.blueSoft, edge: C.blue, top: "VISION TOKENS", bottom: "patches → image residuals" },
    { x: 586, w: 188, fill: "#E9EDF1", edge: C.deep, top: "INSTRUCTION TOKENS", bottom: "Create a \"Psychedelic\"…" },
    { x: 796, w: 94, fill: C.orangeSoft, edge: C.orange, top: "JSON", bottom: "{ \"x\": " },
    { x: 912, w: 112, fill: C.orangeSoft, edge: C.orange, top: "X DIGITS", bottom: "482  primary" },
    { x: 1046, w: 106, fill: C.greenSoft, edge: C.green, top: "Y DIGITS", bottom: "351  matched" },
  ];
  stages.forEach((stage, index) => {
    rect(slide, stage.x, 258, stage.w, 82, stage.fill, true, stage.edge);
    text(slide, stage.top, stage.x + 12, 270, stage.w - 24, 18, { fontSize: 11, bold: true, color: stage.edge, alignment: "center" });
    text(slide, stage.bottom, stage.x + 10, 295, stage.w - 20, 32, { fontSize: 12, bold: index === 3, alignment: "center", verticalAlignment: "middle" });
    if (index < stages.length - 1) text(slide, "+", stage.x + stage.w + 6, 282, 16, 22, { fontSize: 20, bold: true, color: C.muted, alignment: "center" });
  });

  text(slide, "Interchange clean activation into the corrupted run", 372, 382, 720, 22, { fontSize: 16, bold: true });
  rect(slide, 372, 418, 346, 142, C.blueSoft, true, C.blue);
  label(slide, "1. VISUAL EVIDENCE", 390, 438, 170, C.blue);
  text(slide, "Patch the target or distractor image-region residual. This tests sensory evidence, not instruction selectivity.", 390, 480, 304, 60, { fontSize: 14 });
  rect(slide, 746, 418, 414, 142, C.orangeSoft, true, C.orange);
  label(slide, "2. ACTION READOUT", 764, 438, 170, C.orange);
  text(slide, "Phase A patches all x+y positions. Phase B repeats the selected layer with x-only residual and head patches; y is matched in this horizontal panel.", 764, 480, 370, 60, { fontSize: 14 });
  text(slide, "Primary measure: Mxy = log p(target x,y) − log p(distractor x,y). Phase-B measure: Mx only. Compare clean→corrupt recovery and paired clean ablation in Qwen and Holo.", 372, 590, 788, 38, { fontSize: 13, color: C.muted });
  footer(slide, "Four equal-tile pairs × two mirrored instructions. Native 2880×1800 input. Official VisualLocalizerOutput JSON contract.");
  slide.speakerNotes.textFrame.setText("Completed preregistered experiment. Prompt contract: src/demo/screenspot.py build_screenspot_request. Phase-A manifests: benchmarks/activation_patching/qwen_holo_action_panel_v1. Phase-B manifests: benchmarks/activation_patching/qwen_holo_action_panel_v1_phase_b_layer15. Native 2880x1800; image tokens precede instruction tokens.");
}

// Insert slide 2: completed causal result
{
  const slide = presentation.slides.add();
  slide.background.fill = C.paper;
  title(slide, "Qwen to Holo causal contrast", "Fine-tuning strengthens a mid-layer action state", 14);
  text(slide, "No single-head circuit: eight mirrored swaps; Holo processor, official JSON contract, and native 2880×1800 input held fixed", 64, 120, 1090, 24, { fontSize: 14, color: C.muted });

  rect(slide, 64, 160, 550, 150, C.greenSoft, true, C.green);
  label(slide, "PHASE B · LAYER 15 · X-ONLY RESIDUAL", 84, 180, 260, C.green);
  text(slide, "Holo  +1.01 nats", 84, 222, 230, 31, { fontSize: 24, bold: true, color: C.green });
  text(slide, "Qwen  +0.10 nats", 330, 222, 210, 31, { fontSize: 24, bold: true, color: C.deep });
  text(slide, "Median clean→corrupt restoration · Holo positive in 7/8 pairs; paired Δ = +0.72 nats median", 84, 266, 490, 25, { fontSize: 13, color: C.muted });

  rect(slide, 638, 160, 578, 150, C.white, true, C.line);
  label(slide, "NORMALIZED RECOVERY", 658, 180, 170, C.blue);
  text(slide, "Holo  6.7%", 658, 222, 200, 31, { fontSize: 24, bold: true, color: C.blue });
  text(slide, "Qwen  1.3%", 880, 222, 200, 31, { fontSize: 24, bold: true, color: C.deep });
  text(slide, "Median residual recovery fraction · x-token teacher-forcing only", 658, 266, 480, 25, { fontSize: 13, color: C.muted });

  text(slide, "WHAT THE INTERVENTIONS ESTABLISH", 64, 342, 520, 18, { fontSize: 11, bold: true, color: C.blue });
  const findings = [
    ["1", "Mid-layer action state", "Layer-15 coordinate residual from the clean run causally restores the corrupted Holo coordinate preference more than Qwen.", C.greenSoft, C.green],
    ["2", "Not stronger necessity", "Zero-ablation is positive in both models, but its median effect is smaller in Holo (4.51 vs 6.27 nats).", "#E9EDF1", C.deep],
    ["3", "No single-head circuit", "All 16 heads were swept. Holo head 1 restores +0.54 nats, but Qwen also uses it (+0.36); the Holo effect is distributed.", C.orangeSoft, C.orange],
  ];
  findings.forEach(([num, heading, body, fill, accent], index) => {
    const y = 372 + index * 74;
    rect(slide, 64, y, 1152, 58, fill, true, accent);
    label(slide, num, 82, y + 15, 30, accent);
    text(slide, heading, 132, y + 11, 235, 20, { fontSize: 15, bold: true, color: accent });
    text(slide, body, 380, y + 10, 810, 38, { fontSize: 14, verticalAlignment: "middle" });
  });

  rect(slide, 64, 608, 1152, 42, C.deep, true);
  text(slide, "Working conclusion: Holo fine-tuning increases recoverability of a distributed layer-15 coordinate-action state. It does not justify a single-head circuit claim or a visual-understanding claim.", 84, 618, 1112, 20, { fontSize: 14, bold: true, color: C.white, alignment: "center" });
  footer(slide, "Within-image causal panel only: 4 tile pairs × 2 mirrored instructions. Results are not a benchmark accuracy estimate or an independent-image generalization claim.");
  slide.speakerNotes.textFrame.setText("Phase A results: 8 mirrored official localization requests, native 2880x1800. Layer-15 full coordinate residual restoration median: Holo 1.254 nats vs Qwen 0.298. Phase B was preregistered after the Phase-A Holo gate: x-only residual and all 16 heads at layer 15. Phase-B residual medians: Holo restoration 1.010 nats, recovery 0.067, ablation 4.514; Qwen restoration 0.102, recovery 0.013, ablation 6.274. Paired Holo-Qwen restoration difference median +0.718 in 7/8. Head 1: Holo median +0.535 vs Qwen +0.356, so no Holo-exclusive head. Sources: data/remote-results/qwen-holo-action-panel-v1-20260922 and data/remote-results/qwen-holo-action-panel-v1-20260922-phase-b-layer15.");
}

await fs.mkdir(STAGING_DIR, { recursive: true });
const candidatePath = path.join(STAGING_DIR, "candidate-causal-inserts-v6.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
  requiredNativeTableOwnerSlides: [],
  fontPolicy: { basis: "design", families: [family] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(STAGING_DIR, "causal-intervention-inserts-v6.validation.json"),
});

for (let index = 0; index < presentation.slides.items.length; index += 1) {
  const slide = presentation.slides.getItem(index);
  const png = await slide.export({ format: "png", scale: 1.5 });
  await fs.writeFile(path.join(TMP_DIR, `slide-${index + 1}.png`), new Uint8Array(await png.arrayBuffer()));
}
