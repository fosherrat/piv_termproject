// Build slides/piv_pulse_coding.pptx (16:9) in the style of the presenter's earlier deck:
// navy title bar, navy footer with date / page / name / logo, light-blue boxes, grey key-message box.
// usage: NODE_PATH=<dir with pptxgenjs, react-icons, react, react-dom, sharp> node slides/build_deck.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const ROOT = path.resolve(__dirname, "..");
const F = (p) => path.join(ROOT, p);
const SKILL = process.env.PPTX_SKILL;
const LOGO = process.env.LOGO || F("slides/logo.png");
const RES = JSON.parse(fs.readFileSync(F("results/eval_paper.json"), "utf8"));
const DATE = "2026-11-26";
const NAME = "Minseo Park, SNU";

const THEME = {
  name: "PIV pulse coding",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "191919", lt1: "FFFFFF", dk2: "0F0F70", lt2: "E8E8E8",
    accent1: "257DE0", accent2: "F47230", accent3: "00B894", accent4: "F2A600",
    accent5: "DCE6F4", accent6: "7F7F7F", hlink: "257DE0", folHlink: "6A4CE2",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Resolving the directional ambiguity of pulse-coded double-exposure PIV";
pres.author = "Minseo Park";
const C = pres.SchemeColor;
const W = 13.333, H = 7.5;

// ------------------------------------------------------------------ layouts
const footer = () => [
  { rect: { x: 0, y: 6.95, w: W, h: 0.55, fill: { color: C.text2 } } },
  { text: { text: DATE, options: { x: 0.25, y: 7.0, w: 2.5, h: 0.45, fontSize: 12, color: C.background1, valign: "middle", margin: 0 } } },
  { text: { text: NAME, options: { x: 9.6, y: 7.0, w: 2.9, h: 0.45, fontSize: 12, color: C.background1, align: "right", valign: "middle", margin: 0 } } },
  { image: { path: LOGO, x: 12.72, y: 7.0, w: 0.44, h: 0.45 } },
];
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { rect: { x: 0, y: 0, w: W, h: 0.8, fill: { color: C.text2 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.05, w: W - 1, h: 0.7, fontSize: 26, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0 }, text: "" } },
    ...footer(),
  ],
  slideNumber: { x: 6.17, y: 7.0, w: 1.0, h: 0.45, fontSize: 12, color: C.background1, align: "center", valign: "middle" },
});
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: C.background1 },
  objects: [
    { rect: { x: 0, y: 0, w: W, h: 0.8, fill: { color: C.text2 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.9, w: W - 1.6, h: 1.8, fontSize: 36, bold: true, color: C.text1, align: "center", valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 1.5, y: 4.1, w: W - 3, h: 1.6, fontSize: 16, color: C.text1, align: "center", valign: "top" }, text: "" } },
    ...footer(),
  ],
  slideNumber: { x: 6.17, y: 7.0, w: 1.0, h: 0.45, fontSize: 12, color: C.background1, align: "center", valign: "middle" },
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: C.background1 },
  objects: [
    { rect: { x: 0, y: 0, w: W, h: 0.8, fill: { color: C.text2 } } },
    { placeholder: { options: { name: "title", type: "title", x: 1, y: 2.6, w: W - 2, h: 1.1, fontSize: 36, bold: true, color: C.text1, align: "center", valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 1.5, y: 3.7, w: W - 3, h: 0.8, fontSize: 18, color: C.accent6, align: "center", valign: "top" }, text: "" } },
    ...footer(),
  ],
  slideNumber: { x: 6.17, y: 7.0, w: 1.0, h: 0.45, fontSize: 12, color: C.background1, align: "center", valign: "middle" },
});

// ------------------------------------------------------------------ helpers
const T = 1.0, B = 6.75; // content area top / bottom
function img(slide, file, x, y, w, h, name) {
  slide.addImage({ path: F(file), x, y, w, h, altText: name || path.basename(file), objectName: name || path.basename(file) });
}
// fit an image (pixel size pw x ph) into a box, keeping aspect; returns placement
function fit(pw, ph, x, y, w, h, align = "center") {
  const s = Math.min(w / pw, h / ph);
  const ww = pw * s, hh = ph * s;
  const xx = align === "left" ? x : x + (w - ww) / 2;
  return { x: xx, y: y + (h - hh) / 2, w: ww, h: hh };
}
const SIZES = {};
function px(file) {
  if (!SIZES[file]) {
    const buf = fs.readFileSync(F(file));
    SIZES[file] = [buf.readUInt32BE(16), buf.readUInt32BE(20)];
  }
  return SIZES[file];
}
function pic(slide, file, x, y, w, h, align) {
  const [pw, ph] = px(file);
  const p = fit(pw, ph, x, y, w, h, align);
  img(slide, file, p.x, p.y, p.w, p.h);
  return p;
}
// equation image rendered at 22 pt / 220 dpi; scale to `pt`
function eq(slide, name, x, y, pt = 20, align = "left", boxw = null) {
  const file = `slides/fig/${name}.png`;
  const [pw, ph] = px(file);
  const k = pt / 22;
  const w = (pw / 220) * k, h = (ph / 220) * k;
  const xx = align === "center" && boxw ? x + (boxw - w) / 2 : x;
  img(slide, file, xx, y, w, h, name);
  return { w, h };
}
function keybox(slide, text, y = 5.95, h = 0.7) {
  slide.addShape(pres.shapes.RECTANGLE, { x: 0, y, w: W, h, fill: { color: C.accent6 }, line: { type: "none" }, objectName: "key message" });
  slide.addText(text, { x: 0.5, y, w: W - 1, h, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0 });
}
function card(slide, x, y, w, h, head, body, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: opts.fill || C.accent5 }, line: { color: opts.line || C.accent5, width: 1 }, objectName: "card " + head });
  const runs = [{ text: head, options: { bold: true, fontSize: opts.headSize || 17, color: opts.headColor || C.text2, breakLine: !!body } }];
  if (body) {
    const lines = Array.isArray(body) ? body : [body];
    lines.forEach((l, i) => runs.push({ text: l, options: { fontSize: opts.bodySize || 15, color: C.text1, breakLine: i < lines.length - 1, paraSpaceBefore: i === 0 ? 4 : 2 } }));
  }
  slide.addText(runs, { x: x + 0.18, y: y + 0.1, w: w - 0.36, h: h - 0.2, valign: opts.valign || "top", isTextBox: true, margin: 0, align: opts.align || "left" });
}
function bullets(slide, items, x, y, w, h, size = 17) {
  const runs = items.map((t, i) => {
    const sub = typeof t === "object" && t.sub;
    const txt = typeof t === "object" ? t.text : t;
    return { text: txt, options: { bullet: sub ? { indent: 18 } : true, indentLevel: sub ? 1 : 0, fontSize: sub ? size - 2 : size, color: sub ? C.accent6 : C.text1, breakLine: i < items.length - 1, paraSpaceAfter: 6 } };
  });
  slide.addText(runs, { x, y, w, h, valign: "top", isTextBox: true, margin: 0 });
}
function stat(slide, x, y, w, big, label, color, size = 40) {
  slide.addText(big, { x, y, w, h: 0.85, fontSize: size, bold: true, color: color || C.text2, align: "left", valign: "bottom", isTextBox: true, margin: 0 });
  slide.addText(label, { x, y: y + 0.88, w, h: 0.62, fontSize: 14, color: C.accent6, align: "left", valign: "top", isTextBox: true, margin: 0 });
}
async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
function circleIcon(slide, data, x, y, d, fill) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" } });
  const p = d * 0.25;
  slide.addImage({ data, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p });
}
function numCircle(slide, n, x, y, d, fill) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill || C.text2 }, line: { type: "none" } });
  slide.addText(String(n), { x, y, w: d, h: d, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", isTextBox: true, margin: 0 });
}
// particle-pair cartoon in a dark box: pairs = [[x, y, bright1, bright2]] in box-relative inches
function particles(slide, x, y, w, h, pairs, d, label) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: "101018" }, line: { type: "none" }, objectName: "particle cartoon" });
  pairs.forEach(([px_, py_, b1, b2]) => {
    const g1 = Math.round(255 * b1).toString(16).padStart(2, "0"), g2 = Math.round(255 * b2).toString(16).padStart(2, "0");
    slide.addShape(pres.shapes.OVAL, { x: x + px_, y: y + py_, w: 0.2, h: 0.2, fill: { color: g1 + g1 + g1 }, line: { type: "none" } });
    slide.addShape(pres.shapes.OVAL, { x: x + px_ + d[0], y: y + py_ + d[1], w: 0.2, h: 0.2, fill: { color: g2 + g2 + g2 }, line: { type: "none" } });
  });
  if (label) slide.addText(label, { x, y: y + h + 0.05, w, h: 0.4, fontSize: 14, color: C.text1, align: "center", isTextBox: true, margin: 0 });
}
function title(slide, t) { slide.addText(t, { placeholder: "title" }); }
function notes(slide, t) { slide.addNotes(t); }

// ------------------------------------------------------------------ slides
async function build() {
  const I = {
    chart: await icon(fa.FaChartLine, "FFFFFF"), brain: await icon(fa.FaBrain, "FFFFFF"),
    balance: await icon(fa.FaBalanceScale, "FFFFFF"), warn: await icon(fa.FaExclamationTriangle, "FFFFFF"),
    flask: await icon(fa.FaFlask, "FFFFFF"), camera: await icon(fa.FaCamera, "FFFFFF"),
    ruler: await icon(fa.FaRulerCombined, "FFFFFF"), sigma: await icon(fa.FaSuperscript, "FFFFFF"),
  };
  let s;

  // 1 title
  pres.addSection({ title: "Introduction" });
  s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
  s.addText("Resolving the directional ambiguity of\npulse-coded double-exposure PIV\nby learned exposure separation", { placeholder: "title" });
  s.addText("2023-16505, Minseo Park\nDepartment of Aerospace Engineering\nPIV term project", { placeholder: "body" });
  notes(s, "[0:20] 제목 소개. 한 장에 두 번 노출된 PIV 영상에서 속도의 방향을 구하는 문제를, 펄스 밝기 코딩과 U-Net 분리로 푼 연구입니다.");

  // 2 outline
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  title(s, "Outline");
  const out = [["Problem", "why a double exposure hides the flow direction"],
               ["Theory", "what pulse coding can and cannot resolve"],
               ["Method", "U-Net separation + standard multi-pass cross-correlation"],
               ["Fair comparison", "same processing for every method, 4 datasets"],
               ["Results", "outliers, direction errors, accuracy"],
               ["Inside the network", "what the U-Net actually decides"],
               ["Limitations & conclusion", ""]];
  out.forEach(([h, b], i) => {
    const y = 1.2 + i * 0.72;
    numCircle(s, i + 1, 2.2, y, 0.5);
    s.addText([{ text: h, options: { bold: true, fontSize: 20, color: C.text1 } },
               { text: b ? "   " + b : "", options: { fontSize: 16, color: C.accent6 } }],
              { x: 2.95, y, w: 8.5, h: 0.5, valign: "middle", isTextBox: true, margin: 0 });
  });
  notes(s, "[0:20] 순서: 문제 → 이론 → 방법 → 공정한 비교 방법 → 결과 → 네트워크가 실제로 하는 일 → 한계와 결론.");

  // 3 section
  pres.addSection({ title: "Problem and theory" });
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Problem and theory" });
  s.addText("Problem and theory", { placeholder: "title" });
  s.addText("Direction ambiguity, pulse coding and the light sheet", { placeholder: "body" });
  notes(s, "[0:05]");

  // 4 direction ambiguity
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and theory" });
  title(s, "Double exposure: the direction is lost");
  const pairsEq = [[0.4, 0.4, 0.9, 0.9], [1.6, 0.9, 0.6, 0.6], [0.7, 1.7, 0.75, 0.75], [2.1, 2.1, 0.5, 0.5], [2.6, 0.5, 0.85, 0.85], [1.2, 2.6, 0.65, 0.65], [3.0, 1.5, 0.8, 0.8]];
  particles(s, 0.6, 1.25, 3.8, 3.4, pairsEq, [0.45, 0.2], "both pulses on ONE frame, equal energy");
  pic(s, "slides/fig/s_ac_plane.png", 4.8, 1.15, 3.6, 3.9);
  bullets(s, [
    "Auto-correlation: self peak + two equal peaks at ±d",
    "R(s) = R(−s) for every image",
    "Swap the exposures and the sign of d → same image",
    { text: "no estimator, learned or not, can recover the sign", sub: true },
    "Classical fixes: image shift, colour coding, pulse coding",
  ], 8.8, 1.35, 4.1, 4.3, 17);
  keybox(s, "Equal pulses: the flow direction is not identifiable from the image");
  notes(s, "[1:30] 두 펄스를 한 장에 찍으면 auto-correlation이 원점 자기 피크와 ±d에 같은 높이의 피크를 냅니다. " +
    "입자 위치를 d만큼 옮기고 부호를 바꾸면 똑같은 영상이 나오므로(명제 1), 어떤 알고리즘으로도 방향을 알 수 없습니다. 해결책 중 하나가 pulse coding입니다.");

  // 5 pulse coding and light sheet
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and theory" });
  title(s, "Pulse coding, and why brightness alone fails");
  const pairsPc = [[0.4, 0.4, 0.95, 0.66], [1.6, 0.9, 0.35, 0.25], [0.7, 1.7, 0.8, 0.56], [2.1, 2.1, 0.5, 0.35], [2.6, 0.5, 0.2, 0.14], [1.2, 2.6, 0.65, 0.45], [3.0, 1.5, 0.9, 0.63]];
  particles(s, 0.6, 1.25, 3.8, 3.0, pairsPc, [0.45, 0.2], "pulse 2 weaker: I₂ = r·I₁ (r < 1)");
  eq(s, "eq_model", 0.6, 4.75, 15);
  pic(s, "slides/fig/s_bayes.png", 4.8, 1.1, 5.0, 3.6);
  stat(s, 10.2, 1.5, 2.8, "26–40 %", "particles misassigned by the best brightness rule (r = 0.5–0.9)", C.accent2);
  stat(s, 10.2, 3.3, 2.8, "> 10×", "brightness variation from the Gaussian light sheet vs. coding ratio");
  keybox(s, "The direction is carried by particle PAIRS, not by single particles");
  notes(s, "[1:30] 두 번째 펄스를 r배로 약하게 하면 원리상 밝은 쪽이 먼저입니다. 하지만 입자 밝기는 light sheet 안의 z 위치에 따라 10배 이상 변합니다. " +
    "입자 하나만 보고 판정하면 최선의 규칙(Bayes)도 26–40 %를 틀립니다. 그래서 '같은 변위를 가진 쌍'이라는 문맥이 필요합니다.");

  // 6 theory 1
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and theory" });
  title(s, "Theory ①: when does out-of-plane motion reverse a pair?");
  card(s, 0.5, 1.1, 6.4, 1.75, "Pair i is reversed if bᵢ ≥ aᵢ", "z ~ U(−1, 1) in the sheet, out-of-plane step δ", { bodySize: 15 });
  eq(s, "eq_flip", 0.7, 2.0, 18);
  card(s, 0.5, 3.0, 6.4, 1.2, "No pair is ever reversed iff", null);
  eq(s, "eq_rstar", 0.9, 3.5, 20);
  stat(s, 0.7, 4.2, 2.9, "r* = 0.55", "σ = 0.4, w = 0.1 (our set-up)", C.accent1, 36);
  stat(s, 3.9, 4.2, 3.0, "25 %", "pairs reversed at r = 0.9", C.accent2, 36);
  pic(s, "slides/fig/s_flip.png", 7.2, 1.05, 5.8, 4.8);
  keybox(s, "Weak coding (r → 1) is limited by the light sheet, not only by the algorithm");
  notes(s, "[2:00] 쌍의 순서가 뒤집히는 확률을 닫힌 형태로 유도했습니다. 균일한 z와 면외 이동 δ에서 이 식이 나오고, " +
    "모든 |δ| ≤ w에서 뒤집힘이 없을 조건이 r ≤ r* = exp[−w(1−w/2)/σ²]입니다. 우리 조건에서 r* = 0.55이고, r = 0.9이면 쌍의 25 %가 뒤집힙니다. " +
    "즉 약한 코딩의 한계는 알고리즘이 아니라 물리에서 옵니다.");

  // 7 theory 2
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and theory" });
  title(s, "Theory ②: cross-correlating separated frames is a vote");
  card(s, 0.5, 1.1, 6.05, 1.55, "Separated frames (soft mask m)", null);
  eq(s, "eq_vote", 0.7, 1.65, 17);
  card(s, 6.8, 1.1, 6.05, 1.55, "Nonlinear auto-correlation", null, { fill: "FDEEE6", line: "FDEEE6", headColor: C.accent2 });
  eq(s, "eq_asym", 7.0, 1.65, 17);
  bullets(s, ["each pair votes ≈ ±1, whatever r is", "no pixel-accurate separation needed"], 0.7, 2.75, 5.8, 0.9, 15);
  bullets(s, ["each pair votes ∝ (a − b) → vanishes as r → 1", "margin (1−r)/(1+r) = 0.18 at r = 0.7"], 7.0, 2.75, 5.8, 0.9, 15);
  pic(s, "slides/fig/s_planes.png", 1.4, 3.6, 10.5, 2.3);
  keybox(s, "More pairs per window → direction error falls exponentially (Hoeffding bound)");
  notes(s, "[2:00] 분리한 두 프레임의 상관에서 +d와 −d 피크 차이는 Σ aᵢbᵢ(mᵢᴬ − mᵢᴮ)에 비례합니다. 즉 쌍마다 한 표씩 던지는 가중 투표이고, " +
    "픽셀 단위로 완벽히 분리할 필요가 없습니다. 비선형 auto-correlation도 투표지만 표의 크기가 밝기 차이 (a − b)에 비례해서 r → 1이면 사라집니다. " +
    "아래 그림: 일반 AC는 ±d 피크가 같고, 비선형 AC는 +d가 약간 높고, 분리 후 CC는 +d 하나만 남습니다(파란 원 = 정답 방향).");

  // 8 section
  pres.addSection({ title: "Method and comparison" });
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Method and comparison" });
  s.addText("Method and fair comparison", { placeholder: "title" });
  s.addText("Turn a double-exposure problem into a double-frame problem", { placeholder: "body" });
  notes(s, "[0:05]");

  // 9 pipeline
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method and comparison" });
  title(s, "Proposed: separate, then cross-correlate");
  pic(s, "slides/fig/s_arch.png", 0.5, 1.0, 12.3, 3.7);
  eq(s, "eq_split", 0.6, 4.75, 18, "center", 12.1);
  keybox(s, "The network only decides where intensity belongs; displacement comes from standard multi-pass CC");
  notes(s, "[1:15] 3단계 U-Net이 sigmoid 마스크 m을 출력하고, Â = mI', B̂ = (1−m)I'로 나눕니다. 밝기는 보존됩니다. " +
    "나눈 두 장을 표준 multi-pass cross-correlation(window deformation 포함)으로 처리하므로 기존 PIV 처리 체계를 그대로 씁니다. " +
    "수용 영역이 약 100 px라 이웃 쌍을 볼 수 있습니다.");

  // 10 training
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method and comparison" });
  title(s, "Training: synthetic data + label-free fine-tuning");
  pic(s, "slides/fig/s_sep_example.png", 0.6, 1.0, 12.1, 3.2);
  const cw = 3.9;
  card(s, 0.5, 4.35, cw, 1.35, "Synthetic training", ["6000 images, r ∈ [0.45, 0.95]", "light sheet, noise, density, vortices"]);
  card(s, 0.5 + cw + 0.32, 4.35, cw, 1.35, "Loss", ["target t = A / (A + B)", "intensity-weighted BCE"]);
  card(s, 0.5 + 2 * (cw + 0.32), 4.35, cw, 1.35, "Self-supervised fine-tuning", ["real frames A, B → I = A + rB", "labels for free, no annotation"]);
  keybox(s, "The evaluation recordings (test1, case C) were never used for training");
  notes(s, "[1:00] 위 그림은 r = 0.7 합성 영상의 실제 분리 결과입니다. 파랑이 첫 번째, 주황이 두 번째 노출입니다. " +
    "합성 데이터 6000장으로 학습하고, 손실은 밝기 가중 BCE입니다. 실제 실험 두 프레임 A, B만 있으면 A + rB의 정답을 공짜로 얻으므로 라벨 없이 미세조정할 수 있습니다. " +
    "평가에 쓰는 test1과 case C는 학습에 쓰지 않았습니다.");

  // 11 fair protocol
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method and comparison" });
  title(s, "Fair comparison: identical processing for every method");
  const meth = [["Nonlinear auto-correlation", "classical direction fix with pulse coding", C.accent2],
                ["Auto-correlation, true sign †", "idealised: sign given by the reference", C.accent3],
                ["U-Net + CC (ours)", "separation → cross-correlation", C.accent1],
                ["True A, B ‡", "upper bound, synthetic data only", C.accent4]];
  s.addText("Compared methods", { x: 0.5, y: 1.05, w: 6, h: 0.45, fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0 });
  meth.forEach(([h, b, col], i) => {
    const y = 1.6 + i * 1.05;
    s.addShape(pres.shapes.OVAL, { x: 0.55, y: y + 0.15, w: 0.38, h: 0.38, fill: { color: col }, line: { type: "none" } });
    s.addText([{ text: h, options: { bold: true, fontSize: 17, color: C.text1, breakLine: true } },
               { text: b, options: { fontSize: 15, color: C.accent6 } }], { x: 1.15, y, w: 5.2, h: 0.85, valign: "middle", isTextBox: true, margin: 0 });
  });
  s.addText("Shared pipeline", { x: 7.0, y: 1.05, w: 5.8, h: 0.45, fontSize: 20, bold: true, color: C.text2, isTextBox: true, margin: 0 });
  const steps = ["same double exposure I", "3 passes: 64 → 32 → 32 px, 50 % overlap", "same centre mask, sub-pixel fit, validation", "report raw AND + UOD", "same evaluation vectors (from reference)"];
  steps.forEach((t, i) => {
    const y = 1.6 + i * 0.85;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y, w: 5.8, h: 0.62, rectRadius: 0.1, fill: { color: C.accent5 }, line: { type: "none" } });
    s.addText(t, { x: 7.2, y, w: 5.4, h: 0.62, fontSize: 16, color: C.text1, valign: "middle", isTextBox: true, margin: 0 });
    if (i < steps.length - 1) s.addShape(pres.shapes.DOWN_ARROW, { x: 9.75, y: y + 0.64, w: 0.3, h: 0.2, fill: { color: C.accent6 }, line: { type: "none" } });
  });
  keybox(s, "Auto-correlation gets multi-pass too: predictor-guided peak search at ±p");
  notes(s, "[1:30] 비교군 4개입니다. 비선형 AC는 실제로 쓸 수 있는 고전 기법, 부호 oracle AC는 부호를 알려준 이상적 기준, 정답 A/B는 상한입니다. " +
    "공정성을 위해 모든 방법에 같은 영상, 같은 3-pass 일정, 같은 중심 마스크와 검증을 적용했고, auto-correlation도 multi-pass로 처리했습니다. " +
    "UOD 후처리는 적용 전후를 모두 보고합니다(UOD가 와류 근처 정상 벡터를 지워 정답 A/B도 나빠지기 때문).");

  // 12 datasets
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method and comparison" });
  title(s, "Four validation datasets");
  const ds = [["s_ds_synth", "Synthetic (ours)", "light sheet, vortices", "n = 2691"],
              ["s_ds_lml", "Synthetic (LML)", "independent generator, overlapping pair images", "n = 2700"],
              ["s_ds_test1", "Real: test1", "OpenPIV recording, unseen", "n = 638"],
              ["s_ds_caseC", "Real: case C", "PIV Challenge 2001, impeller + reflections, unseen", "n = 1559"]];
  ds.forEach(([f, h, b, n], i) => {
    const x = 0.55 + i * 3.1;
    img(s, `slides/fig/${f}.png`, x, 1.15, 2.85, 2.85, h);
    s.addText([{ text: h, options: { bold: true, fontSize: 17, color: C.text2, breakLine: true } },
               { text: b, options: { fontSize: 14, color: C.text1, breakLine: true } },
               { text: n + " vectors / r", options: { fontSize: 13, color: C.accent6 } }],
              { x, y: 4.1, w: 2.85, h: 1.6, valign: "top", isTextBox: true, margin: 0 });
  });
  keybox(s, "r = 0.9, 0.7, 0.5 on every dataset; real double exposures built as I = A + rB");
  notes(s, "[1:00] 자체 합성, 다른 연구실(Lille LML)의 생성기, 실제 영상 두 개입니다. LML은 입자 지름이 변위와 비슷해서 두 상이 겹치는 어려운 조건입니다. " +
    "case C는 PIV Challenge의 임펠러 영상으로 강한 반사가 있습니다. 실제 영상 둘 다 학습에 쓰지 않았습니다. 실제 영상의 double exposure는 두 프레임을 합성해 만들었습니다.");

  // 13 section
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Results" });
  s.addText("Results", { placeholder: "title" });
  s.addText("All numbers: multi-pass, no post-processing, r = 0.7 unless noted", { placeholder: "body" });
  notes(s, "[0:05]");

  // 14 main result chart
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  title(s, "Lowest outlier rate on every dataset");
  const mm = RES._meta.main_model;
  const dsk = ["synth", "lml", "test1", "caseC"], dsl = ["Synthetic (ours)", "Synthetic (LML)", "Real: test1", "Real: case C"];
  const val = (m) => dsk.map((d) => Math.round(1000 * RES[`${d}/r0.7`][m === "ours" ? `unet:${mm[d]}` : m].outlier) / 10);
  s.addChart(pres.charts.BAR, [
    { name: "Nonlinear AC", labels: dsl, values: val("asym") },
    { name: "AC, true sign †", labels: dsl, values: val("oracle") },
    { name: "U-Net + CC (ours)", labels: dsl, values: val("ours") },
  ], {
    x: 0.5, y: 1.0, w: 8.4, h: 4.85, barDir: "col", barGapWidthPct: 60,
    chartColors: [HEX.accent2, HEX.accent3, HEX.accent1],
    valAxisMinVal: 0, valAxisMaxVal: 60, valAxisMajorUnit: 10, valAxisTitle: "outlier rate (%)", showValAxisTitle: true,
    valAxisTitleFontSize: 13, valAxisTitleColor: "666666", valAxisLabelColor: "666666", valAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt",
    catAxisLabelColor: "191919", catAxisLabelFontSize: 13, catAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E6E6E6", size: 0.75 }, catGridLine: { style: "none" },
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: "191919", dataLabelFormatCode: "0.0", dataLabelFontFace: "+mn-lt",
    showLegend: true, legendPos: "t", legendFontSize: 13, legendFontFace: "+mn-lt",
  });
  stat(s, 9.3, 1.2, 3.6, "12 / 12", "conditions (4 datasets × 3 r): lowest outliers of the realisable methods", C.accent1);
  stat(s, 9.3, 2.95, 3.6, "0.5 vs 35 %", "LML: overlapping pair images defeat auto-correlation", C.accent2);
  stat(s, 9.3, 4.6, 3.6, "12.5 vs 52 %", "case C (real impeller, reflections)", C.text2);
  keybox(s, "r ≤ 0.7: equal to or better than auto-correlation that is TOLD the sign");
  notes(s, "[2:00] r = 0.7, 후처리 없는 결과입니다(로그 축). 4개 데이터셋 × 3개 r 모두에서 실제로 쓸 수 있는 방법 중 outlier가 가장 적습니다. " +
    "합성 0.9 % vs 8.6 %, LML 0.5 % vs 35 %, test1 0.8 % vs 21 %, case C 12.5 % vs 52 %. r ≤ 0.7에서는 부호를 알려준 AC보다도 같거나 낫습니다. " +
    "UOD를 적용하면 고전 기법이 많이 따라오지만(test1 21 → 0.9 %), 여전히 제안 방법이 가장 낮습니다.");

  // 15 vectors LML
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  title(s, "Vector fields: overlapping particle images (LML)");
  pic(s, "paper/fig_vec_lml.png", 0.4, 1.0, 12.5, 3.85);
  card(s, 0.5, 5.0, 3.95, 0.85, "Nonlinear AC", "scattered sign flips and random peaks", { bodySize: 14, headSize: 15, headColor: C.accent2 });
  card(s, 4.69, 5.0, 3.95, 0.85, "AC, true sign", "self peak overlaps ±d → magnitude errors", { bodySize: 14, headSize: 15, headColor: "00866B" });
  card(s, 8.88, 5.0, 3.95, 0.85, "U-Net + CC", "no self peak after separation", { bodySize: 14, headSize: 15 });
  keybox(s, "Red = outlier (> 1 px). Separation works even when the two images of a pair overlap", 6.05, 0.7);
  notes(s, "[0:45] LML 영상, 빨강이 outlier입니다. 비선형 AC는 부호 반전과 무작위 피크가 흩어져 있고, 부호를 알려준 AC도 자기 피크가 ±d와 겹쳐서 크기 오차가 납니다. " +
    "분리 후 CC는 자기 피크가 없어서 이 이미지에서 outlier가 0입니다.");

  // 16 vectors case C
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  title(s, "Vector fields: PIV Challenge case C (real)");
  const pc = pic(s, "paper/fig_vec_caseC.png", 0.4, 1.0, 8.9, 5.8, "left");
  card(s, 9.6, 1.2, 3.3, 1.5, "Where errors remain", "along the blade, in the shadow and at the flare", { bodySize: 15 });
  card(s, 9.6, 2.95, 3.3, 1.5, "Mostly magnitude", "direction errors only 2.1 % at r = 0.7", { bodySize: 15 });
  card(s, 9.6, 4.7, 3.3, 1.6, "Reference uncertain there", "reference is CC of the real frames", { bodySize: 15 });
  notes(s, "[0:45] 실제 임펠러 영상입니다. 남은 오차는 블레이드, 그림자, 반사 근처에 모여 있고 대부분 크기 오차입니다(방향 오류 2.1 %). " +
    "그 영역은 기준값 자체도 불확실합니다.");

  // 17 direction errors vs theory
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  title(s, "Direction errors follow the theory");
  pic(s, "slides/fig/s_theory_b.png", 0.5, 1.0, 6.9, 4.8);
  s.addText("Measured vs. ideal pair vote (31 pairs / window)", { x: 7.8, y: 1.15, w: 5.1, h: 0.45, fontSize: 18, bold: true, color: C.text2, isTextBox: true, margin: 0 });
  stat(s, 7.8, 1.65, 2.5, "5.7 / 5.9 %", "r = 0.95  measured / predicted", C.accent1, 34);
  stat(s, 10.5, 1.65, 2.5, "0.2 / 0.5 %", "r = 0.9  measured / predicted", C.accent1, 34);
  stat(s, 7.8, 3.6, 2.5, "0.11 px", "median error, ours (true frames 0.095)", C.text2, 34);
  stat(s, 10.5, 3.6, 2.5, "0.26 px", "median error, nonlinear AC", C.accent2, 34);
  keybox(s, "The network performs at the level of an ideal brightness-order classifier");
  notes(s, "[1:15] 선은 이론(이상적인 쌍 분류기의 투표), 점은 실측입니다. r = 0.95에서 5.7 % vs 5.9 %, r = 0.9에서 0.2 % vs 0.5 %로 맞습니다. " +
    "낮은 r에서 바닥에 붙은 점은 2700개 중 1개 수준입니다. 정확도도 좋습니다: median 오차 0.11 px로 정답 프레임(0.095)에 가깝고 AC는 0.26 px입니다.");

  // 18 inside the network
  pres.addSection({ title: "Inside the network" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Inside the network" });
  title(s, "What the U-Net actually decides");
  pic(s, "slides/fig/s_probe.png", 0.4, 1.0, 6.2, 4.15);
  pic(s, "slides/fig/s_margin.png", 6.75, 1.0, 6.2, 4.15);
  s.addText([{ text: "Probe: ", options: { bold: true } }, { text: "a step in the brightness ORDER (ρ = 0.3 … 0.85 → same Δm); reversed pairs among normal neighbours are corrected (92–96 %)" }],
            { x: 0.5, y: 5.2, w: 6.0, h: 0.7, fontSize: 14, color: C.text1, valign: "top", isTextBox: true, margin: 0 });
  s.addText([{ text: "Margin: ", options: { bold: true } }, { text: "AC median 0.14 ≈ (1−r)/(1+r) = 0.18;  separated frames: −d peak absent (margin 1)" }],
            { x: 6.85, y: 5.2, w: 6.0, h: 0.7, fontSize: 14, color: C.text1, valign: "top", isTextBox: true, margin: 0 });
  keybox(s, "Pair order + neighbour context → a confident ±1 vote per pair", 6.05, 0.7);
  notes(s, "[1:30] 모든 쌍의 밝기 비 ρ를 바꾼 탐침 영상을 넣어 봤습니다. Δm은 ρ < 1이면 0.3이든 0.85든 +0.75, ρ > 1이면 −0.7인 계단 함수입니다. " +
    "즉 밝기 차이의 크기가 아니라 순서로 판정합니다(이론의 ±1 투표). 주황 점: 정상 쌍들 사이에 섞인 뒤집힌 쌍도 92–96 %를 이웃 변위를 보고 바로잡습니다. " +
    "오른쪽: 창 단위 여유가 AC는 (1−r)/(1+r) 근처, 분리 후 CC는 1입니다.");

  // 19 processing matters
  pres.addSection({ title: "Limitations and conclusion" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Limitations and conclusion" });
  title(s, "Processing matters as much as the estimator");
  pic(s, "paper/fig_process.png", 0.4, 1.0, 12.5, 4.5);
  keybox(s, "test1, r = 0.7:  26.5 % vs 0.2 % with unequal processing  →  0.9 % vs 0.2 % when matched", 5.75, 0.9);
  notes(s, "[1:00] 단계별 outlier 비율입니다(SP: single-pass, MP: multi-pass). 처리 조건이 추정기만큼 중요합니다. " +
    "고전 기법을 single-pass, 후처리 없이 돌리고 제안 방법만 multi-pass + UOD로 돌리면 test1에서 26.5 % vs 0.2 %처럼 보이지만, 맞추면 0.9 % vs 0.2 %입니다. " +
    "그래서 이 연구에서는 모든 방법의 처리를 맞췄습니다.");

  // 20 limitations
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Limitations and conclusion" });
  title(s, "Limitations");
  const lim = [[I.warn, C.accent2, "Weak coding (r = 0.9)", "auto-correlation with the true sign is better on real data (test1 0.5 vs 6.1 %, case C 15.7 vs 31.9 %, + UOD) — predicted by r* = 0.55"],
               [I.camera, C.text2, "Synthesised double exposures", "I = A + rB from double-frame recordings; real pulse-coded shots add flare, saturation, pulse-to-pulse jitter"],
               [I.ruler, C.text2, "Reference on real data", "multi-pass CC of the real frames — the same estimator family as ours"],
               [I.flask, C.text2, "Scope", "two real recordings; analysis assumes non-overlapping images and uniform d per window"]];
  lim.forEach(([ic, col, h, b], i) => {
    const y = 1.15 + i * 1.25;
    circleIcon(s, ic, 0.6, y + 0.1, 0.75, col);
    s.addText([{ text: h, options: { bold: true, fontSize: 18, color: C.text1, breakLine: true } },
               { text: b, options: { fontSize: 15, color: C.accent6 } }], { x: 1.6, y, w: 11.2, h: 1.05, valign: "middle", isTextBox: true, margin: 0 });
  });
  notes(s, "[1:00] 한계입니다. 첫째, r = 0.9에서는 부호를 알려준 AC가 실제 영상에서 더 낫습니다. 이론에서 r* = 0.55를 넘으면 쌍의 순서가 뒤집히기 때문에 예상된 결과입니다. " +
    "둘째, 실제 영상의 double exposure는 두 프레임을 합성한 것이라 실제 촬영의 반사, 포화, 펄스 에너지 변동은 반영되지 않았습니다. " +
    "셋째, 실제 영상의 기준값이 CC 기반입니다. 넷째, 실제 영상은 두 개뿐입니다.");

  // 21 conclusion
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Limitations and conclusion" });
  title(s, "Conclusion");
  const con = [[I.sigma, "Theory", ["direction lives in pairs", "order is safe for r ≤ r* = exp[−w(1−w/2)/σ²]", "separated CC = weighted pair vote"]],
               [I.brain, "Method", ["U-Net soft mask → Â, B̂", "standard multi-pass CC", "label-free fine-tuning on real frames"]],
               [I.chart, "Evidence", ["lowest outliers in 12 / 12 conditions", "matches ideal-vote prediction", "fair: identical processing for all"]]];
  con.forEach(([ic, h, b], i) => {
    const x = 0.55 + i * 4.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.25, w: 3.9, h: 4.4, rectRadius: 0.15, fill: { color: C.accent5 }, line: { type: "none" }, objectName: "conclusion " + h });
    circleIcon(s, ic, x + 1.5, 1.5, 0.9, C.text2);
    s.addText(h, { x, y: 2.55, w: 3.9, h: 0.5, fontSize: 22, bold: true, color: C.text2, align: "center", isTextBox: true, margin: 0 });
    bullets(s, b, x + 0.3, 3.2, 3.35, 2.3, 16);
  });
  keybox(s, "Pulse coding + learned separation resolves the direction for r ≤ 0.7");
  notes(s, "[0:40] 세 줄 요약: 이론적으로 방향 정보는 쌍에 있고 r ≤ r*이면 안전합니다. 방법은 U-Net 분리 후 표준 CC이고, 이는 쌍들의 투표입니다. " +
    "처리 조건을 맞춘 공정한 비교에서 12개 조건 모두 가장 낮은 outlier를 보였고, 이론 예측과도 맞습니다.");

  // 22 Q&A
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Limitations and conclusion" });
  s.addText("Thank you", { placeholder: "title" });
  s.addText("Questions?", { placeholder: "body" });

  // backup
  pres.addSection({ title: "Backup" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  title(s, "Backup: outlier rate (%), all conditions");
  const hdr = ["Dataset", "r", "Nonlin. AC", "AC †", "Ours", "Nonlin. AC + UOD", "AC † + UOD", "Ours + UOD", "True A,B"];
  const th = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.dk2 }, align: "center" } });
  const rows = [hdr.map(th)];
  dsk.forEach((d, di) => [0.9, 0.7, 0.5].forEach((r) => {
    const t = RES[`${d}/r${r}`], u = `unet:${mm[d]}`;
    const f = (k) => (k in t ? (100 * t[k].outlier).toFixed(1) : "–");
    rows.push([r === 0.9 ? dsl[di] : "", String(r), f("asym"), f("oracle"), { text: f(u), options: { bold: true, color: HEX.accent1 } },
               f("asym+uod"), f("oracle+uod"), { text: f(u + "+uod"), options: { bold: true, color: HEX.accent1 } }, f("gt")]);
  }));
  s.addTable(rows, { x: 0.5, y: 1.05, w: 12.3, colW: [2.1, 0.6, 1.35, 1.2, 1.2, 1.75, 1.45, 1.4, 1.25], fontSize: 12, fontFace: "Calibri",
                      align: "center", valign: "middle", rowH: 0.36, border: { type: "solid", pt: 0.5, color: "D0D0D0" } });
  notes(s, "[백업] 논문 표 2 전체. 후처리 없음과 UOD 적용 후. † 부호를 알려준 이상적 기준, ‡ 정답 프레임(합성만).");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  title(s, "Backup: vector fields, synthetic and test1");
  pic(s, "paper/fig_vec_synth.png", 0.4, 0.95, 12.5, 3.0);
  pic(s, "paper/fig_vec_test1.png", 0.4, 4.0, 12.5, 2.8);
  notes(s, "[백업] 자체 합성과 test1의 벡터장 (r = 0.7, 후처리 없음).");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  title(s, "Backup: error distribution and fine-tuning");
  pic(s, "paper/fig_cdf.png", 0.4, 0.95, 12.5, 3.6);
  const ftr = [["", "r", "Zero-shot", "Fine-tuned"].map(th)];
  [["test1", "Real: test1"], ["caseC", "Real: case C"]].forEach(([d, l]) => [0.9, 0.7, 0.5].forEach((r) => {
    const t = RES[`${d}/r${r}`];
    ftr.push([r === 0.9 ? l : "", String(r), (100 * t["unet:zeroshot+uod"].outlier).toFixed(1) + " %", (100 * t["unet:finetuned+uod"].outlier).toFixed(1) + " %"]);
  }));
  s.addTable(ftr, { x: 3.4, y: 4.65, w: 6.5, colW: [1.9, 0.8, 1.9, 1.9], fontSize: 12, fontFace: "Calibri", align: "center", valign: "middle", rowH: 0.3, border: { type: "solid", pt: 0.5, color: "D0D0D0" } });
  notes(s, "[백업] 오차 누적분포(r = 0.7)와 미세조정 효과(outlier, multi-pass + UOD). 실제 영상 둘 다 미세조정에 쓰지 않았습니다.");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  title(s, "Backup: why the window decision is robust");
  eq(s, "eq_hoeff", 0.8, 1.3, 22);
  bullets(s, ["N(eff) = (Σw)² / Σw²: effective number of pairs in the window (wᵢ = aᵢbᵢ)",
              "ε̄: weighted fraction of reversed pairs (Prop. 2)",
              "assumptions: non-overlapping particle images, uniform d in the window, independent reversals given δ",
              "multi-pass: leakage peaks move to −p after symmetric deformation → mask follows −p"], 0.8, 2.3, 11.8, 3.2, 17);
  notes(s, "[백업] Hoeffding 상한. 창 안 유효 쌍 수가 많을수록 방향 오류가 지수적으로 감소합니다.");

  const out_ = F("slides/piv_pulse_coding.pptx");
  await pres.writeFile({ fileName: out_ });
  const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));
  await applyTheme(out_, THEME);
  console.log("saved", out_);
}
build().catch((e) => { console.error(e); process.exit(1); });
