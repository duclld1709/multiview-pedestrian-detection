"use strict";
// MV Label: review multi-view pseudo labels. projectBox / matchViews / labelBox repeat labeling.py exactly
// (16-gon person cylinder, same visibility rule, greedy IoU matching), so what is drawn is what gets exported.

const $ = (id) => document.getElementById(id);
const SEGMENTS = 16;
const COLORS = { ok: "#3ecf8e", check: "#ff7a3d", suggested: "#b98cff", manual: "#4aa8ff" };
const EDITED = "#ffd60a";     // a person the reviewer moved, resized or whose boxes they switched
const SELECTED = "#ff2bd6";   // magenta with a black rim: stands out on bright pavement and dark clothing
const STATUS_ORDER = { suggested: 0, check: 1, manual: 2, ok: 3 };
const VI_STATUS = { ok: "ổn", check: "cần kiểm tra", suggested: "đề xuất", manual: "thêm tay" };
const ORIGIN = { model: "model BEV", "model-low": "model (điểm yếu)", yolo: "YOLO đa view", manual: "thêm tay" };

const S = {
  project: null, cams: [], params: null, area: null, frames: [], frame: null, data: null,
  views: [], sel: null, selYolo: null, focus: { kind: "bev", cam: null }, mouse: null,
  show: { yolo: true, proj: true, gt: false, follow: false, grid: false, obstaclesOnly: false },
  undo: [], tiles: [], maximized: null,
  saveTimer: null, jobTimer: null, images: new Map(), bev: null, drag: null, perView: false,
  siteMap: null, mapEdit: false, mapFocus: false, mapTool: "select", mapDraft: [],
  mapSelected: null, mapSelectedIds: [],
  mapProjectCameras: true, mapProjectionMode: "surface", mapOnlySelected: false,
  mapShowCoverage: false, mapCoverageCamera: "all", mapSightMode: "fan",
  mapView: { zoom: 1, panX: 0, panY: 0 }, mapPan: null,
};

// ------------------------------------------------------------------------------------------------ http

async function api(path, options = {}) {
  const init = { method: options.method || "GET", headers: {} };
  if (options.body !== undefined) {
    init.body = JSON.stringify(options.body);
    init.headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, init);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function toast(text, error = false) {
  const node = $("toast");
  node.textContent = text;
  node.className = "toast" + (error ? " error" : "");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => node.classList.add("hidden"), error ? 6000 : 2500);
}

// ------------------------------------------------------------------------------------------------ geometry

const ANGLES = Array.from({ length: SEGMENTS }, (_, i) => (2 * Math.PI * i) / SEGMENTS);

function toCamera(cam, X) {
  const R = cam.R, t = cam.t;
  return [R[0][0] * X[0] + R[0][1] * X[1] + R[0][2] * X[2] + t[0],
          R[1][0] * X[0] + R[1][1] * X[1] + R[1][2] * X[2] + t[1],
          R[2][0] * X[0] + R[2][1] * X[1] + R[2][2] * X[2] + t[2]];
}

function toPixel(cam, c) {
  const K = cam.K;
  const u = K[0][0] * c[0] + K[0][1] * c[1] + K[0][2] * c[2];
  const v = K[1][0] * c[0] + K[1][1] * c[1] + K[1][2] * c[2];
  const w = K[2][0] * c[0] + K[2][1] * c[1] + K[2][2] * c[2];
  return [u / w, v / w];
}

function projectPoint(cam, X) {
  const c = toCamera(cam, X);
  return c[2] > 0.1 ? toPixel(cam, c) : null;
}

function projectBox(cam, x, y, radius, height) {
  let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
  for (const z of [0, height]) {
    for (const a of ANGLES) {
      const c = toCamera(cam, [x + radius * Math.cos(a), y + radius * Math.sin(a), z]);
      if (c[2] <= 0.1) return null;
      const [u, v] = toPixel(cam, c);
      left = Math.min(left, u); right = Math.max(right, u); top = Math.min(top, v); bottom = Math.max(bottom, v);
    }
  }
  const area = (right - left) * (bottom - top);
  const cl = Math.max(left, 0), ct = Math.max(top, 0), cr = Math.min(right, cam.width - 1), cb = Math.min(bottom, cam.height - 1);
  if (cr <= cl || cb <= ct || bottom - top < 12 || (cr - cl) * (cb - ct) < 0.3 * area) return null;
  const r1 = (v) => Math.round(v * 10) / 10;
  return [r1(cl), r1(ct), r1(cr), r1(cb)];
}

function inv3(m) {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
          [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
          [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}

function ground(cam, u, v) {
  cam.Kinv = cam.Kinv || inv3(cam.K);
  const k = cam.Kinv, R = cam.R;
  const r = [k[0][0] * u + k[0][1] * v + k[0][2], k[1][0] * u + k[1][1] * v + k[1][2], k[2][0] * u + k[2][1] * v + k[2][2]];
  const ray = [0, 1, 2].map((j) => R[0][j] * r[0] + R[1][j] * r[1] + R[2][j] * r[2]);
  const centre = cam.centre;
  if (Math.abs(ray[2]) < 1e-9) return null;
  const depth = -centre[2] / ray[2];
  if (depth <= 0) return null;
  return [centre[0] + depth * ray[0], centre[1] + depth * ray[1]];
}

function iou(a, b) {
  const ix = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const iy = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (ix <= 0 || iy <= 0) return 0;
  const inter = ix * iy;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

function cameraActive(index) {
  return !S.data?.inactive_cameras?.includes(index);
}

function matchViews(persons) {
  const params = S.params, yolo = S.data.yolo;
  const views = persons.map(() => ({}));
  for (const cam of S.cams) {
    if (!cameraActive(cam.index)) continue;
    const boxes = yolo[String(cam.index)] || [];
    const pairs = [];
    persons.forEach((person, p) => {
      const box = projectBox(cam, person.x, person.y, person.r ?? params.radius, person.h || params.height);
      if (!box) return;
      views[p][cam.index] = { proj: box, yolo: null, iou: 0 };
      boxes.forEach((det, y) => {
        const overlap = iou(box, det);
        if (overlap >= params.iou_gate) pairs.push([-overlap, p, y]);
      });
    });
    pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    const usedP = new Set(), usedY = new Set();
    for (const [neg, p, y] of pairs) {
      if (usedP.has(p) || usedY.has(y)) continue;
      usedP.add(p); usedY.add(y);
      Object.assign(views[p][cam.index], { yolo: y, iou: Math.round(-neg * 1000) / 1000 });
    }
  }
  return views;
}

// override per camera: "proj" | "yolo" | "off" | [x1, y1, x2, y2] (a box drawn or resized by hand in that camera only)
function manualBox(person, camIndex) {
  const value = (person.override || {})[String(camIndex)];
  return Array.isArray(value) ? value : null;
}

function boxSource(person, camIndex) {
  if (manualBox(person, camIndex)) return "manual";
  return (person.override || {})[String(camIndex)] || (S.params.box_mode === "yolo" ? "yolo" : "proj");
}

function labelBox(person, camIndex, view) {
  if (!cameraActive(camIndex)) return null;
  const manual = manualBox(person, camIndex);
  if (manual) return manual;
  const mode = boxSource(person, camIndex);
  if (!view || mode === "off") return null;
  if (mode === "yolo" && view.yolo !== null) return S.data.yolo[String(camIndex)][view.yolo].slice(0, 4);
  return view.proj;
}

function included(person) {
  return person.status !== "suggested" || person.verified;
}

function edited(person) {
  if (person.status === "manual") return false;   // added by hand: stays blue
  const moved = person.auto && Math.hypot(person.x - person.auto[0], person.y - person.auto[1]) > 0.005;
  const resized = Math.abs((person.h || S.params.height) - S.params.height) > 1e-6;
  return Boolean(moved || resized || Object.keys(person.override || {}).length);
}

function colorOf(person) {
  return edited(person) ? EDITED : COLORS[person.status] || COLORS.manual;
}

// ------------------------------------------------------------------------------------------------ project

async function loadState() {
  const state = await api("/api/state");
  openSetup.root = state.default_root;
  if (state.project) setProject(state.project);
  else openSetup(state.default_root);
}

function setProject(project) {
  S.project = project;
  S.cams = project.cameras.map((cam) => ({ ...cam }));
  S.params = project.params;
  S.area = project.area;
  setMapProject(project);
  S.frames = project.frames;
  showModelPanel(project);
  S.images.clear();
  $("boxMode").value = S.params.box_mode;
  const detectorName = (project.config.yolo_weights || "yolo26s.pt").split(/[\\/]/).pop();
  const model = project.model ? (project.model.experimental ?
    `${project.model.method} · thử nghiệm, đệm ${project.model.black_views} camera đen` :
    `${project.model.method} (MODA ${project.model.moda ?? "?"})`) :
    `${detectorName} · chưa dùng model BEV`;
  const pixels = project.image_processing?.source_pixels === "undistorted" ?
    "ảnh đã khử méo · intrinsic_zero · không khử méo lại" :
    project.image_processing?.source_pixels === "raw" ?
      `ảnh raw · intrinsic_original · ${project.image_processing.applied_undistort ? "khử méo 1 lần" : "chưa khử méo"}` : "";
  $("projectInfo").textContent = `${project.config.root} · ${S.cams.length}/${project.all_cameras.length} camera · ` +
    `${S.frames.length} frame · ${pixels ? pixels + " · " : ""}${model} · ${project.config.out_dir}`;
  if (project.image_processing?.warning) $("projectInfo").textContent += ` · ⚠ ${project.image_processing.warning}`;
  $("btnExport").disabled = false;
  $("btnMetrics").disabled = !project.gt_supported;
  $("gtDir").value = project.gt_dir || "";
  $("showGt").parentElement.classList.toggle("hidden", !project.has_gt);
  $("legendGt").classList.toggle("hidden", !project.has_gt);
  $("legendGtDiff").classList.toggle("hidden", !project.has_gt);
  $("emptyHint").classList.add("hidden");
  buildTiles();
  renderFrameList();
  pollJob(project.job);
  const first = S.frames.find((f) => f.status === "done") || S.frames[0];
  if (first) selectFrame(S.frame !== null && S.frames.some((f) => f.frame === S.frame) ? S.frame : first.frame);
}

function renderFrameList() {
  const list = $("frameList");
  list.innerHTML = "";
  for (const item of S.frames) {
    const li = document.createElement("li");
    li.className = item.frame === S.frame ? "active" : "";
    const status = item.status === "done" ? "" : item.status === "error" ? " ⚠" : " …";
    li.innerHTML = `<span class="done">${item.reviewed ? "✓" : ""}</span><span class="num">${item.frame}${status}</span>` +
      (item.status === "done" ? `<span class="badge" title="Tổng vị trí gồm nhãn đã giữ và đề xuất">${item.people + item.suggested}</span>` +
        (item.check ? `<span class="badge check">${item.check}</span>` : "") +
        (item.suggested ? `<span class="badge suggested">${item.suggested}</span>` : "") + diffBadge(item.diff) : "");
    li.title = item.status === "done" ? `${item.people + item.suggested} vị trí · ${item.people} nhãn đã giữ · ${item.check} cần kiểm tra · ${item.suggested} đề xuất` +
      (item.diff ? `\nSo với label gốc (${item.diff.gt} người): thừa ${item.diff.fp} · thiếu ${item.diff.fn}` +
        (item.diff.error_cm != null ? ` · lệch TB ${item.diff.error_cm} cm` : "") : "") : item.status;
    li.onclick = () => selectFrame(item.frame);
    list.appendChild(li);
  }
  const reviewed = S.frames.filter((f) => f.reviewed).length;
  $("framesProgress").textContent = `${reviewed}/${S.frames.length}`;
}

function diffBadge(diff) {
  if (!diff) return "";
  const off = diff.fp + diff.fn;
  return off ? `<span class="badge diff">±${off}</span>` : '<span class="badge same">=</span>';
}

function updateSummary(summary) {
  const index = S.frames.findIndex((f) => f.frame === summary.frame);
  if (index >= 0) S.frames[index] = summary;
  renderFrameList();
}

function pollJob(job) {
  clearTimeout(S.jobTimer);
  showJob(job);
  if (!job || !job.running) return;
  S.jobTimer = setTimeout(async () => {
    try {
      const state = await api("/api/job");
      const wasPending = S.frames.find((f) => f.frame === S.frame)?.status !== "done";
      S.frames = state.frames;
      if (state.model_status) renderModelStatus(state.model_status);
      renderFrameList();
      if (wasPending && S.frames.find((f) => f.frame === S.frame)?.status === "done") selectFrame(S.frame);
      pollJob(state.job);
      if (!state.job.running) {
        if (state.job.error) toast(state.job.error, true);
        else toast("Auto-label xong");
        refreshMetrics();
      }
    } catch (error) {
      toast(error.message, true);
    }
  }, 1000);
}

function showJob(job) {
  if (S.project) S.project.job = job;
  updateActiveModelChoice();
  const bar = $("jobBar");
  if (!job || (!job.running && !job.error)) { bar.classList.add("hidden"); return; }
  bar.classList.remove("hidden");
  const fraction = job.total ? job.done / job.total : 0;
  bar.querySelector(".job-fill").style.width = `${100 * fraction}%`;
  bar.querySelector(".job-text").textContent = job.error ? "Lỗi: " + job.error :
    `Auto-label ${job.done}/${job.total}` + (job.eta_s != null ? ` · còn ~${job.eta_s}s` : "");
}

// ------------------------------------------------------------------------------------------------ model catalogue and runtime status

function fillModelOptions(select, models, selected) {
  select.replaceChildren(new Option("Không dùng BEV · chỉ YOLO đa view", ""));
  for (const model of models) {
    const state = model.experimental && model.compatible ? "Thử nghiệm · đệm camera đen" :
      model.compatible ? "Tương thích" : "Chưa chạy được";
    const option = new Option(`${model.best ? "★ " : ""}${model.method} · ${model.dataset} · ${model.num_cam} cam · ${model.name} · ${state}`, model.id);
    option.title = model.reasons?.join(" ") || "Tương thích với dataset và số camera của project.";
    select.add(option);
  }
  if (selected && !models.some((m) => m.id === selected)) select.add(new Option(`${selected} · chưa đăng ký`, selected));
  select.value = selected || "";
}

function modelDescription(model, selected) {
  if (!selected) return "Chưa chọn model BEV. Auto-label chạy YOLO từng ảnh rồi ghép đa view.";
  if (!model) return "Model này không còn trong danh sách đăng ký.";
  return `${model.experimental && model.compatible ? "Chạy thử nghiệm" : model.compatible ? "Tương thích" : "Chưa chạy được"}: ${model.dataset} · ${model.num_cam} camera. ` +
    (model.reasons?.join(" ") || model.warning || "Weight sẽ được nạp khi chạy auto-label.") +
    `\nWeight: ${model.weights}${model.zip_member ? " → " + model.zip_member : ""}`;
}

function showModelPanel(project) {
  $("modelPanel").classList.remove("hidden");
  $("modelError").textContent = "";
  fillModelOptions($("activeModel"), project.models || [], project.config.model_id);
  renderModelStatus(project.model_status);
  updateActiveModelChoice();
}

function renderModelStatus(status) {
  if (!status) {
    $("bevModelStatus").textContent = "Khởi động lại server để tải danh sách và tình trạng model.";
    $("yoloModelStatus").textContent = "";
    return;
  }
  S.project.model_status = status;
  const labels = {not_selected: "Chưa chọn", blocked: "Không chạy được", ready: "Chưa nạp", running: "Đang nạp/chạy", loaded: "Đã chạy", error: "Lỗi"};
  for (const [engine, id, name] of [["bev", "bevModelStatus", "BEV"], ["yolo", "yoloModelStatus", "YOLO"]]) {
    const item = status[engine];
    const selected = engine === "bev" ? item.model?.method : item.weights?.split(/[\\/]/).pop();
    $(id).dataset.state = item.state;
    $(id).textContent = `${name}${selected ? " · " + selected : ""}: ${labels[item.state] || item.state}. ${item.message}` +
      (item.outputs != null ? ` Frame ${item.last_frame}: ${item.outputs} ${engine === "bev" ? "điểm BEV" : "box YOLO"} · ${item.ms} ms.` : "");
  }
  $("modelCatalogSummary").textContent = `${status.registered} model đã đăng ký · ${status.compatible} tương thích`;
}

function updateActiveModelChoice() {
  if (!S.project) return;
  const selected = $("activeModel").value;
  const model = S.project.models?.find((m) => m.id === selected);
  $("activeModelInfo").textContent = modelDescription(model, selected);
  $("btnApplyModel").disabled = !S.project.model_status || !!S.project.job?.running ||
    selected === (S.project.config.model_id || "") || (!!selected && !model?.compatible);
}

async function refreshModelCatalog() {
  try {
    const result = await api("/api/models");
    S.project.models = result.models;
    fillModelOptions($("activeModel"), result.models, $("activeModel").value);
    renderModelStatus(result.model_status);
    updateActiveModelChoice();
    $("modelError").textContent = "";
  } catch (error) { $("modelError").textContent = error.message; }
}

async function applyProjectModel() {
  $("btnApplyModel").disabled = true;
  try {
    await flushSave();
    const project = await api("/api/model", {method: "POST", body: {model_id: $("activeModel").value || null}});
    setProject(project);
    toast("Đã lưu lựa chọn model cho lần auto-label tiếp theo.");
  } catch (error) { $("modelError").textContent = error.message; }
  finally { updateActiveModelChoice(); }
}

// ------------------------------------------------------------------------------------------------ frames

async function selectFrame(frame) {
  await flushSave();
  const version = S.project.image_version;
  S.frame = frame;
  S.sel = null; S.selYolo = null; S.undo = [];
  S.data = null; S.views = [];
  renderFrameList();
  const summary = S.frames.find((f) => f.frame === frame);
  if (!summary || summary.status !== "done") {
    S.data = null;
    $("frameTitle").textContent = `Frame ${frame} · đang auto-label…`;
    drawAll();
    return;
  }
  $("frameTitle").textContent = `Frame ${frame} · đang tải…`;
  drawAll();
  try {
    const data = await api(`/api/frame/${frame}`);
    if (S.frame !== frame || S.project.image_version !== version) return;
    S.data = data;
  } catch (error) {
    toast(error.message, true);
    return;
  }
  if (S.frame !== frame) return;
  for (const person of S.data.persons) person.override = person.override || {};
  refresh();
  prefetch(frame);
}

function prefetch(frame) {
  const index = S.frames.findIndex((f) => f.frame === frame);
  const next = S.frames[index + 1];
  if (next) for (const cam of S.cams) image(next.frame, cam.index);
}

function stepFrame(delta) {
  const index = S.frames.findIndex((f) => f.frame === S.frame);
  const next = S.frames[Math.min(S.frames.length - 1, Math.max(0, index + delta))];
  if (next && next.frame !== S.frame) selectFrame(next.frame);
}

function image(frame, camIndex) {
  const version = S.project.image_version;
  const key = `${version}/${frame}/${camIndex}`;
  let img = S.images.get(key);
  if (!img) {
    img = new Image();
    img.onload = () => { if (frame === S.frame && version === S.project.image_version) drawTiles(); };
    img.src = `/api/image/${frame}/${camIndex}?v=${encodeURIComponent(version)}`;
    S.images.set(key, img);
    if (S.images.size > 80) S.images.delete(S.images.keys().next().value);
  }
  return img;
}

function refresh() {
  if (!S.data) return drawAll();
  S.views = matchViews(S.data.persons);
  const persons = S.data.persons;
  const counted = persons.filter(included).length;
  const check = persons.filter((p) => p.status === "check" && !p.verified).length;
  const suggested = persons.filter((p) => p.status === "suggested" && !p.verified).length;
  $("frameTitle").innerHTML = `Frame ${S.frame} · ${persons.length} vị trí · ${counted} nhãn đã giữ` +
    (check ? ` · <span style="color:var(--check)">${check} cần kiểm tra</span>` : "") +
    (suggested ? ` · <span style="color:var(--suggested)">${suggested} đề xuất</span>` : "") +
    (S.data.reviewed ? ` · <span style="color:var(--ok)">đã review</span>` : "");
  drawAll();
}

function drawAll() {
  drawTiles();
  drawBev();
  renderInspector();
}

// ------------------------------------------------------------------------------------------------ edits

function person(uid = S.sel) {
  return S.data?.persons.find((p) => p.uid === uid) || null;
}

function snapshot() {
  S.undo.push(JSON.stringify({ persons: S.data.persons, reviewed: S.data.reviewed }));
  if (S.undo.length > 100) S.undo.shift();
}

function changed() {
  refresh();
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(save, 350);
}

async function save() {
  S.saveTimer = null;
  if (!S.data) return;
  const frame = S.frame;
  try {
    const summary = await api(`/api/frame/${frame}`, { method: "PUT", body: { persons: S.data.persons, reviewed: S.data.reviewed } });
    updateSummary(summary);
  } catch (error) {
    toast("Lưu thất bại: " + error.message, true);
  }
}

async function flushSave() {
  if (S.saveTimer) {
    clearTimeout(S.saveTimer);
    await save();
  }
}

function undo() {
  const last = S.undo.pop();
  if (!last) return;
  const state = JSON.parse(last);
  S.data.persons = state.persons;
  S.data.reviewed = state.reviewed;
  if (!person()) S.sel = null;
  changed();
}

function nextUid() {
  let n = 1;
  const used = new Set(S.data.persons.map((p) => p.uid));
  while (used.has(`m${n}`)) n++;
  return `m${n}`;
}

function addPerson(x, y) {
  if (!S.data) return;
  snapshot();
  const p = { uid: nextUid(), x: round4(x), y: round4(y), h: S.params.height, score: null, origin: "manual",
              status: "manual", verified: true, override: {}, auto: null };
  S.data.persons.push(p);
  S.sel = p.uid; S.selYolo = null;
  changed();
  snap(true);
}

function addFromYolo() {
  if (!S.selYolo) return;
  const cam = S.cams.find((c) => c.index === S.selYolo.cam);
  const det = S.data.yolo[String(cam.index)][S.selYolo.index];
  const foot = ground(cam, (det[0] + det[2]) / 2, det[3]);
  if (!foot) return toast("Chân của box này không nằm trên mặt sàn", true);
  // the lowest pixel is the near side of the feet: push the point away from the camera by foot_bias
  const dx = foot[0] - cam.centre[0], dy = foot[1] - cam.centre[1], d = Math.hypot(dx, dy) || 1;
  addPerson(foot[0] + (dx / d) * S.params.foot_bias, foot[1] + (dy / d) * S.params.foot_bias);
}

function moveSelected(dx, dy) {
  const p = person();
  if (!p) return;
  if (!S.moving) { snapshot(); S.moving = true; }
  moveLinkedPerson(p, p.x + dx, p.y + dy);
  changed();
}

function round4(v) { return Math.round(v * 10000) / 10000; }

async function snap(quiet = false) {
  const p = person();
  if (!p) return;
  try {
    const result = await api(`/api/frame/${S.frame}/snap`, { method: "POST", body: { person: p } });
    if (!result.feet) { if (!quiet) toast("Không có box YOLO khớp để snap"); return; }
    if (!quiet) snapshot();
    p.x = result.x; p.y = result.y;
    changed();
    if (!quiet) toast(`Snap theo ${result.feet} view YOLO`);
  } catch (error) {
    toast(error.message, true);
  }
}

function accept() {
  const p = person();
  if (!p) return;
  snapshot();
  p.verified = true;
  changed();
  selectNext(1);
}

function removePersonEntirely() {
  if (!person()) return;
  snapshot();
  S.data.persons = S.data.persons.filter((p) => p.uid !== S.sel);
  S.sel = null;
  changed();
}

function removeSelected() {
  if (!person()) return;
  if (!S.perView) return removePersonEntirely();
  if (S.focus.cam === null) return toast("Chọn camera cần xóa box, hoặc bấm 'tắt' tại hàng camera bên phải", true);
  setOverride("off");
  toast(`Đã xóa box nhãn ở C${S.focus.cam + 1}; người và các camera khác được giữ`);
}

function setOverride(mode) {
  const p = person();
  const cam = S.focus.cam;
  if (!p || cam === null) return toast("Bấm vào một camera trước để chọn view");
  snapshot();
  p.override[String(cam)] = mode;
  changed();
}

function changeHeight(delta) {
  const p = person();
  if (!p) return;
  snapshot();
  p.h = Math.round(Math.max(1.0, Math.min(2.2, (p.h || S.params.height) + delta)) * 100) / 100;
  changed();
}

function reviewOrder() {
  return [...S.data.persons].sort((a, b) =>
    (a.verified - b.verified) || (STATUS_ORDER[a.status] - STATUS_ORDER[b.status]) || (a.y - b.y) || (a.x - b.x));
}

function selectNext(direction) {
  if (!S.data || !S.data.persons.length) return;
  const order = reviewOrder();
  const pending = order.filter((p) => !p.verified && p.status !== "ok");
  const pool = pending.length ? pending : order;
  let index = pool.findIndex((p) => p.uid === S.sel);
  index = index < 0 ? (direction > 0 ? 0 : pool.length - 1) : (index + direction + pool.length) % pool.length;
  S.sel = pool[index].uid;
  S.selYolo = null;
  drawAll();
}

async function markReviewed() {
  if (!S.data) return;
  snapshot();
  S.data.reviewed = true;
  for (const p of S.data.persons) if (p.status !== "suggested") p.verified = true;
  changed();
  await flushSave();
  const index = S.frames.findIndex((f) => f.frame === S.frame);
  const next = S.frames.slice(index + 1).find((f) => !f.reviewed) || S.frames[index + 1];
  if (next) selectFrame(next.frame);
  else toast("Đã review frame cuối");
}

// ------------------------------------------------------------------------------------------------ camera tiles

function buildTiles() {
  const views = $("views");
  views.innerHTML = "";
  S.tiles = S.cams.map((cam) => {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.innerHTML = `<canvas></canvas><div class="label"></div>
      <div class="flip-controls" role="group" aria-label="Lật ảnh camera">
        <button type="button" data-flip="x" title="Lật ngang ảnh" aria-label="Lật ngang ảnh">⇄</button>
        <button type="button" data-flip="y" title="Lật dọc ảnh" aria-label="Lật dọc ảnh">⇅</button>
        <button type="button" data-flip="reset" title="Bỏ lật ảnh" aria-label="Bỏ lật ảnh">0</button>
      </div>`;
    views.appendChild(tile);
    tile.querySelector(".flip-controls").setAttribute("aria-label", `Lật ảnh camera ${cam.name}`);
    const t = { cam, el: tile, canvas: tile.querySelector("canvas"), label: tile.querySelector(".label"),
                zoom: null, flipX: false, flipY: false };
    tile.querySelectorAll("[data-flip]").forEach((button) => {
      button.onclick = () => {
        const axis = button.dataset.flip;
        if (axis === "reset") { t.flipX = false; t.flipY = false; }
        else if (axis === "x") t.flipX = !t.flipX;
        else t.flipY = !t.flipY;
        t.zoom = null;
        updateFlipButtons(t);
        drawTile(t);
      };
    });
    updateFlipButtons(t);
    attachTile(t);
    return t;
  });
  layoutTiles();
}

function updateFlipButtons(t) {
  for (const button of t.el.querySelectorAll("[data-flip]")) {
    const active = button.dataset.flip === "x" ? t.flipX : button.dataset.flip === "y" && t.flipY;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }
}

function layoutTiles() {
  if (S.mapFocus) return;
  // the column count that gives the largest tiles inside the view area
  // the view grid gets the height left above a BEV strip of at least 180 px; the strip takes the rest
  const views = $("views");
  const n = S.maximized !== null ? 1 : S.tiles.length;
  if (!n) return;
  const aspect = S.tiles[0].cam.width / S.tiles[0].cam.height;
  const center = document.querySelector(".center");
  const available = center.clientHeight - document.querySelector(".frame-bar").offsetHeight;
  const mapHeight = S.mapEdit ? 380 :
    S.project?.kind === "wavetrack" ? Math.min(520, Math.max(320, available * 0.43)) : 180;
  const W = views.clientWidth - 8, H = Math.max(120, available - mapHeight - 8), gap = 4;
  let best = { width: 0, columns: 1, rows: 1 };
  for (let columns = 1; columns <= n; columns++) {
    const rows = Math.ceil(n / columns);
    const width = Math.min((W - gap * (columns - 1)) / columns, ((H - gap * (rows - 1)) / rows) * aspect);
    // near-equal tile sizes: fewer rows leave more height to the BEV strip
    if (width > best.width * 1.06 || (width > best.width * 0.94 && rows < best.rows)) best = { width, columns, rows };
  }
  views.style.gridTemplateColumns = `repeat(${best.columns}, ${Math.floor(best.width)}px)`;
  views.style.height = `${Math.ceil(best.rows * (best.width / aspect + 2) + gap * (best.rows - 1) + 8)}px`;
  views.classList.toggle("maximized", S.maximized !== null);
  for (const t of S.tiles) t.el.classList.toggle("max", t.cam.index === S.maximized);
}

function tileTransform(t) {
  const W = t.canvas.width, H = t.canvas.height, cam = t.cam;
  const fit = Math.min(W / cam.width, H / cam.height);
  const base = { s: fit, ox: (W - cam.width * fit) / 2, oy: (H - cam.height * fit) / 2,
                 width: cam.width, height: cam.height, flipX: t.flipX, flipY: t.flipY };
  if (S.show.follow && S.data) {
    const p = person();
    const view = p && S.views[S.data.persons.indexOf(p)]?.[cam.index];
    if (view) {
      const [l, tp, r, b] = view.proj;
      const s = Math.max(fit, Math.min((0.6 * H) / (b - tp), 6 * fit));
      const centerU = t.flipX ? cam.width - (l + r) / 2 : (l + r) / 2;
      const centerV = t.flipY ? cam.height - (tp + b) / 2 : (tp + b) / 2;
      return clampView(t, { s, ox: W / 2 - centerU * s, oy: H / 2 - centerV * s });
    }
  }
  return t.zoom ? clampView(t, t.zoom) : base;
}

function clampView(t, tr) {
  // keep the image covering the tile (no black border while zoomed)
  const W = t.canvas.width, H = t.canvas.height;
  const w = t.cam.width * tr.s, h = t.cam.height * tr.s;
  const ox = w <= W ? (W - w) / 2 : Math.min(0, Math.max(W - w, tr.ox));
  const oy = h <= H ? (H - h) / 2 : Math.min(0, Math.max(H - h, tr.oy));
  return { s: tr.s, ox, oy, width: t.cam.width, height: t.cam.height, flipX: t.flipX, flipY: t.flipY };
}

function toScreen(tr, u, v) {
  return [(tr.flipX ? tr.width - u : u) * tr.s + tr.ox,
          (tr.flipY ? tr.height - v : v) * tr.s + tr.oy];
}
function toImage(tr, X, Y) {
  const u = (X - tr.ox) / tr.s, v = (Y - tr.oy) / tr.s;
  return [tr.flipX ? tr.width - u : u, tr.flipY ? tr.height - v : v];
}

function drawTiles() {
  drawCrops();
  if (S.mapFocus) return;
  for (const t of S.tiles) drawTile(t);
}

function strokeBox(ctx, tr, box, color, width, dash = []) {
  const [x1, y1, x2, y2] = screenBox(tr, box);
  ctx.setLineDash(dash);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);
  return [x1, y1, x2, y2];
}

function drawTile(t) {
  const dpr = window.devicePixelRatio || 1;
  const W = Math.round(t.canvas.clientWidth * dpr), H = Math.round(t.canvas.clientHeight * dpr);
  if (!W || !H) return;
  if (t.canvas.width !== W || t.canvas.height !== H) { t.canvas.width = W; t.canvas.height = H; }
  const ctx = t.canvas.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, W, H);
  t.el.classList.toggle("focus", S.focus.kind === "cam" && S.focus.cam === t.cam.index);
  if (S.frame === null) return;
  if (!cameraActive(t.cam.index)) {
    t.label.textContent = `C${t.cam.index + 1} ${t.cam.name} · đã hết video`;
    t.label.title = S.data?.sources?.[String(t.cam.index)]?.folder || t.cam.name;
    return;
  }
  const tr = tileTransform(t);
  const img = image(S.frame, t.cam.index);
  if (img.complete && img.naturalWidth) {
    ctx.save();
    ctx.translate(tr.ox + (tr.flipX ? t.cam.width * tr.s : 0),
                  tr.oy + (tr.flipY ? t.cam.height * tr.s : 0));
    ctx.scale(tr.flipX ? -tr.s : tr.s, tr.flipY ? -tr.s : tr.s);
    ctx.drawImage(img, 0, 0, t.cam.width, t.cam.height);
    ctx.restore();
  }
  if (S.show.grid && S.area) drawGroundGrid(ctx, t, tr, dpr);
  if (S.mapEdit && S.mapProjectCameras && S.siteMap) drawSiteMapOnCamera(ctx, t, tr, dpr);
  const obstaclesOnly = S.mapEdit && S.show.obstaclesOnly;
  const key = String(t.cam.index);
  let people = 0, proposals = 0;
  const yoloCount = S.data?.yolo[key]?.length || 0;
  if (S.data && !obstaclesOnly) {
    const yolo = S.data.yolo[key] || [];
    const matched = new Set();
    S.views.forEach((v) => { if (v[t.cam.index]?.yolo != null) matched.add(v[t.cam.index].yolo); });
    const match = S.show.gt ? gtMatch() : null;
    if (match) {
      const missed = new Set(match.missed);
      S.data.gt.forEach((g, j) => {
        if (!g.boxes[key]) return;
        if (missed.has(j)) strokeBox(ctx, tr, g.boxes[key], GT_MISS, 2 * dpr);
        else strokeBox(ctx, tr, g.boxes[key], "#22d3ee", 1.2 * dpr, [3 * dpr, 3 * dpr]);
      });
    }
    if (S.show.yolo) {
      yolo.forEach((det, i) => {
        const selected = S.selYolo && S.selYolo.cam === t.cam.index && S.selYolo.index === i;
        if (matched.has(i)) strokeBox(ctx, tr, det, "rgba(255,255,255,0.5)", 1 * dpr);
        else if (selected) {
          strokeBox(ctx, tr, det, "#000000", 6 * dpr);
          strokeBox(ctx, tr, det, SELECTED, 3 * dpr, [5 * dpr, 3 * dpr]);
        } else strokeBox(ctx, tr, det, "#ff6b6b", 1.3 * dpr, [2 * dpr, 3 * dpr]);
      });
    }
    let selectedBox = null;
    S.data.persons.forEach((p, k) => {
      const view = S.views[k][t.cam.index];
      const box = labelBox(p, t.cam.index, view);
      if (!view && !box) return;
      const source = boxSource(p, t.cam.index);
      if (source === "off") return;
      const isSel = p.uid === S.sel;
      const color = isSel ? SELECTED : colorOf(p);
      const dash = p.status === "suggested" && !p.verified ? [6 * dpr, 4 * dpr] : [];
      if (included(p)) people += box ? 1 : 0;
      else if (box || view?.proj) proposals += 1;
      if (S.show.proj && box && view && (source === "manual" || (source === "yolo" && view.yolo !== null))) {
        strokeBox(ctx, tr, view.proj, color, 1 * dpr, [4 * dpr, 3 * dpr]);
      }
      const shown = box || view.proj;
      if (isSel) selectedBox = shown;
      if (isSel) strokeBox(ctx, tr, shown, "#000000", 7 * dpr);
      const rect = strokeBox(ctx, tr, shown, color, (isSel ? 3.5 : p.verified ? 2 : 1.6) * dpr,
                             isSel ? [] : box ? dash : [1 * dpr, 4 * dpr]);
      const foot = projectPoint(t.cam, [p.x, p.y, 0]);
      if (foot) {
        const [fx, fy] = toScreen(tr, foot[0], foot[1]);
        ctx.fillStyle = color;
        ctx.beginPath(); ctx.arc(fx, fy, (isSel ? 4 : 2.5) * dpr, 0, 2 * Math.PI); ctx.fill();
      }
      if (p.verified) {
        ctx.fillStyle = color;
        ctx.font = `${11 * dpr}px Segoe UI`;
        ctx.fillText("✓", rect[0] + 2 * dpr, rect[1] + 11 * dpr);
      }
    });
    if (selectedBox) drawHandles(ctx, tr, selectedBox, dpr);
  }
  if (S.mapEdit && S.mapProjectCameras && S.siteMap) drawMapCameraHandles(ctx, t, tr, dpr);
  const zoom = tr.s / Math.min(W / t.cam.width, H / t.cam.height);
  t.label.innerHTML = `C${t.cam.index + 1} <span class="muted">${t.cam.name}` +
    (S.data ? obstaclesOnly ? " · chỉ xem vật cản" :
      ` · ${yoloCount} YOLO · ${people} nhãn` + (proposals ? ` · ${proposals} đề xuất` : "") : " · đang xử lý") +
    (zoom > 1.05 ? ` · ×${zoom.toFixed(1)}` : "") + "</span>";
  const source = S.data?.sources?.[String(t.cam.index)];
  t.label.title = source ? `${source.folder} / ${source.file || "đã hết video"}` : t.cam.name;
  if (source?.file) {
    const detail = document.createElement("span");
    detail.className = "muted";
    detail.textContent = ` · ${source.file}` + (source.time_s != null ? ` @ ${source.time_s.toFixed(2)} s` : "");
    t.label.append(detail);
  }
}

// ------------------------------------------------------------------------------------------------ ground grid

const NEAR = 0.1;   // metres in front of the camera: ground lines are cut there before projecting

function frontPolygon(cam, points) {
  // Sutherland-Hodgman against the plane z = NEAR in camera coordinates
  const cs = points.map((X) => toCamera(cam, X));
  const out = [];
  cs.forEach((a, i) => {
    const b = cs[(i + 1) % cs.length];
    if (a[2] >= NEAR) out.push(a);
    if ((a[2] >= NEAR) !== (b[2] >= NEAR)) {
      const k = (NEAR - a[2]) / (b[2] - a[2]);
      out.push(a.map((v, j) => v + k * (b[j] - v)));
    }
  });
  return out;
}

function groundSegment(cam, tr, a, b) {
  // screen end points of the ground segment a-b (undistorted frames: straight lines stay straight)
  let ca = toCamera(cam, [a[0], a[1], 0]), cb = toCamera(cam, [b[0], b[1], 0]);
  if (ca[2] < NEAR && cb[2] < NEAR) return null;
  if (ca[2] < NEAR) ca = ca.map((v, j) => v + ((NEAR - ca[2]) / (cb[2] - ca[2])) * (cb[j] - v));
  if (cb[2] < NEAR) cb = cb.map((v, j) => v + ((NEAR - cb[2]) / (ca[2] - cb[2])) * (ca[j] - v));
  return [toScreen(tr, ...toPixel(cam, ca)), toScreen(tr, ...toPixel(cam, cb))];
}

function strokeGround(ctx, cam, tr, a, b, color, width) {
  const seg = groundSegment(cam, tr, a, b);
  if (!seg) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(...seg[0]); ctx.lineTo(...seg[1]); ctx.stroke();
}

function drawGroundGrid(ctx, t, tr, dpr) {
  // The BEV area (what gets labelled) tinted on the ground, its 1 m grid, metre marks every 5 m on the border,
  // and a 25 cm grid around the selected person for fine alignment.
  const cam = t.cam;
  const [x0, x1, y0, y1] = S.area;
  ctx.save();
  ctx.beginPath(); ctx.rect(tr.ox, tr.oy, cam.width * tr.s, cam.height * tr.s); ctx.clip();
  const poly = frontPolygon(cam, [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]);
  if (poly.length >= 3) {
    ctx.beginPath();
    poly.forEach((c, i) => { const [X, Y] = toScreen(tr, ...toPixel(cam, c)); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
    ctx.closePath();
    ctx.fillStyle = "rgba(56,189,248,0.13)";
    ctx.fill();
  }
  const major = (v) => Math.abs(Math.round(v)) % 5 === 0;
  for (let x = Math.ceil(x0); x <= x1; x++) {
    strokeGround(ctx, cam, tr, [x, y0], [x, y1], `rgba(255,255,255,${major(x) ? 0.5 : 0.22})`, (major(x) ? 1.3 : 0.8) * dpr);
  }
  for (let y = Math.ceil(y0); y <= y1; y++) {
    strokeGround(ctx, cam, tr, [x0, y], [x1, y], `rgba(255,255,255,${major(y) ? 0.5 : 0.22})`, (major(y) ? 1.3 : 0.8) * dpr);
  }
  const border = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  for (const [width, color] of [[4.5, "#000"], [2, "#fff"]]) {
    border.forEach((a, i) => strokeGround(ctx, cam, tr, a, border[(i + 1) % 4], color, width * dpr));
  }
  ctx.font = `${10 * dpr}px Segoe UI`;
  ctx.textBaseline = "middle";
  const mark = (X, text) => {
    const c = toCamera(cam, [X[0], X[1], 0]);
    if (c[2] < NEAR) return;
    const [u, v] = toScreen(tr, ...toPixel(cam, c));
    if (u < tr.ox || v < tr.oy || u > tr.ox + cam.width * tr.s || v > tr.oy + cam.height * tr.s) return;
    const w = ctx.measureText(text).width + 6 * dpr;
    ctx.fillStyle = "rgba(0,0,0,0.65)"; ctx.fillRect(u - w / 2, v - 7 * dpr, w, 14 * dpr);
    ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.fillText(text, u, v);
  };
  for (let x = Math.ceil(x0); x <= x1; x++) if (major(x)) { mark([x, y0], `x ${x}`); mark([x, y1], `x ${x}`); }
  for (let y = Math.ceil(y0); y <= y1; y++) if (major(y)) { mark([x0, y], `y ${y}`); mark([x1, y], `y ${y}`); }
  const p = S.mapEdit && S.show.obstaclesOnly ? null : person();
  if (p) {
    const step = 0.25, cx = Math.round(p.x / step) * step, cy = Math.round(p.y / step) * step;
    for (let k = -4; k <= 4; k++) {
      strokeGround(ctx, cam, tr, [cx + k * step, cy - 1], [cx + k * step, cy + 1], "rgba(255,43,214,0.55)", 0.9 * dpr);
      strokeGround(ctx, cam, tr, [cx - 1, cy + k * step], [cx + 1, cy + k * step], "rgba(255,43,214,0.55)", 0.9 * dpr);
    }
  }
  ctx.restore();
}

function insideArea(p) {
  const [x0, x1, y0, y1] = S.area;
  return p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1;
}

function tilePoint(t, event) {
  const rect = t.canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const X = (event.clientX - rect.left) * dpr, Y = (event.clientY - rect.top) * dpr;
  return { X, Y, img: toImage(tileTransform(t), X, Y) };
}

// ------------------------------------------------------------------------------------------------ box handles
// The selected person's box has 8 resize handles (they change the box of this camera only) and a move grip above
// its top edge; clicking inside a box only selects it, so a box is never moved while being resized.

const HANDLE = 5;   // half size of a resize handle, css px
const CURSORS = { n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", ne: "nesw-resize", sw: "nesw-resize",
                  nw: "nwse-resize", se: "nwse-resize", move: "move" };

function handlePoints(tr, box, dpr) {
  const [x1, y1, x2, y2] = screenBox(tr, box);
  const xm = (x1 + x2) / 2, ym = (y1 + y2) / 2;
  return { nw: [x1, y1], n: [xm, y1], ne: [x2, y1], e: [x2, ym], se: [x2, y2], s: [xm, y2], sw: [x1, y2], w: [x1, ym],
           move: [xm, y1 - 16 * dpr] };
}

function screenBox(tr, box) {
  const [x1, y1] = toScreen(tr, box[0], box[1]);
  const [x2, y2] = toScreen(tr, box[2], box[3]);
  return [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)];
}

function drawHandles(ctx, tr, box, dpr) {
  const points = handlePoints(tr, box, dpr);
  const h = HANDLE * dpr;
  for (const [name, [X, Y]] of Object.entries(points)) {
    if (name === "move") continue;
    ctx.fillStyle = "#fff"; ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5 * dpr;
    ctx.fillRect(X - h, Y - h, 2 * h, 2 * h); ctx.strokeRect(X - h, Y - h, 2 * h, 2 * h);
  }
  const [X, Y] = points.move;
  line(ctx, [X, Y + 8 * dpr], [X, points.n[1]], SELECTED, dpr);
  ctx.beginPath(); ctx.arc(X, Y, 8 * dpr, 0, 2 * Math.PI);
  ctx.fillStyle = SELECTED; ctx.fill(); ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5 * dpr; ctx.stroke();
  ctx.fillStyle = "#fff"; ctx.font = `${11 * dpr}px Segoe UI`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("✥", X, Y + 0.5 * dpr);
  ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
}

function selectedTileBox(t) {
  const p = person();
  if (!p || !S.data) return null;
  if (boxSource(p, t.cam.index) === "off") return null;
  const view = S.views[S.data.persons.indexOf(p)]?.[t.cam.index];
  return labelBox(p, t.cam.index, view) || view?.proj || null;
}

function hitHandle(t, X, Y) {
  const box = selectedTileBox(t);
  if (!box) return null;
  const dpr = window.devicePixelRatio || 1;
  const points = handlePoints(tileTransform(t), box, dpr);
  const [mx, my] = points.move;
  if (Math.hypot(X - mx, Y - my) <= 10 * dpr) return "move";
  const reach = (HANDLE + 3) * dpr;
  for (const [name, [hx, hy]] of Object.entries(points)) {
    if (name !== "move" && Math.abs(X - hx) <= reach && Math.abs(Y - hy) <= reach) return name;
  }
  return null;
}

function clampBox(cam, box) {
  const [x1, y1, x2, y2] = box;
  const r1 = (v) => Math.round(v * 10) / 10;
  return [r1(Math.max(0, Math.min(x1, x2))), r1(Math.max(0, Math.min(y1, y2))),
          r1(Math.min(cam.width - 1, Math.max(x1, x2))), r1(Math.min(cam.height - 1, Math.max(y1, y2)))];
}

function setManualBox(p, cam, box) {
  const b = clampBox(cam, box);
  if (b[2] - b[0] >= 4 && b[3] - b[1] >= 4) p.override[String(cam.index)] = b;
}

function setPerView(on) {
  S.perView = on;
  $("perView").checked = on;
  try { localStorage.setItem("mvlabel.perView", on ? "1" : "0"); } catch (error) { /* private window */ }
  if (S.data) renderInspector();
}

function moveLinkedPerson(p, x, y) {
  const old = [p.x, p.y, 0], next = [round4(x), round4(y), 0];
  p.override ||= {};
  for (const cam of S.cams) {
    const key = String(cam.index), box = manualBox(p, cam.index);
    if (box) {
      const before = projectPoint(cam, old), after = projectPoint(cam, next);
      if (before && after) {
        const du = after[0] - before[0], dv = after[1] - before[1];
        setManualBox(p, cam, [box[0] + du, box[1] + dv, box[2] + du, box[3] + dv]);
      }
    } else if (boxSource(p, cam.index) !== 'off') {
      // A fixed YOLO detection must not pin a linked label to its previous location.
      p.override[key] = 'proj';
    }
  }
  p.x = next[0]; p.y = next[1];
}

function startBoxMove(t, pt) {
  const p = person(), box = selectedTileBox(t);
  if (!p || !box) return;
  const foot = projectPoint(t.cam, [p.x, p.y, 0]);
  S.drag = { kind: 'boxmove', handle: 'move', t, start: pt.img, box: box.slice(),
             offset: foot ? [foot[0] - pt.img[0], foot[1] - pt.img[1]] : [0, 0], moved: false };
  S.selYolo = null;
}

function addBoxAtCursor() {
  if (!S.data) return;
  if (S.mouse?.kind !== "cam") return toast("Đưa chuột lên ảnh camera (tại chân người) rồi bấm T");
  const t = S.mouse.t, cam = t.cam;
  const [u, v] = tilePoint(t, S.mouse.event).img;
  if (u < 0 || v < 0 || u >= cam.width || v >= cam.height) return toast("Con trỏ nằm ngoài ảnh");
  const g = ground(cam, u, v);
  // the cursor is the feet: the box takes the size of the person cylinder standing there
  const cylinder = g && projectBox(cam, g[0], g[1], S.params.radius, S.params.height);
  const w = cylinder ? cylinder[2] - cylinder[0] : 60, h = cylinder ? cylinder[3] - cylinder[1] : 150;
  const box = [u - w / 2, v - h, u + w / 2, v];
  const sel = person();
  snapshot();
  S.focus = { kind: "cam", cam: cam.index };
  if (sel && !labelBox(sel, cam.index, S.views[S.data.persons.indexOf(sel)]?.[cam.index])) {
    setManualBox(sel, cam, box);   // the selected person had no box in this camera: give it this one
    toast(`Thêm box C${cam.index + 1} cho người đang chọn`);
  } else {
    if (!g) { S.undo.pop(); return toast("Điểm chân không nằm trên mặt sàn: đưa con trỏ xuống chân người", true); }
    const p = { uid: nextUid(), x: round4(g[0]), y: round4(g[1]), h: S.params.height, score: null, origin: "manual",
                status: "manual", verified: true, override: {}, auto: null };
    setManualBox(p, cam, box);
    if (S.perView) for (const other of S.cams) if (other.index !== cam.index) p.override[String(other.index)] = "off";
    S.data.persons.push(p);
    S.sel = p.uid;
    toast(S.perView ? `Thêm người chỉ có box ở C${cam.index + 1}` : "Thêm người: box tay ở camera này, các camera khác theo box chiếu");
  }
  S.selYolo = null;
  changed();
}

function hitPerson(t, u, v) {
  // overlapping boxes: the person nearest the camera (lowest box bottom) is the one in front
  let best = null;
  S.data.persons.forEach((p, k) => {
    if (boxSource(p, t.cam.index) === "off") return;
    const view = S.views[k][t.cam.index];
    const box = labelBox(p, t.cam.index, view) || view?.proj;
    if (!box) return;
    if (u >= box[0] && u <= box[2] && v >= box[1] && v <= box[3] && (!best || box[3] > best.bottom)) {
      best = { uid: p.uid, bottom: box[3] };
    }
  });
  return best?.uid || null;
}

function hitYolo(t, u, v) {
  const matched = new Set();
  S.views.forEach((view) => { if (view[t.cam.index]?.yolo != null) matched.add(view[t.cam.index].yolo); });
  const yolo = S.data.yolo[String(t.cam.index)] || [];
  let best = null;
  yolo.forEach((det, i) => {
    if (matched.has(i) || u < det[0] || u > det[2] || v < det[1] || v > det[3]) return;
    if (!best || det[3] > best.bottom) best = { index: i, bottom: det[3] };
  });
  return best ? best.index : null;
}

function attachTile(t) {
  const c = t.canvas;
  c.addEventListener("mousedown", (event) => {
    if (!S.data) return;
    S.focus = { kind: "cam", cam: t.cam.index };
    const pt = tilePoint(t, event);
    if (S.mapEdit && S.mapTool !== "select" && event.button === 0) {
      if (!cameraActive(t.cam.index))
        return toast("Camera này đã hết video ở frame hiện tại; hãy chọn frame có ảnh để chấm điểm.", true);
      const sourceImage = image(S.frame, t.cam.index);
      if (!sourceImage.complete || !sourceImage.naturalWidth)
        return toast("Ảnh camera chưa tải xong; hãy thử lại sau khi ảnh hiện lên.", true);
      const point = ground(t.cam, ...pt.img);
      if (point) addMapPoint(point);
      else toast("Điểm ảnh này không cắt mặt sàn phía trước camera", true);
      return;
    }
    if (event.button === 2 || event.button === 1) {
      const tr = tileTransform(t);
      S.drag = { kind: "pan", t, X: pt.X, Y: pt.Y, tr: { ...tr } };
      event.preventDefault();
      return;
    }
    if (S.mapEdit && S.mapTool === "select" && event.button === 0) {
      const point = hitMapCameraPoint(t, pt.X, pt.Y);
      if (point) {
        const additive = event.ctrlKey || event.metaKey;
        selectMapObject(point.id, additive);
        if (additive) return;
        S.mapActivePoint = { id: point.id, index: point.index };
        const offset = [point.pixel[0] - pt.img[0], point.pixel[1] - pt.img[1]];
        S.drag = { kind: "map-point", t, id: point.id, index: point.index,
          offset, X: pt.X, Y: pt.Y, moved: false };
        renderMapEditor(); drawBev(); drawTiles();
        return;
      }
    }
    if (S.mapEdit && S.show.obstaclesOnly) return;
    const handle = hitHandle(t, pt.X, pt.Y);
    if (handle) {
      if (handle === 'move') {
        startBoxMove(t, pt);
        return drawAll();
      }
      const p = person();
      const foot = projectPoint(t.cam, [p.x, p.y, 0]);
      S.drag = { kind: handle === "move" ? "boxmove" : "resize", handle, t, start: pt.img, box: selectedTileBox(t).slice(),
                 manual: manualBox(p, t.cam.index)?.slice() || null,
                 offset: foot ? [foot[0] - pt.img[0], foot[1] - pt.img[1]] : [0, 0], moved: false };
      S.selYolo = null;
      return drawAll();
    }
    const uid = hitPerson(t, ...pt.img);
    if (uid) {
      S.sel = uid; S.selYolo = null;
      startBoxMove(t, pt);
    } else {
      const y = S.show.yolo ? hitYolo(t, ...pt.img) : null;
      S.selYolo = y !== null ? { cam: t.cam.index, index: y } : null;
      S.sel = null;
    }
    drawAll();
  });
  c.addEventListener("mousemove", (event) => {
    S.mouse = { kind: "cam", t, event };
    const pt = tilePoint(t, event);
    if (!S.drag || S.drag.t !== t) {
      const drawingMap = S.mapEdit && S.mapTool !== "select";
      const mapPoint = S.mapEdit && S.mapTool === "select" ? hitMapCameraPoint(t, pt.X, pt.Y) : null;
      const handle = S.data && !S.drag && !drawingMap && !(S.mapEdit && S.show.obstaclesOnly) ?
        hitHandle(t, pt.X, pt.Y) : null;
      const body = S.data && !S.mapEdit && hitPerson(t, ...pt.img);
      c.style.cursor = drawingMap ? "crosshair" : mapPoint ? "grab" : handle ? CURSORS[handle] : body ? "grab" : "";
      return;
    }
    if (S.drag.kind === "pan") {
      t.zoom = { s: S.drag.tr.s, ox: S.drag.tr.ox + pt.X - S.drag.X, oy: S.drag.tr.oy + pt.Y - S.drag.Y };
      S.show.follow = false; $("followZoom").checked = false;
      drawTile(t);
      return;
    }
    if (S.drag.kind === "map-point") {
      if (!S.drag.moved && Math.hypot(pt.X - S.drag.X, pt.Y - S.drag.Y) < 2 * (window.devicePixelRatio || 1)) return;
      moveMapCameraPoint(S.drag, pt.img);
      return;
    }
    const p = person();
    if (!p) return;
    const du = pt.img[0] - S.drag.start[0], dv = pt.img[1] - S.drag.start[1];
    if (!S.drag.moved) { if (Math.hypot(du, dv) * tileTransform(t).s < 2) return; snapshot(); S.drag.moved = true; }
    const shift = (b) => [b[0] + du, b[1] + dv, b[2] + du, b[3] + dv];
    if (S.drag.kind === "resize") {
      const b = S.drag.box.slice();
      let h = S.drag.handle;
      if (t.flipX) h = h.replace(/[we]/g, (side) => side === "w" ? "e" : "w");
      if (t.flipY) h = h.replace(/[ns]/g, (side) => side === "n" ? "s" : "n");
      if (h.includes("w")) b[0] += du;
      if (h.includes("e")) b[2] += du;
      if (h.includes("n")) b[1] += dv;
      if (h.includes("s")) b[3] += dv;
      setManualBox(p, t.cam, b);
    } else if (S.perView) {
      setManualBox(p, t.cam, shift(S.drag.box));   // this camera only: the person and the other views stay
    } else {
      // linked: the feet follow the grip, every projected box follows; a hand-drawn box here moves along
      const g = ground(t.cam, pt.img[0] + S.drag.offset[0], pt.img[1] + S.drag.offset[1]);
      if (!g) return;
      moveLinkedPerson(p, g[0], g[1]);
    }
    refresh();
  });
  c.addEventListener("wheel", (event) => {
    event.preventDefault();
    const pt = tilePoint(t, event);
    const tr = tileTransform(t);
    const fit = Math.min(c.width / t.cam.width, c.height / t.cam.height);
    const factor = event.deltaY < 0 ? 1.25 : 0.8;
    const s = Math.max(fit, Math.min(fit * 12, tr.s * factor));
    const k = s / tr.s;
    t.zoom = { s, ox: pt.X - (pt.X - tr.ox) * k, oy: pt.Y - (pt.Y - tr.oy) * k };
    if (s <= fit * 1.001) t.zoom = null;
    S.show.follow = false; $("followZoom").checked = false;
    drawTile(t);
  }, { passive: false });
  c.addEventListener("dblclick", () => {
    S.maximized = S.maximized === t.cam.index ? null : t.cam.index;
    layoutTiles();
    requestAnimationFrame(drawTiles);
  });
  c.addEventListener("contextmenu", (event) => event.preventDefault());
  c.addEventListener("mouseleave", () => { if (S.mouse?.t === t) S.mouse = null; });
}

window.addEventListener("mouseup", () => {
  if (S.drag?.kind === "map-point") {
    const moved = S.drag.moved;
    S.drag = null;
    if (moved) saveSiteMap(mapCopy());
    return;
  }
  if (["boxmove", "resize"].includes(S.drag?.kind) && S.drag.moved) changed();
  S.drag = null;
});

// ------------------------------------------------------------------------------------------------ BEV

function bevMapping() {
  const canvas = $("bev");
  const [x0, x1, y0, y1] = S.siteMap?.bounds || S.area;
  const pad = S.siteMap ? 0.25 : 0.6;
  const horizontalY = S.siteMap ? false : (y1 - y0) >= (x1 - x0);
  const flipVertical = !!S.siteMap?.flip_vertical;
  const flipHorizontal = !!S.siteMap?.flip_horizontal;
  const spanU = horizontalY ? y1 - y0 + 2 * pad : x1 - x0 + 2 * pad;
  const spanV = horizontalY ? x1 - x0 + 2 * pad : y1 - y0 + 2 * pad;
  const dpr = window.devicePixelRatio || 1;
  const W = Math.round(canvas.clientWidth * dpr), H = Math.round(canvas.clientHeight * dpr);
  const zoom = S.siteMap ? S.mapView.zoom : 1;
  const s = Math.min(W / spanU, H / spanV) * zoom;
  const du = (W - spanU * s) / 2 + (S.siteMap ? S.mapView.panX * dpr : 0);
  const dv = (H - spanV * s) / 2 + (S.siteMap ? S.mapView.panY * dpr : 0);
  const orient = (x, y) => horizontalY ? [y - y0 + pad, x - x0 + pad] :
    [flipHorizontal ? x1 - x + pad : x - x0 + pad,
     flipVertical ? y1 - y + pad : y - y0 + pad];
  const deorient = (u, v) => horizontalY ? [v + x0 - pad, u + y0 - pad] :
    [flipHorizontal ? x1 + pad - u : u + x0 - pad,
     flipVertical ? y1 + pad - v : v + y0 - pad];
  return {
    W, H, s, flipVertical, flipHorizontal, zoom, du, dv,
    to: (x, y) => { const [u, v] = orient(x, y); return [du + u * s, dv + v * s]; },
    from: (u, v) => deorient((u - du) / s, (v - dv) / s),
    // screen arrow -> world step
    step: (u, v) => horizontalY ? [v, u] : [flipHorizontal ? -u : u, flipVertical ? -v : v],
  };
}

function drawBev() {
  const canvas = $("bev");
  if (!S.area) return;
  const m = bevMapping();
  S.bev = m;
  canvas.width = m.W; canvas.height = m.H;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  ctx.fillStyle = S.siteMap ? "#e7ecee" : "#0b0e12";
  ctx.fillRect(0, 0, m.W, m.H);
  const [x0, x1, y0, y1] = S.siteMap?.bounds || S.area;
  const a = m.to(x0, y0), b = m.to(x1, y1);
  const left = Math.min(a[0], b[0]), top = Math.min(a[1], b[1]);
  const width = Math.abs(b[0] - a[0]), height = Math.abs(b[1] - a[1]);
  if (S.siteMap) {
    ctx.fillStyle = "#fff";
    if (S.siteMap.floor_regions?.length) { mapFloorPath(ctx, m); ctx.fill(); }
    else ctx.fillRect(left, top, width, height);
  }
  if (S.siteMap && S.mapShowCoverage) drawMapCoverage(ctx, m, dpr);
  ctx.save();
  if (S.siteMap?.floor_regions?.length) { mapFloorPath(ctx, m); ctx.clip(); }
  for (let x = Math.ceil(x0); x <= x1; x++) line(ctx, m.to(x, y0), m.to(x, y1),
    S.siteMap ? x % 5 === 0 ? "#c5cbca" : "#e1e4e0" : x % 5 === 0 ? "#2a323c" : "#171c22", dpr);
  for (let y = Math.ceil(y0); y <= y1; y++) line(ctx, m.to(x0, y), m.to(x1, y),
    S.siteMap ? y % 5 === 0 ? "#c5cbca" : "#e1e4e0" : y % 5 === 0 ? "#2a323c" : "#171c22", dpr);
  ctx.restore();
  ctx.strokeStyle = S.siteMap ? "#9ca8a8" : "#46505d"; ctx.lineWidth = 1 * dpr;
  ctx.strokeRect(left, top, width, height);
  if (S.siteMap) {
    ctx.font = `${10 * dpr}px Consolas, monospace`;
    ctx.fillStyle = "#57676b";
    for (let x = Math.ceil(x0); x <= x1; x++) if (x % 2 === 0) {
      const [u, v] = m.to(x, y0);
      ctx.fillText(`x ${x}`, u + 3 * dpr, v + (m.flipVertical ? 13 : -4) * dpr);
    }
    for (let y = Math.ceil(y0); y <= y1; y++) if (y % 2 === 0) {
      const [u, v] = m.to(x0, y);
      ctx.fillText(`y ${y}`, u + (m.flipHorizontal ? -25 : 3) * dpr, v - 3 * dpr);
    }
  }
  if (S.siteMap) drawSiteMap(ctx, m, dpr);
  if (!S.data || (S.mapEdit && S.show.obstaclesOnly)) return;
  const r = Math.max(3, 0.25 * m.s);
  const match = S.show.gt ? gtMatch() : null;
  const extra = new Set(match ? match.extra.map((p) => p.uid) : []);
  if (match) {
    const missed = new Set(match.missed);
    S.data.gt.forEach((g, j) => {
      const [u, v] = m.to(g.x, g.y);
      ctx.strokeStyle = missed.has(j) ? GT_MISS : "#22d3ee";
      ctx.lineWidth = (missed.has(j) ? 2 : 1) * dpr;
      ctx.strokeRect(u - r * 0.8, v - r * 0.8, r * 1.6, r * 1.6);
    });
    for (const [uid, hit] of match.byUid) {
      const p = S.data.persons.find((q) => q.uid === uid);
      line(ctx, m.to(p.x, p.y), m.to(S.data.gt[hit.g].x, S.data.gt[hit.g].y), "#22d3ee", dpr);
    }
  }
  for (const p of S.data.persons) {
    const [u, v] = m.to(p.x, p.y);
    const color = colorOf(p);
    if (extra.has(p.uid)) {
      ctx.beginPath(); ctx.arc(u, v, r + 3 * dpr, 0, 2 * Math.PI);
      ctx.strokeStyle = GT_MISS; ctx.lineWidth = 2 * dpr; ctx.stroke();
    }
    if (p.auto && (Math.abs(p.auto[0] - p.x) > 1e-3 || Math.abs(p.auto[1] - p.y) > 1e-3)) {
      const [au, av] = m.to(p.auto[0], p.auto[1]);
      line(ctx, [au, av], [u, v], "rgba(255,255,255,0.35)", dpr);
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.fillRect(au - 1.5 * dpr, av - 1.5 * dpr, 3 * dpr, 3 * dpr);
    }
    ctx.beginPath(); ctx.arc(u, v, r, 0, 2 * Math.PI);
    if (p.verified || p.status !== "suggested") { ctx.fillStyle = color + (p.verified ? "" : "99"); ctx.fill(); }
    ctx.setLineDash(p.status === "suggested" && !p.verified ? [3 * dpr, 2 * dpr] : []);
    ctx.strokeStyle = color; ctx.lineWidth = 1.5 * dpr; ctx.stroke(); ctx.setLineDash([]);
    if (p.uid === S.sel) {
      ctx.beginPath(); ctx.arc(u, v, r + 4 * dpr, 0, 2 * Math.PI);
      ctx.strokeStyle = "#000"; ctx.lineWidth = 5 * dpr; ctx.stroke();
      ctx.strokeStyle = SELECTED; ctx.lineWidth = 2.5 * dpr; ctx.stroke();
    }
  }
}

const GT_MISS = "#ff3b3b";

// Labels against the original ones of this frame, live while editing: pairs within 0.5 m, nearest first
// (the frame list and the metrics panel use the server's Hungarian matching; they agree unless people stand < 0.5 m apart).
function gtMatch() {
  if (!S.data?.gt) return null;
  const labels = S.data.persons.filter((p) => p.status !== "suggested" || p.verified);
  const pairs = [];
  labels.forEach((p) => S.data.gt.forEach((g, j) => {
    const d = Math.hypot(p.x - g.x, p.y - g.y);
    if (d <= 0.5) pairs.push([d, p.uid, j]);
  }));
  pairs.sort((a, b) => a[0] - b[0]);
  const byUid = new Map(), used = new Set();
  for (const [d, uid, j] of pairs) {
    if (byUid.has(uid) || used.has(j)) continue;
    byUid.set(uid, { g: j, d });
    used.add(j);
  }
  return { byUid, labels, missed: S.data.gt.map((_, j) => j).filter((j) => !used.has(j)),
           extra: labels.filter((p) => !byUid.has(p.uid)) };
}

function line(ctx, a, b, color, dpr) {
  ctx.strokeStyle = color; ctx.lineWidth = 1 * dpr;
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
}

function bevPoint(event) {
  const canvas = $("bev");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  return S.bev.from((event.clientX - rect.left) * dpr, (event.clientY - rect.top) * dpr);
}

function bevHit(x, y) {
  // nearest person within 0.7 m, or within 10 screen px when the map is drawn small
  const radius = Math.max(0.7, (10 * (window.devicePixelRatio || 1)) / S.bev.s);
  let best = null, bestD = radius;
  for (const p of S.data.persons) {
    const d = Math.hypot(p.x - x, p.y - y);
    if (d < bestD) { best = p; bestD = d; }
  }
  return best;
}

function attachBev() {
  const canvas = $("bev");
  canvas.addEventListener("mousedown", (event) => {
    if (S.siteMap && (event.button === 1 || event.button === 2 || event.button === 0 && event.shiftKey)) {
      S.mapPan = { X: event.clientX, Y: event.clientY,
        panX: S.mapView.panX, panY: S.mapView.panY };
      canvas.style.cursor = "grabbing";
      event.preventDefault();
      return;
    }
    if (S.mapEdit) return mapPointerDown(event);
    if (!S.data || !S.bev) return;
    S.focus = { kind: "bev", cam: S.focus.cam };
    const [x, y] = bevPoint(event);
    const best = bevHit(x, y);
    S.sel = best ? best.uid : null;
    S.selYolo = null;
    if (best) {
      S.drag = { kind: "bev", offset: [best.x - x, best.y - y], moved: false };
      canvas.style.cursor = "grabbing";
    }
    drawAll();
  });
  canvas.addEventListener("mousemove", (event) => {
    if (S.mapPan) {
      S.mapView.panX = S.mapPan.panX + event.clientX - S.mapPan.X;
      S.mapView.panY = S.mapPan.panY + event.clientY - S.mapPan.Y;
      drawBev();
      return;
    }
    if (S.mapEdit) return mapPointerMove(event);
    S.mouse = { kind: "bev", event };
    if (S.drag?.kind !== "bev") {
      // a pointing hand over people, so it is clear where a click selects someone
      canvas.style.cursor = S.data && S.bev && bevHit(...bevPoint(event)) ? "pointer" : "crosshair";
      return;
    }
    const [x, y] = bevPoint(event);
    const p = person();
    if (!p) return;
    if (!S.drag.moved) { snapshot(); S.drag.moved = true; }
    moveLinkedPerson(p, x + S.drag.offset[0], y + S.drag.offset[1]);
    refresh();
  });
  canvas.addEventListener("mouseup", () => {
    if (S.mapPan) { S.mapPan = null; canvas.style.cursor = "crosshair"; drawBev(); return; }
    if (S.mapEdit) return mapPointerUp();
    if (S.drag?.kind === "bev" && S.drag.moved) changed();
    canvas.style.cursor = "pointer";
  });
  canvas.addEventListener("dblclick", (event) => {
    if (S.mapEdit) { event.preventDefault(); return; }
    if (!S.data) return;
    const [x, y] = bevPoint(event);
    addPerson(x, y);
  });
  canvas.addEventListener("mouseleave", () => { if (S.mouse?.kind === "bev") S.mouse = null; });
  canvas.addEventListener("wheel", (event) => {
    if (!S.siteMap) return;
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    zoomMapAt(event.deltaY < 0 ? 1.25 : 0.8,
      (event.clientX - rect.left) * dpr, (event.clientY - rect.top) * dpr);
  }, { passive: false });
  window.addEventListener("mouseup", () => {
    if (S.mapPan) { S.mapPan = null; canvas.style.cursor = "crosshair"; drawBev(); }
  });
  canvas.addEventListener("contextmenu", (event) => { if (S.siteMap) event.preventDefault(); });
}

// ------------------------------------------------------------------------------------------------ inspector

function renderInspector() {
  const box = $("inspector");
  if (!S.data) { box.innerHTML = ""; return; }
  if (S.selYolo) {
    const det = S.data.yolo[String(S.selYolo.cam)][S.selYolo.index];
    box.innerHTML = `<h3><span class="dot" style="background:#ff6b6b"></span>Box YOLO chưa khớp · C${S.selYolo.cam + 1}</h3>
      <div class="kv"><span>conf</span><span>${det[4].toFixed(2)}</span><span>box</span><span>${det.slice(0, 4).map((v) => Math.round(v)).join(", ")}</span></div>
      <div class="row"><button class="primary" id="btnAddYolo">Thêm người từ box này <kbd>A</kbd></button></div>
      <p class="muted small">Điểm chân của box được chiếu xuống mặt sàn rồi snap theo mọi view YOLO khớp.</p>`;
    $("btnAddYolo").onclick = addFromYolo;
    return;
  }
  const p = person();
  if (!p) {
    const persons = S.data.persons;
    const pending = persons.filter((q) => !q.verified && q.status !== "ok").length;
    const proposals = persons.filter((q) => !included(q)).length;
    const yoloCount = S.cams.reduce((count, cam) => count + (S.data.yolo[String(cam.index)]?.length || 0), 0);
    box.innerHTML = `<p class="muted">Chọn một người trên camera hoặc BEV. ${pending ? `<b>${pending}</b> người cần xem — bấm <kbd>Tab</kbd>.` : "Không còn người cần kiểm tra."}</p>
      <p class="small">YOLO: <b>${yoloCount} box trên ảnh</b> · ghép đa view: <b>${persons.length} vị trí</b>.</p>
      ${proposals ? '<p class="muted small">Nét tím là đề xuất chưa duyệt. Chọn đề xuất rồi bấm <kbd>Enter</kbd> để giữ làm nhãn.</p>' : ""}
      ${!persons.length && yoloCount ? `<p class="muted small">Đã có box YOLO nhưng chưa ghép được vị trí chung. Cần box phù hợp ở ít nhất ${S.params.suggest_views} camera trong vùng gán nhãn.</p>` : ""}
      <div class="kv">${Object.entries(VI_STATUS).map(([k, v]) =>
        `<span><span class="dot" style="background:${COLORS[k]};display:inline-block;width:9px;height:9px;border-radius:50%"></span> ${v}</span><span>${persons.filter((q) => q.status === k && !edited(q)).length}</span>`).join("")}
        <span><span class="dot" style="background:${EDITED};display:inline-block;width:9px;height:9px;border-radius:50%"></span> đã chỉnh</span><span>${persons.filter(edited).length}</span></div>
      ${gtSummary()}
      <p class="muted small">Ý nghĩa các ký hiệu: xem chú thích ở thanh Bird's-eye view.</p>`;
    return;
  }
  const k = S.data.persons.indexOf(p);
  const views = S.views[k] || {};
  const color = colorOf(p);
  const moved = p.auto ? Math.hypot(p.x - p.auto[0], p.y - p.auto[1]) : null;
  const rows = S.cams.map((cam) => {
    if (!cameraActive(cam.index)) return `<tr><td>C${cam.index + 1}</td><td colspan="2" class="muted">đã hết video</td></tr>`;
    const v = views[cam.index];
    const mode = boxSource(p, cam.index);
    const focus = S.focus.cam === cam.index ? " class=\"focus\"" : "";
    const manual = mode === "manual" ? '<button class="on" disabled title="box chỉnh tay ở camera này; bấm chiếu/YOLO/tắt để bỏ">tay</button>' : "";
    if (!v) {
      return manual ? `<tr${focus}><td>C${cam.index + 1}</td><td class="muted">ngoài khung hình</td><td><span class="seg">${manual}<button data-cam="${cam.index}" data-mode="off">tắt</button></span></td></tr>`
        : `<tr${focus}><td>C${cam.index + 1}</td><td colspan="2" class="muted">ngoài khung hình</td></tr>`;
    }
    const seg = ["proj", "yolo", "off"].map((m) =>
      `<button data-cam="${cam.index}" data-mode="${m}" class="${mode === m ? "on" : ""}" ${m === "yolo" && v.yolo === null ? "disabled" : ""} title="${m === "off" ? `Xóa box nhãn ở C${cam.index + 1}; BEV và camera khác giữ nguyên` : "Chọn nguồn box cho camera này"}">${{ proj: "chiếu", yolo: "YOLO", off: "tắt" }[m]}</button>`).join("");
    return `<tr${focus}><td>C${cam.index + 1}</td><td>${v.yolo !== null ? v.iou.toFixed(2) : '<span class="muted">không khớp</span>'}</td><td><span class="seg">${manual}${seg}</span></td></tr>`;
  }).join("");
  const support = Object.values(views).filter((v) => v.yolo !== null).length;
  box.innerHTML = `<h3><span class="dot" style="background:${color}"></span>${VI_STATUS[p.status] || p.status}${edited(p) ? " · đã chỉnh" : ""}${p.verified ? " · đã duyệt ✓" : ""}</h3>
    <div class="kv">
      <span>ID người</span><span id="selectedPersonUid" style="user-select:text"></span>
      <span>Frame</span><span id="selectedPersonFrame" style="user-select:text"></span>
      <span>nguồn</span><span>${ORIGIN[p.origin] || p.origin}${p.score != null ? ` · score ${(+p.score).toFixed(2)}` : ""}</span>
      <span>vị trí</span><span>x ${p.x.toFixed(3)} m · y ${p.y.toFixed(3)} m</span>
      <span>chiều cao</span><span>${(p.h || S.params.height).toFixed(2)} m <kbd>[</kbd> <kbd>]</kbd></span>
      <span>YOLO khớp</span><span>${support} / ${Object.keys(views).length} view thấy người này</span>
      ${moved != null ? `<span>đã dời</span><span>${(100 * moved).toFixed(1)} cm so với auto</span>` : ""}
      ${gtPersonRow(p)}
      ${S.project.kind && !insideArea(p) ? `<span>⚠</span><span style="color:var(--danger)">ngoài vùng label: sẽ không được export</span>` : ""}
    </div>
    <table><tr><th>Cam</th><th>IoU chiếu↔YOLO</th><th>Box ghi vào label</th></tr>${rows}</table>
    ${S.perView ? '<p class="muted small">Chỉnh riêng camera: Del xóa box nhãn của camera đang chọn. Box YOLO gốc vẫn có thể hiện nếu bật YOLO; box đó không được xuất khi camera đã tắt.</p>' : ""}
    <div class="row">
      <button class="primary" id="btnAccept">Chấp nhận <kbd>Enter</kbd></button>
      <button id="btnSnap">Snap YOLO <kbd>S</kbd></button>
      ${p.auto ? '<button id="btnReset">Về vị trí auto</button>' : ""}
      <button id="btnDelete" ${S.perView && S.focus.cam === null ? "disabled" : ""}>${S.perView ? S.focus.cam === null ? "Chọn camera để xóa box" : `Xóa box nhãn C${S.focus.cam + 1} <kbd>Del</kbd>` : "Xóa người mọi camera <kbd>Del</kbd>"}</button>
    </div>`;
  box.querySelector("#selectedPersonUid").textContent = String(p.uid);
  box.querySelector("#selectedPersonFrame").textContent = String(S.frame);
  box.querySelectorAll(".seg button").forEach((button) => {
    button.onclick = () => {
      S.focus = { kind: "cam", cam: +button.dataset.cam };
      setOverride(button.dataset.mode);
    };
  });
  $("btnAccept").onclick = accept;
  $("btnSnap").onclick = () => snap();
  $("btnDelete").onclick = removeSelected;
  if ($("btnReset")) $("btnReset").onclick = () => { snapshot(); p.x = p.auto[0]; p.y = p.auto[1]; changed(); };
}

function gtSummary() {
  const match = gtMatch();
  if (!match) return "";
  const errors = [...match.byUid.values()].map((hit) => hit.d);
  const mean = errors.length ? (100 * errors.reduce((a, b) => a + b, 0) / errors.length).toFixed(1) : "–";
  return `<div class="kv" style="margin-top:6px">
      <span>label gốc</span><span>${S.data.gt.length} người · khớp ${match.byUid.size} · lệch TB ${mean} cm</span>
      <span>thiếu</span><span${match.missed.length ? ' style="color:var(--danger)"' : ""}>${match.missed.length} (ô đỏ trên BEV)</span>
      <span>thừa</span><span${match.extra.length ? ' style="color:var(--danger)"' : ""}>${match.extra.length} (vòng đỏ)</span></div>`;
}

function gtPersonRow(p) {
  const match = gtMatch();
  if (!match) return "";
  const hit = match.byUid.get(p.uid);
  if (hit) {
    const g = S.data.gt[hit.g];
    return `<span>label gốc</span><span>lệch ${(100 * hit.d).toFixed(1)} cm (personID ${g.id}, x ${g.x.toFixed(3)} · y ${g.y.toFixed(3)})</span>`;
  }
  if (!match.labels.includes(p)) return '<span>label gốc</span><span class="muted">đề xuất chưa duyệt: chưa tính</span>';
  return '<span>label gốc</span><span style="color:var(--danger)">không có người gốc nào trong 0.5 m (thừa)</span>';
}

// ------------------------------------------------------------------------------------------------ metrics

async function refreshMetrics() {
  if (!S.project || $("metrics").classList.contains("hidden")) return;
  const body = $("metricsBody");
  if (!S.project.has_gt) {
    body.innerHTML = '<p class="muted">Không tìm thấy label gốc cho các frame của project ở thư mục này. Chọn thư mục annotations_positions rồi bấm <b>Nạp</b>.</p>';
    return;
  }
  try {
    const m = await api("/api/metrics");
    if (!m) { body.innerHTML = '<p class="muted">Chưa có frame nào xong (hoặc có label gốc).</p>'; return; }
    const row = (name, v) => `<tr><td>${name}</td><td>${v.moda ?? "–"}</td><td>${v.precision ?? "–"}</td><td>${v.recall ?? "–"}</td><td>${v.error_cm ?? "–"}</td><td>${v.box_iou ?? "–"}</td></tr>`;
    const worst = S.frames.filter((f) => f.diff && f.diff.fp + f.diff.fn > 0)
      .sort((a, b) => (b.diff.fp + b.diff.fn) - (a.diff.fp + a.diff.fn) || (b.diff.error_cm ?? 0) - (a.diff.error_cm ?? 0)).slice(0, 12);
    body.innerHTML = `<div class="pane-title" style="padding:0 0 6px;border:none">So với label gốc <span class="muted">${m.current.frames} frame, ngưỡng 0.5 m</span></div>
      <table><tr><th></th><th>MODA</th><th>Prec</th><th>Recall</th><th>cm</th><th>IoU box</th></tr>
      ${row("Model BEV", m.model)}${row("Auto label", m.auto)}${row("Sau review", m.current)}</table>
      <p class="muted small">"cm": sai số vị trí trung bình của các cặp khớp. "IoU box": box label so với box gốc trên các camera đã chọn.</p>
      ${worst.length ? `<div class="pane-title" style="padding:6px 0;border:none">Frame lệch nhiều nhất <span class="muted">bấm để mở</span></div>
      <table><tr><th>Frame</th><th>Gốc</th><th>Thừa</th><th>Thiếu</th><th>cm</th></tr>
      ${worst.map((f) => `<tr class="link" data-frame="${f.frame}"><td>${f.frame}</td><td>${f.diff.gt}</td><td>${f.diff.fp}</td><td>${f.diff.fn}</td><td>${f.diff.error_cm ?? "–"}</td></tr>`).join("")}</table>`
        : '<p class="muted small">Mọi frame đã xong đều khớp số người với label gốc.</p>'}`;
    body.querySelectorAll("tr.link").forEach((tr) => tr.onclick = () => selectFrame(+tr.dataset.frame));
  } catch (error) {
    toast(error.message, true);
  }
}

async function loadGt() {
  try {
    await flushSave();
    const project = await api("/api/gt", { method: "POST", body: { path: $("gtDir").value } });
    setProject(project);
    if (project.has_gt && !S.show.gt) { $("showGt").checked = true; $("showGt").onchange(); }
    toast(project.has_gt ? `Đã nạp label gốc: ${project.gt_dir}` : "Thư mục không có label cho các frame của project", !project.has_gt);
    refreshMetrics();
  } catch (error) {
    toast(error.message, true);
  }
}

// ------------------------------------------------------------------------------------------------ setup

let probeInfo = null;
let videoRoot = null;
let sourceDirty = false;

function sourceSettings() {
  if (videoRoot !== $("fRoot").value.trim()) return {};
  const videos = {};
  $("videoSources").querySelectorAll(".video-source").forEach((row) => {
    const files = [...row.querySelectorAll(".video-file")].map((select) => select.value);
    videos[row.dataset.folder] = { ...JSON.parse(row.dataset.sync || "{}"), file: files[0], ...(files.length > 1 ? { files } : {}),
      offset_s: Number(row.querySelector(".video-offset").value) };
  });
  return { videos, timeline_fps: Number($("fTimelineFps").value) };
}

function invalidateSources() {
  sourceDirty = true;
  $("btnCreate").disabled = true;
  $("videoPreviews").replaceChildren();
  $("videoNote").textContent = "Bấm Đọc dataset + calibration để cập nhật khoảng frame với lựa chọn mới.";
}

function invalidateDataset() {
  if (probeInfo && $("fCalib").value === probeInfo.calib) $("fCalib").value = "";
  sourceDirty = true;
  $("btnCreate").disabled = true;
  $("probeResult").classList.add("hidden");
}

function showVideoSources(info) {
  const video = info.source?.mode === "videos";
  $("videoSetup").classList.toggle("hidden", !video);
  $("videoSources").replaceChildren();
  $("videoPreviews").replaceChildren();
  videoRoot = video ? $("fRoot").value.trim() : null;
  sourceDirty = false;
  if (!video) return;
  $("syncProfile").classList.toggle("hidden", !info.sync_profile);
  if (info.sync_profile) {
    const lags = Object.values(info.sync_profile.independent_validation || {})
      .map((result) => result.max_absolute_lag_ms).filter((value) => Number.isFinite(value));
    $("syncSummary").textContent = info.sync_profile.summary + (lags.length ?
      ` Kiểm tra âm thanh độc lập còn sai lệch tối đa ${Math.ceil(Math.max(...lags))} ms; chưa bảo đảm khớp chính xác từng frame.` : "");
    $("btnApplySync").onclick = () => runProbe(info.sync_profile.settings);
  }
  $("fTimelineFps").value = info.source.timeline_fps;
  $("fTimelineFps").oninput = invalidateSources;
  for (const cam of info.source.cameras) {
    const row = document.createElement("div");
    row.className = "video-source";
    row.dataset.folder = cam.folder;
    row.dataset.sync = JSON.stringify(info.source.videos[cam.folder].read_mode === "timestamp" ? {
      rate: cam.rate, end_s: cam.end_s, read_mode: cam.read_mode,
    } : {});
    const name = document.createElement("strong");
    name.textContent = cam.folder;
    const modeLabel = document.createElement("label");
    modeLabel.textContent = "Cách dùng video";
    const mode = document.createElement("select");
    mode.className = "video-mode";
    mode.add(new Option("Giữ một video", "single"));
    if (cam.choices.length > 1) mode.add(new Option("Ghép nhiều video theo thứ tự", "concat"));
    let files = cam.clips?.map((clip) => clip.file) || [cam.file];
    mode.value = files.length > 1 ? "concat" : "single";
    modeLabel.append(mode);
    const offsetLabel = document.createElement("label");
    offsetLabel.textContent = "Mốc đầu (giây)";
    const offset = document.createElement("input");
    offset.className = "video-offset";
    offset.type = "number"; offset.min = "0"; offset.step = "any"; offset.value = cam.offset_s;
    offset.oninput = invalidateSources;
    offsetLabel.append(offset);
    const list = document.createElement("div");
    list.className = "video-clips";
    const tools = document.createElement("div");
    tools.className = "video-clip-actions";
    const add = document.createElement("button");
    add.type = "button"; add.textContent = "+ Thêm clip"; add.className = "video-add";
    const previewJoin = document.createElement("button");
    previewJoin.type = "button"; previewJoin.textContent = "Xem điểm nối"; previewJoin.className = "video-preview-join";
    previewJoin.onclick = () => previewSources(cam.folder);
    tools.append(add, previewJoin);
    const timeline = document.createElement("div");
    timeline.className = "muted small video-timeline";

    function renderClips() {
      if (!files.every((file, i) => file === cam.clips[i]?.file) || files.length !== cam.clips.length) row.dataset.sync = "{}";
      list.replaceChildren();
      files.forEach((file, index) => {
        const clipRow = document.createElement("div");
        clipRow.className = "video-clip";
        const number = document.createElement("span");
        number.textContent = `${index + 1}.`;
        const select = document.createElement("select");
        select.className = "video-file";
        for (const choice of cam.choices) {
          if (choice === file || !files.includes(choice)) select.add(new Option(choice, choice));
        }
        select.value = file;
        select.onchange = () => { files[index] = select.value; renderClips(); invalidateSources(); };
        clipRow.append(number, select);
        if (mode.value === "concat") {
          for (const [label, delta] of [["↑", -1], ["↓", 1]]) {
            const button = document.createElement("button");
            button.type = "button"; button.textContent = label;
            button.title = delta < 0 ? "Chuyển clip lên trước" : "Chuyển clip xuống sau";
            button.disabled = index + delta < 0 || index + delta >= files.length;
            button.onclick = () => {
              [files[index], files[index + delta]] = [files[index + delta], files[index]];
              renderClips(); invalidateSources();
            };
            clipRow.append(button);
          }
          const remove = document.createElement("button");
          remove.type = "button"; remove.textContent = "Bỏ"; remove.disabled = files.length <= 2;
          remove.onclick = () => { files.splice(index, 1); renderClips(); invalidateSources(); };
          clipRow.append(remove);
        }
        list.append(clipRow);
      });
      tools.classList.toggle("hidden", mode.value !== "concat");
      add.disabled = files.length >= cam.choices.length;
      const sameClips = files.length === cam.clips.length && files.every((file, i) => file === cam.clips[i].file);
      timeline.textContent = files.length > 1 ? `${files.join(" → ")} · nối ngay khi clip trước kết thúc, không chèn khoảng trống.` :
        `${files[0]} · hết video sẽ hiển thị nền đen.`;
      if (sameClips) {
        timeline.textContent += ` Tổng ${cam.duration_s.toFixed(3)} giây.`;
        if (files.length > 1) timeline.textContent += ` Điểm nối: ${cam.clips.slice(1).map((clip) => clip.start_s.toFixed(3) + " s").join(", ")}.`;
      }
    }
    add.onclick = () => {
      const next = cam.choices.find((file) => !files.includes(file));
      if (next) { files.push(next); renderClips(); invalidateSources(); }
    };
    mode.onchange = () => {
      if (mode.value === "single") files = files.slice(0, 1);
      else if (files.length === 1) files.push(cam.choices.find((file) => !files.includes(file)));
      renderClips(); invalidateSources();
    };
    row.append(name, modeLabel, offsetLabel, list, tools, timeline);
    $("videoSources").append(row);
    renderClips();
  }
  const corrected = info.source.cameras.every((cam) => cam.read_mode === "timestamp");
  $("videoNote").textContent = corrected ?
    `Đã áp dụng căn thời gian và bù trôi đồng hồ; đọc timestamp video. Mỗi camera xử lý từ mốc đầu đến mốc cuối đã lưu, lấy ${info.source.timeline_fps} frame/giây. Tạo project mới để label trên các mốc này.` :
    `Frame n tương ứng n / ${info.source.timeline_fps} giây sau mốc đầu của từng camera (tính trên chuỗi clip đã ghép). Timeline chạy đến camera dài nhất; camera hết video hiển thị nền đen và không tạo box.`;
}

async function previewSources(joins = null) {
  const button = $("btnPreviewSources");
  button.disabled = true;
  $("setupError").textContent = "";
  try {
    const result = await api("/api/source-preview", { method: "POST", body: {
      root: $("fRoot").value.trim(), ...sourceSettings(), joins,
    } });
    $("videoPreviews").replaceChildren();
    for (const preview of result.images) {
      const figure = document.createElement("figure");
      const img = document.createElement("img");
      img.src = preview.image; img.alt = preview.folder;
      const caption = document.createElement("figcaption");
      caption.textContent = preview.caption || `${preview.folder} · ${preview.offset_s} s`;
      figure.append(img, caption);
      $("videoPreviews").append(figure);
    }
  } catch (error) {
    $("setupError").textContent = error.message;
  } finally { button.disabled = false; }
}

function openSetup(defaultRoot) {
  openSetup.root = defaultRoot || openSetup.root;
  if (!$("fRoot").value) $("fRoot").value = S.project?.config.root || openSetup.root || "";
  $("setupError").textContent = "";
  $("setupDialog").showModal();
}

async function runProbe(settings = null) {
  $("setupError").textContent = "";
  sourceDirty = true;
  $("btnCreate").disabled = true;
  if (!$("fRoot").value.trim()) { $("setupError").textContent = "Chọn thư mục dataset trước."; return; }
  try {
    probeInfo = await api("/api/probe", { method: "POST", body: {
      root: $("fRoot").value.trim(), calib: $("fCalib").value.trim(), ...(settings || sourceSettings()),
    } });
  } catch (error) {
    if (videoRoot !== $("fRoot").value.trim()) $("probeResult").classList.add("hidden");
    $("setupError").textContent = error.message;
    return;
  }
  const info = probeInfo;
  showVideoSources(info);
  $("probeResult").classList.remove("hidden");
  $("fCalib").value = info.calib;
  const c = info.calibration;
  const pixelMode = c.source_pixels === "undistorted" ? " · <b>ảnh đã khử méo, intrinsic_zero, không xử lý lần 2</b>" :
    c.source_pixels === "raw" ? " · <b>ảnh raw, intrinsic_original, khử méo 1 lần</b>" : "";
  $("probeSummary").innerHTML = `Calibration <b>${c.layout}</b> (${c.unit}${c.distortion ? ", có méo: ảnh sẽ được khử méo" : ""}) · ` +
    `${info.cameras.length} camera · ${info.frames.count} frame (${info.frames.first}…${info.frames.last}, bước ${info.frames.step})` +
    pixelMode +
    (c.bev_grid ? ` · <b>rig BEV ${c.bev_grid.x}×${c.bev_grid.y}</b>` : "") +
    (info.ground_truth ? " · <b>có GT</b> để đối chiếu" : " · không có GT");
  $("camChecks").innerHTML = info.cameras.map((cam) =>
    `<label><input type="checkbox" value="${cam.index}" checked> C${cam.index + 1} <span class="muted">${cam.name} · ${cam.width}×${cam.height}</span></label>`).join("");
  $("camChecks").querySelectorAll("input").forEach((input) => input.onchange = updateCamCount);
  updateCamCount();
  const isWildtrack = info.kind === "wildtrack" && info.frames.last >= 1995;
  $("fStart").value = isWildtrack ? 1800 : info.frames.first;
  $("fEnd").value = info.frames.last;
  updateFrameInputs();
  const current = S.project?.config.root === info.root ? S.project.config.model_id : undefined;
  fillModelOptions($("fModel"), info.models, current === undefined ? info.models.find((m) => m.best)?.id : current);
  $("fYolo").innerHTML = info.yolo.map((y) => `<option value="${y.path}">${y.name}</option>`).join("") || '<option value="">yolo26s (tải tự động)</option>';
  $("fOut").value = info.out_dir;
  $("fGt").value = info.gt_dir || "";
  $("fGt").disabled = !info.gt_supported;
  $("fGt").placeholder = info.gt_supported ? "" : "chỉ so được trên lưới Wildtrack / MultiviewX";
  updateSetupModelChoice();
}

function updateSetupModelChoice() {
  const selected = $("fModel").value;
  const model = probeInfo?.models.find((m) => m.id === selected);
  const models = probeInfo?.models || [];
  $("setupModelStatus").textContent = `${models.length} model đã đăng ký · ${models.filter((m) => m.compatible).length} tương thích.\n` +
    modelDescription(model, selected);
  $("setupModelStatus").style.whiteSpace = "pre-wrap";
  updateCamCount();
}

function updateFrameInputs() {
  const strategy = $("fStrategy").value;
  $("fCount").disabled = strategy === "all" || strategy === "dataset";
  $("fStart").disabled = $("fEnd").disabled = strategy === "dataset";
  if (!probeInfo) return;
  const frames = probeInfo.frames;
  const wildtrackTrain = probeInfo.kind === "wildtrack" && frames.last >= 1995 &&
    (strategy === "dataset" || +$("fStart").value < 1800);
  $("frameNote").textContent = (strategy === "dataset" ? `Lấy toàn bộ ${frames.count} frame (${frames.first}…${frames.last}). ` : "") +
    (wildtrackTrain ? "Checkpoint đã được train trên frame 0–1795 của Wildtrack: label lại chúng thì kết quả (và MODA so với GT) bị lạc quan." :
     probeInfo.kind === "wildtrack" && frames.last >= 1995 ? "Mặc định là đoạn test của Wildtrack (frame ≥ 1800)." : "");
}

function updateCamCount() {
  const boxes = [...$("camChecks").querySelectorAll("input")];
  const on = boxes.filter((b) => b.checked).length;
  $("camCount").textContent = `(${on}/${boxes.length})`;
  $("camNote").textContent = on < boxes.length ?
    `Model BEV vẫn nhận đủ ${boxes.length} camera (trọng số cố định số camera); YOLO, box và label chỉ làm trên ${on} camera đã chọn.` : "";
  const selected = $("fModel").value;
  $("btnCreate").disabled = !on || sourceDirty || (!!selected && !probeInfo?.models.find((m) => m.id === selected)?.compatible);
}

async function createProject() {
  $("setupError").textContent = "";
  if (sourceDirty) { $("setupError").textContent = "Đọc lại dataset sau khi đổi nguồn video."; return; }
  const cameras = [...$("camChecks").querySelectorAll("input")].filter((b) => b.checked).map((b) => +b.value);
  const number = (id) => ($(id).value === "" ? null : +$(id).value);
  const body = {
    root: $("fRoot").value.trim(), calib: $("fCalib").value.trim(), cameras,
    count: number("fCount"), strategy: $("fStrategy").value, start: number("fStart"), end: number("fEnd"),
    model_id: $("fModel").value, yolo_weights: $("fYolo").value, out_dir: $("fOut").value.trim(),
    gt_dir: $("fGt").disabled ? "" : $("fGt").value.trim(),
    overwrite: $("fOverwrite").checked,
    ...sourceSettings(),
    params: { yolo_conf: number("fConf"), radius: number("fRadius"), height: number("fHeight"),
              box_mode: $("fBoxMode").value, suggest_views: number("fViews") },
  };
  $("btnCreate").disabled = true;
  try {
    const project = await api("/api/create", { method: "POST", body });
    $("setupDialog").close();
    S.frame = null;
    setProject(project);
  } catch (error) {
    $("setupError").textContent = error.message;
  } finally {
    $("btnCreate").disabled = false;
  }
}

async function openProject(path) {
  try {
    const project = await api("/api/open", { method: "POST", body: { path } });
    $("setupDialog").close();
    S.frame = null;
    setProject(project);
  } catch (error) {
    $("setupError").textContent = error.message;
  }
}

// ------------------------------------------------------------------------------------------------ folder picker

let pickerResolve = null;

function pickFolder(start, title = "Chọn thư mục") {
  $("pickerTitle").textContent = title;
  $("pickerDialog").showModal();
  browse(start || "");
  return new Promise((resolve) => { pickerResolve = resolve; });
}

async function browse(path) {
  try {
    const listing = await api(`/api/fs?files=1&path=${encodeURIComponent(path)}`);
    $("pickerPath").value = listing.path;
    $("pickerUp").onclick = () => browse(listing.parent ?? "");
    const list = $("pickerList");
    list.innerHTML = "";
    for (const entry of listing.entries) {
      const li = document.createElement("li");
      li.className = entry.dir ? "" : "file";
      li.textContent = (entry.dir ? "📁 " : "📄 ") + entry.name;
      if (entry.dir) li.onclick = () => browse(entry.path);
      list.appendChild(li);
    }
  } catch (error) {
    toast(error.message, true);
  }
}

function closePicker(value) {
  $("pickerDialog").close();
  if (pickerResolve) pickerResolve(value);
  pickerResolve = null;
}

// ------------------------------------------------------------------------------------------------ export

async function doExport() {
  $("exportError").textContent = "";
  $("btnDoExport").disabled = true;
  try {
    await flushSave();
    const result = await api("/api/export", { method: "POST", body: {
      path: $("eOut").value.trim(), copy_images: $("eCopy").checked, only_reviewed: $("eReviewed").checked } });
    const gt = result.vs_ground_truth;
    $("exportResult").classList.remove("hidden");
    $("exportResult").textContent = `Đã ghi ${result.frames.length} frame, ${result.people} người (${result.reviewed} frame đã review)\n` +
      `→ ${result.path}\n` + (result.outside_grid ? `${result.outside_grid} người ngoài lưới BEV bị bỏ.\n` : "") +
      (gt ? `So với GT: MODA ${gt.moda} · precision ${gt.precision} · recall ${gt.recall} · sai số ${gt.error_cm} cm · IoU box ${gt.box_iou}` : "");
  } catch (error) {
    $("exportError").textContent = error.message;
  } finally {
    $("btnDoExport").disabled = false;
  }
}

// ------------------------------------------------------------------------------------------------ keyboard

function keyMove(event) {
  const p = person();
  if (!p) return false;
  const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  const [du, dv] = arrows[event.key];
  const factor = event.shiftKey ? 5 : 1;
  if (S.focus.kind === "cam" && S.focus.cam !== null && S.perView) {
    // per-camera editing: shift this camera's box only
    const t = S.tiles.find((tile) => tile.cam.index === S.focus.cam);
    const box = t && selectedTileBox(t);
    if (!box) return true;
    const pixelDu = t.flipX ? -du : du, pixelDv = t.flipY ? -dv : dv;
    if (!S.moving) { snapshot(); S.moving = true; }
    setManualBox(p, t.cam, [box[0] + 2 * factor * pixelDu, box[1] + 2 * factor * pixelDv,
                            box[2] + 2 * factor * pixelDu, box[3] + 2 * factor * pixelDv]);
    changed();
    return true;
  }
  if (S.focus.kind === "cam" && S.focus.cam !== null) {
    // move the foot pixel in the focused camera, then put the person where that pixel meets the ground
    const cam = S.cams.find((c) => c.index === S.focus.cam);
    const t = S.tiles.find((tile) => tile.cam.index === S.focus.cam);
    const pixelDu = t?.flipX ? -du : du, pixelDv = t?.flipY ? -dv : dv;
    const foot = projectPoint(cam, [p.x, p.y, 0]);
    const g = foot && ground(cam, foot[0] + 2 * factor * pixelDu, foot[1] + 2 * factor * pixelDv);
    if (g) moveSelected(g[0] - p.x, g[1] - p.y);
    return true;
  }
  const [dx, dy] = S.bev.step(du, dv);
  moveSelected(0.025 * factor * dx, 0.025 * factor * dy);
  return true;
}

function mouseGround() {
  if (!S.mouse) return null;
  if (S.mouse.kind === "bev") return bevPoint(S.mouse.event);
  const pt = tilePoint(S.mouse.t, S.mouse.event);
  return ground(S.mouse.t.cam, pt.img[0], pt.img[1]);
}

document.addEventListener("keydown", (event) => {
  if (event.target.closest("input:not([type=checkbox]), select, textarea, dialog[open]")) return;
  if (S.mapEdit) {
    if (event.key.startsWith("Arrow") && nudgeMapSelection(event)) { event.preventDefault(); return; }
    if (event.key === "Escape") { S.mapDraft = []; renderMapDraft(); drawBev(); event.preventDefault(); }
    return;
  }
  if (!S.data && ![",", ".", "PageUp", "PageDown"].includes(event.key)) return;
  const key = event.key;
  if (key.startsWith("Arrow")) {
    if (keyMove(event)) event.preventDefault();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === "z") { undo(); event.preventDefault(); return; }
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const actions = {
    j: () => setCrops(!CROPS.open), J: () => setCrops(!CROPS.open),
    Tab: () => selectNext(event.shiftKey ? -1 : 1),
    Enter: accept,
    Delete: removeSelected,
    Backspace: removeSelected,
    Escape: () => { S.sel = null; S.selYolo = null; drawAll(); },
    s: () => snap(), S: () => snap(),
    p: () => setOverride("proj"), y: () => setOverride("yolo"), o: () => setOverride("off"),
    "[": () => changeHeight(-0.05), "]": () => changeHeight(0.05),
    n: () => { const g = mouseGround(); if (g) addPerson(g[0], g[1]); else toast("Đưa chuột lên ảnh hoặc BEV rồi bấm N"); },
    a: addFromYolo,
    t: addBoxAtCursor,
    r: () => { setPerView(!S.perView); toast(S.perView ? "Chỉnh riêng từng camera: kéo/resize chỉ đổi box của camera đó" : "Chỉnh liên kết: dời người thì box mọi camera đi theo"); },
    f: markReviewed,
    ",": () => stepFrame(-1), ".": () => stepFrame(1), PageUp: () => stepFrame(-1), PageDown: () => stepFrame(1),
    g: () => { $("showGt").checked = !$("showGt").checked; $("showGt").onchange(); },
    b: () => { $("showYolo").checked = !$("showYolo").checked; $("showYolo").onchange(); },
    z: () => { $("followZoom").checked = !$("followZoom").checked; $("followZoom").onchange(); },
    l: () => { $("showGrid").checked = !$("showGrid").checked; $("showGrid").onchange(); },
  };
  const action = actions[key] || actions[key.toLowerCase()];
  if (action) { action(); event.preventDefault(); }
});

document.addEventListener("keyup", (event) => { if (event.key.startsWith("Arrow")) S.moving = false; });

// ------------------------------------------------------------------------------------------------ wiring

function wire() {
  wireCrops();
  $("fModel").onchange = updateSetupModelChoice;
  $("activeModel").onchange = updateActiveModelChoice;
  $("btnApplyModel").onclick = applyProjectModel;
  $("btnModelRefresh").onclick = refreshModelCatalog;
  $("btnImportFromSetup").onclick = () => $("btnImportLabels").click();
  $("btnImportLabels").onclick = () => {
    $("iRoot").value ||= S.project?.config.root || $("fRoot").value || openSetup.root || "";
    $("importError").textContent = "";
    $("importDialog").showModal();
  };
  $("btnImportRun").onclick = async () => {
    const button = $("btnImportRun");
    button.disabled = true;
    $("importError").textContent = "";
    try {
      await flushSave();
      const project = await api("/api/import-annotations", {method: "POST", body: {
        root: $("iRoot").value.trim(), calib: $("iCalib").value.trim(),
        labels_dir: $("iLabels").value.trim(), out_dir: $("iOut").value.trim()
      }});
      $("importDialog").close();
      $("setupDialog").close();
      S.frame = null;
      setProject(project);
      toast("Đã nhập nhãn. Tinh chỉnh rồi nhấn F để chốt từng frame.");
    } catch (error) { $("importError").textContent = error.message; }
    finally { button.disabled = false; }
  };
  $("btnSetup").onclick = () => openSetup();
  $("btnProbe").onclick = () => runProbe();
  $("fRoot").oninput = invalidateDataset;
  $("btnPreviewSources").onclick = () => previewSources();
  $("btnCreate").onclick = createProject;
  $("camAll").onclick = () => { $("camChecks").querySelectorAll("input").forEach((b) => b.checked = true); updateCamCount(); };
  $("camNone").onclick = () => { $("camChecks").querySelectorAll("input").forEach((b) => b.checked = false); updateCamCount(); };
  $("fStrategy").onchange = updateFrameInputs;
  $("fStart").oninput = updateFrameInputs;
  $("btnOpenProject").onclick = async () => {
    const path = await pickFolder($("fOut").value || "", "Chọn thư mục project (có label_project.json)");
    if (path) openProject(path);
  };
  document.querySelectorAll("[data-browse]").forEach((button) => {
    button.onclick = async () => {
      const target = $(button.dataset.browse);
      const path = await pickFolder(target.value);
      if (path) {
        target.value = path;
        if (target.id === "fRoot") invalidateDataset();
      }
    };
  });
  $("pickerGo").onclick = () => browse($("pickerPath").value);
  $("pickerPath").onkeydown = (event) => { if (event.key === "Enter") { event.preventDefault(); browse($("pickerPath").value); } };
  $("pickerCancel").onclick = () => closePicker(null);
  $("pickerUse").onclick = () => closePicker($("pickerPath").value);
  $("btnPrev").onclick = () => stepFrame(-1);
  $("btnNext").onclick = () => stepFrame(1);
  $("btnReviewed").onclick = markReviewed;
  $("btnRelabel").onclick = async () => {
    if (S.frame === null || !confirm("Chạy lại model + YOLO cho frame này? Các chỉnh sửa của frame sẽ bị thay.")) return;
    try {
      const job = await api(`/api/frame/${S.frame}/relabel`, { method: "POST" });
      S.frames.find((f) => f.frame === S.frame).status = "pending";
      S.data = null;
      drawAll();
      pollJob(job);
    } catch (error) {
      toast(error.message, true);
    }
  };
  // toggles must not keep the keyboard focus (arrow keys move people)
  document.querySelectorAll(".frame-bar input, .frame-bar select").forEach((el) => el.addEventListener("change", () => el.blur()));
  $("showYolo").onchange = () => { S.show.yolo = $("showYolo").checked; drawTiles(); };
  $("showProj").onchange = () => { S.show.proj = $("showProj").checked; drawTiles(); };
  $("showGt").onchange = () => { S.show.gt = $("showGt").checked; drawTiles(); drawBev(); };
  $("followZoom").onchange = () => { S.show.follow = $("followZoom").checked; drawTiles(); };
  $("perView").onchange = () => setPerView($("perView").checked);
  try { setPerView(localStorage.getItem("mvlabel.perView") === "1"); } catch (error) { setPerView(false); }
  $("showGrid").onchange = () => {
    S.show.grid = $("showGrid").checked;
    $("legendGrid").classList.toggle("hidden", !S.show.grid);
    try { localStorage.setItem("mvlabel.grid", S.show.grid ? "1" : "0"); } catch (error) { /* private window */ }
    drawTiles();
    drawBev();
  };
  try { $("showGrid").checked = localStorage.getItem("mvlabel.grid") === "1"; } catch (error) { /* private window */ }
  $("showGrid").onchange();
  $("boxMode").onchange = async () => {
    S.params.box_mode = $("boxMode").value;
    try { await api("/api/params", { method: "POST", body: { box_mode: S.params.box_mode } }); } catch (error) { toast(error.message, true); }
    refresh();
    refreshMetrics();
  };
  $("btnGtLoad").onclick = loadGt;
  $("btnMetrics").onclick = () => {
    $("metrics").classList.toggle("hidden");
    refreshMetrics();
  };
  $("btnExport").onclick = () => {
    $("eOut").value = $("eOut").value || `${S.project.config.out_dir}\\export`;
    $("exportResult").classList.add("hidden");
    $("exportError").textContent = "";
    $("exportDialog").showModal();
  };
  $("btnDoExport").onclick = doExport;
  attachBev();
  wireSiteMap();
  window.addEventListener("resize", () => requestAnimationFrame(() => { layoutTiles(); drawAll(); }));
  setInterval(() => { if (!$("metrics").classList.contains("hidden") && !S.saveTimer) refreshMetrics(); }, 15000);
}

wire();
loadState().catch((error) => toast(error.message, true));
