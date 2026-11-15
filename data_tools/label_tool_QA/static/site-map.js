"use strict";
// WaveTrack floor plan is a visual reference. The model ROI and labels stay in their calibration coordinates.

const MAP_SIGHT_COLOR = "#357f9b";
const MAP_CAMERA_COLORS = ["#1479b2", "#e87d1e", "#258b45", "#cf3445", "#8058a8"];

function setMapProject(project) {
  S.siteMap = project.kind === "wavetrack" ? structuredClone(project.site_map) : null;
  S.mapEdit = false;
  S.mapFocus = false;
  S.mapTool = "select";
  S.mapShowCoverage = false;
  S.mapCoverageCamera = "all";
  S.mapSightMode = "fan";
  S.mapFanCache = null;
  $("mapSightMode").value = "fan";
  S.mapProjectCameras = true;
  S.mapProjectionMode = "surface";
  S.mapOnlySelected = false;
  S.show.obstaclesOnly = false;
  S.mapDraft = [];
  S.mapSelected = S.siteMap?.objects[0]?.id || null;
  S.mapSelectedIds = S.mapSelected ? [S.mapSelected] : [];
  S.mapActivePoint = null;
  clearTimeout(S.mapNudgeTimer);
  S.mapNudgeTimer = null;
  S.mapCoverageCache = null;
  S.mapMouse = null;
  S.mapDragging = null;
  S.mapSaveToken = 0;
  S.mapSaveQueue = Promise.resolve();
  S.mapView = { zoom: 1, panX: 0, panY: 0 };
  S.mapPan = null;
  $("mapControls").classList.toggle("hidden", !S.siteMap);
  $("btnSiteMap").classList.toggle("hidden", !S.siteMap);
  $("bevHeading").textContent = S.siteMap ? "Sơ đồ hiện trường · WaveTrack" : "Bird's-eye view";
  $("mapTool").value = "select";
  $("mapFlipHorizontal").checked = !!S.siteMap?.flip_horizontal;
  $("mapFlipVertical").checked = !!S.siteMap?.flip_vertical;
  $("mapProjectCameras").checked = true;
  $("mapProjectionMode").value = "surface";
  $("mapOnlySelected").checked = false;
  $("mapSightToggle").checked = false;
  $("mapShowCoverage").checked = false;
  const coverageCamera = $("mapCoverageCamera");
  coverageCamera.replaceChildren(new Option("Tất cả", "all"),
    ...project.all_cameras.map((cam) => new Option(`C${cam.index + 1} · ${cam.name}`, String(cam.index))));
  coverageCamera.value = S.mapCoverageCamera;
  coverageCamera.disabled = false;
  $("mapHidePeople").checked = false;
  renderMapFlipControls();
  renderMapZoom();
  updateMapLayout();
  renderMapEditor();
}

function renderMapFlipControls() {
  if (!S.siteMap) return;
  for (const [id, active] of [["btnBevFlipHorizontal", !!S.siteMap.flip_horizontal],
                             ["btnBevFlipVertical", !!S.siteMap.flip_vertical]]) {
    const button = $(id);
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }
  $("btnBevFlipReset").disabled = !S.siteMap.flip_horizontal && !S.siteMap.flip_vertical;
  $("mapFlipHorizontal").checked = !!S.siteMap.flip_horizontal;
  $("mapFlipVertical").checked = !!S.siteMap.flip_vertical;
}

function renderMapZoom() {
  if (!S.siteMap) return;
  $("btnMapZoomReset").textContent = `${Math.round(S.mapView.zoom * 100)}%`;
  $("btnMapZoomOut").disabled = S.mapView.zoom <= 1.001;
  $("btnMapZoomIn").disabled = S.mapView.zoom >= 12;
}

function zoomMapAt(factor, X = null, Y = null) {
  if (!S.siteMap || !S.bev) return;
  const dpr = window.devicePixelRatio || 1;
  const anchorX = X ?? S.bev.W / 2, anchorY = Y ?? S.bev.H / 2;
  const world = S.bev.from(anchorX, anchorY);
  const zoom = Math.max(1, Math.min(12, S.mapView.zoom * factor));
  if (Math.abs(zoom - S.mapView.zoom) < 1e-5) return;
  S.mapView.zoom = zoom;
  if (zoom <= 1.001) {
    S.mapView.panX = 0; S.mapView.panY = 0;
  } else {
    const point = bevMapping().to(...world);
    S.mapView.panX += (anchorX - point[0]) / dpr;
    S.mapView.panY += (anchorY - point[1]) / dpr;
  }
  renderMapZoom(); drawBev();
}

function updateMapLayout() {
  const center = document.querySelector(".center");
  document.querySelector(".layout").classList.toggle("map-plan-focus", !!S.siteMap && S.mapFocus);
  center.classList.toggle("map-wave", !!S.siteMap);
  center.classList.toggle("map-edit", S.mapEdit);
  center.classList.toggle("map-focus", S.mapFocus);
  $("siteMapEditor").classList.toggle("hidden", !S.mapEdit);
  $("btnMapExpand").classList.toggle("active", S.mapFocus);
  $("btnMapEdit").classList.toggle("active", S.mapEdit);
  $("btnSiteMap").classList.toggle("active", S.mapEdit && S.mapFocus);
  $("btnSiteMap").textContent = S.mapEdit && S.mapFocus ? "Đóng sơ đồ" : "✎ Vẽ hiện trường";
  $("btnMapExpand").textContent = S.mapFocus ? "Hiện lại camera" : "Mở bản đồ lớn";
  $("btnMapEdit").textContent = S.mapEdit ? "Đóng chỉnh sửa" : "Vẽ hiện trường";
  $("btnMapUseCamera").disabled = !S.mapFocus;
  $("btnMapUseCamera").textContent = S.mapFocus ? "Chấm trên ảnh camera → map" : "Ảnh camera đang mở · click chân vật thể";
  requestAnimationFrame(() => { layoutTiles(); drawAll(); });
}

function mapObject() {
  return S.siteMap?.objects.find((item) => item.id === S.mapSelected) || null;
}

function reconcileMapSelection() {
  const valid = new Set(S.siteMap?.objects.map((item) => item.id) || []);
  S.mapSelectedIds = S.mapSelectedIds.filter((id) => valid.has(id));
  S.mapSelected = S.mapSelectedIds.at(-1) || null;
}

function selectMapObject(id, additive = false) {
  S.mapActivePoint = null;
  if (!additive) S.mapSelectedIds = id ? [id] : [];
  else if (id) {
    const index = S.mapSelectedIds.indexOf(id);
    if (index < 0) S.mapSelectedIds.push(id);
    else S.mapSelectedIds.splice(index, 1);
  }
  S.mapSelected = S.mapSelectedIds.at(-1) || null;
  renderMapEditor(); drawBev(); drawTiles();
}

function mapCopy() { return structuredClone(S.siteMap); }

// A shared corner is one physical junction, even when two walls store it.
// Resolve every target before moving so a multi-selection never moves it twice.
function applyMapPointMoves(map, moves) {
  const targets = new Map();
  for (const { id, index, point } of moves) {
    const item = map.objects.find((o) => o.id === id);
    if (!item) continue;
    if (item.kind === "pillar") { targets.set(item.center, point); continue; }
    const old = item.points[index];
    const matching = (p) => Math.hypot(p[0] - old[0], p[1] - old[1]) < 0.00015;
    for (const wall of map.objects.filter((o) => o.kind === "wall"))
      for (const p of wall.points) if (matching(p)) targets.set(p, point);
    for (const region of map.floor_regions || [])
      for (const p of region.points) if (matching(p)) targets.set(p, point);
  }
  for (const [target, point] of targets) {
    target[0] = +point[0].toFixed(4); target[1] = +point[1].toFixed(4);
  }
}

function insideFloor(point, regions = S.siteMap?.floor_regions) {
  if (!regions?.length) return true;
  return regions.some(({ points }) => {
    let inside = false;
    for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
      const a = points[i], b = points[j];
      if (pointSegmentDistance(point, a, b) < 1e-7) return true;
      if ((a[1] > point[1]) !== (b[1] > point[1]) &&
          point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
    }
    return inside;
  });
}

function mapFloorPath(ctx, m) {
  ctx.beginPath();
  for (const region of S.siteMap.floor_regions || []) {
    region.points.forEach((p, i) => { if (i) ctx.lineTo(...m.to(...p)); else ctx.moveTo(...m.to(...p)); });
    ctx.closePath();
  }
}

function mapWallSegments(map = S.siteMap) {
  return map.objects.filter((o) => o.kind === "wall" && o.occludes !== false).flatMap((o) =>
    o.points.slice(1).map((b, i) => ({ id: o.id, a: o.points[i], b, height: o.height, width: o.width })));
}

function saveSiteMap(candidate) {
  clearTimeout(S.mapNudgeTimer);
  S.mapNudgeTimer = null;
  const value = structuredClone(candidate);
  const token = ++S.mapSaveToken;
  const currentProject = S.project;
  S.siteMap = value;
  reconcileMapSelection();
  $("mapSaveStatus").textContent = "Đang lưu…";
  renderMapEditor(); drawBev(); drawTiles();
  S.mapSaveQueue = S.mapSaveQueue.then(async () => {
    if (S.project !== currentProject) return;
    try {
      const clean = await api("/api/site-map", { method: "PUT", body: value });
      currentProject.site_map = clean;
      if (token === S.mapSaveToken) {
        S.siteMap = clean;
        reconcileMapSelection();
        $("mapSaveStatus").textContent = "Đã lưu · chung rig";
        renderMapEditor(); drawBev(); drawTiles();
      }
    } catch (error) {
      if (token === S.mapSaveToken) {
        S.siteMap = structuredClone(currentProject.site_map);
        reconcileMapSelection();
        $("mapSaveStatus").textContent = "Chưa lưu";
        renderMapEditor(); drawBev(); drawTiles();
      }
      toast(error.message, true);
    }
  });
  return S.mapSaveQueue;
}

function mapId() {
  return `m${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function insideMap([x, y]) {
  const b = S.siteMap.bounds;
  return b[0] <= x && x <= b[1] && b[2] <= y && y <= b[3];
}

function mapLength(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }

function wallDeviation(item) {
  if (item.kind !== "wall" || item.points.length < 3) return 0;
  const first = item.points[0], last = item.points.at(-1);
  return Math.max(...item.points.slice(1, -1).map((p) => pointSegmentDistance(p, first, last)));
}

function addMapPoint(point) {
  if (!S.mapEdit || !S.siteMap || S.mapTool === "select") return;
  const p = point.map((value) => Math.round(value * 10000) / 10000);
  if (!insideMap(p)) return toast("Điểm nằm ngoài khung nhìn; hãy mở rộng khung bản đồ", true);
  if (S.mapTool === "measure" && S.mapDraft.length === 2) S.mapDraft = [];
  S.mapDraft.push(p);
  if (S.mapTool === "pillar_radius" && S.mapDraft.length === 2) {
    const radius = mapLength(...S.mapDraft);
    if (radius < 0.02 || radius > 5) { S.mapDraft = []; return toast("Bán kính cột phải từ 0,02 đến 5 m", true); }
    const candidate = mapCopy(), id = mapId();
    candidate.objects.push({ id, kind: "pillar", name: `Cột ${candidate.objects.filter((x) => x.kind === "pillar").length + 1}`,
                             center: S.mapDraft[0], radius, height: 2.8 });
    S.mapDraft = []; S.mapSelectedIds = [id]; S.mapSelected = id; saveSiteMap(candidate);
  } else if (S.mapTool === "pillar_diameter" && S.mapDraft.length === 2) {
    const diameter = mapLength(...S.mapDraft);
    if (diameter < 0.04 || diameter > 10) { S.mapDraft = []; return toast("Đường kính cột phải từ 0,04 đến 10 m", true); }
    const candidate = mapCopy(), id = mapId(), [a, b] = S.mapDraft;
    candidate.objects.push({ id, kind: "pillar", name: `Cột ${candidate.objects.filter((x) => x.kind === "pillar").length + 1}`,
                             center: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], radius: diameter / 2, height: 2.8 });
    S.mapDraft = []; S.mapSelectedIds = [id]; S.mapSelected = id; saveSiteMap(candidate);
  }
  renderMapDraft();
  drawBev();
  drawTiles();
}

function finishMapWall() {
  if (S.mapDraft.length < 2) return toast("Cần ít nhất hai điểm để tạo vách", true);
  const candidate = mapCopy(), id = mapId();
  const wall = { id, kind: "wall", name: `Vách ${candidate.objects.filter((x) => x.kind === "wall").length + 1}`,
                 points: S.mapDraft, width: 0.12, height: 2.8 };
  if (!$("mapNewWallOccludes").checked) wall.occludes = false;
  candidate.objects.push(wall);
  S.mapDraft = []; S.mapSelectedIds = [id]; S.mapSelected = id;
  saveSiteMap(candidate);
  renderMapDraft();
}

function renderMapDraft() {
  const n = S.mapDraft.length, tool = S.mapTool;
  const status = tool === "wall" ? `${n} điểm · click tiếp để nối vách, rồi bấm Hoàn tất vách.` :
    tool === "pillar_radius" ? (n ? "Chọn một điểm ở mép cột để đo bán kính." : "Chọn tâm cột, rồi chọn một điểm ở mép.") :
    tool === "pillar_diameter" ? (n ? "Chọn mép đối diện để đo đường kính." : "Chọn hai mép đối diện của cột.") :
    tool === "measure" ? (n === 2 ? `Khoảng cách: ${mapLength(...S.mapDraft).toFixed(3)} m · click để đo lại.` :
      n ? "Chọn điểm thứ hai để đo khoảng cách." : "Chọn hai điểm để đo khoảng cách.") :
    "Chọn vật thể để sửa; kéo các điểm điều khiển trên bản đồ.";
  $("mapDraftStatus").textContent = status;
  const pointLog = $("mapDraftPoints");
  pointLog.classList.toggle("hidden", !n);
  pointLog.textContent = S.mapDraft.map(([x, y], i) =>
    `P${i + 1}  X ${x.toFixed(3)} m · Y ${y.toFixed(3)} m`).join("\n");
  $("btnMapFinish").classList.toggle("hidden", tool !== "wall");
  $("btnMapFinish").disabled = n < 2;
  $("btnMapCancel").disabled = !n;
}

function renderMapEditor() {
  if (!S.siteMap) return;
  const doors = S.siteMap.objects.filter((o) => o.role === "door");
  const openDoors = doors.filter((o) => o.occludes === false).length;
  $("mapStructureStatus").textContent = doors.length ?
    `Tường đen · cửa nâu. ${doors.length - openDoors} cửa đóng, ${openDoors} cửa mở. Chọn cửa để đổi trạng thái. Góc nối giữa các vách di chuyển cùng nhau.` :
    "Góc nối giữa các vách di chuyển cùng nhau khi chỉnh trên ảnh hoặc BEV.";
  renderMapDraft();
  const list = $("mapObjectList");
  const scrollTop = list.scrollTop;
  list.replaceChildren();
  $("mapObjectCount").textContent = `${S.mapSelectedIds.length}/${S.siteMap.objects.length} đã chọn`;
  for (const item of S.siteMap.objects) {
    const button = document.createElement("button");
    button.type = "button";
    button.classList.toggle("selected", S.mapSelectedIds.includes(item.id));
    button.classList.toggle("primary-selected", item.id === S.mapSelected);
    button.setAttribute("aria-pressed", S.mapSelectedIds.includes(item.id) ? "true" : "false");
    const label = document.createElement("span"), size = document.createElement("small");
    label.textContent = item.name;
    size.textContent = item.kind === "pillar" ? `R ${item.radius.toFixed(2)} · H ${item.height.toFixed(1)} m` :
      `${item.points.length} điểm · H ${item.height.toFixed(1)} m` +
      (item.points.length > 2 ? ` · lệch ${(wallDeviation(item) * 100).toFixed(0)} cm` : "");
    if (item.role === "door") size.textContent += item.occludes === false ? " · cửa mở" : " · cửa đóng";
    else if (item.occludes === false || item.estimated) size.textContent += " · ước lượng";
    button.append(label, size);
    button.onclick = (event) => selectMapObject(item.id, event.ctrlKey || event.metaKey);
    list.appendChild(button);
  }
  list.scrollTop = scrollTop;
  const item = mapObject();
  $("mapObjectFields").classList.toggle("hidden", !item);
  if (item) {
    $("mapObjectName").value = item.name;
    $("mapObjectSizeLabel").textContent = item.kind === "pillar" ? "Đường kính cột (m)" : "Bề dày vách (m)";
    $("mapObjectSize").value = item.kind === "pillar" ? +(item.radius * 2).toFixed(4) : item.width;
    $("mapRadiusRow").classList.toggle("hidden", item.kind !== "pillar");
    if (item.kind === "pillar") $("mapObjectRadius").value = item.radius;
    $("mapObjectHeight").value = item.height;
    $("mapObjectOccludes").checked = item.occludes !== false;
    $("mapObjectOccludesLabel").textContent = item.role === "door" ? "Đóng cửa · chặn tầm nhìn camera" : "Vật thể này chặn tầm nhìn camera";
    const fields = $("mapPointFields");
    fields.replaceChildren();
    const points = item.kind === "pillar" ? [item.center] : item.points;
    points.forEach((point, i) => {
      const row = document.createElement("div"); row.className = "point-row";
      row.classList.toggle("active", S.mapActivePoint?.id === item.id && S.mapActivePoint.index === i);
      const label = document.createElement("button");
      label.type = "button";
      label.textContent = item.kind === "pillar" ? "Tâm" : `P${i + 1}`;
      label.title = "Chọn điểm này để căn bằng phím mũi tên";
      label.onclick = () => {
        S.mapActivePoint = { id: item.id, index: i };
        renderMapEditor(); drawBev();
      };
      row.appendChild(label);
      for (let axis = 0; axis < 2; axis++) {
        const input = document.createElement("input");
        input.type = "number"; input.step = "0.01"; input.value = point[axis];
        input.title = axis ? "Y (m)" : "X (m)";
        input.onchange = () => {
          const candidate = mapCopy(), target = candidate.objects.find((x) => x.id === item.id);
          const updated = [...(target.kind === "pillar" ? target.center : target.points[i])];
          updated[axis] = Number(input.value);
          applyMapPointMoves(candidate, [{ id: item.id, index: i, point: updated }]);
          saveSiteMap(candidate);
        };
        row.appendChild(input);
      }
      fields.appendChild(row);
    });
    const selectedPoint = S.mapActivePoint?.id === item.id ? S.mapActivePoint.index : null;
    $("btnMapStraightenPoint").classList.toggle("hidden", item.kind !== "wall" || item.points.length < 3);
    $("btnMapStraightenPoint").disabled = selectedPoint === null || selectedPoint === 0 ||
      selectedPoint === item.points.length - 1;
    const canStraightenWall = item.kind === "wall" && item.points.length > 2;
    $("btnMapStraightenWall").classList.toggle("hidden", !canStraightenWall);
    $("btnMapStraightenWall").disabled = !canStraightenWall || wallDeviation(item) < 0.0001;
    $("mapWallDeviation").classList.toggle("hidden", !canStraightenWall);
    if (canStraightenWall) $("mapWallDeviation").textContent =
      `Điểm giữa lệch tối đa ${(wallDeviation(item) * 100).toFixed(1)} cm so với đường nối hai đầu. Đường vách sẽ gấp khúc khi phóng to.`;
  }
  ["mapX0", "mapX1", "mapY0", "mapY1"].forEach((id, index) => { $(id).value = S.siteMap.bounds[index]; });
  renderMapFlipControls();
}

function pointSegmentDistance(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], den = dx * dx + dy * dy;
  const t = den ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / den)) : 0;
  return mapLength(p, [a[0] + t * dx, a[1] + t * dy]);
}

function mapHit(point) {
  const grip = Math.max(0.12, 11 * (window.devicePixelRatio || 1) / S.bev.s);
  for (const item of [...S.siteMap.objects].reverse()) {
    const points = item.kind === "pillar" ? [item.center] : item.points;
    for (let i = 0; i < points.length; i++)
      if (mapLength(point, points[i]) < grip) return { id: item.id, index: i };
    if (item.kind === "pillar" && mapLength(point, item.center) <= item.radius + grip / 2)
      return { id: item.id, index: null };
    if (item.kind === "wall" && item.points.some((p, i) => i &&
      pointSegmentDistance(point, item.points[i - 1], p) <= item.width / 2 + grip / 2))
      return { id: item.id, index: null };
  }
  return null;
}

function mapCameraPoints(tile, transform = tileTransform(tile)) {
  if (!S.mapEdit || S.mapTool !== "select" || !S.mapProjectCameras || !S.siteMap ||
      !cameraActive(tile.cam.index)) return [];
  const points = [];
  for (const item of S.siteMap.objects) {
    if (S.mapOnlySelected && !S.mapSelectedIds.includes(item.id)) continue;
    const anchors = item.kind === "pillar" ? [item.center] : item.points;
    anchors.forEach((point, index) => {
      const pixel = projectPoint(tile.cam, [...point, 0]);
      if (!pixel || pixel[0] < 0 || pixel[0] >= tile.cam.width ||
          pixel[1] < 0 || pixel[1] >= tile.cam.height) return;
      const [X, Y] = toScreen(transform, ...pixel);
      points.push({ id: item.id, index, X, Y, pixel,
        selected: S.mapSelectedIds.includes(item.id),
        active: S.mapActivePoint?.id === item.id && S.mapActivePoint.index === index });
    });
  }
  return points;
}

function hitMapCameraPoint(tile, X, Y) {
  const radius = 12 * (window.devicePixelRatio || 1);
  let best = null, distance = radius;
  for (const point of mapCameraPoints(tile)) {
    const d = Math.hypot(point.X - X, point.Y - Y);
    if (d < distance || d === distance && point.selected && !best?.selected) {
      best = point; distance = d;
    }
  }
  return best;
}

function drawMapCameraHandles(ctx, tile, transform, dpr) {
  const points = mapCameraPoints(tile, transform);
  if (!points.length) return;
  ctx.save();
  ctx.beginPath(); ctx.rect(transform.ox, transform.oy,
    tile.cam.width * transform.s, tile.cam.height * transform.s); ctx.clip();
  ctx.font = `${11 * dpr}px Consolas, monospace`;
  for (const point of points) {
    const radius = (point.active ? 6 : point.selected ? 5 : 3.2) * dpr;
    ctx.beginPath(); ctx.arc(point.X, point.Y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = point.active ? "#ffdb59" : point.selected ? "#fff" : "rgba(220,248,251,.68)";
    ctx.fill();
    ctx.strokeStyle = point.active ? "#5e3500" : "#11323a";
    ctx.lineWidth = (point.selected ? 2 : 1.3) * dpr; ctx.stroke();
    if (point.selected) {
      const label = `P${point.index + 1}`;
      ctx.lineWidth = 3 * dpr; ctx.strokeStyle = "rgba(0,0,0,.85)";
      ctx.strokeText(label, point.X + 8 * dpr, point.Y - 7 * dpr);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, point.X + 8 * dpr, point.Y - 7 * dpr);
    }
  }
  ctx.restore();
}

function moveMapCameraPoint(drag, imagePoint) {
  const point = ground(drag.t.cam, imagePoint[0] + drag.offset[0], imagePoint[1] + drag.offset[1]);
  if (!point || !insideMap(point)) return false;
  const item = S.siteMap.objects.find((value) => value.id === drag.id);
  if (!item) return false;
  applyMapPointMoves(S.siteMap, [{ id: item.id, index: drag.index, point }]);
  drag.moved = true;
  drawBev(); drawTiles();
  return true;
}

function mapPointerDown(event) {
  if (!S.bev) return;
  if (event.button === 2) { S.mapDraft = []; renderMapDraft(); drawBev(); drawTiles(); return; }
  if (event.button !== 0) return;
  const point = bevPoint(event);
  if (S.mapTool !== "select") return addMapPoint(point);
  const hit = mapHit(point);
  const additive = event.ctrlKey || event.metaKey;
  selectMapObject(hit?.id || null, additive);
  if (!additive && hit?.index !== null && hit?.index !== undefined) {
    S.mapActivePoint = { id: hit.id, index: hit.index };
    renderMapEditor(); drawBev();
  }
  S.mapDragging = !additive && hit?.index !== null && hit?.index !== undefined ?
    { ...hit, moved: false } : null;
}

function mapPointerMove(event) {
  S.mapMouse = bevPoint(event);
  if (S.mapDragging && insideMap(S.mapMouse)) {
    const item = mapObject();
    if (item) {
      const point = S.mapMouse.map((x) => Math.round(x * 10000) / 10000);
      applyMapPointMoves(S.siteMap, [{ id: item.id, index: S.mapDragging.index, point }]);
      S.mapDragging.moved = true;
      drawBev();
      drawTiles();
    }
  } else if (S.mapDraft.length) drawBev();
}

function mapPointerUp() {
  const moved = S.mapDragging?.moved;
  S.mapDragging = null;
  if (moved) saveSiteMap(mapCopy());
}

function commitMapNudge() {
  if (!S.mapNudgeTimer) return;
  clearTimeout(S.mapNudgeTimer);
  S.mapNudgeTimer = null;
  saveSiteMap(mapCopy());
}

function nudgeMapSelection(event) {
  if (!S.siteMap || !S.bev || S.mapTool !== "select" || !S.mapSelectedIds.length ||
      event.ctrlKey || event.metaKey) return false;
  const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  const direction = arrows[event.key];
  if (!direction) return false;
  const step = event.altKey ? 0.001 : event.shiftKey ? 0.1 : 0.01;
  const [dx, dy] = S.bev.step(direction[0] * step, direction[1] * step);
  const candidate = mapCopy();
  const active = S.mapActivePoint;
  const chosen = active ? [active.id] : S.mapSelectedIds;
  const moves = [];
  for (const item of candidate.objects) {
    if (!chosen.includes(item.id)) continue;
    const points = item.kind === "pillar" ? [item.center] : item.points;
    for (const [index, point] of points.entries()) {
      if (active && index !== active.index) continue;
      const moved = [+(point[0] + dx).toFixed(4), +(point[1] + dy).toFixed(4)];
      if (!insideMap(moved)) { toast("Vật thể đã chạm mép bản đồ", true); return true; }
      moves.push({ id: item.id, index, point: moved });
    }
  }
  applyMapPointMoves(candidate, moves);
  S.siteMap = candidate;
  $("mapSaveStatus").textContent = "Chưa lưu · nhả phím mũi tên";
  renderMapEditor(); drawBev(); drawTiles();
  clearTimeout(S.mapNudgeTimer);
  const project = S.project;
  S.mapNudgeTimer = setTimeout(() => { if (S.project === project) commitMapNudge(); }, 350);
  return true;
}

function floorRayBlocked(cam, point, walls, pillars = []) {
  const origin = cam.centre, dx = point[0] - origin[0], dy = point[1] - origin[1];
  for (const wall of walls) {
    const ex = wall.b[0] - wall.a[0], ey = wall.b[1] - wall.a[1];
    const length = Math.hypot(ex, ey);
    if (length < 1e-8) continue;
    const ux = ex / length, uy = ey / length;
    const ox = origin[0] - wall.a[0], oy = origin[1] - wall.a[1];
    const half = Math.max(1e-6, (wall.width || 0) / 2);
    // Intersect the full wall prism, including its end faces and height.
    let enter = 1e-7, leave = 1 - 1e-7;
    for (const [start, direction, lo, hi] of [
      [ox * ux + oy * uy, dx * ux + dy * uy, 0, length],
      [-ox * uy + oy * ux, -dx * uy + dy * ux, -half, half],
      [origin[2], -origin[2], 0, wall.height]]) {
      if (Math.abs(direction) < 1e-10) {
        if (start < lo - 1e-9 || start > hi + 1e-9) { leave = -1; break; }
      } else {
        const a = (lo - start) / direction, b = (hi - start) / direction;
        enter = Math.max(enter, Math.min(a, b)); leave = Math.min(leave, Math.max(a, b));
      }
    }
    if (enter <= leave) return true;
  }
  const a = dx * dx + dy * dy;
  for (const pillar of pillars) {
    if (a < 1e-9) continue;
    const ox = origin[0] - pillar.center[0], oy = origin[1] - pillar.center[1];
    const b = 2 * (ox * dx + oy * dy);
    const c = ox * ox + oy * oy - pillar.radius * pillar.radius;
    const discriminant = b * b - 4 * a * c;
    if (discriminant < 0) continue;
    const enter = Math.max(1e-7, (-b - Math.sqrt(discriminant)) / (2 * a),
      origin[2] > 0 ? 1 - pillar.height / origin[2] : 0);
    const leave = Math.min(1 - 1e-7, (-b + Math.sqrt(discriminant)) / (2 * a));
    if (enter <= leave) return true;
  }
  return false;
}

function cameraFloorClass(cam, point, walls, pillars) {
  if (!insideFloor(point)) return 0;
  const pixel = projectPoint(cam, [...point, 0]);
  if (!pixel || pixel[0] < 0 || pixel[0] >= cam.width || pixel[1] < 0 || pixel[1] >= cam.height) return 0;
  return floorRayBlocked(cam, point, walls, pillars) ? 2 : 1;
}

function traceCoverageBoundary(ctx, visible, nx, ny, xs, ys, mapping, isInside) {
  // Locate the real transition on each cell edge. Midpoints of binary cells form
  // a staircase when the map is enlarged, even when the FOV edge is straight.
  const segments = [[], [3, 0], [0, 1], [3, 1], [1, 2], [3, 0, 1, 2], [0, 2], [3, 2],
    [2, 3], [0, 2], [0, 1, 2, 3], [1, 2], [3, 1], [0, 1], [3, 0], []];
  const at = (i, j) => i >= 0 && i < nx && j >= 0 && j < ny && visible[j * nx + i] ? 1 : 0;
  const centerX = (i) => xs[0] + (i + 0.5) * (xs[nx] - xs[0]) / nx;
  const centerY = (j) => ys[0] + (j + 0.5) * (ys[ny] - ys[0]) / ny;
  const edgesCache = new Map();
  function edge(horizontal, i, j) {
    const key = `${horizontal ? "h" : "v"}:${i}:${j}`;
    if (edgesCache.has(key)) return edgesCache.get(key);
    const x0 = centerX(i), y0 = centerY(j);
    const x1 = horizontal ? centerX(i + 1) : x0;
    const y1 = horizontal ? y0 : centerY(j + 1);
    let lo = 0, hi = 1;
    // The mask outside the map has no world-space sample. Close that border
    // at the cell midpoint; refine all interior transitions against geometry.
    const insideGrid = i >= 0 && j >= 0 &&
      (horizontal ? i + 1 < nx && j < ny : i < nx && j + 1 < ny);
    if (insideGrid) {
      const startInside = !!at(i, j);
      for (let k = 0; k < 8; k++) {
        const mid = (lo + hi) / 2;
        if (!!isInside(x0 + (x1 - x0) * mid, y0 + (y1 - y0) * mid) === startInside) lo = mid;
        else hi = mid;
      }
    }
    const t = (lo + hi) / 2;
    const point = mapping.to(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
    edgesCache.set(key, point);
    return point;
  }
  ctx.beginPath();
  for (let j = -1; j < ny; j++) for (let i = -1; i < nx; i++) {
    const kind = at(i, j) + 2 * at(i + 1, j) + 4 * at(i + 1, j + 1) + 8 * at(i, j + 1);
    const edges = segments[kind];
    if (!edges.length) continue;
    const point = (index) => index === 0 ? edge(true, i, j) :
      index === 1 ? edge(false, i + 1, j) :
      index === 2 ? edge(true, i, j + 1) : edge(false, i, j);
    for (let k = 0; k < edges.length; k += 2) {
      ctx.moveTo(...point(edges[k])); ctx.lineTo(...point(edges[k + 1]));
    }
  }
}

// Top-down diagram of the horizontal image row through the optical axis.
// This is deliberately separate from the footprint of the image on Z=0.
function cameraFanAngles(cam) {
  const k = inv3(cam.K), R = cam.R;
  const heading = Math.atan2(R[2][1], R[2][0]);
  const edges = [0, cam.width].map((u) => {
    const v = cam.K[1][2];
    const ray = k.map((row) => row[0] * u + row[1] * v + row[2]);
    const x = R[0][0] * ray[0] + R[1][0] * ray[1] + R[2][0] * ray[2];
    const y = R[0][1] * ray[0] + R[1][1] * ray[1] + R[2][1] * ray[2];
    return Math.atan2(Math.sin(Math.atan2(y, x) - heading), Math.cos(Math.atan2(y, x) - heading));
  });
  return { heading, lo: Math.min(...edges), hi: Math.max(...edges) };
}

function fanRayDistance(cam, angle, limit, walls, pillars) {
  const [cx, cy, z] = cam.centre, dx = Math.cos(angle), dy = Math.sin(angle);
  let nearest = limit;
  for (const wall of walls) {
    if (wall.height < z) continue;
    const ex = wall.b[0] - wall.a[0], ey = wall.b[1] - wall.a[1], length = Math.hypot(ex, ey);
    if (length < 1e-8) continue;
    const ux = ex / length, uy = ey / length, ox = cx - wall.a[0], oy = cy - wall.a[1];
    const half = Math.max(1e-6, (wall.width || 0) / 2);
    let enter = 0, leave = nearest;
    for (const [start, direction, low, high] of [
      [ox * ux + oy * uy, dx * ux + dy * uy, 0, length],
      [-ox * uy + oy * ux, -dx * uy + dy * ux, -half, half]]) {
      if (Math.abs(direction) < 1e-10) {
        if (start < low || start > high) { leave = -1; break; }
      } else {
        const a = (low - start) / direction, b = (high - start) / direction;
        enter = Math.max(enter, Math.min(a, b)); leave = Math.min(leave, Math.max(a, b));
      }
    }
    if (enter <= leave) nearest = Math.min(nearest, enter);
  }
  for (const pillar of pillars) {
    if (pillar.height < z) continue;
    const ox = cx - pillar.center[0], oy = cy - pillar.center[1];
    const b = ox * dx + oy * dy, c = ox * ox + oy * oy - pillar.radius ** 2;
    const discriminant = b * b - c;
    if (discriminant < 0) continue;
    if (c <= 0) { nearest = 0; continue; }
    const hit = -b - Math.sqrt(discriminant);
    if (hit >= 0) nearest = Math.min(nearest, hit);
  }
  return nearest;
}

function buildCameraFan(cam, bounds, walls, pillars) {
  const { heading, lo, hi } = cameraFanAngles(cam);
  const origin = cam.centre.slice(0, 2);
  const limit = Math.max(...[bounds[0], bounds[1]].flatMap(x => [bounds[2], bounds[3]].map(y => Math.hypot(x-origin[0], y-origin[1])))) + 1;
  const angles = Array.from({ length: 161 }, (_, i) => lo + (hi - lo) * i / 160);
  const addAngle = (angle) => {
    const offset = Math.atan2(Math.sin(angle - heading), Math.cos(angle - heading));
    for (const delta of [-1e-6, 0, 1e-6]) if (offset + delta > lo && offset + delta < hi) angles.push(offset + delta);
  };
  for (const wall of walls) {
    const ex = wall.b[0] - wall.a[0], ey = wall.b[1] - wall.a[1], length = Math.hypot(ex, ey);
    if (!length) continue;
    const nx = -ey / length * wall.width / 2, ny = ex / length * wall.width / 2;
    for (const p of [wall.a, wall.b]) for (const sign of [-1, 1])
      addAngle(Math.atan2(p[1] + sign * ny - origin[1], p[0] + sign * nx - origin[0]));
  }
  for (const pillar of pillars) {
    const dx = pillar.center[0] - origin[0], dy = pillar.center[1] - origin[1];
    const angle = Math.atan2(dy, dx), tangent = Math.asin(Math.min(1, pillar.radius / Math.hypot(dx, dy)));
    addAngle(angle - tangent); addAngle(angle + tangent);
  }
  angles.sort((a, b) => a - b);
  const at = (angle, distance) => [origin[0] + Math.cos(angle) * distance, origin[1] + Math.sin(angle) * distance];
  const visible = angles.map(offset => at(heading + offset, fanRayDistance(cam, heading + offset, limit, walls, pillars)));
  const far = angles.map(offset => at(heading + offset, limit));
  const rays = Array.from({ length: 7 }, (_, i) => {
    const angle = heading + lo + (hi - lo) * i / 6;
    return at(angle, fanRayDistance(cam, angle, limit, walls, pillars));
  });
  const axis = at(heading, fanRayDistance(cam, heading, limit, walls, pillars));
  return { cam, origin, visible, far, rays, axis, angle: (hi-lo)*180/Math.PI };
}

function drawCameraFans(ctx, m, dpr) {
  const map = S.siteMap, mode = S.mapCoverageCamera;
  const cameras = S.project.all_cameras.filter(cam => mode === "all" || String(cam.index) === mode);
  const key = JSON.stringify([mode, map.bounds, map.objects,
    cameras.map(cam => [cam.K, cam.R, cam.centre, cam.width])]);
  if (S.mapFanCache?.key !== key) {
    const walls = mapWallSegments(map), pillars = map.objects.filter(o => o.kind === "pillar" && o.occludes !== false);
    S.mapFanCache = { key, fans: cameras.map(cam => buildCameraFan(cam, map.bounds, walls, pillars)) };
  }
  const fans = S.mapFanCache.fans;
  const path = (out, origin, points) => {
    out.beginPath(); out.moveTo(...m.to(...origin));
    for (const p of points) out.lineTo(...m.to(...p));
    out.closePath();
  };
  ctx.save();
  if (map.floor_regions?.length) { mapFloorPath(ctx, m); ctx.clip(); }
  else {
    const a=m.to(map.bounds[0],map.bounds[2]), b=m.to(map.bounds[1],map.bounds[3]);
    ctx.beginPath(); ctx.rect(Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.abs(b[0]-a[0]),Math.abs(b[1]-a[1])); ctx.clip();
  }
  // A shadow is gray only where no selected camera sees: subtract the union
  // of visible fans from the union of unoccluded viewing cones.
  const mask = S.mapFanMask || (S.mapFanMask = document.createElement("canvas"));
  if (mask.width !== m.W || mask.height !== m.H) { mask.width=m.W; mask.height=m.H; }
  const out=mask.getContext("2d"); out.clearRect(0,0,m.W,m.H);
  out.fillStyle="#7b848a";
  for (const fan of fans) { path(out,fan.origin,fan.far); out.fill(); }
  out.globalCompositeOperation="destination-out";
  for (const fan of fans) { path(out,fan.origin,fan.visible); out.fill(); }
  out.globalCompositeOperation="source-over";
  ctx.globalAlpha=.6; ctx.drawImage(mask,0,0); ctx.globalAlpha=1;
  for (const fan of fans) {
    const color=MAP_CAMERA_COLORS[fan.cam.index % MAP_CAMERA_COLORS.length];
    path(ctx,fan.origin,fan.visible);
    ctx.fillStyle=color; ctx.globalAlpha=mode === "all" ? .15 : .28; ctx.fill();
    ctx.globalAlpha=.9; ctx.strokeStyle=color; ctx.lineWidth=(mode === "all" ? 1.2 : 1.8)*dpr; ctx.stroke();
    const start=m.to(...fan.origin);
    ctx.globalAlpha=mode === "all" ? .24 : .42; ctx.lineWidth=dpr;
    for (const ray of fan.rays.slice(1,-1)) {
      ctx.beginPath(); ctx.moveTo(...start); ctx.lineTo(...m.to(...ray)); ctx.stroke();
    }
    ctx.globalAlpha=.85; ctx.setLineDash([5*dpr,4*dpr]);
    ctx.beginPath(); ctx.moveTo(...start); ctx.lineTo(...m.to(...fan.axis)); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.restore();
}

function drawMapCoverage(ctx, m, dpr) {
  if (!m.W || !m.H) return;
  if (S.mapSightMode === "fan") return drawCameraFans(ctx, m, dpr);
  const map = S.siteMap;
  const mode = S.mapCoverageCamera;
  const key = JSON.stringify([mode, m.flipVertical, m.flipHorizontal, map.bounds, map.objects, map.floor_regions,
    S.project.all_cameras.map((cam) => [cam.K, cam.R, cam.t, cam.centre, cam.width, cam.height])]);
  const drawCached = () => {
    const cache = S.mapCoverageCache;
    const ratio = m.s / cache.s;
    ctx.save();
    if (map.floor_regions?.length) { mapFloorPath(ctx, m); ctx.clip(); }
    ctx.translate(m.du - cache.du * ratio, m.dv - cache.dv * ratio);
    ctx.scale(ratio, ratio);
    ctx.drawImage(cache.canvas, 0, 0);
    ctx.restore();
  };
  if (S.mapCoverageCache?.key === key) {
    drawCached();
    return;
  }
  const editing = S.mapDragging?.moved || S.drag?.kind === "map-point" && S.drag.moved || S.mapNudgeTimer;
  if (!S.mapCoverageCache || !editing) {
    const overlay = document.createElement("canvas");
    overlay.width = m.W; overlay.height = m.H;
    const out = overlay.getContext("2d");
    const [x0, x1, y0, y1] = map.bounds;
    // Sight only needs roughly decimetre precision; denser grids multiply the
    // expensive calibrated ray/obstacle checks for every camera.
    const span = Math.max(x1 - x0, y1 - y0);
    const step = span / (mode === "all" ? 140 : 190);
    const nx = Math.ceil((x1 - x0) / step), ny = Math.ceil((y1 - y0) / step);
    const xs = Array.from({ length: nx + 1 }, (_, i) => x0 + (x1 - x0) * i / nx);
    const ys = Array.from({ length: ny + 1 }, (_, j) => y0 + (y1 - y0) * j / ny);
    const px = xs.map((x) => m.to(x, y0)[0]), py = ys.map((y) => m.to(x0, y)[1]);
    const walls = mapWallSegments(map);
    const pillars = map.objects.filter((item) => item.kind === "pillar" && item.occludes !== false);
    const masks = [];
    const counts = new Uint8Array(nx * ny);
    const blocked = new Uint8Array(nx * ny);
    const cameras = mode === "all" ? S.project.all_cameras :
      S.project.all_cameras.filter((cam) => String(cam.index) === mode);
    for (const cam of cameras) {
      const visible = new Uint8Array(nx * ny);
      const frustum = new Uint8Array(nx * ny);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const point = [(xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2];
        const index = j * nx + i, kind = cameraFloorClass(cam, point, walls, pillars);
        if (kind) frustum[index] = 1;
        if (kind === 1) { visible[index] = 1; counts[index]++; }
        else if (kind === 2) blocked[index]++;
      }
      masks.push({ cam, visible, frustum });
    }
    const raster = document.createElement("canvas");
    raster.width = nx; raster.height = ny;
    const rasterCtx = raster.getContext("2d");
    const pixels = rasterCtx.createImageData(nx, ny);
    const sightColor = mode === "all" ? MAP_SIGHT_COLOR :
      MAP_CAMERA_COLORS[cameras[0]?.index % MAP_CAMERA_COLORS.length || 0];
    const sightRgb = [1, 3, 5].map((i) => parseInt(sightColor.slice(i, i + 2), 16));
    for (let index = 0; index < counts.length; index++) {
      const n = counts[index], offset = index * 4;
      const rgb = n ? sightRgb : [123, 132, 138];
      pixels.data[offset] = rgb[0];
      pixels.data[offset + 1] = rgb[1];
      pixels.data[offset + 2] = rgb[2];
      pixels.data[offset + 3] = n ? 84 : blocked[index] ? 180 : 0;
    }
    rasterCtx.putImageData(pixels, 0, 0);
    out.save();
    out.imageSmoothingEnabled = true;
    out.translate(px[0], py[0]);
    out.scale((px[nx] - px[0]) / nx, (py[ny] - py[0]) / ny);
    out.drawImage(raster, 0, 0);
    out.restore();
    out.lineCap = "round";
    out.lineJoin = "round";
    for (const { cam, visible, frustum } of masks) {
      const cameraColor = MAP_CAMERA_COLORS[cam.index % MAP_CAMERA_COLORS.length];
      if (mode !== "all") {
        traceCoverageBoundary(out, frustum, nx, ny, xs, ys, m, (x, y) => {
          if (!insideFloor([x, y])) return false;
          const pixel = projectPoint(cam, [x, y, 0]);
          return pixel && pixel[0] >= 0 && pixel[0] < cam.width && pixel[1] >= 0 && pixel[1] < cam.height;
        });
        out.strokeStyle = cameraColor;
        out.globalAlpha = 0.6;
        out.lineWidth = 1.15 * dpr;
        out.setLineDash([4 * dpr, 4 * dpr]); out.stroke(); out.setLineDash([]);
        out.globalAlpha = 1;
      }
      traceCoverageBoundary(out, visible, nx, ny, xs, ys, m,
        (x, y) => cameraFloorClass(cam, [x, y], walls, pillars) === 1);
      out.strokeStyle = cameraColor;
      out.lineWidth = (mode === "all" ? 1.45 : 2.25) * dpr;
      out.stroke();
    }
    S.mapCoverageCache = { key, mode, canvas: overlay, s: m.s, du: m.du, dv: m.dv };
  }
  drawCached();
}

function drawSiteMap(ctx, m, dpr) {
  const [x0, x1, y0, y1] = S.area;
  const a = m.to(x0, y0), b = m.to(x1, y1);
  const left = Math.min(a[0], b[0]), top = Math.min(a[1], b[1]);
  const width = Math.abs(b[0] - a[0]), height = Math.abs(b[1] - a[1]);
  ctx.fillStyle = "rgba(197, 49, 65, .055)";
  ctx.fillRect(left, top, width, height);
  ctx.strokeStyle = "#bf3446"; ctx.lineWidth = 2 * dpr;
  ctx.strokeRect(left, top, width, height);
  ctx.font = `${11 * dpr}px Consolas, monospace`;
  ctx.fillStyle = "#a52b3b";
  ctx.fillText(`ROI MODEL ${(+x1 - x0).toFixed(2)} × ${(+y1 - y0).toFixed(2)} m`, left + 7 * dpr, top + 15 * dpr);
  for (const item of S.siteMap.objects) {
    const selected = S.mapEdit && S.mapSelectedIds.includes(item.id);
    const primary = selected && item.id === S.mapSelected;
    if (item.kind === "wall") {
      const points = item.points.map((p) => m.to(...p));
      ctx.beginPath(); ctx.moveTo(...points[0]);
      points.slice(1).forEach((p) => ctx.lineTo(...p));
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      const guide = item.occludes === false;
      const width = guide ? 2 * dpr : Math.max(3 * dpr, item.width * m.s);
      if (guide) ctx.setLineDash([9 * dpr, 5 * dpr]);
      if (selected) {
        ctx.strokeStyle = primary ? "#e69b2d" : "#2d91ad";
        ctx.lineWidth = width + 5 * dpr; ctx.stroke();
      }
      ctx.strokeStyle = item.role === "door" ? "#966124" : guide ? "#6b7d84" : "#171c1e";
      ctx.lineWidth = width; ctx.stroke();
      if (guide) ctx.setLineDash([]);
      if (selected && (S.mapFocus || S.mapEdit)) {
        ctx.fillStyle = "#344348";
        ctx.fillText(item.name, points[0][0] + 7 * dpr, points[0][1] - 7 * dpr);
      }
      if (primary) points.forEach((p, index) =>
        mapHandle(ctx, p, dpr, S.mapActivePoint?.id === item.id && S.mapActivePoint.index === index));
    } else {
      const p = m.to(...item.center), radius = Math.max(2 * dpr, item.radius * m.s);
      ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, 2 * Math.PI);
      ctx.fillStyle = item.occludes === false ? "#d1a270" : "#ef8c24"; ctx.fill();
      ctx.strokeStyle = primary ? "#9b4517" : selected ? "#2d91ad" : "#392b20";
      ctx.lineWidth = (selected ? 3 : 1.5) * dpr; ctx.stroke();
      if (selected && (S.mapFocus || S.mapEdit)) {
        ctx.fillStyle = "#6d3a17";
        ctx.fillText(`${item.name} · ⌀${(item.radius * 2).toFixed(2)}`, p[0] + radius + 6 * dpr, p[1]);
      }
      if (primary) mapHandle(ctx, p, dpr, S.mapActivePoint?.id === item.id);
    }
  }
  for (const cam of S.project.all_cameras) {
    const color = MAP_CAMERA_COLORS[cam.index % MAP_CAMERA_COLORS.length];
    const p = m.to(cam.centre[0], cam.centre[1]);
    const forward = m.to(cam.centre[0] + cam.R[2][0], cam.centre[1] + cam.R[2][1]);
    const dir = [forward[0] - p[0], forward[1] - p[1]];
    const norm = Math.hypot(...dir) || 1;
    const tip = [p[0] + dir[0] / norm * 19 * dpr, p[1] + dir[1] / norm * 19 * dpr];
    ctx.beginPath(); ctx.moveTo(...tip);
    ctx.lineTo(p[0] - dir[1] / norm * 7 * dpr, p[1] + dir[0] / norm * 7 * dpr);
    ctx.lineTo(p[0] + dir[1] / norm * 7 * dpr, p[1] - dir[0] / norm * 7 * dpr);
    ctx.closePath(); ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = "#08131c"; ctx.lineWidth = 1.5 * dpr; ctx.stroke();
    const fan = S.mapShowCoverage && S.mapSightMode === "fan" ? S.mapFanCache?.fans.find(f => f.cam.index === cam.index) : null;
    ctx.fillStyle = color;
    ctx.fillText(`C${cam.index + 1}${fan ? ` · ${fan.angle.toFixed(0)}°` : ""}`, p[0] + 9 * dpr, p[1] - 7 * dpr);
  }
  drawMapScale(ctx, m, dpr);
  if (S.mapFocus && S.mapEdit) drawSelectedWallDimensions(ctx, m, dpr);
  const draft = S.mapDraft;
  if (S.mapEdit && draft.length) {
    const points = [...draft, ...(S.mapMouse && draft.length < 2 || S.mapTool === "wall" && S.mapMouse ? [S.mapMouse] : [])];
    ctx.beginPath(); ctx.moveTo(...m.to(...points[0]));
    points.slice(1).forEach((p) => ctx.lineTo(...m.to(...p)));
    ctx.strokeStyle = "#b1eff4"; ctx.lineWidth = 2 * dpr; ctx.setLineDash([5 * dpr, 4 * dpr]); ctx.stroke(); ctx.setLineDash([]);
    for (const point of draft) mapHandle(ctx, m.to(...point), dpr);
    if ((S.mapTool === "pillar_radius" || S.mapTool === "pillar_diameter") && points.length > 1) {
      const center = S.mapTool === "pillar_radius" ? points[0] :
        [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2];
      const radius = S.mapTool === "pillar_radius" ? mapLength(points[0], points[1]) : mapLength(points[0], points[1]) / 2;
      const c = m.to(...center);
      ctx.beginPath(); ctx.arc(c[0], c[1], radius * m.s, 0, 2 * Math.PI);
      ctx.strokeStyle = "#ffb298"; ctx.lineWidth = 2 * dpr; ctx.stroke();
    }
  }
}

function drawMapScale(ctx, m, dpr) {
  const metre = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    .find((length) => length * m.s >= 62 * dpr) || 10;
  const width = metre * m.s;
  const x = 17 * dpr, y = m.H - 19 * dpr;
  ctx.save();
  ctx.fillStyle = "rgba(255,255,255,.9)";
  ctx.fillRect(x - 8 * dpr, y - 25 * dpr, width + 19 * dpr, 34 * dpr);
  ctx.strokeStyle = "#35464b"; ctx.lineWidth = 2 * dpr;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + width, y);
  ctx.moveTo(x, y - 5 * dpr); ctx.lineTo(x, y + 5 * dpr);
  ctx.moveTo(x + width, y - 5 * dpr); ctx.lineTo(x + width, y + 5 * dpr); ctx.stroke();
  ctx.fillStyle = "#35464b"; ctx.font = `${11 * dpr}px Consolas, monospace`;
  ctx.fillText(`${metre} m`, x, y - 9 * dpr);
  ctx.restore();
}

function drawSelectedWallDimensions(ctx, m, dpr) {
  const item = mapObject();
  if (item?.kind !== "wall") return;
  ctx.save();
  ctx.font = `${10 * dpr}px Consolas, monospace`;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (let i = 1; i < item.points.length; i++) {
    const a = m.to(...item.points[i - 1]), b = m.to(...item.points[i]);
    const dx = b[0] - a[0], dy = b[1] - a[1], span = Math.hypot(dx, dy);
    if (span < 28 * dpr) continue;
    const nx = -dy / span, ny = dx / span, offset = 18 * dpr;
    const p = [a[0] + nx * offset, a[1] + ny * offset];
    const q = [b[0] + nx * offset, b[1] + ny * offset];
    ctx.strokeStyle = "#53717b"; ctx.lineWidth = 1 * dpr;
    ctx.beginPath(); ctx.moveTo(...p); ctx.lineTo(...q);
    for (const v of [p, q]) {
      ctx.moveTo(v[0] - nx * 5 * dpr, v[1] - ny * 5 * dpr);
      ctx.lineTo(v[0] + nx * 5 * dpr, v[1] + ny * 5 * dpr);
    }
    ctx.stroke();
    const label = `${mapLength(item.points[i - 1], item.points[i]).toFixed(2)} m`;
    const cx = (p[0] + q[0]) / 2, cy = (p[1] + q[1]) / 2;
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(255,255,255,.92)";
    ctx.fillRect(cx - textWidth / 2 - 3 * dpr, cy - 7 * dpr, textWidth + 6 * dpr, 14 * dpr);
    ctx.fillStyle = "#35515c"; ctx.fillText(label, cx, cy);
  }
  ctx.restore();
}

function mapHandle(ctx, point, dpr, active = false) {
  ctx.beginPath(); ctx.arc(point[0], point[1], (active ? 6.5 : 5) * dpr, 0, 2 * Math.PI);
  ctx.fillStyle = active ? "#ffd652" : "#eafcff"; ctx.fill();
  ctx.strokeStyle = active ? "#713d10" : "#123339"; ctx.lineWidth = 2 * dpr; ctx.stroke();
}

function drawSiteMapOnCamera(ctx, tile, transform, dpr) {
  const cam = tile.cam;
  const selected = S.mapSelected;
  const items = S.siteMap.objects
    .filter((item) => !S.mapOnlySelected || S.mapSelectedIds.includes(item.id))
    .sort((a, b) => Number(S.mapSelectedIds.includes(a.id)) - Number(S.mapSelectedIds.includes(b.id)) ||
      Number(a.id === selected) - Number(b.id === selected));
  const wallSegments = mapWallSegments();
  const pillars = S.siteMap.objects.filter((item) => item.kind === "pillar" && item.occludes !== false);
  const screen = ([x, y, z]) => {
    const pixel = projectPoint(cam, [x, y, z]);
    return pixel ? toScreen(transform, ...pixel) : null;
  };
  const near = 0.1;
  const screenCamera = (point) => toScreen(transform, ...toPixel(cam, point));
  function nearIntersection(a, b) {
    const t = (near - a[2]) / (b[2] - a[2]);
    return a.map((value, i) => value + t * (b[i] - value));
  }
  function projectedSegment(a, b) {
    let ca = toCamera(cam, a), cb = toCamera(cam, b);
    if (ca[2] < near && cb[2] < near) return null;
    if (ca[2] < near) ca = nearIntersection(ca, cb);
    else if (cb[2] < near) cb = nearIntersection(cb, ca);
    return [screenCamera(ca), screenCamera(cb)];
  }
  function projectedFace(worldPoints, fill) {
    const source = worldPoints.map((point) => toCamera(cam, point));
    const clipped = [];
    for (let i = 0; i < source.length; i++) {
      const a = source[i], b = source[(i + 1) % source.length];
      const aIn = a[2] >= near, bIn = b[2] >= near;
      if (aIn) clipped.push(a);
      if (aIn !== bIn) clipped.push(nearIntersection(a, b));
    }
    if (clipped.length < 3) return false;
    ctx.beginPath();
    clipped.map(screenCamera).forEach((point, i) => {
      if (i) ctx.lineTo(...point); else ctx.moveTo(...point);
    });
    ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
    return true;
  }
  function projectedLine(a, b, color, width, dash = []) {
    const segment = projectedSegment(a, b);
    if (segment) line(segment, color, width, false, dash);
  }
  function line(points, color, width, closed = false, dash = []) {
    if (points.some((p) => !p)) return;
    ctx.beginPath();
    points.forEach((p, i) => { if (i) ctx.lineTo(...p); else ctx.moveTo(...p); });
    if (closed) ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = width * dpr;
    ctx.setLineDash(dash.map((value) => value * dpr));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  function wall(a, b, item, active, primary) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const length = Math.hypot(dx, dy);
    if (length < 0.01) return;
    const nx = -dy / length * item.width / 2, ny = dx / length * item.width / 2;
    const floor = [[a[0] + nx, a[1] + ny], [b[0] + nx, b[1] + ny],
                   [b[0] - nx, b[1] - ny], [a[0] - nx, a[1] - ny]];
    const base = floor.map(([x, y]) => screen([x, y, 0]));
    const top = floor.map(([x, y]) => screen([x, y, item.height]));
    const vertices = [...base, ...top];
    if (vertices.some((p) => !p)) return;
    const color = primary ? "#ffe3a3" : active ? "#a9eaf0" : "rgba(255, 192, 108, .55)";
    if (active) {
      ctx.beginPath();
      for (const [i, p] of [base[0], base[1], top[1], top[0]].entries()) {
        if (i) ctx.lineTo(...p); else ctx.moveTo(...p);
      }
      ctx.closePath(); ctx.fillStyle = primary ? "rgba(255, 190, 96, .08)" : "rgba(126, 221, 234, .08)"; ctx.fill();
    }
    line(base, color, active ? 2.5 : 1.3, true);
    line(top, color, active ? 2.5 : 1.3, true);
    for (let i = 0; i < 4; i++) line([base[i], top[i]], color, active ? 2 : 1);
    const xs = vertices.map((p) => p[0]), ys = vertices.map((p) => p[1]);
    const left = Math.min(...xs), right = Math.max(...xs);
    const upper = Math.min(...ys), lower = Math.max(...ys);
    if (right < 0 || left > tile.canvas.width || lower < 0 || upper > tile.canvas.height) return;
    if (right - left > 4 * tile.canvas.width || lower - upper > 4 * tile.canvas.height) return;
    ctx.strokeStyle = primary ? "#fff0c8" : active ? "#b9f1f5" : "rgba(255, 218, 156, .46)";
    ctx.lineWidth = (active ? 2.2 : 1.1) * dpr;
    ctx.setLineDash([6 * dpr, 4 * dpr]);
    ctx.strokeRect(left, upper, right - left, lower - upper);
    ctx.setLineDash([]);
  }
  function wallSurface(a, b, item, active, primary) {
    const dx = b[0] - a[0], dy = b[1] - a[1], length = Math.hypot(dx, dy);
    if (length < 0.01) return;
    const facing = (-dy * (cam.centre[0] - a[0]) + dx * (cam.centre[1] - a[1])) >= 0 ? 1 : -1;
    const nx = facing * -dy / length * item.width / 2;
    const ny = facing * dx / length * item.width / 2;
    const p = [a[0] + nx, a[1] + ny], q = [b[0] + nx, b[1] + ny];
    if (!projectedFace([[...p, 0], [...q, 0], [...q, item.height], [...p, item.height]],
      primary ? "rgba(216, 57, 60, .34)" : active ? "rgba(222, 60, 62, .29)" : "rgba(212, 53, 57, .23)")) return;
    projectedLine([...p, item.height], [...q, item.height], primary ? "#e93438" : "rgba(191, 38, 48, .85)", 1.7);
    projectedLine([...p, 0], [...p, item.height], "rgba(191, 38, 48, .82)", 1.6);
    projectedLine([...q, 0], [...q, item.height], "rgba(191, 38, 48, .82)", 1.6);
    projectedLine([...p, 0], [...q, 0], "rgba(67, 42, 10, .82)", 5);
    projectedLine([...p, 0], [...q, 0], primary ? "#fff36b" : "#ffe335", 2.8);
  }
  function visibleWallSurface(a, b, item, active, primary) {
    const length = mapLength(a, b);
    const count = Math.max(1, Math.ceil(length / 0.35));
    const blockers = wallSegments.filter((segment) => segment.id !== item.id);
    const point = (t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    let start = null;
    for (let i = 0; i < count; i++) {
      const visible = !floorRayBlocked(cam, point((i + 0.5) / count), blockers, pillars);
      if (visible && start === null) start = i / count;
      if (!visible && start !== null) {
        wallSurface(point(start), point(i / count), item, active, primary);
        start = null;
      }
    }
    if (start !== null) wallSurface(point(start), b, item, active, primary);
  }
  function pillarSurface(item, active, primary) {
    const count = 32, base = [], top = [];
    for (let i = 0; i < count; i++) {
      const angle = i * 2 * Math.PI / count;
      const x = item.center[0] + item.radius * Math.cos(angle);
      const y = item.center[1] + item.radius * Math.sin(angle);
      base.push(screen([x, y, 0])); top.push(screen([x, y, item.height]));
    }
    if ([...base, ...top].some((point) => !point)) return;
    const points = [...base, ...top].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const lower = [], upper = [];
    for (const point of points) {
      while (lower.length > 1 && cross(lower.at(-2), lower.at(-1), point) <= 0) lower.pop();
      lower.push(point);
    }
    for (const point of [...points].reverse()) {
      while (upper.length > 1 && cross(upper.at(-2), upper.at(-1), point) <= 0) upper.pop();
      upper.push(point);
    }
    const hull = [...lower.slice(0, -1), ...upper.slice(0, -1)];
    if (hull.length < 3) return;
    ctx.beginPath(); hull.forEach((point, i) => { if (i) ctx.lineTo(...point); else ctx.moveTo(...point); });
    ctx.closePath();
    ctx.fillStyle = primary ? "rgba(53, 151, 216, .49)" : active ? "rgba(60, 154, 219, .40)" : "rgba(67, 153, 213, .34)";
    ctx.fill(); ctx.strokeStyle = "#2088c2"; ctx.lineWidth = 1.8 * dpr; ctx.stroke();
    line(base, "#26a6dc", 2, true);
    line(top, "#26a6dc", 1.7, true);
  }
  function pillar(item, active, primary) {
    const count = 32, center = item.center, radius = item.radius;
    const floor = Array.from({ length: count }, (_, i) => {
      const angle = i * 2 * Math.PI / count;
      return [center[0] + radius * Math.cos(angle), center[1] + radius * Math.sin(angle)];
    });
    const base = floor.map(([x, y]) => screen([x, y, 0]));
    const top = floor.map(([x, y]) => screen([x, y, item.height]));
    if ([...base, ...top].some((p) => !p)) return;
    const color = primary ? "#ffd0bb" : active ? "#a9eaf0" : "rgba(255, 148, 115, .68)";
    if (active) for (let i = 0; i < count; i++) {
      const next = (i + 1) % count;
      ctx.beginPath(); ctx.moveTo(...base[i]); ctx.lineTo(...base[next]);
      ctx.lineTo(...top[next]); ctx.lineTo(...top[i]); ctx.closePath();
      ctx.fillStyle = primary ? "rgba(255, 145, 112, .025)" : "rgba(126, 221, 234, .025)"; ctx.fill();
    }
    line(base, color, active ? 2.5 : 1.5, true);
    line(top, color, active ? 2.5 : 1.5, true);
    for (let i = 0; i < count; i += 4)
      line([base[i], top[i]], color, active ? 2 : 1);
  }
  ctx.save();
  ctx.beginPath(); ctx.rect(transform.ox, transform.oy,
    cam.width * transform.s, cam.height * transform.s); ctx.clip();
  for (const item of items) {
    const active = S.mapSelectedIds.includes(item.id);
    const primary = item.id === selected;
    if (item.occludes === false) {
      if (item.kind === "wall") item.points.slice(1).forEach((p, i) =>
        line([screen([...item.points[i], 0]), screen([...p, 0])],
          primary ? "#fff0ad" : "#c3e9ec", active ? 3.5 : 2.5, false, [8, 5]));
      else {
        const points = Array.from({ length: 32 }, (_, i) => {
          const angle = i * 2 * Math.PI / 32;
          return screen([item.center[0] + item.radius * Math.cos(angle),
                         item.center[1] + item.radius * Math.sin(angle), 0]);
        });
        line(points, primary ? "#fff0ad" : "#c3e9ec", 2.5, true, [8, 5]);
      }
      continue;
    }
    if (S.mapProjectionMode === "floor") {
      if (item.kind === "wall") item.points.slice(1).forEach((p, i) =>
        line([screen([...item.points[i], 0]), screen([...p, 0])],
             primary ? "#ffe3a3" : active ? "#a9eaf0" : "rgba(255, 195, 106, .94)",
             active ? 3.5 : 3, false, [7, 4]));
      else {
        const points = Array.from({ length: 32 }, (_, i) => {
          const angle = i * 2 * Math.PI / 32;
          return screen([item.center[0] + item.radius * Math.cos(angle),
                         item.center[1] + item.radius * Math.sin(angle), 0]);
        });
        line(points, primary ? "#ffd0bb" : active ? "#a9eaf0" : "rgba(255, 133, 104, .94)",
             active ? 3.5 : 3, true, [7, 4]);
      }
    } else if (S.mapProjectionMode === "surface") {
      if (item.kind === "wall") item.points.slice(1).forEach((p, i) =>
        visibleWallSurface(item.points[i], p, item, active, primary));
      else pillarSurface(item, active, primary);
    } else if (item.kind === "wall") {
      item.points.slice(1).forEach((p, i) => wall(item.points[i], p, item, active, primary));
    } else pillar(item, active, primary);
  }
  if (S.mapDraft.length >= 2 && S.mapTool === "wall")
    line(S.mapDraft.map(([x, y]) => screen([x, y, 0])), "rgba(169, 239, 244, .95)", 3, false, [7, 4]);
  ctx.font = `${11 * dpr}px Consolas, monospace`;
  for (const [i, [x, y]] of S.mapDraft.entries()) {
    const point = screen([x, y, 0]);
    if (!point) continue;
    ctx.beginPath(); ctx.arc(point[0], point[1], 5 * dpr, 0, 2 * Math.PI);
    ctx.fillStyle = "#b9f5f8"; ctx.fill();
    ctx.strokeStyle = "#073139"; ctx.lineWidth = 2 * dpr; ctx.stroke();
    ctx.fillStyle = "#d9fbfd";
    ctx.fillText(S.mapTool === "pillar_radius" && i === 0 ? "Tâm" : `P${i + 1}`,
                 point[0] + 8 * dpr, point[1] - 8 * dpr);
  }
  ctx.restore();
}

function wireSiteMap() {
  const setMapSight = (on) => {
    S.mapShowCoverage = on;
    $("mapSightToggle").checked = on;
    $("mapShowCoverage").checked = on;
    drawBev();
  };
  const changeFlip = (key, value) => {
    if (!S.siteMap) return;
    const candidate = mapCopy();
    candidate[key] = value;
    saveSiteMap(candidate);
  };
  $("btnBevFlipHorizontal").onclick = () => changeFlip("flip_horizontal", !S.siteMap.flip_horizontal);
  $("btnBevFlipVertical").onclick = () => changeFlip("flip_vertical", !S.siteMap.flip_vertical);
  $("btnBevFlipReset").onclick = () => {
    if (!S.siteMap) return;
    const candidate = mapCopy();
    candidate.flip_horizontal = false;
    candidate.flip_vertical = false;
    saveSiteMap(candidate);
  };
  $("btnMapZoomOut").onclick = () => zoomMapAt(0.8);
  $("btnMapZoomIn").onclick = () => zoomMapAt(1.25);
  $("btnMapZoomReset").onclick = () => {
    S.mapView = { zoom: 1, panX: 0, panY: 0 };
    renderMapZoom(); drawBev();
  };
  $("btnSiteMap").onclick = () => {
    const open = !(S.mapEdit && S.mapFocus);
    S.mapEdit = open;
    S.mapFocus = open;
    S.mapDraft = [];
    updateMapLayout();
    renderMapEditor();
  };
  $("btnMapExpand").onclick = () => { S.mapFocus = !S.mapFocus; updateMapLayout(); };
  $("btnMapUseCamera").onclick = () => {
    if (S.mapTool === "select") {
      S.mapTool = "wall";
      $("mapTool").value = "wall";
      renderMapDraft();
    }
    S.mapProjectCameras = true;
    $("mapProjectCameras").checked = true;
    S.mapFocus = false;
    updateMapLayout();
    toast("Click đúng chân vách/cột trên ảnh; điểm sẽ xuất hiện trên map và các camera khác.");
  };
  $("btnMapEdit").onclick = () => { S.mapEdit = !S.mapEdit; S.mapDraft = []; updateMapLayout(); renderMapEditor(); };
  $("mapTool").onchange = () => {
    S.mapTool = $("mapTool").value; S.mapDraft = []; S.mapDragging = null; S.mapActivePoint = null;
    renderMapDraft(); drawBev(); drawTiles();
  };
  $("mapFlipHorizontal").onchange = () => changeFlip("flip_horizontal", $("mapFlipHorizontal").checked);
  $("mapFlipVertical").onchange = () => changeFlip("flip_vertical", $("mapFlipVertical").checked);
  $("mapProjectCameras").onchange = () => {
    S.mapProjectCameras = $("mapProjectCameras").checked;
    if (!S.mapProjectCameras && S.show.obstaclesOnly) {
      S.show.obstaclesOnly = false;
      $("mapHidePeople").checked = false;
    }
    drawTiles(); drawBev();
  };
  $("mapSightToggle").onchange = () => setMapSight($("mapSightToggle").checked);
  $("mapShowCoverage").onchange = () => setMapSight($("mapShowCoverage").checked);
  $("mapCoverageCamera").onchange = () => {
    S.mapCoverageCamera = $("mapCoverageCamera").value;
    setMapSight(true);
  };
  $("mapSightMode").onchange = () => {
    S.mapSightMode = $("mapSightMode").value;
    setMapSight(true);
  };
  $("mapProjectionMode").onchange = () => {
    S.mapProjectionMode = $("mapProjectionMode").value;
    drawTiles();
  };
  $("mapOnlySelected").onchange = () => {
    S.mapOnlySelected = $("mapOnlySelected").checked;
    drawTiles();
  };
  $("mapHidePeople").onchange = () => {
    S.show.obstaclesOnly = $("mapHidePeople").checked;
    if (S.show.obstaclesOnly) {
      S.mapProjectCameras = true;
      $("mapProjectCameras").checked = true;
    }
    drawTiles(); drawBev();
  };
  $("btnMapFinish").onclick = finishMapWall;
  $("btnMapCancel").onclick = () => { S.mapDraft = []; renderMapDraft(); drawBev(); drawTiles(); };
  $("mapObjectName").onchange = () => {
    const candidate = mapCopy(), item = candidate.objects.find((x) => x.id === S.mapSelected);
    if (item) { item.name = $("mapObjectName").value.trim(); saveSiteMap(candidate); }
  };
  $("mapObjectSize").onchange = () => {
    const candidate = mapCopy(), item = candidate.objects.find((x) => x.id === S.mapSelected);
    if (item) {
      if (item.kind === "pillar") item.radius = Number($("mapObjectSize").value) / 2;
      else item.width = Number($("mapObjectSize").value);
      saveSiteMap(candidate);
    }
  };
  $("mapObjectRadius").onchange = () => {
    const candidate = mapCopy(), item = candidate.objects.find((x) => x.id === S.mapSelected);
    if (item?.kind === "pillar") { item.radius = Number($("mapObjectRadius").value); saveSiteMap(candidate); }
  };
  $("mapObjectHeight").onchange = () => {
    const candidate = mapCopy(), item = candidate.objects.find((x) => x.id === S.mapSelected);
    if (item) { item.height = Number($("mapObjectHeight").value); saveSiteMap(candidate); }
  };
  $("mapObjectOccludes").onchange = () => {
    const candidate = mapCopy(), item = candidate.objects.find((x) => x.id === S.mapSelected);
    if (!item) return;
    if ($("mapObjectOccludes").checked) delete item.occludes;
    else item.occludes = false;
    saveSiteMap(candidate);
  };
  $("btnMapStraightenPoint").onclick = () => {
    const active = S.mapActivePoint;
    const candidate = mapCopy();
    const item = candidate.objects.find((value) => value.id === active?.id);
    const i = active?.index;
    if (item?.kind !== "wall" || !Number.isInteger(i) || i <= 0 || i >= item.points.length - 1) return;
    const a = item.points[i - 1], p = item.points[i], b = item.points[i + 1];
    const dx = b[0] - a[0], dy = b[1] - a[1], squared = dx * dx + dy * dy;
    if (squared < 1e-8) return toast("Hai điểm kề quá gần để nắn thẳng", true);
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / squared));
    applyMapPointMoves(candidate, [{ id: item.id, index: i, point: [a[0] + t * dx, a[1] + t * dy] }]);
    saveSiteMap(candidate);
  };
  $("btnMapStraightenWall").onclick = () => {
    const candidate = mapCopy();
    const item = candidate.objects.find((value) => value.id === S.mapSelected);
    if (item?.kind !== "wall" || item.points.length < 3) return;
    const a = item.points[0], b = item.points.at(-1);
    const dx = b[0] - a[0], dy = b[1] - a[1], squared = dx * dx + dy * dy;
    if (squared < 1e-8) return toast("Hai đầu vách quá gần để nắn thẳng", true);
    const moves = item.points.map((point, index) => {
      if (index === 0 || index === item.points.length - 1) return { id: item.id, index, point };
      const t = Math.max(0, Math.min(1,
        ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / squared));
      return { id: item.id, index, point: [a[0] + t * dx, a[1] + t * dy] };
    });
    applyMapPointMoves(candidate, moves);
    saveSiteMap(candidate);
  };
  $("btnMapDelete").onclick = () => {
    const candidate = mapCopy();
    candidate.objects = candidate.objects.filter((x) => x.id !== S.mapSelected);
    S.mapSelectedIds = S.mapSelectedIds.filter((id) => id !== S.mapSelected);
    S.mapSelected = S.mapSelectedIds.at(-1) || null;
    saveSiteMap(candidate);
  };
  $("btnMapBounds").onclick = () => {
    const candidate = mapCopy();
    candidate.bounds = ["mapX0", "mapX1", "mapY0", "mapY1"].map((id) => Number($(id).value));
    saveSiteMap(candidate);
  };
  $("btnMapDownload").onclick = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(S.siteMap, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url; link.download = "wavetrack-site-map.json"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  $("btnMapImport").onclick = () => $("mapImportFile").click();
  $("mapImportFile").onchange = async () => {
    const file = $("mapImportFile").files[0];
    $("mapImportFile").value = "";
    if (!file) return;
    if (file.size > 1024 * 1024) return toast("Sơ đồ JSON tối đa 1 MB", true);
    try { await saveSiteMap(JSON.parse(await file.text())); }
    catch (error) { toast(`Không đọc được JSON: ${error.message}`, true); }
  };
  window.addEventListener("mouseup", () => { if (S.mapDragging) mapPointerUp(); });
  window.addEventListener("keyup", (event) => { if (event.key.startsWith("Arrow")) commitMapNudge(); });
  window.addEventListener("blur", commitMapNudge);
}
