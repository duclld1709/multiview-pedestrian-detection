"use strict";

const CROPS = { open: false, width: 220, cards: [], signature: "" };

// Boxes use calibrated camera coordinates; image pixels may have a different size.
function cropRegion(box, cam, img) {
  if (!box || box.length !== 4 || !box.every(Number.isFinite) ||
      !img.naturalWidth || !img.naturalHeight || !(cam.width > 0 && cam.height > 0)) return null;
  const x1 = Math.max(0, box[0]), y1 = Math.max(0, box[1]);
  const x2 = Math.min(cam.width, box[2]), y2 = Math.min(cam.height, box[3]);
  if (x2 <= x1 || y2 <= y1) return null;
  const sx = img.naturalWidth / cam.width, sy = img.naturalHeight / cam.height;
  return [x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy];
}

function setCrops(open) {
  CROPS.open = open;
  $("cropPanel").classList.toggle("hidden", !open);
  document.querySelector(".layout").classList.toggle("with-crops", open);
  $("btnCrops").setAttribute("aria-pressed", String(open));
  try { localStorage.setItem("mvlabel.crops", open ? "1" : "0"); } catch (_) { /* private window */ }
  requestAnimationFrame(() => { layoutTiles(); drawAll(); });
}

function setCropWidth(width) {
  const layout = document.querySelector(".layout");
  CROPS.width = Math.round(Math.max(120, Math.min(width, Math.max(120, layout.clientWidth * .35))));
  layout.style.setProperty("--crop-width", `${CROPS.width}px`);
  try { localStorage.setItem("mvlabel.cropWidth", String(CROPS.width)); } catch (_) { /* private window */ }
  requestAnimationFrame(() => { layoutTiles(); drawAll(); });
}

function drawCrops() {
  if (!CROPS.open) return;
  const selected = person();
  $("cropPerson").textContent = selected ? `Người ${selected.uid} · frame ${S.frame}` : "Chọn người trên BEV";
  const signature = JSON.stringify(S.cams.map(c => [c.index, c.name]));
  if (signature !== CROPS.signature) {
    CROPS.signature = signature;
    $("cropList").replaceChildren();
    CROPS.cards = S.cams.map(cam => {
      const card = document.createElement("div"); card.className = "crop-card";
      const label = document.createElement("div"); label.className = "crop-label";
      label.textContent = `C${cam.index + 1} · ${cam.name}`;
      const canvas = document.createElement("canvas");
      canvas.setAttribute("aria-label", `Crop ${cam.name}`);
      card.append(label, canvas); $("cropList").append(card);
      return { cam, canvas };
    });
  }
  const views = selected ? S.views[S.data.persons.indexOf(selected)] : null;
  for (const { cam, canvas } of CROPS.cards) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#101216"; ctx.fillRect(0, 0, w, h);
    let message = "Chọn người trên BEV";
    if (selected && S.frame !== null) {
      const box = labelBox(selected, cam.index, views?.[cam.index]);
      message = cameraActive(cam.index) ? "Không có box trong ảnh" : "Camera đã hết video";
      if (box) {
        const img = image(S.frame, cam.index);
        const region = img.complete && cropRegion(box, cam, img);
        message = img.complete ? "Không có ảnh crop" : "Đang tải ảnh…";
        if (region) {
          const [, , cw, ch] = region;
          const scale = Math.min(w / cw, h / ch);
          ctx.drawImage(img, ...region, (w - cw * scale) / 2, (h - ch * scale) / 2, cw * scale, ch * scale);
          continue;
        }
      }
    }
    ctx.fillStyle = "#aeb7c3"; ctx.font = `${12 * dpr}px Segoe UI`;
    ctx.fillText(message, 8 * dpr, Math.min(h - 2, 24 * dpr), Math.max(1, w - 16 * dpr));
  }
}

function wireCrops() {
  $("btnCrops").onclick = () => { setCrops(!CROPS.open); $("btnCrops").blur(); };
  $("btnCloseCrops").onclick = () => setCrops(false);
  const handle = $("cropResize");
  let drag = null;
  handle.onpointerdown = event => {
    if (event.button !== 0) return;
    drag = { x: event.clientX, width: $("cropPanel").getBoundingClientRect().width };
    handle.setPointerCapture(event.pointerId); event.preventDefault();
  };
  handle.onpointermove = event => { if (drag) setCropWidth(drag.width + event.clientX - drag.x); };
  handle.onpointerup = handle.onpointercancel = handle.onlostpointercapture = () => { drag = null; };
  handle.onkeydown = event => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault(); event.stopPropagation();
      setCropWidth(CROPS.width + (event.key === "ArrowRight" ? 20 : -20));
    }
  };
  try {
    const width = Number(localStorage.getItem("mvlabel.cropWidth"));
    if (Number.isFinite(width) && width >= 120) setCropWidth(width);
    setCrops(localStorage.getItem("mvlabel.crops") === "1");
  } catch (_) { setCrops(false); }
}
