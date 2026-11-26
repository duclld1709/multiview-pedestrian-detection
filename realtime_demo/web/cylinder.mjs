import {project} from '../../map_visualization/physics.mjs';

// Same calibrated surface construction as label_tools.draw_cylinder_3d.
export function cylinderGeometry(cam, {x, y}) {
  const radius = .25, height = 1.8, segments = 64, levels = 6;
  const dx = cam.center[0] - x, dy = cam.center[1] - y;
  const distance = Math.hypot(dx, dy);
  if (distance <= radius) return null;
  const rings = [];
  for (let level = 0; level < levels; level++) {
    const ring = [];
    for (let k = 0; k <= segments; k++) {
      const angle = k * 2 * Math.PI / segments;
      const {uv, validRay} = project(cam, [x + radius * Math.cos(angle),
        y + radius * Math.sin(angle), level * height / (levels - 1)]);
      if (!validRay || !uv.every(Number.isFinite)) return null;
      ring.push(uv);
    }
    rings.push(ring);
  }
  const lightAngle = 50 * Math.PI / 180;
  const light = [(dx * Math.cos(lightAngle) - dy * Math.sin(lightAngle)) / distance,
    (dx * Math.sin(lightAngle) + dy * Math.cos(lightAngle)) / distance];
  const facing = [], shades = [];
  for (let k = 0; k < segments; k++) {
    const angle = (k + .5) * 2 * Math.PI / segments;
    const nx = Math.cos(angle), ny = Math.sin(angle);
    const vx = dx - radius * nx, vy = dy - radius * ny;
    const cosine = (nx * vx + ny * vy) / Math.hypot(vx, vy);
    facing.push(cosine > 0);
    shades.push(.35 + .45 * Math.max(0, nx * light[0] + ny * light[1]) + .35 * cosine ** 2);
  }
  return {rings, facing, shades, topVisible: cam.center[2] > height};
}

export function drawCylinder(ctx, cam, person, color, sx, sy) {
  const geometry = cylinderGeometry(cam, person);
  if (!geometry) return;
  const {rings, facing, shades, topVisible} = geometry;
  const rgb = [1, 3, 5].map(offset => parseInt(color.slice(offset, offset + 2), 16));
  const tint = shade => `rgb(${rgb.map(value => Math.min(255, Math.round(value * shade))).join(',')})`;
  const path = (points, closed = false) => {
    ctx.beginPath();
    points.forEach(([x, y], i) => ctx[i ? 'lineTo' : 'moveTo'](x * sx, y * sy));
    if (closed) ctx.closePath();
  };
  ctx.save();
  ctx.globalAlpha = .38;
  for (let k = 0; k < facing.length; k++) {
    if (!facing[k]) continue;
    path([...rings.map(ring => ring[k]), ...rings.map(ring => ring[k + 1]).reverse()], true);
    ctx.fillStyle = tint(shades[k]);
    ctx.fill();
  }
  if (topVisible) {
    path(rings.at(-1), true);
    ctx.fillStyle = tint(1.25);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  for (const level of [0, rings.length - 1]) {
    for (let k = 0; k < facing.length; k++) {
      if (facing[k] || (level > 0 && topVisible) || k % 2 === 0) {
        path(rings[level].slice(k, k + 2));
        ctx.stroke();
      }
    }
  }
  for (let k = 0; k < facing.length; k++) {
    if (facing[k] !== facing[(k + 1) % facing.length]) {
      path(rings.map(ring => ring[k + 1]));
      ctx.stroke();
    }
  }
  ctx.restore();
}
