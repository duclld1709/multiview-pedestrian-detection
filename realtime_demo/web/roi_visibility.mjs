import {defaultClosedDoors, groundVisible, insidePolygon, obstacles, project} from '../../map_visualization/physics.mjs';

const SAMPLE_SPACING = 0.005; // metres
const FLOOR_TOLERANCE = 0.00001; // include points on the floor outline

function onFloor(point, polygon) {
  if (insidePolygon(point, polygon)) return true;
  const [x, y] = point;
  return polygon.some(([ax, ay], index) => {
    const [bx, by] = polygon[(index + 1) % polygon.length];
    const dx = bx - ax, dy = by - ay;
    const length = Math.hypot(dx, dy);
    if (!length) return false;
    const along = ((x - ax) * dx + (y - ay) * dy) / length;
    const across = ((y - ay) * dx - (x - ax) * dy) / length;
    return along >= -FLOOR_TOLERANCE && along <= length + FLOOR_TOLERANCE && Math.abs(across) <= FLOOR_TOLERANCE;
  });
}

function visibleRuns(length, spacing, visible) {
  const steps = Math.max(1, Math.ceil(length / spacing));
  const runs = [];
  let wasVisible = visible(0);
  let start = wasVisible ? 0 : null;
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    const isVisible = visible(t);
    if (wasVisible !== isVisible) {
      let lo = (i - 1) / steps, hi = t;
      // Keep the endpoint on the visible side of the transition.
      for (let pass = 0; pass < 12; pass++) {
        const mid = (lo + hi) / 2;
        if (visible(mid) === wasVisible) lo = mid;
        else hi = mid;
      }
      if (isVisible) start = hi;
      else if (lo > start) runs.push([start, lo]);
    }
    wasVisible = isVisible;
  }
  if (wasVisible && start < 1) runs.push([start, 1]);
  return runs;
}

function gridCoordinates(min, max, pitch) {
  const values = [];
  if (!Number.isFinite(pitch) || pitch <= 0) return values;
  for (let i = Math.floor(min / pitch) + 1; i * pitch < max - FLOOR_TOLERANCE; i++) {
    const coordinate = i * pitch;
    if (coordinate > min + FLOOR_TOLERANCE) values.push(coordinate);
  }
  return values;
}

export function cameraRoiSegments({cameras, roi, site, pitch = []}, spacing = SAMPLE_SPACING) {
  const {x0, y0, x1, y1} = roi;
  const corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  const solids = obstacles(site, defaultClosedDoors(site));
  const floors = site.floor_regions.map(floor => floor.points);
  const border = corners.map((point, i) => [point, corners[(i + 1) % corners.length]]);
  const [pitchX, pitchY] = pitch;
  const grid = [
    ...gridCoordinates(x0, x1, pitchX).map(x => [[x, y0], [x, y1]]),
    ...gridCoordinates(y0, y1, pitchY).map(y => [[x0, y], [x1, y]]),
  ];
  function visibleSegments(cam, [[x, y], [endX, endY]]) {
    const dx = endX - x, dy = endY - y;
    const point = t => [x + t * dx, y + t * dy, 0];
    const visible = t => {
      const p = point(t);
      return groundVisible(cam, p, solids) && floors.some(floor => onFloor(p, floor));
    };
    const length = Math.hypot(dx, dy);
    return visibleRuns(length, spacing, visible).flatMap(([from, to]) => {
      const steps = cam.dist ? Math.max(1, Math.ceil((to - from) * length / .25)) : 1;
      const points = Array.from({length: steps + 1}, (_, i) => project(cam, point(from + (to - from) * i / steps)).uv);
      return points.slice(1).map((end, i) => [points[i], end]);
    });
  }
  return cameras.map(cam => ({
    border: border.map(edge => visibleSegments(cam, edge)),
    grid: grid.map(edge => visibleSegments(cam, edge)),
  }));
}
