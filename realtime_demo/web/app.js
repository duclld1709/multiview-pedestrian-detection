import {cameraRoiSegments} from './roi_visibility.mjs';
import {bodyVisible, obstacles} from '/map_visualization/physics.mjs';

const $=id=>document.getElementById(id);
let scene,state,renderedKey='',rendering=false,lastPeople=[];
let renderedGeneration;
let sceneRevision, datasetPending=false, cachePending=false;
let refreshedOnOpen = false;
let shownDatasetRoot, shownCacheDirectory, datasetListKey, sourceListKey;
let clockAnchor=0, clockAnchorAt=performance.now();
function acceptState(value){
  if (state && value.scene_revision < state.scene_revision) return;
  if (value.phase === 'error' && state?.phase !== 'error') openMenu($('settings-menu'));
  state=value;clockAnchor=value.playhead_s||0;clockAnchorAt=performance.now();
}
const cards=new Map();
const formatTime=t=>{const m=Math.floor(t/60),s=Math.floor(t%60),ms=Math.floor((t%1)*1000);return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(ms).padStart(3,'0')}`};
const fixed=(n,d=1)=>Number.isFinite(n)?n.toFixed(d):'—';
async function api(path,body){const r=await fetch('/api/'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await r.json();if(!r.ok)throw Error(data.error||r.statusText);return data;}
let cameraPending = false;
let mp4OptionsPending = false;
let controlError = '';

async function action(path, body={}) {
  const scenarioAction = path.startsWith('scenarios/');
  const setupAction = path === 'datasets/scan' && $('dataset-setup').open;
  controlError = '';
  if (scenarioAction) $('scenario-error').hidden = true;
  try {
    acceptState(await api(path, {...body, scene_revision:state?.scene_revision}));
    updateControls();
    await syncScene();
    return true;
  } catch (error) {
    const message = setupAction ? $('dataset-setup-error') : scenarioAction ? $('scenario-error') : $('message');
    message.textContent = error.message;
    message.hidden = false;
    if (!setupAction) {
      if (!scenarioAction) controlError = error.message;
      openMenu($(scenarioAction ? 'scenario-menu' : 'settings-menu'));
    }
    return false;
  }
}

function makeCards() {
  for (const cam of scene.cameras) {
    const div = document.createElement('div');
    div.className = 'camera-card';
    div.dataset.name = cam.name;
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 360;
    const tag = document.createElement('span');
    tag.className = 'camera-tag';
    tag.textContent = cam.name.replace('Cam', 'C');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'camera-toggle';
    button.dataset.camera = cam.id;
    button.textContent = 'Off';
    button.disabled = true;
    button.onclick = async () => {
      const selected = state.selected.includes(cam.id)
        ? state.selected.filter(id => id !== cam.id)
        : [...state.selected, cam.id];
      cameraPending = true;
      updateCameraButtons();
      try {
        await action('config', {cameras: selected});
      } finally {
        cameraPending = false;
        updateCameraButtons();
      }
    };
    div.append(canvas, tag, button);
    $('cameras').append(div);
    cards.set(cam.id, {div, canvas, button, image: null});
  }
  drawCameras();
  drawBev([]);
}

async function syncScene() {
  if (sceneRevision === state.scene_revision) return;
  const value = await api('scene');
  if (value.scene_revision !== state.scene_revision || value.scene_revision === sceneRevision) return;
  sceneRevision = value.scene_revision;
  scene = value.cameras.length ? value : null;
  cards.clear();
  $('cameras').replaceChildren();
  $('bev').getContext('2d').clearRect(0, 0, $('bev').width, $('bev').height);
  document.querySelector('.workspace').dataset.dataset = state.dataset || '';
  if (state.dataset) $('model').replaceChildren(...value.checkpoints.map(name => new Option(name, name)));
  else $('model').replaceChildren();
  $('model').value = state.checkpoint || value.default_checkpoint || '';
  mapKey = mapImage = renderedGeneration = undefined;
  renderedKey = '';
  lastPeople = [];
  showScenarioDraft('');
  if (scene) {
    roiPaths = makeRoiPaths(scene);
    const wallsAndPillars = scene.site.objects.filter(object =>
      object.kind === 'pillar' || (object.kind === 'wall' && object.role !== 'door'));
    bboxOccluders = obstacles({objects: wallsAndPillars});
    makeCards();
  }
  updateControls();
}

const compactMap = matchMedia('(max-width: 650px) and (orientation: portrait)');
let mapKey, mapImage;
let roiPaths;
let bboxOccluders;

function makeRoiPaths(scene) {
  const spacing = state.dataset === 'wavetrack' ? undefined : .04;
  return cameraRoiSegments(scene, spacing).map(({border, grid}, i) => {
    const [width, height] = scene.cameras[i].size;
    const sx = 640 / width, sy = 360 / height;
    const pathFor = edges => {
      const path = new Path2D();
      for (const edge of edges) {
        for (const [[x0, y0], [x1, y1]] of edge) {
          path.moveTo(x0 * sx, y0 * sy);
          path.lineTo(x1 * sx, y1 * sy);
        }
      }
      return path;
    };
    return {border: pathFor(border), grid: pathFor(grid)};
  });
}

function drawCameras() {
  if (!scene) return;
  const showOccluded = $('show-occluded-bbox').checked;
  for (const cam of scene.cameras) {
    const {canvas, image} = cards.get(cam.id);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (image) {
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = '#101b2b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#6d849d';
      ctx.textAlign = 'center';
      ctx.font = '14px Segoe UI';
      ctx.fillText('Waiting for video frames', 320, 185);
    }
    const sx = canvas.width / cam.size[0], sy = canvas.height / cam.size[1];
    const {border, grid} = roiPaths[cam.id];
    if ($('show-grid').checked) {
      ctx.strokeStyle = '#ff666680';
      ctx.lineWidth = 1;
      ctx.stroke(grid);
    }
    if ($('show-roi').checked) {
      ctx.strokeStyle = '#56dfbc';
      ctx.lineWidth = 2;
      ctx.stroke(border);
    }
    if (!$('show-bbox').checked) continue;
    ctx.lineWidth = 2;
    ctx.font = '11px Segoe UI';
    ctx.textAlign = 'left';
    for (const person of lastPeople) {
      const box = person.boxes[cam.id];
      if (!box || (!showOccluded && !bodyVisible(cam, person, box, bboxOccluders))) continue;
      const [left, top, right, bottom] = box;
      const x = left * sx, y = top * sy;
      ctx.strokeStyle = person.inside ? '#ffbf69' : '#aabbd0';
      ctx.strokeRect(x, y, (right - left) * sx, (bottom - top) * sy);
      const labelY = Math.max(14, y);
      ctx.fillStyle = '#0a101bda';
      ctx.fillRect(x, labelY - 15, 48, 15);
      ctx.fillStyle = '#ffdbab';
      ctx.fillText(Math.round(person.score * 100) + '%', x + 4, labelY - 4);
    }
  }
}

function drawBev(people = lastPeople) {
  if (!scene) return;
  const selected = state?.selected ?? scene.cameras.map(cam => cam.id);
  const cameraMask = selected.reduce((mask, id) => mask + 2 ** id, 0);
  if (!cameraMask) return;
  const key = [+$('show-roi').checked, +!compactMap.matches, cameraMask].join('/') + '?scene=' + sceneRevision;
  if (key !== mapKey) {
    mapKey = key;
    const image = new Image();
    image.onload = () => {
      if (mapKey !== key) return;
      mapImage = image;
      drawBev();
    };
    image.src = '/api/site-map/' + key;
  }
  if (!mapImage) return;

  const canvas = $('bev'), ctx = canvas.getContext('2d');
  const {extent, padding, scale} = scene.map;
  const [xMin, , , yMax] = scene.site.bounds;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(mapImage, 0, 0);
  ctx.save();
  ctx.beginPath();
  ctx.rect(padding, padding, extent, extent);
  ctx.clip();
  for (const person of people) {
    const x = Math.round(padding + (person.x - xMin) * scale);
    const y = Math.round(padding + (yMax - person.y) * scale);
    ctx.beginPath();
    ctx.arc(x, y, 6.5, 0, 2 * Math.PI);
    ctx.fillStyle = person.inside ? '#ed9744' : '#9bafc6';
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#eff3f4';
    ctx.stroke();
  }
  ctx.restore();
}
compactMap.addEventListener('change', () => drawBev());
function updateCameraButtons() {
  const canChange = state.phase === 'paused' && state.source_ready && state.camera_toggles && !cameraPending;
  for (const [id, {div, button}] of cards) {
    const active = state.selected.includes(id);
    const lastActive = active && state.selected.length === 1;
    button.disabled = !canChange || lastActive;
    button.textContent = active ? 'Off' : 'On';
    button.setAttribute('aria-label', (active ? 'Turn off ' : 'Turn on ') + div.dataset.name);
    button.title = !state.camera_toggles ? 'This checkpoint uses all cameras'
      : lastActive ? 'Keep at least one camera on' : button.getAttribute('aria-label');
    div.classList.toggle('off', !active);
  }
}

function updateBenchmark() {
  const rows = state.benchmark;
  $('benchmark-summary').textContent = rows.length
    ? 'Selected batch size ' + state.batch + ' · ' + sourceNames[rows[0].source] +
      ' · Prefetch ' + (rows[0].prefetch ? 'on' : 'off')
    : 'No results yet';
  $('benchmark-rows').replaceChildren();
  for (const row of rows) {
    const tr = $('benchmark-rows').insertRow();
    if (row.batch === state.batch) tr.className = 'chosen';
    const values = row.error
      ? [row.batch, row.error, '—', '—', '—', '—']
      : [row.batch, fixed(row.fps), fixed(row.median_ms) + ' ms', fixed(row.p95_ms) + ' ms',
         fixed(row.input_span_ms) + ' ms', row.cameras];
    for (const value of values) tr.insertCell().textContent = value;
  }
}

const sourceNames = {cache:'FP16 cache', mp4:'Direct MP4', images:'Direct images'};
let cacheSourceKey;
function syncFolderInput(id, previous, value) {
  if (previous !== value) $(id).value = value || '';
  return value;
}

function updateCacheControls(idle) {
  const {dataset, cache_directory, cache_media: media, cache_fps, cache_progress: progress, cache_build} = state;
  $('cache-form').hidden = !media;
  if (!media) {
    cacheSourceKey = undefined;
    return;
  }
  $('cache-fields').disabled = !idle || cachePending;
  shownCacheDirectory = syncFolderInput('cache-directory', shownCacheDirectory, cache_directory);
  const key = [dataset, media.kind, media.fps].join('/');
  if (key !== cacheSourceKey) {
    $('cache-fps').max = media.fps;
    $('cache-fps').value = Math.min(15, media.fps);
    cacheSourceKey = key;
  }
  const fps = $('cache-fps').valueAsNumber;
  const valid = Number.isFinite(fps) && 0 < fps && fps <= media.fps;
  $('cache-generate').disabled = !(state.phase === 'paused' && state.source_ready && valid);
  const sets = Math.ceil(media.total / media.fps * fps);
  const input = state.input_size;
  let size = '';
  if (valid && input) {
    const megabytes = sets * input[0] * input[1] * media.cameras * 6 / 1e6;
    const volume = megabytes < 1000 ? fixed(megabytes) + ' MB' : fixed(megabytes / 1000, 2) + ' GB';
    size = ' · ' + volume + ' + JPEGs';
  }
  $('cache-estimate').textContent = 'Full dataset · ' + sourceNames[media.kind] + ' · Max ' + media.fps + ' FPS' + size;
  $('cache-ready').textContent = cache_fps ? 'Cache available at ' + cache_fps + ' FPS' : '';
  const building = state.phase === 'caching';
  $('cache-job').hidden = !building;
  $('cache-cancel').disabled = cache_build === 'cancelling';
  $('cache-progress').value = progress.total ? 100 * progress.completed / progress.total : 0;
  $('cache-job-label').textContent = (cache_build === 'cancelling' ? 'Cancelling · ' : 'Generating · ') +
    progress.completed + ' / ' + progress.total + ' frame sets' +
    (progress.fps > 0 ? ' · ' + fixed(progress.fps) + ' sets/s' : '');
}

function updateDatasetControls(idle, phase) {
  const {datasets, dataset, dataset_root, dataset_errors, needs_dataset_folder, legacy} = state;
  const active = datasets.find(item => item.id === dataset);
  const picker = $('dataset-select');
  const key = JSON.stringify(datasets);
  if (key !== datasetListKey) {
    picker.replaceChildren(...datasets.map(({id, name}) => new Option(name, id)));
    datasetListKey = key;
  }
  picker.value = dataset || '';
  picker.disabled = !datasets.length || datasetPending || !(idle || phase === 'running');
  $('dataset-menu').hidden = legacy || !dataset_root;
  $('dataset-label').textContent = active?.name || 'Dataset';
  $('dataset-form').hidden = legacy;
  $('dataset-fields').disabled = datasetPending || !(idle || phase === 'running');
  shownDatasetRoot = syncFolderInput('dataset-root', shownDatasetRoot, dataset_root);
  $('dataset-scan-report').textContent = Object.entries(dataset_errors)
    .map(([name, message]) => name + ' — ' + message).join('\n');
  if (needs_dataset_folder && !$('dataset-setup').open) $('dataset-setup').showModal();
  if (!needs_dataset_folder && $('dataset-setup').open) $('dataset-setup').close();

  updateCacheControls(idle);

  const sourceKey = state.sources.join(',');
  if (sourceKey !== sourceListKey) {
    $('source').replaceChildren(...state.sources.map(source => new Option(sourceNames[source], source)));
    sourceListKey = sourceKey;
  }
}

function updateControls() {
  const {phase, source_ready: ready, metrics} = state;
  const paused = phase === 'paused';
  const idle = ['idle', 'paused', 'error'].includes(phase);
  updateDatasetControls(idle, phase);
  $('message').textContent = controlError || state.message;
  $('model').disabled = !idle || !state.dataset || !$('model').options.length;
  $('load').disabled = $('model').disabled;
  if (state.checkpoint && state.checkpoint !== shownCheckpoint) {
    $('model').value = state.checkpoint;
    shownCheckpoint = state.checkpoint;
  }
  $('model-fallback').hidden = !state.using_wave_weights;
  if (state.using_wave_weights) $('model-fallback').textContent = 'Using WaveTrack weights for this dataset';
  $('source').disabled = !idle || !state.dataset || mp4OptionsPending;
  $('source').value = state.source;
  updateMp4Options();
  $('benchmark').disabled = !paused || !ready || mp4OptionsPending;
  const toggle = $('toggle');
  const running = phase === 'running';
  const playbackAction = running ? 'Pause inference' : 'Resume inference';
  toggle.disabled = !(running || (paused && ready)) || mp4OptionsPending;
  toggle.classList.toggle('playing', running);
  toggle.setAttribute('aria-label', playbackAction);
  toggle.title = playbackAction;
  const timeline = $('timeline');
  timeline.disabled = !paused || !ready;
  timeline.min = state.timeline_start;
  timeline.max = Math.max(state.timeline_start, state.timeline_end - 1);
  if (document.activeElement !== timeline) {
    const frame = state.mode === 'realtime' ? Math.floor(state.playhead_s * state.source_fps) : state.next_frame;
    timeline.value = Math.max(Number(timeline.min), Math.min(Number(timeline.max), frame));
  }
  paintTimeline();
  $('batch').disabled = !paused || !ready || state.mode === 'realtime';
  $('playback-mode').disabled = !paused || !ready;
  $('playback-mode').value = state.mode;
  $('batch').value = state.batch;
  updateCameraButtons();

  $('duration').textContent = '/ ' + formatTime(state.timeline_end_s);
  $('fps').textContent = fixed(metrics.delivered_fps);
  $('gpu').textContent = fixed(metrics.infer_ms) + ' ms';
  $('cache-meta').textContent = ready
    ? state.input_size.join(' × ') + ' · ' + state.total_frames + ' frame sets · ' + state.source_fps + ' FPS · ' +
      (state.source === 'cache' ? fixed(state.cache_size_gb, 2) + ' GB cache' : sourceNames[state.source])
    : '';
  updateScenarios();
  if ($('benchmark-panel').open) updateBenchmark();
}

async function render() {
  if (rendering || !scene || sceneRevision !== state?.scene_revision) return;
  const {latest, selected, generation} = state;
  if (generation !== renderedGeneration) {
    renderedGeneration = generation;
    lastPeople = [];
    for (const card of cards.values()) card.image = null;
    drawCameras();
    drawBev();
    $('position').textContent = '';
  }
  if (!state.source_ready) return;
  const frame = latest?.frame ?? Math.min(state.next_frame, state.timeline_end - 1);
  const key = [generation, frame, selected.join(','), !!latest].join(':');
  if (key === renderedKey) return;
  rendering = true;
  try {
    const loaded = await Promise.all(selected.map(async id => {
      const image = new Image();
      image.src = '/api/frame/' + generation + '/' + frame + '/' + id;
      await image.decode();
      return [id, image];
    }));
    if (state.generation !== generation) return;
    for (const [id, image] of loaded) cards.get(id).image = image;
    lastPeople = latest?.people || [];
    renderedKey = key;
    drawCameras();
    drawBev();
    if (state.mode !== 'realtime') $('time').textContent = formatTime(frame / state.source_fps);
    $('position').textContent = latest
      ? (frame + 1) + ' / ' + state.total_frames + ' frame sets · Prediction at ' + formatTime(latest.time_s)
      : 'No predictions yet';
  } catch (error) {
    if (state.generation === generation) console.warn('Frame display:', error.message);
  } finally {
    rendering = false;
  }
}

let shownCheckpoint;
$('load').onclick = () => action('load', {checkpoint: $('model').value});
$('source').onchange = () => action('source', {source: $('source').value});
$('dataset-settings').onclick = event => {
  event.stopPropagation();
  openMenu($('settings-menu'));
  $('dataset-root').focus();
};
$('dataset-select').onchange = async () => {
  datasetPending = true;
  try {
    const dataset = $('dataset-select').value;
    updateControls();
    await action('datasets/select', {dataset});
  } finally {
    datasetPending = false;
    updateControls();
  }
};
async function scanDataset(event, input) {
  event.preventDefault();
  datasetPending = true;
  $('dataset-setup-error').hidden = true;
  $('dataset-setup-scan').disabled = true;
  updateControls();
  try {
    if (await action('datasets/scan', {directory: $(input).value}) && !state.datasets.length) {
      openMenu($('settings-menu'));
    }
  } finally {
    datasetPending = false;
    $('dataset-setup-scan').disabled = false;
    updateControls();
  }
}
$('dataset-setup').addEventListener('cancel', event => event.preventDefault());
$('dataset-setup-form').onsubmit = event => scanDataset(event, 'dataset-setup-root');
$('dataset-form').onsubmit = event => scanDataset(event, 'dataset-root');
$('cache-form').onsubmit = async event => {
  event.preventDefault();
  const generate = event.submitter === $('cache-generate');
  const directory = $('cache-directory').value;
  const fps = $('cache-fps').valueAsNumber;
  cachePending = true;
  updateControls();
  try {
    await (generate ? action('cache/generate', {directory, fps}) : action('cache-directory', {directory}));
  } finally {
    cachePending = false;
    updateControls();
  }
};
$('cache-fps').oninput = () => updateCacheControls(['idle', 'paused', 'error'].includes(state.phase));
$('cache-cancel').onclick = () => action('cache/cancel');

function updateMp4Options() {
  const editable = ['idle', 'paused'].includes(state.phase);
  $('mp4-remap-label').hidden = state.source !== 'mp4' || !state.can_fuse_remap;
  $('mp4-options').disabled = !editable || mp4OptionsPending;
  if (!mp4OptionsPending) {
    $('mp4-fused-remap').checked = state.mp4_options.fused_remap;
    $('mp4-prefetch').checked = state.mp4_options.prefetch;
  }
}

async function changeMp4Options() {
  const options = {fused_remap: $('mp4-fused-remap').checked, prefetch: $('mp4-prefetch').checked};
  mp4OptionsPending = true;
  updateControls();
  try {
    await action('mp4-options', options);
  } finally {
    mp4OptionsPending = false;
    updateControls();
  }
}

$('mp4-fused-remap').onchange = $('mp4-prefetch').onchange = changeMp4Options;
$('benchmark').onclick = () => action('benchmark');
$('toggle').onclick = () => action(state.phase === 'running' ? 'pause' : 'resume');
function paintTimeline() {
  const timeline = $('timeline');
  const span = Number(timeline.max) - Number(timeline.min);
  const percent = span > 0 ? 100 * (timeline.value - timeline.min) / span : 0;
  timeline.style.setProperty('--played', percent + '%');
}
$('timeline').addEventListener('input', () => {
  paintTimeline();
  $('time').textContent = formatTime(+$('timeline').value / state.source_fps);
});
$('timeline').addEventListener('change', () => action('config', {frame: +$('timeline').value}));
$('batch').onchange = () => action('config', {batch: +$('batch').value});
$('playback-mode').onchange = () => action('config', {mode: $('playback-mode').value});
$('show-bbox').onchange = drawCameras;
$('show-occluded-bbox').onchange = drawCameras;
$('show-roi').onchange = () => { drawCameras(); drawBev(); };
$('show-grid').onchange = drawCameras;

let scenarioListKey;
let scenarioPending = false;

function showScenarioDraft(id) {
  const scenario = state.scenarios.find(item => item.id === id);
  $('scenario-list').value = scenario?.id || '';
  $('scenario-name').value = scenario?.name || '';
  $('scenario-start').value = scenario?.start_s ?? '';
  $('scenario-end').value = scenario?.end_s ?? '';
  $('scenario-error').hidden = true;
  updateScenarioButtons();
}

function updateScenarioButtons() {
  const id = $('scenario-list').value;
  $('scenario-fields').disabled = state.phase !== 'paused' || !state.source_ready || scenarioPending;
  $('scenario-save').textContent = id ? 'Save' : 'Add';
  $('scenario-delete').disabled = !id;
  $('scenario-enter').disabled = !id;
  $('scenario-exit').disabled = !state.active_scenario;
}

function updateScenarios() {
  const list = $('scenario-list');
  const key = JSON.stringify(state.scenarios);
  if (key !== scenarioListKey) {
    const id = list.value;
    list.replaceChildren(new Option('New scenario', ''));
    for (const scenario of state.scenarios) list.add(new Option(scenario.name, scenario.id));
    if (state.scenarios.some(scenario => scenario.id === id)) list.value = id;
    else if (id) showScenarioDraft('');
    scenarioListKey = key;
  }
  const active = state.active_scenario;
  $('scenario-menu').classList.toggle('active', !!active);
  $('scenario-active').textContent = active ? active.name : 'Free playback';
  $('scenario-menu').querySelector('summary').title = $('scenario-active').textContent;
  for (const id of ['scenario-start', 'scenario-end']) $(id).max = state.video_duration_s;
  updateScenarioButtons();
}

async function scenarioAction(operation, body) {
  scenarioPending = true;
  updateScenarioButtons();
  try {
    return await action('scenarios/' + operation, body);
  } finally {
    scenarioPending = false;
    updateScenarioButtons();
  }
}

$('scenario-list').onchange = () => showScenarioDraft($('scenario-list').value);
$('scenario-form').onsubmit = async event => {
  event.preventDefault();
  const id = $('scenario-list').value;
  const saved = await scenarioAction('save', {id: id || null, name: $('scenario-name').value,
    start_s: $('scenario-start').valueAsNumber, end_s: $('scenario-end').valueAsNumber});
  if (saved) showScenarioDraft(id || state.scenarios.at(-1).id);
};
$('scenario-delete').onclick = async () => {
  if (await scenarioAction('delete', {id: $('scenario-list').value})) showScenarioDraft('');
};
$('scenario-enter').onclick = () => scenarioAction('select', {id: $('scenario-list').value});
$('scenario-exit').onclick = () => scenarioAction('select', {id: null});

const menus = [...document.querySelectorAll('.menu')];
function openMenu(active) {
  for (const menu of menus) if (menu !== active) menu.open = false;
  if (active) active.open = true;
}
$('benchmark-panel').addEventListener('toggle', () => {
  if (state && $('benchmark-panel').open) updateBenchmark();
});
document.addEventListener('click', event => {
  const active = event.target.closest('.menu');
  for (const menu of menus) if (menu !== active) menu.open = false;
});
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  const active = menus.find(menu => menu.open);
  if (active) {
    openMenu(null);
    active.querySelector('summary').focus();
  }
});

async function poll() {
  try {
    acceptState(await api('state'));
    updateControls();
    await syncScene();
    await render();
    if (!refreshedOnOpen) {
      refreshedOnOpen = true;
      if (!state.legacy && state.dataset_root) void action('datasets/refresh');
    }
  } catch (error) {
    $('message').textContent = 'Unable to update the demo. ' + error.message;
    openMenu($('settings-menu'));
  }
  setTimeout(poll, state?.phase === 'running' ? 25 : 300);
}
poll();

function animateClock(){
  if(state?.mode==='realtime' && state.source_ready && document.activeElement!==$('timeline')){
    const duration=state.timeline_end_s;
    const elapsed=state.clock_running?(performance.now()-clockAnchorAt)/1000:0;
    const t=Math.min(duration,clockAnchor+elapsed);
    $('time').textContent=formatTime(t);
    $('timeline').value=Math.min(state.timeline_end-1,Math.floor(t*state.source_fps));
    paintTimeline();
  }
  requestAnimationFrame(animateClock);
}
requestAnimationFrame(animateClock);
