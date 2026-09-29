// Interfaz de LSM Coach: recibe el estado por WebSocket y lo dibuja.
const $ = (id) => document.getElementById(id);
const FINGER_RANGE = [0, 180];
const AXIS_RANGE = { roll: [-180, 180], pitch: [-90, 90] };
const AXIS_LABEL = { roll: 'giro', pitch: 'inclinación' };

let catalog = [];       // /api/signs
let state = null;       // último estado del servidor
let ws = null;
let shownTarget = null; // seña cuya tarjeta está dibujada
let lastMessageId = 0;
let toastTimer = null;
let group = loadGroup(); // grupo de señas visible: '1', '2' o 'all'

function loadGroup() {
  try { return localStorage.getItem('lsm-group') || '1'; } catch { return '1'; }
}

// ---------- conexión ----------
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => setPill('st-link', 'ok', 'Servidor');
  ws.onclose = () => {
    setPill('st-link', 'bad', 'Sin servidor');
    setVerdict('wait', 'Reconectando…');
    setTimeout(connect, 1000);
  };
  ws.onmessage = (e) => render(JSON.parse(e.data));
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

async function loadCatalog() {
  catalog = await (await fetch('/api/signs')).json();
  renderSignButtons();
  shownTarget = null;
}

const inGroup = (s) => group === 'all' || String(s.level) === group;

function setGroup(g) {
  group = g;
  try { localStorage.setItem('lsm-group', g); } catch { /* sin almacenamiento: no pasa nada */ }
  renderSignButtons();
}

function renderSignButtons() {
  document.querySelectorAll('#groups button').forEach((b) =>
    b.setAttribute('aria-selected', String(b.dataset.group === group)));
  const box = $('sign-buttons');
  box.classList.toggle('compact', group === 'all');
  box.replaceChildren(...catalog.filter(inGroup).map((s) => {
    const b = document.createElement('button');
    b.innerHTML = `${s.sign}${s.dynamic ? '<span class="mov">↻</span>' : ''}`;
    b.dataset.sign = s.sign;
    b.title = s.descripcion + (s.calibrated ? '' : ' (aún no evaluable)');
    b.classList.toggle('uncal', !s.calibrated);
    b.setAttribute('aria-pressed', String(s.sign === shownTarget));
    b.onclick = () => { chooseSign(s.sign); b.blur(); };  // sin foco: Espacio no lo "presiona"
    return b;
  }));
}

function chooseSign(sign) {
  const info = catalog.find((c) => c.sign === sign);
  if (info && !inGroup(info)) setGroup('all');
  send({ type: 'target', sign });
}

// ---------- dibujo ----------
function render(s) {
  state = s;
  if (!s.target) return;  // el servidor aún no procesa la primera imagen
  if (s.target !== shownTarget) renderTarget(s.target);
  renderVerdict(s);
  renderIssues(s);
  renderShape(s);
  renderFingers(s.fingers || []);
  renderOrientation(s.orientation || []);
  renderStatus(s);
  renderProgress(s.progress || {});
  if (s.message && s.message.id !== lastMessageId) {
    lastMessageId = s.message.id;
    toast(s.message.text, s.message.kind);
  }
}

function renderTarget(sign) {
  shownTarget = sign;
  const info = catalog.find((c) => c.sign === sign) || { sign, descripcion: '', error_tipico: '', shape: {} };
  $('target-letter').textContent = sign;
  $('target-desc').textContent = info.descripcion;
  const ref = $('target-ref');
  ref.hidden = !info.ref;
  if (info.ref) { ref.src = info.ref; ref.alt = `Referencia de la ${sign}`; }
  $('target-typical').textContent = info.error_tipico ? `Error típico: ${info.error_tipico}` : '';
  const names = { pulgar: 'Pulgar', indice: 'Índice', medio: 'Medio', anular: 'Anular', menique: 'Meñique' };
  $('target-shape').replaceChildren(...Object.entries(info.shape).map(([f, how]) => {
    const li = document.createElement('li');
    li.innerHTML = `${names[f]}: <b>${how}</b>`;
    return li;
  }));
  document.querySelectorAll('#sign-buttons button').forEach((b) =>
    b.setAttribute('aria-pressed', String(b.dataset.sign === sign)));
  $('fingers').replaceChildren();  // se reconstruyen con los rangos de la nueva seña
  $('orient').replaceChildren();
}

function setVerdict(kind, text) {
  const v = $('verdict');
  v.className = `verdict ${kind}`;
  v.textContent = text;
}

function renderVerdict(s) {
  if (s.camera_live === false) setVerdict('bad', 'Sin imagen de la cámara: revisa la conexión');
  else if (s.verdict === 'ok') setVerdict('ok', '✓ ¡Seña correcta!');
  else if (s.verdict === 'fix') setVerdict('bad', `✗ Corrige: ${s.failed_parameters.join(' y ').toLowerCase()}`);
  else if (s.verdict === 'uncalibrated') {
    const dyn = catalog.find((c) => c.sign === s.target)?.dynamic;
    setVerdict('wait', dyn ? `La ${s.target} lleva movimiento: pronto se evaluará` : `La seña ${s.target} aún no está calibrada`);
  }
  else setVerdict('wait', 'Muestra tu mano derecha a la cámara');
}

function renderIssues(s) {
  const ul = $('issues');
  let items;
  if (s.verdict === 'ok') {
    items = [li('ok', '✓ Todo bien, mantén la seña')];
  } else if (s.verdict === 'fix') {
    items = s.issues.map((i) => {
      const el = li('', i.action);
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = i.parameter;
      el.prepend(chip);
      return el;
    });
  } else if (s.verdict === 'uncalibrated') {
    const dyn = catalog.find((c) => c.sign === s.target)?.dynamic;
    items = [li('wait', dyn ? 'Mira la animación de referencia para practicarla'
      : 'Graba muestras de esta seña en Calibración')];
  } else {
    items = [li('wait', 'No veo tu mano')];
  }
  ul.replaceChildren(...items);
}

function li(cls, text) {
  const el = document.createElement('li');
  if (cls) el.className = cls;
  el.append(text);
  return el;
}

// Barra con la franja del rango correcto y un marcador con el valor actual
function gaugeRow(container, key, name, value, min, max, ok, [lo, hi], unit = '°') {
  let row = container.querySelector(`[data-key="${key}"]`);
  if (!row) {
    row = document.createElement('div');
    row.className = 'row';
    row.dataset.key = key;
    row.innerHTML = '<span class="name"></span><div class="track"><div class="band"></div><div class="marker"></div></div><span class="val"></span>';
    container.append(row);
  }
  const pct = (x) => `${Math.max(0, Math.min(100, ((x - lo) / (hi - lo)) * 100))}%`;
  row.querySelector('.name').textContent = name;
  const band = row.querySelector('.band');
  band.hidden = min == null;
  if (min != null) {
    band.style.left = pct(min);
    band.style.width = `calc(${pct(max)} - ${pct(min)})`;
  }
  const marker = row.querySelector('.marker');
  marker.hidden = value == null;
  if (value != null) marker.style.left = pct(value);
  row.querySelector('.val').textContent = value == null ? '—' : `${Math.round(value)}${unit}`;
  row.className = `row ${ok == null ? '' : ok ? 'ok' : 'bad'}`;
}

// Qué letra reconoce el clasificador (IA) y qué tan seguro está de la seña objetivo
function renderShape(s) {
  const el = $('shape');
  el.hidden = !s.shape;
  if (!s.shape) return;
  const wrong = s.shape.best !== s.target || s.shape.error >= 0.5;
  el.className = `shape-line ${wrong ? 'bad' : ''}`;
  const err = s.shape.error >= 0.2 ? ` · posible error típico: <b>${Math.round(s.shape.error * 100)}%</b>` : '';
  el.innerHTML = `Forma reconocida: <b>${s.shape.best}</b>${err}`;
}

function renderFingers(fingers) {
  const box = $('fingers');
  if (!fingers.length) {
    box.querySelectorAll('.row').forEach((r) => { r.querySelector('.marker').hidden = true; r.querySelector('.val').textContent = '—'; r.className = 'row'; });
    return;
  }
  for (const f of fingers) gaugeRow(box, f.id, f.label, f.value, f.min, f.max, f.ok, FINGER_RANGE);
}

function renderOrientation(axes) {
  $('orient-block').hidden = !axes.length;
  for (const a of axes) {
    gaugeRow($('orient'), a.axis, AXIS_LABEL[a.axis] || a.axis, a.value, a.min, a.max,
      a.value == null ? false : a.ok, AXIS_RANGE[a.axis] || [-180, 180]);
  }
}

function setPill(id, kind, text) {
  const p = $(id);
  p.className = `pill ${kind}`;
  p.textContent = text;
}

function renderStatus(s) {
  const imu = s.imu || {};
  if (imu.mode === 'none') setPill('st-imu', 'wait', 'Sin muñequera');
  else if (imu.connected) setPill('st-imu', 'ok', `Muñequera · giro ${Math.round(imu.roll)}° incl. ${Math.round(imu.pitch)}°`);
  else setPill('st-imu', 'bad', 'Buscando muñequera…');

  const hand = s.handedness === 'Right' ? ' · mano derecha' : s.handedness === 'Left' ? ' · mano izquierda' : '';
  if (s.camera_live === false) setPill('st-cam', 'bad', 'Cámara sin imagen');
  else setPill('st-cam', s.fps >= 12 ? 'ok' : 'wait', `Cámara ${Math.round(s.fps)} fps${hand}`);

  const sync = s.sync || {};
  if (!sync.cloud) setPill('st-cloud', 'wait', 'Nube: sin configurar');
  else if (sync.error) setPill('st-cloud', 'bad', `Nube: sin conexión (${sync.pending} pendientes)`);
  else if (sync.pending) setPill('st-cloud', 'wait', `Nube: ${sync.pending} por subir`);
  else setPill('st-cloud', 'ok', 'Nube: al día');

  const v2 = $('video2');
  if (s.cameras > 1 && v2.hidden) { v2.src = '/video/1'; v2.hidden = false; }
  if (document.activeElement !== $('person') && s.person) $('person').value = s.person;
}

function renderProgress(progress) {
  $('progress').replaceChildren(...Object.entries(progress).map(([sign, n]) => {
    const el = document.createElement('span');
    el.textContent = `${sign} ✓${n > 1 ? ` ×${n}` : ''}`;
    return el;
  }));
}

function toast(text, kind = 'ok') {
  const t = $('toast');
  t.textContent = text;
  t.className = `toast ${kind}`;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2500);
}

// ---------- voz (sin internet: usa las voces del sistema) ----------
function pickVoice() {
  const voices = speechSynthesis.getVoices();
  return voices.find((v) => v.lang === 'es-MX' && v.localService)
    || voices.find((v) => v.lang.startsWith('es') && v.localService)
    || voices.find((v) => v.lang.startsWith('es'));
}

function speak() {
  let text;
  if (!state || !state.hand) text = 'No veo tu mano';
  else if (state.verdict === 'ok') text = 'Todo bien';
  else if (state.verdict === 'uncalibrated') text = 'Esta seña aún no está calibrada';
  else text = [...new Set(state.issues.map((i) => i.action))].slice(0, 3).join('. ');
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'es-MX';
  const voice = pickVoice();
  if (voice) u.voice = voice;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

// ---------- controles ----------
$('speak').onclick = (e) => { speak(); e.currentTarget.blur(); };
$('calib-toggle').onclick = () => {
  const panel = $('calib');
  panel.hidden = !panel.hidden;
  $('calib-toggle').setAttribute('aria-expanded', String(!panel.hidden));
};
$('person').onchange = (e) => send({ type: 'person', name: e.target.value });
$('rec-ok').onclick = () => send({ type: 'record', is_error: false });
$('rec-err').onclick = () => send({ type: 'record', is_error: true });
$('reload').onclick = async () => { send({ type: 'reload' }); setTimeout(loadCatalog, 300); };
document.querySelectorAll('#groups button').forEach((b) => {
  b.onclick = () => { setGroup(b.dataset.group); b.blur(); };
});

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); speak(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const key = e.key.toUpperCase();
  if (catalog.some((c) => c.sign === key)) chooseSign(key);  // la tecla de la letra la elige
});

loadCatalog().then(connect);
