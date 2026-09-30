// Interfaz de LSM Coach: recibe el estado por WebSocket (10 Hz) y lo dibuja.
// Sin dependencias: corre igual en la Mac y en la UNO Q (Chromium en modo quiosco).
const $ = (id) => document.getElementById(id);

// Animaciones con Motion (motion.dev, copia local en web/vendor). Si no cargó, no se anima
// nada pero la app sigue funcionando igual.
const M = window.Motion || null;
const animate = (el, keyframes, options) => { try { return M ? M.animate(el, keyframes, options) : null; } catch { return null; } };
const stagger = (s) => (M && M.stagger ? M.stagger(s) : 0);
const SPRING = { type: 'spring', stiffness: 420, damping: 20, mass: .8 };
const FINGER_RANGE = [0, 180];
const AXIS_RANGE = { roll: [-180, 180], pitch: [-90, 90] };
const AXIS_LABEL = { roll: 'giro', pitch: 'inclinación' };
const RING = 283;                 // perímetro del anillo (2π·45)
const WELCOME_AFTER_MS = 2500;    // sin mano este tiempo -> pantalla de bienvenida
const AUTO_VOICE_PERSIST_MS = 2200;  // el error debe durar esto para decirlo en voz alta
const AUTO_VOICE_GAP_MS = 6000;      // y no se repite antes de este tiempo
const NEXT_SIGN_DELAY_MS = 1800;     // en práctica: pausa para la celebración antes de la siguiente
const WORD_NEXT_DELAY_MS = 900;      // al deletrear una palabra: pausa más corta entre letras
// Palabras del Nivel 3 del reto (vocabulario funcional). Por ahora se practican deletreadas
// con las letras estáticas; la seña propia de cada palabra (con ubicación y movimiento) va aparte.
const WORDS = ['HOLA', 'GRACIAS', 'POR FAVOR', 'AYUDA', 'MAMÁ'];
// letras a deletrear: sin espacios y sin acentos (la Á se deletrea como A)
const wordLetters = (w) => w.toUpperCase().replace(/Á/g, 'A').replace(/[^A-ZÑ]/g, '').split('');

let catalog = [];
let state = null;
let ws = null;
let shownTarget = null;
let lastMessageId = 0, lastAchievementId = 0, lastAlertId = 0;
let toastTimer = null;
let group = loadPref('lsm-group', '1');
let voiceAuto = loadPref('lsm-voice', '1') === '1';
let lastVerdict = null;
let noHandSince = null;
let voice = { action: null, since: 0, lastSpoken: 0 };
let practice = null;   // {signs, i, stats: {sign: {start, alerts, seconds, done}}}
let audioCtx = null;

function loadPref(key, def) { try { return localStorage.getItem(key) ?? def; } catch { return def; } }
function savePref(key, val) { try { localStorage.setItem(key, val); } catch { /* sin almacenamiento */ } }

// ---------- conexión ----------
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => setPill('st-link', 'ok', 'En línea');
  ws.onclose = () => {
    setPill('st-link', 'bad', 'Sin servidor');
    setVerdict('wait', '…', 'Reconectando…');
    setTimeout(connect, 1000);
  };
  ws.onmessage = (e) => render(JSON.parse(e.data));
}
function send(msg) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg)); }

async function loadCatalog() {
  catalog = await (await fetch('/api/signs')).json();
  renderSignButtons();
  renderPath();
  shownTarget = null;
}
const inGroup = (s) => s.word ? group === 'words' : group === 'all' || String(s.level) === group;
const level1 = () => catalog.filter((s) => s.level === 1).map((s) => s.sign);
const isWord = (sign) => !!catalog.find((c) => c.sign === sign)?.word;

function setGroup(g) {
  group = g;
  savePref('lsm-group', g);
  renderSignButtons();
}

function renderSignButtons() {
  document.querySelectorAll('#groups button').forEach((b) =>
    b.setAttribute('aria-selected', String(b.dataset.group === group)));
  const box = $('sign-buttons');
  box.classList.toggle('compact', group === 'all');
  const progress = (state && state.progress) || {};
  if (group === 'words') {  // Nivel 3: la seña propia de cada palabra (o su deletreo, desde la tarjeta)
    const words = catalog.filter((c) => c.word);
    box.replaceChildren(...(words.length ? words : WORDS.map((w) => ({ sign: w }))).map((s) => {
      const b = document.createElement('button');
      b.className = 'word';
      b.textContent = s.sign;
      b.title = s.calibrated ? `Seña de ${s.sign}` : `${s.sign}: aún sin muestras de la seña (se puede deletrear)`;
      b.classList.toggle('uncal', !s.calibrated);
      b.classList.toggle('done', !!progress[s.sign]);
      b.setAttribute('aria-pressed', String(s.sign === shownTarget));
      b.onclick = () => { animate(b, { scale: [1, .94, 1] }, { duration: .25 }); chooseSign(s.sign); b.blur(); };
      return b;
    }));
    return;
  }
  box.replaceChildren(...catalog.filter(inGroup).map((s) => {
    const b = document.createElement('button');
    b.innerHTML = `${s.sign}${s.dynamic ? '<span class="mov">↻</span>' : ''}`;
    b.dataset.sign = s.sign;
    b.title = s.descripcion + (s.calibrated ? '' : ' (aún no evaluable)');
    b.classList.toggle('uncal', !s.calibrated);
    b.classList.toggle('done', !!progress[s.sign]);
    b.setAttribute('aria-pressed', String(s.sign === shownTarget));
    b.onclick = () => { animate(b, { scale: [1, .9, 1] }, { duration: .25 }); chooseSign(s.sign); b.blur(); };
    return b;
  }));
}

function renderPath() {
  const progress = (state && state.progress) || {};
  if (practice?.word) {  // deletreo: las letras de la palabra, en orden
    $('path').replaceChildren(...practice.signs.map((sign, i) => {
      const el = document.createElement('span');
      el.textContent = sign;
      el.className = 'small' + (i < practice.i || practice.stats[i].done ? ' done' : i === practice.i ? ' current' : '');
      return el;
    }));
    return;
  }
  $('path').replaceChildren(...level1().map((sign) => {
    const el = document.createElement('span');
    el.textContent = sign;
    el.classList.toggle('done', !!progress[sign]);
    el.classList.toggle('current', sign === shownTarget);
    el.title = progress[sign] ? `${sign}: lograda` : `${sign}: pendiente`;
    return el;
  }));
}

function chooseSign(sign) {
  const info = catalog.find((c) => c.sign === sign);
  if (info && !inGroup(info) && !practice) setGroup(info.word ? 'words' : 'all');
  send({ type: 'target', sign });
}

// ---------- dibujo ----------
function render(s) {
  state = s;
  if (!s.target) return;
  if (s.target !== shownTarget) renderTarget(s.target);
  renderVerdict(s);
  renderInstruction(s);
  renderHand(s);
  renderShape(s);
  renderFingers(s.fingers || []);
  renderOrientation(s.orientation || []);
  renderStatus(s);
  renderWelcome(s);
  handleEvents(s);
  autoVoice(s);
  lastVerdict = s.verdict;
}

function renderTarget(sign) {
  shownTarget = sign;
  const info = catalog.find((c) => c.sign === sign) || { sign, descripcion: '', error_tipico: '', shape: {}, level: 0 };
  const letter = $('target-letter');
  letter.textContent = sign;
  letter.classList.toggle('word', !!info.word);
  animate(letter, { scale: [.55, 1], opacity: [0, 1] }, SPRING);  // solo escala: no se mueve de sitio
  $('target-level').textContent = info.level ? `· Nivel ${info.level}` : '';
  $('target-desc').textContent = info.descripcion;
  $('target-typical').textContent = info.word && info.parametros ? `Se evalúa: ${info.parametros}`
    : info.error_tipico ? `Error típico: ${info.error_tipico}` : '';
  $('target-spell').hidden = !info.word;
  $('target-spell').onclick = (e) => { startPractice(wordLetters(sign), sign); e.currentTarget.blur(); };
  $('welcome-letter').textContent = sign;
  const ref = $('target-ref');
  ref.hidden = !info.ref;
  if (info.ref) { ref.src = info.ref; ref.alt = `Referencia de la ${sign}`; animate(ref, { opacity: [0, 1], y: [10, 0] }, { duration: .35 }); }
  document.querySelectorAll('#sign-buttons button').forEach((b) =>
    b.setAttribute('aria-pressed', String(b.dataset.sign === sign)));
  renderPath();
  $('fingers').replaceChildren();
  $('orient').replaceChildren();
  voice = { action: null, since: 0, lastSpoken: 0 };
  if (practice) practiceStep();
}

function setVerdict(kind, icon, text) {
  const v = $('verdict');
  if (!v.classList.contains(kind)) {
    v.className = `verdict ${kind}`;
    animate(v, { scale: [.9, 1] }, SPRING);
  }
  v.dataset.icon = icon;  // hand | check | x | clock | alert | move (iconos SVG del HTML)
  $('verdict-text').textContent = text;
}

function renderVerdict(s) {
  // el anillo: se llena mientras sostienes la seña correcta; en "settling" cuenta el tiempo de preparación
  const ring = s.verdict === 'ok' ? (s.done ? 1 : s.hold || 0)
    : s.verdict === 'settling' ? 1 - (s.settle ?? 0)
    : s.verdict === 'moving' ? (s.hold || 0) : 0;
  $('ring').style.strokeDashoffset = RING * (1 - ring);
  if (s.camera_live === false) setVerdict('bad', 'alert', 'Sin imagen de la cámara');
  else if (s.verdict === 'ok') setVerdict('ok', 'check', s.motion ? (isWord(s.target) ? '¡Seña correcta!' : '¡Movimiento correcto!') : s.done ? '¡Seña correcta!' : 'Bien… mantenla');
  else if (s.verdict === 'fix') setVerdict('bad', 'x', s.issues[0]?.action || 'Corrige la seña');
  else if (s.verdict === 'settling') setVerdict('wait', 'clock', 'Forma la seña…');
  else if (s.verdict === 'ready') setVerdict('wait', 'move', s.recording ? `Grabando: haz la seña de ${s.target}` : isWord(s.target) ? `Haz la seña de ${s.target}` : `Haz el trazo de la ${s.target}`);
  else if (s.verdict === 'moving') setVerdict('wait', 'clock', s.recording ? 'Grabando la muestra…' : isWord(s.target) ? 'Grabando la seña…' : 'Grabando el trazo…');
  else if (s.verdict === 'uncalibrated') {
    const info = catalog.find((c) => c.sign === s.target);
    setVerdict('wait', 'move', info?.word ? `${s.target}: sin muestras de la seña todavía` : info?.dynamic ? `La ${s.target} lleva movimiento: mira la referencia` : `La ${s.target} aún no está calibrada`);
  } else setVerdict('wait', 'hand', 'Muestra tu mano a la cámara');
}

function renderInstruction(s) {
  const big = $('instruction');
  const ul = $('issues');
  let kind = 'wait', text, items = [];
  if (s.camera_live === false) { kind = 'bad'; text = 'Revisa la conexión de la cámara'; }
  else if (s.verdict === 'ok') { kind = 'ok'; text = s.motion ? '¡Perfecto! Puedes repetirlo o elegir otra letra' : s.done ? '¡Perfecto! Baja la mano para otro intento' : 'Muy bien, mantén la seña'; }
  else if (s.verdict === 'ready') { text = s.recording ? 'Haz la seña completa y detén la mano: se guarda sola' : isWord(s.target) ? 'Haz la seña completa, con el rostro a la vista, y detén la mano' : 'Mira la animación, haz el trazo completo y detén la mano'; }
  else if (s.verdict === 'moving') { text = isWord(s.target) ? 'Sigue la seña… al terminar, deja la mano quieta' : 'Sigue el trazo… al terminar, deja la mano quieta'; }
  else if (s.verdict === 'fix') {
    kind = 'bad';
    text = s.issues[0]?.action || 'Corrige la seña';
    items = s.issues.slice(1).map((i) => {
      const el = document.createElement('li');
      const chip = document.createElement('span'); chip.className = 'chip'; chip.textContent = i.parameter;
      el.append(chip, i.action);
      return el;
    });
  } else if (s.verdict === 'settling') { text = 'Acomoda los dedos como en la referencia'; }
  else if (s.verdict === 'uncalibrated') {
    const info = catalog.find((c) => c.sign === s.target);
    text = info?.word ? 'Graba la seña en Calibración (⚙ → Guardar correcta) o practícala deletreada'
      : info?.dynamic ? 'Practica el movimiento con la animación' : 'Graba muestras de esta seña en Calibración';
  } else text = 'Coloca la mano frente a la cámara';
  big.textContent = text;
  big.className = `big ${kind}`;
  ul.replaceChildren(...items);
}

function renderHand(s) {
  const status = {};
  for (const f of s.fingers || []) status[f.id] = f.ok;
  // en "settling" y en las letras con movimiento (sin rangos por dedo) nada se pinta
  const judging = s.hand && (s.fingers || []).length > 0 && (s.verdict === 'ok' || s.verdict === 'fix');
  document.querySelectorAll('#hand-svg .finger').forEach((el) => {
    const ok = judging ? (s.verdict === 'ok' || status[el.dataset.finger]) : null;
    el.classList.toggle('ok', ok === true);
    el.classList.toggle('bad', ok === false);
  });
  const palm = $('h-palm');
  const shapeBad = s.hand && (s.issues || []).some((i) => i.where === 'forma' || i.where === 'separacion' || i.where === 'hueco');
  palm.classList.toggle('bad', !!shapeBad);
  palm.classList.toggle('ok', !!s.hand && !shapeBad && s.verdict === 'ok');
  const orient = s.orientation || [];
  const wrist = $('h-wrist');
  wrist.classList.toggle('bad', orient.some((a) => a.ok === false && a.value != null));
  wrist.classList.toggle('ok', orient.length > 0 && orient.every((a) => a.ok));
}

function renderWelcome(s) {
  const now = performance.now();
  if (s.hand || s.camera_live === false) noHandSince = null;
  else if (noHandSince == null) noHandSince = now;
  const el = $('welcome');
  const show = noHandSince != null && now - noHandSince > WELCOME_AFTER_MS && $('summary').hidden;
  if (show && el.hidden) { el.hidden = false; animate(el, { opacity: [0, 1] }, { duration: .4 }); animate(el.querySelector('h1'), { y: [16, 0], opacity: [0, 1] }, { duration: .45, delay: .1 }); }
  else if (!show) el.hidden = true;
}

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
  if (min != null) { band.style.left = pct(min); band.style.width = `calc(${pct(max)} - ${pct(min)})`; }
  const marker = row.querySelector('.marker');
  marker.hidden = value == null;
  if (value != null) marker.style.left = pct(value);
  row.querySelector('.val').textContent = value == null ? '—' : `${Math.round(value)}${unit}`;
  row.className = `row ${ok == null ? '' : ok ? 'ok' : 'bad'}`;
}

function renderShape(s) {
  const el = $('shape');
  if (s.motion) {  // letra con movimiento: qué trazo reconoció el modelo de secuencias
    el.hidden = false;
    el.className = `shape-line ${s.motion.best !== s.target ? 'bad' : ''}`;
    el.innerHTML = s.motion.best === 'quieta' ? 'La mano casi no se movió: <b>sin trazo</b>'
      : `El trazo se reconoce como: <b>${s.motion.best}</b> · ${Math.round(s.motion.prob * 100)} %`;
    return;
  }
  el.hidden = !s.shape || !s.hand;
  if (el.hidden) return;
  const wrong = s.shape.best !== s.target || s.shape.error >= 0.4;
  el.className = `shape-line ${wrong ? 'bad' : ''}`;
  const err = s.shape.error >= 0.2 ? ` · posible error típico: <b>${Math.round(s.shape.error * 100)}%</b>` : '';
  el.innerHTML = `La IA reconoce: <b>${s.shape.best}</b>${err}`;
}

function renderFingers(fingers) {
  if ($('details-body').hidden) return;
  const box = $('fingers');
  if (!fingers.length) {
    box.querySelectorAll('.row').forEach((r) => { r.querySelector('.marker').hidden = true; r.querySelector('.val').textContent = '—'; r.className = 'row'; });
    return;
  }
  for (const f of fingers) gaugeRow(box, f.id, f.label, f.value, f.min, f.max, f.ok, FINGER_RANGE);
}

function renderOrientation(axes) {
  $('orient-block').hidden = !axes.length;
  if ($('details-body').hidden) return;
  for (const a of axes) {
    gaugeRow($('orient'), a.axis, AXIS_LABEL[a.axis] || a.axis, a.value, a.min, a.max,
      a.value == null ? false : a.ok, AXIS_RANGE[a.axis] || [-180, 180]);
  }
}

function setPill(id, kind, text) {
  const p = $(id);
  p.classList.remove('ok', 'bad', 'wait');
  p.classList.add(kind);
  p.querySelector('b').textContent = text;
}

function renderStatus(s) {
  const imu = s.imu || {};
  if (imu.mode === 'none') setPill('st-imu', 'wait', 'Sin muñequera');
  else if (imu.connected) setPill('st-imu', 'ok', 'Muñequera');
  else setPill('st-imu', 'bad', 'Buscando muñequera');
  $('st-imu').title = imu.connected ? `giro ${Math.round(imu.roll)}° · inclinación ${Math.round(imu.pitch)}°` : 'Muñequera';

  const hand = s.handedness === 'Right' ? 'mano derecha' : s.handedness === 'Left' ? 'mano izquierda' : '';
  if (s.camera_live === false) setPill('st-cam', 'bad', 'Sin imagen');
  else setPill('st-cam', s.fps >= 8 ? 'ok' : 'wait', hand ? `Cámara · ${hand}` : 'Cámara');  // ámbar si va lenta

  const sync = s.sync || {};
  if (!sync.cloud) setPill('st-cloud', 'wait', 'Local');
  else if (sync.error) setPill('st-cloud', 'bad', `${sync.pending} pendientes`);
  else if (sync.pending) setPill('st-cloud', 'wait', `${sync.pending} por subir`);
  else setPill('st-cloud', 'ok', 'Nube al día');

  const v2 = $('video2');
  if (s.cameras > 1 && v2.hidden) { v2.src = '/video/1'; v2.hidden = false; }
  if (document.activeElement !== $('person') && s.person) $('person').value = s.person;
}

// ---------- eventos: logro, vibración, avisos ----------
function handleEvents(s) {
  if (s.message && s.message.id !== lastMessageId) {
    lastMessageId = s.message.id;
    toast(s.message.text, s.message.kind);
  }
  if (s.alert && s.alert.id !== lastAlertId) {
    lastAlertId = s.alert.id;
    const p = $('st-imu');
    p.classList.add('buzz');
    animate(p, { x: [0, -3, 3, -2, 2, 0] }, { duration: .5 });
    setTimeout(() => p.classList.remove('buzz'), 600);
    if (practice) practice.stats[practice.i].alerts++;
  }
  if (s.achievement && s.achievement.id !== lastAchievementId) {
    const first = lastAchievementId === 0;
    lastAchievementId = s.achievement.id;
    if (first && s.achievement.sign !== s.target) return;  // logro viejo (de antes de abrir la página)
    celebrate(s.achievement);
  }
}

function celebrate(a) {
  animate($('flash'), { opacity: [0, 1, 0] }, { duration: .9, ease: 'easeOut', times: [0, .15, 1] });
  const box = $('confetti');
  const colors = ['#4ade80', '#2dd4bf', '#fbbf24', '#a78bfa', '#fb7185', '#fff'];
  const pieces = Array.from({ length: 36 }, () => {
    const i = document.createElement('i');
    i.style.left = `${Math.random() * 100}%`;
    i.style.background = colors[Math.floor(Math.random() * colors.length)];
    return i;
  });
  box.replaceChildren(...pieces);
  if (M) {
    animate(pieces, { y: ['0vh', '110vh'], rotate: [0, 540], opacity: [1, 1, 0] },
      { duration: 1.5, ease: 'easeIn', delay: stagger(.012) });
  }
  setTimeout(() => box.replaceChildren(), 2100);
  animate($('target-letter'), { scale: [1, 1.15, 1] }, { duration: .5 });
  chime();
  if (practice && a.sign === practice.signs[practice.i]) {
    const st = practice.stats[practice.i];
    if (!st.done) { st.done = true; st.seconds = a.seconds; }
    if (voiceAuto && !practice.word) say(`¡Muy bien! ${a.sign} correcta`);
    setTimeout(practiceNext, practice.word ? WORD_NEXT_DELAY_MS : NEXT_SIGN_DELAY_MS);
  } else if (voiceAuto) say(`¡Muy bien! ${a.sign} correcta`);
  renderSignButtons();
  renderPath();
}

function chime() {  // dos notas cortas, sin archivos de audio
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const t0 = audioCtx.currentTime;
    [[523.25, 0], [783.99, .12]].forEach(([freq, dt]) => {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.type = 'sine'; o.frequency.value = freq;
      g.gain.setValueAtTime(0, t0 + dt);
      g.gain.linearRampToValueAtTime(.25, t0 + dt + .02);
      g.gain.exponentialRampToValueAtTime(.001, t0 + dt + .35);
      o.connect(g).connect(audioCtx.destination);
      o.start(t0 + dt); o.stop(t0 + dt + .4);
    });
  } catch { /* sin audio: no pasa nada */ }
}

function toast(text, kind = 'ok') {
  const t = $('toast');
  t.textContent = text;
  t.className = `toast ${kind}`;
  t.hidden = false;
  animate(t, { y: [-12, 0], opacity: [0, 1] }, { duration: .25 });
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 2500);
}

// ---------- modo práctica ----------
// Una secuencia de letras: el Nivel 1 completo, o una palabra deletreada (word = 'HOLA').
// Las letras pueden repetirse (CASA), así que las estadísticas van por posición, no por letra.
function startPractice(signs, word = null) {
  signs = signs || level1();
  const missing = signs.filter((s) => !catalog.find((c) => c.sign === s)?.calibrated);
  if (missing.length) return toast(`Aún no se puede: falta calibrar ${[...new Set(missing)].join(', ')}`, 'bad');
  practice = { signs, word, i: 0, stats: signs.map(() => ({ start: 0, alerts: 0, seconds: null, done: false })) };
  $('practice-label').textContent = word ? `Terminar ${word}` : 'Terminar práctica';
  $('practice').classList.add('stop');
  $('practice-bar').hidden = false;
  animate($('practice-bar'), { y: [16, 0], opacity: [0, 1] }, { duration: .3 });
  $('summary').hidden = true;
  if (!word) setGroup('1');
  shownTarget = null;  // fuerza a redibujar la tarjeta aunque la letra sea la misma
  chooseSign(signs[0]);
  renderSignButtons();
  if (voiceAuto) say(word ? `Deletrea ${word}. Empieza con la ${signs[0]}` : `Empezamos. Haz la letra ${signs[0]}`);
}

function practiceStep() {
  if (!practice || shownTarget !== practice.signs[practice.i]) return;
  const st = practice.stats[practice.i];
  if (!st.start) st.start = performance.now();
  $('practice-step').textContent = practice.word
    ? `${practice.word} · letra ${practice.i + 1} de ${practice.signs.length}: ${shownTarget}`
    : `Práctica ${practice.i + 1} / ${practice.signs.length} · letra ${shownTarget}`;
  renderPath();
}

function practiceNext() {
  if (!practice) return;
  if (practice.i + 1 >= practice.signs.length) return finishPractice();
  practice.i += 1;
  const next = practice.signs[practice.i];
  shownTarget = null;  // la misma letra dos veces (CASA) también debe reiniciar el intento
  chooseSign(next);
  if (voiceAuto && !practice.word) say(`Ahora la letra ${next}`);
}

function finishPractice() {
  const p = practice;
  practice = null;
  $('practice-label').textContent = 'Practicar Nivel 1';
  $('practice').classList.remove('stop');
  $('practice-bar').hidden = true;
  renderPath();
  renderSignButtons();
  const rows = p.signs.map((s, i) => {
    const st = p.stats[i];
    const secs = st.done ? `${st.seconds} s` : st.start ? `${Math.round((performance.now() - st.start) / 1000)} s` : '—';
    return `<tr><td>${s}</td><td class="${st.done ? 'ok' : 'bad'}">${st.done ? '✓ Lograda' : 'Pendiente'}</td><td>${secs}</td><td>${st.alerts}</td></tr>`;
  });
  $('summary-table').querySelector('tbody').innerHTML = rows.join('');
  const done = p.stats.filter((st) => st.done).length;
  const all = done === p.signs.length;
  const total = p.stats.reduce((t, st) => t + (st.seconds || 0), 0).toFixed(1);
  $('summary-title').lastChild.textContent = p.word
    ? (all ? ` ¡${p.word}! Palabra completa` : ` ${p.word}: ${done} de ${p.signs.length} letras`)
    : (all ? ' ¡Nivel 1 completo!' : ` Resultados: ${done} de ${p.signs.length}`);
  $('summary-foot').textContent = p.word
    ? (all ? `Deletreaste ${p.word} en ${total} s. Elige otra palabra abajo.` : 'Puedes intentarla otra vez desde Palabras.')
    : (all ? 'Lograste todas las señas estáticas del reto. ¡Sigue con el Nivel 2!'
           : 'Las pendientes puedes repetirlas eligiendo la letra abajo.');
  $('summary-again').onclick = () => startPractice(p.signs, p.word);
  $('summary').hidden = false;
  animate($('summary'), { opacity: [0, 1] }, { duration: .25 });
  animate($('summary').querySelector('.modal-card'), { scale: [.92, 1], opacity: [0, 1] }, SPRING);
  animate($('summary-table').querySelectorAll('tbody tr'), { opacity: [0, 1], x: [-10, 0] }, { duration: .3, delay: stagger(.07) });
  if (p.word && all) { chime(); if (voiceAuto) say(`${p.signs.join(' ')}. ¡${p.word}!`); }
  else if (voiceAuto) say(p.word ? `Lograste ${done} de ${p.signs.length} letras`
    : all ? 'Nivel uno completo. ¡Felicidades!' : `Lograste ${done} de ${p.signs.length} señas`);
}

// ---------- voz (sin internet: voces del sistema) ----------
function pickVoice() {
  const voices = speechSynthesis.getVoices();
  return voices.find((v) => v.lang === 'es-MX' && v.localService)
    || voices.find((v) => v.lang.startsWith('es') && v.localService)
    || voices.find((v) => v.lang.startsWith('es'));
}

function say(text) {
  if (!('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = 'es-MX';
  const v = pickVoice();
  if (v) u.voice = v;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
  voice.lastSpoken = performance.now();
}

function speakNow() {
  let text;
  if (!state || !state.hand) text = 'No veo tu mano';
  else if (state.verdict === 'ok') text = 'Todo bien';
  else if (state.verdict === 'uncalibrated') text = 'Esta seña aún no está calibrada';
  else text = [...new Set(state.issues.map((i) => i.action))].slice(0, 3).join('. ');
  say(text);
}

// Dice la corrección sola cuando el mismo error persiste (misma lógica que la vibración)
function autoVoice(s) {
  if (!voiceAuto) return;
  const action = s.verdict === 'fix' ? s.issues[0]?.action : null;
  const now = performance.now();
  if (action !== voice.action) { voice.action = action; voice.since = now; }
  if (!action) return;
  if (now - voice.since >= AUTO_VOICE_PERSIST_MS && now - voice.lastSpoken >= AUTO_VOICE_GAP_MS) say(action);
}

// ---------- controles ----------
$('speak').onclick = (e) => { speakNow(); e.currentTarget.blur(); };
$('voice-auto').onclick = (e) => {
  voiceAuto = !voiceAuto;
  savePref('lsm-voice', voiceAuto ? '1' : '0');
  e.currentTarget.setAttribute('aria-pressed', String(voiceAuto));
  e.currentTarget.blur();
};
$('voice-auto').setAttribute('aria-pressed', String(voiceAuto));
$('practice').onclick = (e) => { practice ? finishPractice() : startPractice(); e.currentTarget.blur(); };
$('practice-skip').onclick = (e) => { practiceNext(); e.currentTarget.blur(); };
$('practice-stop').onclick = () => finishPractice();
$('summary-close').onclick = () => { $('summary').hidden = true; };
$('details-toggle').onclick = (e) => {
  const body = $('details-body');
  body.hidden = !body.hidden;
  e.currentTarget.setAttribute('aria-expanded', String(!body.hidden));
  e.currentTarget.blur();
};
$('calib-toggle').onclick = (e) => {
  const panel = $('calib');
  panel.hidden = !panel.hidden;
  if (!panel.hidden) animate(panel, { y: [12, 0], opacity: [0, 1] }, { duration: .25 });
  e.currentTarget.setAttribute('aria-expanded', String(!panel.hidden));
  e.currentTarget.blur();
};
$('person').onchange = (e) => send({ type: 'person', name: e.target.value });
$('rec-ok').onclick = () => send({ type: 'record', is_error: false });
$('rec-err').onclick = () => send({ type: 'record', is_error: true });
$('reload').onclick = () => { send({ type: 'reload' }); setTimeout(loadCatalog, 300); };
document.querySelectorAll('#groups button').forEach((b) => { b.onclick = () => { setGroup(b.dataset.group); b.blur(); }; });

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.code === 'Space') { e.preventDefault(); speakNow(); return; }
  if (e.key === 'Escape') { $('summary').hidden = true; $('calib').hidden = true; return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const key = e.key.toUpperCase();
  if (key === 'D' && !catalog.some((c) => c.sign === 'D' && inGroup(c) && group !== 'all')) { $('details-toggle').click(); return; }
  if (key === 'P') { $('practice').click(); return; }
  if (catalog.some((c) => c.sign === key)) chooseSign(key);
});

// en quiosco: el cursor desaparece cuando nadie mueve el mouse
let idleTimer = null;
document.addEventListener('mousemove', () => {
  document.body.classList.remove('idle');
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => document.body.classList.add('idle'), 3000);
});
// el audio del navegador necesita un gesto del usuario la primera vez
document.addEventListener('pointerdown', () => { try { audioCtx = audioCtx || new AudioContext(); audioCtx.resume(); } catch { /* sin audio */ } }, { once: true });

// ---------- vista previa del diseño (?preview): estados simulados, SOLO para revisar la
// interfaz sin cámara ni mano. No se usa en la demo: ahí todo viene del servidor real. ----------
function previewLoop() {
  const fingers = (bad) => ['pulgar', 'indice', 'medio', 'anular', 'menique'].map((id) => ({
    id, label: { pulgar: 'pulgar', indice: 'índice', medio: 'medio', anular: 'anular', menique: 'meñique' }[id],
    value: bad.includes(id) ? 120 : 20, min: 0, max: 60, ok: !bad.includes(id) }));
  const base = { target: 'A', calibrated: true, imu: { mode: 'ble', connected: true, roll: 5, pitch: -3 }, fps: 12,
    camera_live: true, cameras: 1, person: 'invitado', sync: { cloud: true, pending: 0 }, progress: {}, message: null,
    orientation: [{ axis: 'roll', value: 5, min: -20, max: 20, ok: true }, { axis: 'pitch', value: -3, min: -25, max: 25, ok: true }] };
  const steps = [
    { hand: false, verdict: 'nohand', issues: [], failed_parameters: [], fingers: [], hold: 0, done: false },
    { hand: true, verdict: 'settling', settle: .45, issues: [], failed_parameters: [], fingers: fingers(['indice']), hold: 0, done: false },
    { hand: true, verdict: 'fix', issues: [{ parameter: 'Configuración', where: 'indice', action: 'Flexiona el índice' }, { parameter: 'Configuración', where: 'medio', action: 'Flexiona el medio' }],
      failed_parameters: ['Configuración'], fingers: fingers(['indice', 'medio']), shape: { best: 'L', error: .1 }, hold: 0, done: false, alert: { id: 1, action: 'Flexiona el índice' } },
    { hand: true, verdict: 'ok', issues: [], failed_parameters: [], fingers: fingers([]), shape: { best: 'A', error: .05 }, hold: .55, done: false },
    { hand: true, verdict: 'ok', issues: [], failed_parameters: [], fingers: fingers([]), shape: { best: 'A', error: .05 }, hold: 1, done: true,
      progress: { A: 1 }, achievement: { id: 2, sign: 'A', seconds: 4.2, failed_parameters: ['Configuración'] } },
  ];
  const fixed = new URLSearchParams(location.search).get('preview');  // ?preview=2 -> solo ese paso
  let i = fixed === '' || fixed === null ? 0 : Number(fixed) || 0;
  const tick = () => { render({ ...base, ...steps[Math.min(i, steps.length - 1)] }); };
  setPill('st-link', 'ok', M ? 'Vista previa' : 'Vista previa · sin Motion');
  if (fixed) { noHandSince = performance.now() - WELCOME_AFTER_MS - 1; tick(); return; }
  tick();
  setInterval(() => { i = (i + 1) % steps.length; if (i === 0) { lastAchievementId = 0; noHandSince = null; } tick(); }, 3500);
  setInterval(tick, 100);
}

if (new URLSearchParams(location.search).has('preview')) loadCatalog().then(previewLoop);
else loadCatalog().then(connect);
