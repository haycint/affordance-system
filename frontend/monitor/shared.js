// Shared WebSocket / three.js helpers for the monitor frontend.
// Loaded via <script type="module"> on watch.html and train.html.

import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

export { THREE, OrbitControls };

let ws = null;
const pending = new Map();
const subscribers = new Map();   // type → [fn]

export function uuid4() {
  return ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
    (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c/4).toString(16));
}

export function getMonitorUUID() {
  let u = sessionStorage.getItem('monitor_uuid');
  if (!u) { u = uuid4(); sessionStorage.setItem('monitor_uuid', u); }
  return u;
}

export function connect(url) {
  return new Promise((resolve, reject) => {
    if (ws && ws.readyState === 1) return resolve();
    let settled = false;
    try { ws = new WebSocket(url); }
    catch (e) { return reject(new Error("invalid URL: " + e.message)); }

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      try { ws.close(); } catch {}
      reject(new Error("connect timeout (后端可能未启动 / 端口不通)"));
    }, 5000);

    ws.onopen = async () => {
      if (settled) return;
      try {
        await call('register_user', { uuid: getMonitorUUID() });
        settled = true; clearTimeout(timer); resolve();
      } catch (e) {
        settled = true; clearTimeout(timer); reject(e);
      }
    };
    ws.onerror = () => {
      if (settled) return;
      settled = true; clearTimeout(timer);
      reject(new Error("WebSocket error (检查 ws://host:port/ws 是否正确)"));
    };
    ws.onclose = ev => {
      if (!settled) {
        settled = true; clearTimeout(timer);
        reject(new Error("WebSocket closed before open (code=" + ev.code + ")"));
      }
      ws = null;
      for (const fn of subscribers.get('__close__') || []) fn();
    };
    ws.onmessage = ev => {
      let msg; try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.request_id && pending.has(msg.request_id)) {
        const { resolve } = pending.get(msg.request_id);
        pending.delete(msg.request_id);
        resolve(msg);
      } else if (msg.type && subscribers.has(msg.type)) {
        for (const fn of subscribers.get(msg.type)) fn(msg);
      }
    };
  });
}

export function disconnect() { if (ws) ws.close(); }

export function call(api, payload) {
  return new Promise((resolve, reject) => {
    if (!ws || ws.readyState !== 1) return reject(new Error("not connected"));
    const id = uuid4();
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ type: api, request_id: id, payload }));
    setTimeout(() => {
      if (pending.has(id)) { pending.delete(id); reject(new Error("timeout")); }
    }, 60000);
  });
}

export function on(eventType, fn) {
  if (!subscribers.has(eventType)) subscribers.set(eventType, []);
  subscribers.get(eventType).push(fn);
}

// ── three.js scene helper ──────────────────────────────────────────────
export function makeScene(canvas) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101418);
  const camera = new THREE.PerspectiveCamera(
    60, canvas.clientWidth / canvas.clientHeight, 0.01, 1000);
  camera.position.set(1.5, 1.5, 1.5);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  scene.add(new THREE.AxesHelper(0.5));
  (function loop() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  })();
  new ResizeObserver(() => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }).observe(canvas);
  return { scene, camera, renderer, controls };
}

export function prefToColor(v) {
  if (v < 0) return [0.6, 0.2, 0.8];   // 高存疑 (-1) → 紫
  const t = Math.max(0, Math.min(1, v));
  return [t, 0.15 + 0.3*(1-t), 1-t];   // 高奖励 → 红，低 → 蓝
}

export function buildPoints(positions, colors, size) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('color',    new THREE.BufferAttribute(colors,    3));
  const mat = new THREE.PointsMaterial({
    size, vertexColors: true, sizeAttenuation: false });
  return new THREE.Points(geom, mat);
}

export function renderCloud(scene, pts, pref, oldObj, size) {
  const N = pts.length;
  const pos = new Float32Array(N*3);
  const col = new Float32Array(N*3);
  for (let i = 0; i < N; i++) {
    pos[i*3] = pts[i][0]; pos[i*3+1] = pts[i][1]; pos[i*3+2] = pts[i][2];
    const c = prefToColor(pref ? pref[i] : 0.5);
    col[i*3] = c[0]; col[i*3+1] = c[1]; col[i*3+2] = c[2];
  }
  if (oldObj) scene.remove(oldObj);
  const obj = buildPoints(pos, col, size);
  scene.add(obj);
  return { obj, positions: pos, preference: pref ? Float32Array.from(pref) : null };
}
