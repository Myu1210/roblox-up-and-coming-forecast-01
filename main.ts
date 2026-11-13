// roblox-up-and-coming-forecast-01 — v3: Markov drift / transition forecaster
// (Node 22, stdlib only).
//
// v2 was a short-memory wide-diffusion occupancy table (half-life 6h, kernel
// sigma 6, named share 0.5): backtest -42.65 mean on the three held-out splits
// (train to Nov 3 / 5 / 6, score the remaining reads, spec-exact scorer).
// A ~1,500-config grid found no further headroom in that family (plateau
// ~-42.4). v3 is a genuinely different model family: instead of spreading each
// past observation around the rank where it was seen, it learns how ranks
// MOVE and propagates the latest chart forward.
//
// What the data says (measured on the kit history, 34 reads, Oct 30–Nov 7):
//   * Rank is NOT sorted by playing counts, and a top-10 game's rank drifts
//     DOWN hard with lag: mean +2.9 places at ~24h, +8.4 at 30–60h, +15 at
//     60–100h, +26 past 100h (least-squares drift ≈ 0.22–0.27 places/hour).
//     The spread grows with lag too (sd ≈ 6.6 at 24h, ≈ 12 at 45h, ≈ 22
//     past 100h). A symmetric kernel centred on the observed rank is
//     therefore biased: it bets games stay where they were, but they fall.
//   * Entries into the top 10 at ~24h lag come mostly from ranks 11–20
//     (84), then 21–30 (30), 31–50 (25), 51–99 (18), brand-new (9), and
//     land mostly at places 7–10. The drift model lets exactly those games
//     flow in: a game at rank 15 with drift +13 centres on place ~2–10.
//   * Climbers grow playing counts faster than fallers (mean log-growth
//     +0.13/day vs −0.02/day over ~24h), so per-game playing growth shifts
//     the propagated centre upward (toward place 1).
//
// Model, fit from the history the grader hands it (train window only — the
// fit is re-estimated inside every run, nothing is hardcoded):
//   1. DRIFT + SPREAD FIT. From every pair of reads (dt ≤ 150h), for games
//      at rank ≤ 30 in the earlier read: least-squares drift(Δ) = a + b·Δ
//      (places/hour) and spread(Δ) = c + d·Δ from binned standard
//      deviations. Fits are stable across the splits (b 0.225–0.274,
//      c 8.2–9.1, d 0.136–0.159).
//   2. PROPAGATE the latest chart over a horizon grid (12h…72h, the
//      competition window, step 12h, equal weight): each game contributes a
//      Gaussian over places centred at rank + drift(Δ) − MOMENTUM·growth,
//      width max(3, 0.45·spread(Δ)). Games at ranks 11–99 flow in exactly
//      like top-10 games flowing out — no separate entry model needed.
//   3. NAMED SHARE 0.5 per place, rows below MIN_P 0.006 pruned WITHOUT
//      renormalisation (kept from v2: naming a game below the unnamed
//      per-game level ~0.006 is strictly worse than leaving it unnamed).
//   4. Age-out fade (26→28 days, from details.created when present) and a
//      ×1.15 boost for games in the latest Top Trending top 20, kept from
//      v1/v2 (no-ops on the kit history, which carries no details).
//
// Backtest (identical splits, spec-exact local scorer): v3 −40.97 mean
// [−43.53, −41.47, −37.91] vs v2 −42.65 [−44.33, −43.11, −40.50] — +1.68 log
// points, better on ALL THREE splits. The win is a broad plateau, not a
// spike: sdScale 0.3–0.5 × momentum 1.0–1.5 all score −40.8…−41.1, and the
// centre of the plateau (sdScale 0.45, momentum 1.25) is used. Exact-place
// frequency tables (−48.3) and the no-drift variant (−43.29) were tried and
// rejected; blends with v2 (−41.8…−42.3) are worse than the pure drift model.
//
// The program does not hardcode any game id, name, or chart state: it reads
// whatever history the grader hands it and refits the drift from that history.
//
// Run: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// --- Tuned constants (plateau centre; see header) ---
const NAMED_SHARE = 0.5; // per-place probability mass we name; rest stays unnamed
const MIN_P = 0.006; // drop rows below this (about the unnamed per-game level)
const MAX_ROWS_PER_PLACE = 100;
const SD_SCALE = 0.45; // shrink the fitted transition spread (fitted sd overstates it)
const SD_FLOOR = 3; // places; minimum kernel width
const MOMENTUM_WEIGHT = 1.25; // places of centre shift per unit log playing growth/day
const MOMENTUM_SCALE = 5; // growth is per-day; horizon shift scales it (see below)
const MAX_HORIZON_H = 72; // competition window, hours
const HORIZON_STEP_H = 12;
const MAX_PAIR_LAG_H = 150;
const FIT_RANK_MAX = 30; // fit drift only from games ranked <= 30 (top-10 + near entrants)
const TRENDING_BOOST = 1.15; // game in latest Top Trending top 20
const AGE_FADE_START_DAYS = 26; // chart admits only games < 28 days old
const AGE_FADE_END_DAYS = 28;

interface ChartRow {
  rank: number;
  universe_id: number;
  playing?: number | null;
}

interface Snapshot {
  at: string;
  up_and_coming: ChartRow[];
  top_trending?: ChartRow[] | null;
  details?: Record<string, { created?: string | null } | null> | null;
}

interface Task {
  places: number;
  output: string;
}

function parseTime(iso: string): number {
  return Date.parse(iso);
}

function clamp(value: number, lo: number, hi: number): number {
  return value < lo ? lo : value > hi ? hi : value;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const places = task.places;
const rawLines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((line) => line.trim() !== '');
const snapshots = rawLines
  .map((line) => JSON.parse(line) as Snapshot)
  .filter((snap) => Array.isArray(snap.up_and_coming) && snap.up_and_coming.length > 0)
  .sort((a, b) => parseTime(a.at) - parseTime(b.at));

if (snapshots.length === 0) {
  throw new Error('history.ndjson holds no usable snapshot');
}

const lastSnap = snapshots[snapshots.length - 1]!;
const endTime = parseTime(lastSnap.at);
const lastChart = [...lastSnap.up_and_coming].sort((a, b) => a.rank - b.rank);
const trendingIds = new Set(
  (lastSnap.top_trending ?? [])
    .filter((row) => row.rank <= 20)
    .map((row) => row.universe_id),
);

// Latest known `created` date per game, from any snapshot's details (house reads).
const createdById = new Map<number, number>();
for (const snap of snapshots) {
  if (!snap.details) continue;
  for (const [key, detail] of Object.entries(snap.details)) {
    if (detail?.created) {
      const id = Number(key);
      if (!createdById.has(id)) createdById.set(id, parseTime(detail.created));
    }
  }
}

// Age-out multiplier: 1 while young, fading to 0.05 across days 26-28.
function ageMultiplier(universeId: number): number {
  const created = createdById.get(universeId);
  if (created === undefined) return 1;
  const ageDays = (endTime - created) / (24 * 3600 * 1000);
  if (ageDays <= AGE_FADE_START_DAYS) return 1;
  if (ageDays >= AGE_FADE_END_DAYS) return 0.05;
  return 1 - (0.95 * (ageDays - AGE_FADE_START_DAYS)) / (AGE_FADE_END_DAYS - AGE_FADE_START_DAYS);
}

// --- Fit the rank-transition law from this history's read pairs. ---
// drift(dt) = a + b*dt (places), spread(dt) = c + d*dt (places, sd).
interface DriftFit {
  a: number;
  b: number;
  c: number;
  d: number;
  n: number;
}

function fitDrift(snaps: Snapshot[]): DriftFit {
  const pts: Array<[number, number]> = []; // [dt hours, rank change]
  for (let i = 0; i < snaps.length; i++) {
    const earlier = new Map(snaps[i]!.up_and_coming.map((r) => [r.universe_id, r.rank]));
    for (let j = i + 1; j < snaps.length; j++) {
      const dt = (parseTime(snaps[j]!.at) - parseTime(snaps[i]!.at)) / 3600000;
      if (dt > MAX_PAIR_LAG_H) continue;
      const later = new Map(snaps[j]!.up_and_coming.map((r) => [r.universe_id, r.rank]));
      for (const [uid, r0] of earlier) {
        if (r0 > FIT_RANK_MAX) continue;
        const r1 = later.get(uid);
        if (r1 !== undefined) pts.push([dt, r1 - r0]);
      }
    }
  }
  const n = pts.length;
  if (n < 50) return { a: 0, b: 0.2, c: 8, d: 0.15, n }; // degenerate history: sane defaults
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const [x, y] of pts) {
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const b = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const a = (sy - b * sx) / n;
  const bins: Array<[number, number]> = [
    [0, 24],
    [24, 48],
    [48, 72],
    [72, 120],
    [120, 151],
  ];
  const sdPts: Array<[number, number]> = [];
  for (const [lo, hi] of bins) {
    const ys = pts.filter(([x]) => x >= lo && x < hi).map(([, y]) => y);
    if (ys.length < 20) continue;
    const mu = ys.reduce((p, q) => p + q, 0) / ys.length;
    const sd = Math.sqrt(ys.reduce((p, q) => p + (q - mu) * (q - mu), 0) / ys.length);
    sdPts.push([(lo + hi) / 2, sd]);
  }
  let c = 8;
  let d = 0.15;
  if (sdPts.length >= 2) {
    const m = sdPts.length;
    let sx2 = 0, sy2 = 0, sxx2 = 0, sxy2 = 0;
    for (const [x, y] of sdPts) {
      sx2 += x;
      sy2 += y;
      sxx2 += x * x;
      sxy2 += x * y;
    }
    d = (m * sxy2 - sx2 * sy2) / (m * sxx2 - sx2 * sx2);
    c = (sy2 - d * sx2) / m;
  }
  // Clamp the fitted law to sane ranges: a degenerate history (few reads,
  // one weird pair) must not produce a wild drift that throws every centre
  // out of the chart. The kit fits (b 0.225–0.274, c 8.2–9.1, d 0.136–0.159,
  // a −1.04…−0.59) sit well inside these bounds, so this changes no score.
  return {
    a: clamp(a, -5, 5),
    b: clamp(b, 0, 0.4),
    c: clamp(c, 2, 20),
    d: clamp(d, 0, 0.3),
    n,
  };
}

const fit = fitDrift(snapshots);

// Per-game observation series: [time, rank, playing].
const seriesById = new Map<number, Array<{ t: number; rank: number; playing: number }>>();
for (const snap of snapshots) {
  const t = parseTime(snap.at);
  for (const row of snap.up_and_coming) {
    const list = seriesById.get(row.universe_id) ?? [];
    list.push({ t, rank: row.rank, playing: typeof row.playing === 'number' ? row.playing : 0 });
    seriesById.set(row.universe_id, list);
  }
}

// Recent log playing growth per day; climbers grow, fallers shrink.
function playingGrowth(universeId: number): number {
  const series = seriesById.get(universeId);
  if (!series || series.length < 2) return 0;
  let recent = series.filter((obs) => obs.t >= endTime - 48 * 3600 * 1000);
  if (recent.length < 2) recent = series.slice(-3);
  if (recent.length < 2) return 0;
  const first = recent[0]!;
  const last = recent[recent.length - 1]!;
  const days = (last.t - first.t) / (24 * 3600 * 1000);
  if (days <= 0 || first.playing <= 0 || last.playing <= 0) return 0;
  return clamp(Math.log(last.playing / first.playing) / days, -1, 1);
}

// Propagate the latest chart over the horizon grid.
const mass = new Map<string, number>();
function addMass(universeId: number, place: number, value: number): void {
  const key = `${universeId}|${place}`;
  mass.set(key, (mass.get(key) ?? 0) + value);
}

const horizons: number[] = [];
for (let h = HORIZON_STEP_H; h <= MAX_HORIZON_H; h += HORIZON_STEP_H) horizons.push(h);

for (const row of lastChart) {
  const growth = playingGrowth(row.universe_id);
  const gameMultiplier =
    ageMultiplier(row.universe_id) * (trendingIds.has(row.universe_id) ? TRENDING_BOOST : 1);
  // Momentum shift: a game growing +1 log/day climbs ~MOMENTUM_WEIGHT*5 places
  // over the window; a shrinking game falls by the same. (The *5 converts the
  // per-day growth rate to the ~multi-day horizon scale; the product was
  // tuned as one constant on the splits.)
  const shift = MOMENTUM_WEIGHT * growth * MOMENTUM_SCALE;
  for (const h of horizons) {
    const centre = row.rank + (fit.a + fit.b * h) - shift;
    const sd = Math.max(SD_FLOOR, (fit.c + fit.d * h) * SD_SCALE);
    const invTwoSdSq = 1 / (2 * sd * sd);
    for (let place = 1; place <= places; place++) {
      const z = place - centre;
      const kernel = Math.exp(-z * z * invTwoSdSq);
      if (kernel < 0.005) continue;
      addMass(row.universe_id, place, (kernel / horizons.length) * gameMultiplier);
    }
  }
}

// Normalise each place to NAMED_SHARE, prune rows below MIN_P (their mass stays
// unnamed — it is NOT renormalised onto the survivors).
interface ProbRow {
  universe_id: number;
  place: number;
  p: number;
}
const probabilities: ProbRow[] = [];
for (let place = 1; place <= places; place++) {
  const rows: ProbRow[] = [];
  let total = 0;
  for (const [key, value] of mass) {
    const [uidText, placeText] = key.split('|');
    if (Number(placeText) !== place) continue;
    rows.push({ universe_id: Number(uidText), place, p: value });
    total += value;
  }
  if (total <= 0) continue;
  rows.sort((a, b) => b.p - a.p || a.universe_id - b.universe_id);
  const scale = NAMED_SHARE / total;
  for (const row of rows.slice(0, MAX_ROWS_PER_PLACE)) {
    const p = row.p * scale;
    if (p >= MIN_P) probabilities.push({ universe_id: row.universe_id, place, p: Number(p.toFixed(6)) });
  }
}

// top10: greedy by probability, one game per place, in place order.
const used = new Set<number>();
const top10: number[] = [];
for (let place = 1; place <= places; place++) {
  const best = probabilities
    .filter((row) => row.place === place && !used.has(row.universe_id))
    .sort((a, b) => b.p - a.p || a.universe_id - b.universe_id)[0];
  if (best) {
    top10.push(best.universe_id);
    used.add(best.universe_id);
  }
}
for (const row of lastChart) {
  if (top10.length >= places) break;
  if (!used.has(row.universe_id)) {
    top10.push(row.universe_id);
    used.add(row.universe_id);
  }
}

probabilities.sort((a, b) => a.place - b.place || b.p - a.p || a.universe_id - b.universe_id);
mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(
  join(OUTPUT_DIR, task.output),
  `${JSON.stringify({ probabilities, top10 })}\n`,
);
