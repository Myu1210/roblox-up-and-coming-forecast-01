// roblox-up-and-coming-forecast-01 — v2: short-memory wide-diffusion occupancy table
// (Node 22, stdlib only).
//
// v1 was a recency-weighted Gaussian kernel forecast (36h e-folding, sigma 2.6,
// 0.65 named share): backtest -45.44 on the three held-out splits, vs the house
// baseline's -54.54. v2 keeps the same skeleton but retunes it into what the
// splits actually reward, after a ~1,500-config grid search on the identical
// splits (train to Nov 3 / 5 / 6, score the remaining reads):
//
//   1. SHORT MEMORY. Half-life 6h instead of a 36h e-folding. The chart's top 10
//      set turns over in days, and a game's rank a week ago says almost nothing
//      about where it sits now; only the last ~day of reads carries real weight.
//      (With the final history's hourly house reads this is still ~a day of
//      reads; with the sparse archive reads it is effectively the last 1-2.)
//   2. WIDE DIFFUSION. Kernel sigma 6 places instead of 2.6. Places churn hard
//      (the kit's own notes: the top 10 changed in 42 of 44 consecutive read
//      pairs, median 6 places), so one read's rank is a weak statement about a
//      game's place three days out. A wide kernel spreads each observation over
//      the whole top 10 instead of betting it stays within +-3 places.
//   3. HALF NAMED, HALF HEDGED. NAMED_SHARE 0.5 instead of 0.65. The other half
//      stays unnamed, spread over the ~90 games the forecast does not name at a
//      place (~0.005 each) — so a surprise entrant costs ~-5.3, not the -9.21
//      floor, and the named mass is spent only where the table is confident.
//   4. PRUNE BELOW THE UNNAMED LEVEL. Rows below MIN_P 0.006 are dropped, and
//      the mass they would have taken stays unnamed instead of being
//      renormalised onto the survivors. Naming a game at p below the unnamed
//      per-game probability (~0.005-0.01) is strictly worse than leaving it
//      unnamed: if it is the actual occupant you score ln(p) instead of the
//      higher ln(unnamed p). So tiny rows are left out on purpose.
//   5. Momentum, age-out and trending-boost kept from v1, unchanged in spirit:
//      a clamped rank-velocity / player-growth shift of the kernel centre, mass
//      faded out as a game approaches the chart's 28-day age limit, games
//      missing from the latest read faded x0.3, games in Top Trending's top 20
//      boosted x1.15. The grid found these second-order (momentum 0 vs 0.35 vs
//      0.7 moves the mean by ~0.1); they are kept because they help on the
//      final's hourly reads, where velocity is measured over real hours.
//
// Backtest (identical splits, spec-exact local scorer, measured by running this
// file on the truncated histories): v2 -42.64 mean
// [-44.22, -43.12, -40.59] vs v1 -45.44 [-46.51, -46.35, -43.47] — +2.80 log
// points, better on all three splits — and vs the house baseline's -54.54.
// The improvement is a broad plateau (HL 4-8h, sigma 5-8, named 0.45-0.55 all
// score -42.4 to -42.9), not a single lucky point, so the middle of the plateau
// is used rather than the argmax. Two-stage set/place models, Laplace kernels,
// timescale mixtures and persistence blends were all tried and scored worse.
//
// The program does not hardcode any game id, name, or chart state: it reads
// whatever history the grader hands it, so the same code forecasts from the
// final (hourly, house-recorder) history.
//
// Run: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// --- Tuned constants (grid-searched on the three held-out splits; see header) ---
const HALF_LIFE_MS = 6 * 3600 * 1000; // recency half-life for read weights
const KERNEL_SIGMA = 6; // places; wide spread of one read's kernel over places
const NAMED_SHARE = 0.5; // per-place probability mass we name; rest stays unnamed
const MOMENTUM_WEIGHT = 0.35; // how much recent rank/player momentum shifts a kernel
const MIN_P = 0.006; // drop rows below this (about the unnamed per-game level)
const MAX_ROWS_PER_PLACE = 100;
const MISSING_LATEST_DECAY = 0.3; // game absent from the latest read
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
const lastIds = new Set(lastChart.map((row) => row.universe_id));
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

// Age-out multiplier: 1 while young, fading to 0.05 across days 26-28.
function ageMultiplier(universeId: number): number {
  const created = createdById.get(universeId);
  if (created === undefined) return 1;
  const ageDays = (endTime - created) / (24 * 3600 * 1000);
  if (ageDays <= AGE_FADE_START_DAYS) return 1;
  if (ageDays >= AGE_FADE_END_DAYS) return 0.05;
  return 1 - (0.95 * (ageDays - AGE_FADE_START_DAYS)) / (AGE_FADE_END_DAYS - AGE_FADE_START_DAYS);
}

// Recent momentum of one game's series, in places of kernel shift.
// Positive = climbing / growing, so its kernel centre moves toward place 1.
// Measured over the last 48h of reads (not just the last 3 rows): with the
// final history's hourly reads, 3 rows would span only 3h and a 1-place wobble
// would read as 8 places/day. Falls back to the last 3 rows when the 48h
// window spans under 6h (sparse archive reads).
function momentumShift(
  series: Array<{ t: number; rank: number; playing: number }>,
  endTime: number,
): number {
  let recent = series.filter((obs) => obs.t >= endTime - 48 * 3600 * 1000);
  const spanHours = (obs: Array<{ t: number }>): number =>
    obs.length >= 2 ? (obs[obs.length - 1]!.t - obs[0]!.t) / 3600000 : 0;
  if (recent.length < 2 || spanHours(recent) < 6) recent = series.slice(-3);
  if (recent.length < 2) return 0;
  const first = recent[0]!;
  const last = recent[recent.length - 1]!;
  const days = (last.t - first.t) / (24 * 3600 * 1000);
  if (days <= 0) return 0;
  const rankVelocity = (first.rank - last.rank) / days; // places/day, + = climbing
  let growth = 0;
  if (first.playing > 0 && last.playing > 0) {
    growth = Math.log(last.playing / first.playing) / days; // log players/day
  }
  const shift =
    MOMENTUM_WEIGHT * clamp(rankVelocity, -3, 3) * 0.5 +
    MOMENTUM_WEIGHT * clamp(growth, -1, 1) * 1.0;
  return clamp(shift, -3, 3);
}

// Accumulate the occupancy mass: mass[(universe_id, place)].
const mass = new Map<string, number>();
function addMass(universeId: number, place: number, value: number): void {
  const key = `${universeId}|${place}`;
  mass.set(key, (mass.get(key) ?? 0) + value);
}

const invTwoSigmaSq = 1 / (2 * KERNEL_SIGMA * KERNEL_SIGMA);
for (const [universeId, series] of seriesById) {
  const shift = momentumShift(series, endTime);
  const gameMultiplier =
    ageMultiplier(universeId) *
    (lastIds.has(universeId) ? 1 : MISSING_LATEST_DECAY) *
    (trendingIds.has(universeId) ? TRENDING_BOOST : 1);
  for (const obs of series) {
    // Half-life weighting: a read 6h old counts half, 12h old a quarter, and
    // anything older than ~2 days is negligible (weight < 0.005 is skipped).
    const weight = Math.pow(0.5, (endTime - obs.t) / HALF_LIFE_MS);
    if (weight < 0.005) continue;
    // Only the freshest reads (last 48h) get the momentum shift applied.
    const appliedShift = obs.t >= endTime - 48 * 3600 * 1000 ? shift : 0;
    const centre = obs.rank - appliedShift;
    for (let place = 1; place <= places; place++) {
      const z = place - centre;
      const kernel = Math.exp(-z * z * invTwoSigmaSq);
      if (kernel < 0.02) continue;
      addMass(universeId, place, weight * kernel * gameMultiplier);
    }
  }
}

// Normalise each place to NAMED_SHARE, prune rows below MIN_P (their mass stays
// unnamed — see header point 4 — it is NOT renormalised onto the survivors).
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
