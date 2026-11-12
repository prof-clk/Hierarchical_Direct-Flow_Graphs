from __future__ import annotations

import colorsys
import heapq
import html
import io
import itertools
import math
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from IPython.display import HTML, display

EPS = 1e-6
BEND_PEN = 60.0                        # each 90° turn = 60 px length penalty
DOOR_BALANCE_PX = 0.0                  # >0: door usage penalty (distributes congestion to doors)
_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIRV = {"E": (1, 0), "W": (-1, 0), "S": (0, 1), "N": (0, -1)}   # SVG: y down, x right
_DNAMES = "EWSN"                       # indices: E0 W1 S2 N3  (reverse direction = idx ^ 1)

def manhattan_fallback(exit_a, exit_b):
    _, y0 = exit_a
    x1, _ = exit_b
    mid = (x1, y0)
    if mid == exit_a or mid == exit_b:
        return [exit_a, exit_b]
    return [exit_a, mid, exit_b]


def _orthogonalize(points):
    out = [points[0]]
    for p in points[1:]:
        q = out[-1]
        if abs(p[0] - q[0]) > 1e-6 and abs(p[1] - q[1]) > 1e-6:
            out.append((p[0], q[1]))
        out.append(p)
    return out


def _simplify_points(pts, eps=1e-6):
    """Removes duplicate and collinear intermediate points; leaves endpoints untouched."""
    dedup = []
    for p in pts:
        if dedup and abs(p[0] - dedup[-1][0]) < eps and abs(p[1] - dedup[-1][1]) < eps:
            continue
        dedup.append(p)
    if len(dedup) < 3:
        return dedup
    res = [dedup[0]]
    for i in range(1, len(dedup) - 1):
        a, b, c = res[-1], dedup[i], dedup[i + 1]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
        if abs(cross) < eps and dot > 0:
            continue
        res.append(b)
    res.append(dedup[-1])
    return res


def _sdir(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    if abs(dx) > EPS and abs(dy) < EPS:
        return "E" if dx > 0 else "W"
    if abs(dy) > EPS and abs(dx) < EPS:
        return "S" if dy > 0 else "N"
    return None


def _path_cost(pts, d_start, d_end):
    """length + BEND_PEN*bends. Start/end directions must be perpendicular to the sides; backtracking is forbidden."""
    dirs = [d for d in (_sdir(p, q) for p, q in itertools.pairwise(pts)) if d]
    if not dirs:
        return math.inf, 0
    L = sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p, q in itertools.pairwise(pts))
    bends, prev = 0, d_start
    for d in dirs:
        if d != prev:
            bends += 1
            prev = d
    if dirs[-1] != d_end:
        bends += 1
    pen = (1e6 if dirs[0] == _OPP[d_start] else 0) + (1e6 if dirs[-1] == _OPP[d_end] else 0)
    return L + BEND_PEN * bends + pen, bends


def _direct_candidates(A, B):
    (x0, y0), (x1, y1) = A, B
    c = [[A, B]] if (abs(x0 - x1) < EPS or abs(y0 - y1) < EPS) else []
    c += [[A, (x1, y0), B], [A, (x0, y1), B]]
    for f in (0.5, 0.25, 0.75):
        mx, my = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        c += [[A, (mx, y0), (mx, y1), B], [A, (x0, my), (x1, my), B]]
    return c


def _port_xy(bb, side, t):
    x, y, w, h = bb
    return {"N": (t, y), "S": (t, y + h), "W": (x, t), "E": (x + w, t)}[side]


def _stub_xy(p, side, L):
    dx, dy = _DIRV[side]
    return (p[0] + dx * L, p[1] + dy * L)


def _side_candidates(Ca, Cb, tol=2.0):
    """Source sides based on the quadrant of the target center; target sides are their counterparts."""
    dx, dy = Cb[0] - Ca[0], Cb[1] - Ca[1]
    s = []
    if abs(dx) > tol:
        s.append("E" if dx > 0 else "W")
    if abs(dy) > tol:
        s.append("S" if dy > 0 else "N")
    if not s:
        s = ["E"]
    s.sort(key=lambda k: (k in "NS") if abs(dx) >= abs(dy) else (k in "EW"))
    return s, [_OPP[k] for k in s]


def _ideal_coord(side, bb, other, corner):
    """Ideal position along the side: center of overlapping regions if any, otherwise center of the side."""
    x, y, w, h = bb
    X, Y, W, H = other
    if side in "NS":
        lo, hi, a0, a1 = max(x, X), min(x + w, X + W), x, x + w
    else:
        lo, hi, a0, a1 = max(y, Y), min(y + h, Y + H), y, y + h
    c = (lo + hi) / 2 if hi - lo > 2 * corner else (a0 + a1) / 2
    return min(max(c, a0 + corner), a1 - corner)


# ---- numpy collision
def _arr(rects):
    return np.array(list(rects), float).reshape(-1, 4)


def _inf(a, m):
    return a + np.array([-m, -m, 2 * m, 2 * m]) if len(a) else a


def a_hit(p, q, arr, m=0.0):
    """if segment is parallel to an axis, checks if it intersects any rectangle in the array (tangent not counted)."""
    if len(arr) == 0:
        return False
    x0, x1 = (p[0], q[0]) if p[0] <= q[0] else (q[0], p[0])
    y0, y1 = (p[1], q[1]) if p[1] <= q[1] else (q[1], p[1])
    return bool(((x0 < arr[:, 0] + arr[:, 2] + m - EPS) & (x1 > arr[:, 0] - m + EPS) &
                 (y0 < arr[:, 1] + arr[:, 3] + m - EPS) & (y1 > arr[:, 1] - m + EPS)).any())


def a_path_hit(pts, arr, m=0.0):
    return any(a_hit(p, q, arr, m) for p, q in itertools.pairwise(pts))


def a_window(arr, x0, y0, x1, y1):
    if len(arr) == 0:
        return arr
    mk = (arr[:, 0] < x1) & (arr[:, 0] + arr[:, 2] > x0) & (arr[:, 1] < y1) & (arr[:, 1] + arr[:, 3] > y0)
    return arr[mk]


# ---- Hanan grid + bending-penalized Dijkstra (NumPy masked)
def r4_hanan(A, B, d_start, d_end, arr):
    rects = arr.tolist()
    xs = sorted({A[0], B[0]} | {v for r in rects for v in (r[0], r[0] + r[2])})
    ys = sorted({A[1], B[1]} | {v for r in rects for v in (r[1], r[1] + r[3])})
    nx, ny = len(xs), len(ys)
    xa, ya = np.array(xs), np.array(ys)
    vb = np.zeros((ny, nx), bool)
    hb = np.zeros((ny, max(nx - 1, 1)), bool)
    wb = np.zeros((max(ny - 1, 1), nx), bool)
    for rx, ry, rw, rh in rects:
        i0, i1 = int(np.searchsorted(xa, rx)), int(np.searchsorted(xa, rx + rw))
        j0, j1 = int(np.searchsorted(ya, ry)), int(np.searchsorted(ya, ry + rh))
        vb[j0 + 1:j1, i0 + 1:i1] = True
        hb[j0 + 1:j1, i0:i1] = True
        wb[j0:j1, i0 + 1:i1] = True
    vbl, hbl, wbl = vb.tolist(), hb.tolist(), wb.tolist()
    xi, yi = {x: i for i, x in enumerate(xs)}, {y: j for j, y in enumerate(ys)}
    si, sj, gi, gj = xi[A[0]], yi[A[1]], xi[B[0]], yi[B[1]]
    if vbl[sj][si] or vbl[gj][gi]:
        return None
    d0 = _DNAMES.index(d_start)
    de = _DNAMES.index(d_end)
    _h = lambda i, j: abs(xs[i] - xs[gi]) + abs(ys[j] - ys[gj]) 
    start = (si, sj, d0)
    best, prev, pq, cnt = {start: 0.0}, {}, [(_h(si, sj), 0, 0.0, start)], 0
    while pq:
        _, _, c, cur = heapq.heappop(pq)
        if best.get(cur, math.inf) < c - 1e-9:
            continue
        i, j, d = cur
        if (i, j) == (gi, gj):
            pts, key = [], cur
            while True:
                pts.append((xs[key[0]], ys[key[1]]))
                if key == start:
                    break
                key = prev[key]
            return _simplify_points(pts[::-1])
        for nd in range(4):
            if nd == (d ^ 1):
                continue
            if nd == 0:
                ok = i + 1 < nx and not hbl[j][i] and not vbl[j][i + 1]
                ni, nj = i + 1, j
            elif nd == 1:
                ok = i - 1 >= 0 and not hbl[j][i - 1] and not vbl[j][i - 1]
                ni, nj = i - 1, j
            elif nd == 2:
                ok = j + 1 < ny and not wbl[j][i] and not vbl[j + 1][i]
                ni, nj = i, j + 1
            else:
                ok = j - 1 >= 0 and not wbl[j - 1][i] and not vbl[j - 1][i]
                ni, nj = i, j - 1
            if not ok:
                continue
            nc = c + abs(xs[ni] - xs[i]) + abs(ys[nj] - ys[j]) + (BEND_PEN if nd != d else 0)
            if (ni, nj) == (gi, gj) and nd != de:
                nc += BEND_PEN
            key = (ni, nj, nd)
            if nc < best.get(key, math.inf) - 1e-9:
                best[key] = nc
                prev[key] = cur
                cnt += 1
                heapq.heappush(pq, (nc + _h(ni, nj), cnt, nc, key))
    return None


def r4_route_between(A, B, ds, de, arr, allow_hanan=True):
    if math.dist(A, B) < EPS:
        return [A]
    best = None
    for c in _direct_candidates(A, B):
        if not a_path_hit(c, arr):
            cost = _path_cost(c, ds, de)[0]
            if best is None or cost < best[0]:
                best = (cost, c)
    if best:
        return _simplify_points(best[1])
    if not allow_hanan:
        return None
    for pad in (200.0, 600.0, None):
        sub = arr if pad is None else a_window(arr, min(A[0], B[0]) - pad, min(A[1], B[1]) - pad,
                                               max(A[0], B[0]) + pad, max(A[1], B[1]) + pad)
        pts = r4_hanan(A, B, ds, de, sub)
        if pts and not a_path_hit(pts, arr):
            return pts
    return None


def r4_spread(ts, lo, hi, gap):
    n = len(ts)
    if n == 1:
        return [min(max(ts[0], lo), hi)]
    order = sorted(range(n), key=lambda i: ts[i])
    v = [ts[i] for i in order]
    for k in range(1, n):
        v[k] = max(v[k], v[k - 1] + gap)
    sh = (sum(ts) - sum(v)) / n
    v = [x + sh for x in v]
    if v[-1] > hi:
        d = v[-1] - hi
        v = [x - d for x in v]
    if v[0] < lo:
        d = lo - v[0]
        v = [x + d for x in v]
    if v[-1] > hi:
        v = [lo + (hi - lo) * k / (n - 1) for k in range(n)]
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = v[pos]
    return out


# ---- parallel strip separation
def r4_orient(p, q, eps=0.6):
    if abs(p[1] - q[1]) < eps and abs(p[0] - q[0]) > eps:
        return "H"
    if abs(p[0] - q[0]) < eps and abs(p[1] - q[1]) > eps:
        return "V"
    return None


def r4_try_move(pts, k, o, c, hard, min_end, apply=True):
    n = len(pts) - 1
    v0, v1 = pts[k], pts[k + 1]
    nv0 = (v0[0], c) if o == "H" else (c, v0[1])
    nv1 = (v1[0], c) if o == "H" else (c, v1[1])
    if a_hit(nv0, nv1, hard):
        return False
    for kk, pv, nv, far in ((k - 1, v0, nv0, pts[k - 1]), (k + 1, v1, nv1, pts[k + 2])):
        old = abs(far[0] - pv[0]) + abs(far[1] - pv[1])
        new = abs(far[0] - nv[0]) + abs(far[1] - nv[1])
        dot = (pv[0] - far[0]) * (nv[0] - far[0]) + (pv[1] - far[1]) * (nv[1] - far[1])
        if dot <= 0 or new < (min_end if kk in (0, n - 1) else 2.0):
            return False
        if new > old and a_hit(pv, nv, hard):
            return False
    if apply:
        pts[k], pts[k + 1] = nv0, nv1
    return True


def r4_separate(routes, hard_for, fixed_arr, gap, min_end, iters=5):
    """Splits overlapping or closely spaced PARALLEL segments lying on the same line into strips.
    Port segments and segments passing through doors are fixed; they do not enter box/wall/door blockers.
    Returns: (number of moved segments, {'fixed':.., 'no_space':..})."""
    stats = {"fixed": 0, "no_space": 0}
    total = 0
    for _it in range(iters):
        fixedflag = [[a_hit(p[k], p[k + 1], fixed_arr) for k in range(len(p) - 1)] for p in routes]
        segs = []
        for ri, pts in enumerate(routes):
            n = len(pts) - 1
            ff = fixedflag[ri]
            for k in range(n):
                p, q = pts[k], pts[k + 1]
                o = r4_orient(p, q)
                if o is None:
                    continue
                c = p[1] if o == "H" else p[0]
                a_, b_ = sorted((p[0], q[0])) if o == "H" else sorted((p[1], q[1]))
                mv = 0 < k < n - 1 and not ff[k] and not ff[k - 1] and not ff[k + 1]
                segs.append({"r": ri, "k": k, "o": o, "c": c, "a": a_, "b": b_, "mv": mv})
        parent = list(range(len(segs)))

        def find(i, parent=parent):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        conflict = False
        for o in "HV":
            idx = sorted([i for i, s in enumerate(segs) if s["o"] == o], key=lambda i: segs[i]["c"])
            for ii, i in enumerate(idx):
                si = segs[i]
                for j in idx[ii + 1:]:
                    sj = segs[j]
                    if sj["c"] - si["c"] >= gap * 0.999:
                        break
                    if si["r"] != sj["r"] and min(si["b"], sj["b"]) - max(si["a"], sj["a"]) > 1.0 \
                            and (si["mv"] or sj["mv"]):
                        parent[find(i)] = find(j)
                        conflict = True
        if not conflict:
            break
        comps = defaultdict(list)
        for i in range(len(segs)):
            comps[find(i)].append(i)
        moved = 0
        for members in comps.values():
            mv = [i for i in members if segs[i]["mv"]]
            stats["fixed"] += len(members) - len(mv)
            if len(members) < 2 or not mv:
                continue
            o = segs[members[0]]["o"]
            placed = [(segs[i]["c"], segs[i]["a"], segs[i]["b"], segs[i]["r"]) for i in members if not segs[i]["mv"]]

            def sign(i, o=o):
                s = segs[i]  # noqa: B023
                pts = routes[s["r"]]
                k = s["k"]

                def sg(nb, cur):
                    d = (nb[1] - cur[1]) if o == "H" else (nb[0] - cur[0])
                    return -1 if d < -0.6 else (1 if d > 0.6 else 0)

                return sg(pts[k - 1], pts[k]) + sg(pts[k + 2], pts[k + 1])

            mv.sort(key=lambda i: (sign(i), segs[i]["c"], segs[i]["r"]))
            cm = float(np.mean([segs[i]["c"] for i in members]))
            for rank, i in enumerate(mv):
                s = segs[i]
                pts = routes[s["r"]]
                k = s["k"]
                p, q = pts[k], pts[k + 1]
                a_, b_ = sorted((p[0], q[0])) if o == "H" else sorted((p[1], q[1]))
                desired = cm + (rank - (len(mv) - 1) / 2.0) * gap
                hard = hard_for(s["r"])
                chosen = None
                for c in sorted((desired + t * gap * 0.5 for t in range(-40, 41)), key=lambda c: abs(c - desired)):
                    if any(pr != s["r"] and abs(c - pc) < gap - 1e-6 and min(b_, pb) - max(a_, pa) > 1.0
                           for pc, pa, pb, pr in placed):
                        continue
                    if r4_try_move(pts, k, o, c, hard, min_end):
                        chosen = c
                        break
                if chosen is None:
                    stats["no_space"] += 1
                    placed.append((s["c"], a_, b_, s["r"]))
                else:
                    placed.append((chosen, a_, b_, s["r"]))
                    moved += 1
        for ri in range(len(routes)):
            routes[ri] = _simplify_points(routes[ri])
        total += moved
        if moved == 0:
            break
    return total, stats

# ============================================================================
# SINGLE SOURCE OF TRUTH: all domain-config flows through here; no function
# reads bare module-level globals anymore. Every function accepts
# `cfg: EventLogConfig`. frozen=True -> protects against accidental
# mutation (a shared object in a multi-user environment cannot be
# modified inadvertently).
# ============================================================================

@dataclass(frozen=True)
class HierLevel:
    code_col: str
    name_col: str | None
    label: str


@dataclass(frozen=True)
class EventLogConfig:
    case_id_col: str
    activity_code_col: str
    activity_name_col: str | None
    timestamp_col: str | None          # point-event log if not None
    start_timestamp_col: str | None    # interval-event log if not None
    end_timestamp_col: str | None
    resource_cols: tuple[str, ...]     # no longer singular — multiple resource dimensions (User, Vehicle, Operator, ...)
    other_cols: tuple[str, ...]
    hierarchy_levels: tuple[HierLevel, ...]
    is_flow_event_col: str | None = None
    flow_attach_level_col: str | None = None
    flow_attach_code_col: str | None = None
    flow_timestamp_col: str | None = None   # The dedicated timestamp column for flow events (Place Order, etc.)

    @property
    def flow_event_ts_col(self) -> str | None:
        """Flow events timestamp column: the explicitly selected one, or the default flow_ts_col."""
        return self.flow_timestamp_col or self.flow_ts_col

    @property
    def show_start_finish_box(self) -> bool:
        return self.start_timestamp_col is not None and self.end_timestamp_col is not None

    @property
    def flow_ts_col(self) -> str | None:
        """Flow events timestamp column: the dedicated column for flow events (Place Order, etc.), or the default."""
        return self.timestamp_col if self.timestamp_col else self.start_timestamp_col

    @property
    def op_ts_col(self) -> str | None:
        """Timestamp of ordinary operation rows (start time if available, else the point timestamp)."""
        return self.start_timestamp_col if self.start_timestamp_col else self.timestamp_col

    @property
    def hier_code_cols(self) -> tuple[str, ...]:
        return tuple(lvl.code_col for lvl in self.hierarchy_levels)

# ============================================================================
# 1) FILE LOADING — not depended on global config, just IO
# ============================================================================
def load_event_log(file_bytes: bytes, file_name: str, sheet_name=0) -> pd.DataFrame:
    """CSV or XLSX raw text (dtype=str) is read as-is. Column meanings
    are not yet known — this is resolved by prepare_dataframe() with the cfg."""
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), dtype=str)
    if file_name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, dtype=str, engine="openpyxl")
    raise ValueError(f"Unsupported file type: {file_name!r} (only .csv / .xlsx)")


# ============================================================================
# 2) VALIDATION + TYPE CONVERSION — the cfg-enabled 
# ============================================================================
def prepare_dataframe(raw: pd.DataFrame, cfg: EventLogConfig) -> pd.DataFrame:
    """Validates required columns and sets their types (str/category/datetime).
    Raises ValueError if any required column is missing — display in the calling
    context with st.error(), similar to the notebook."""
    df = raw.copy()

    required = [cfg.case_id_col, cfg.activity_code_col] + list(cfg.hier_code_cols)
    if cfg.show_start_finish_box:
        required += [cfg.start_timestamp_col, cfg.end_timestamp_col]
    else:
        required += [cfg.timestamp_col]
    if getattr(cfg, "flow_timestamp_col", None):
        required.append(cfg.flow_timestamp_col)
    missing = [c for c in required if c and c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}\nAvailable columns: {list(df.columns)}"
        )

    str_cols = list(cfg.hier_code_cols) + [cfg.activity_code_col]
    for lvl in cfg.hierarchy_levels:
        if lvl.name_col:
            str_cols.append(lvl.name_col)
    if cfg.activity_name_col:
        str_cols.append(cfg.activity_name_col)
    str_cols += list(cfg.resource_cols) + list(cfg.other_cols)
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    ts_cols = [c for c in [cfg.timestamp_col, cfg.start_timestamp_col, cfg.end_timestamp_col,
                           getattr(cfg, 'flow_timestamp_col', None)] if c]
    for col in ts_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in list(cfg.hier_code_cols) + [cfg.activity_code_col]:
        df[col] = df[col].astype("category")
    return df


# ============================================================================
# 3) HELPER FUNCTIONS — do not require cfg
# ============================================================================
def safe_sort(frame: pd.DataFrame, cols) -> pd.DataFrame:
    if isinstance(cols, str):
        cols = [cols]
    if len(frame) == 0:
        return frame.reset_index(drop=True)
    keys = [frame[c].to_numpy() for c in reversed(cols)]
    order = np.lexsort(keys)
    return frame.iloc[order].reset_index(drop=True)


def format_duration(seconds):
    seconds = abs(seconds)
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.0f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"


# ============================================================================
# 4) NODE-KEY — AGG_LEVEL/HIERARCHY_LEVELS is no longer global; it is passed as a parameter:
# cfg + agg_level. agg_level: int (0..len(hierarchy_levels)-1) or
# the string "activity" (the most detailed level).
# ============================================================================
def agg_level_index(cfg: EventLogConfig, agg_level) -> int:
    return len(cfg.hierarchy_levels) if agg_level == "activity" else int(agg_level)


def _base_from_parts(parts, idx: int) -> str:
    """The core of `node_key` — `compute_node_stats`/`build_base_paths` in Section 4 
    — was extracted into a separate function so that it could also generate the *same* base string."""
    return f"lvl{idx}_" + "_".join(re.sub(r"[^a-zA-Z0-9]", "", str(p)) or "NA" for p in parts)


def node_key(level_values, cfg: EventLogConfig, agg_level) -> str:
    idx = agg_level_index(cfg, agg_level)
    parts = [str(v) for v in level_values[: idx + 1]]
    return _base_from_parts(parts, idx)


def norm_code(x) -> str:
    """For code comparison: remove spaces, '9.0' -> '9', ignore leading zeros ('009' == '9')."""
    t = str(x).strip()
    t = re.sub(r"^(-?\d+)\.0+$", r"\1", t)
    return t.lstrip("0") or "0" if t.isdigit() else t.upper()


def flow_node_key(attach_level: int, attach_code, activity_label: str) -> str:
    clean_label = re.sub(r"[^a-zA-Z0-9]", "_", str(activity_label))
    return f"flow_L{attach_level}_{re.sub(r'[^a-zA-Z0-9]', '', str(attach_code))}__{clean_label}"


# ============================================================================
# 5) EVENT LISTS SETUP — fully cfg-based
# ============================================================================
def _non_flow_df(df: pd.DataFrame, cfg: EventLogConfig) -> pd.DataFrame:
    if cfg.is_flow_event_col and cfg.is_flow_event_col in df.columns:
        mask = df[cfg.is_flow_event_col].astype(str).str.lower().isin(["true", "1"])
        return df[~mask]
    return df


def _row_level_values(row, cfg: EventLogConfig):
    return tuple(row[c] for c in cfg.hier_code_cols) + (row[cfg.activity_code_col],)


def make_start_events(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> pd.DataFrame:
    frame = _non_flow_df(df, cfg).copy()
    frame = frame.dropna(subset=[cfg.start_timestamp_col])
    frame["timestamp"] = frame[cfg.start_timestamp_col].astype("datetime64[ns]").astype("int64")
    frame["node_id"] = "start_" + frame.apply(
        lambda r: node_key(_row_level_values(r, cfg), cfg, agg_level), axis=1
    )
    act_label = (frame[cfg.activity_name_col].astype(str)
                 if cfg.activity_name_col else frame[cfg.activity_code_col].astype(str))
    frame["activity"] = "Start: " + act_label
    frame["node_label"] = "Start"
    frame["case_id"] = frame[cfg.case_id_col].astype(str)
    return frame[["case_id", "timestamp", "activity", "node_id", "node_label"]]


def make_finish_events(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> pd.DataFrame:
    frame = _non_flow_df(df, cfg).copy()
    frame = frame.dropna(subset=[cfg.end_timestamp_col])
    frame["timestamp"] = frame[cfg.end_timestamp_col].astype("datetime64[ns]").astype("int64")
    frame["node_id"] = "finish_" + frame.apply(
        lambda r: node_key(_row_level_values(r, cfg), cfg, agg_level), axis=1
    )
    act_label = (frame[cfg.activity_name_col].astype(str)
                 if cfg.activity_name_col else frame[cfg.activity_code_col].astype(str))
    frame["activity"] = "Finish: " + act_label
    frame["node_label"] = "Finish"
    frame["case_id"] = frame[cfg.case_id_col].astype(str)
    return frame[["case_id", "timestamp", "activity", "node_id", "node_label"]]


def make_point_events(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> pd.DataFrame:
    frame = _non_flow_df(df, cfg).copy()
    frame = frame.dropna(subset=[cfg.timestamp_col])
    frame["timestamp"] = frame[cfg.timestamp_col].astype("datetime64[ns]").astype("int64")
    frame["node_id"] = frame.apply(
        lambda r: node_key(_row_level_values(r, cfg), cfg, agg_level), axis=1
    )
    act_label = (frame[cfg.activity_name_col].astype(str)
                 if cfg.activity_name_col else frame[cfg.activity_code_col].astype(str))
    frame["activity"] = act_label
    frame["node_label"] = None
    frame["case_id"] = frame[cfg.case_id_col].astype(str)
    return frame[["case_id", "timestamp", "activity", "node_id", "node_label"]]


def make_flow_events(df: pd.DataFrame, cfg: EventLogConfig):
    """Events like Place Order / Request Order Complete. If not present in the dataset
    (cfg.is_flow_event_col is not set), an empty result is returned."""
    empty = pd.DataFrame(columns=["case_id", "timestamp", "activity", "node_id", "node_label"]), {}
    if not (cfg.is_flow_event_col and cfg.is_flow_event_col in df.columns):
        return empty
    mask = df[cfg.is_flow_event_col].astype(str).str.lower().isin(["true", "1"])
    if not mask.any():
        return empty
    ts_col = cfg.flow_event_ts_col
    if not (cfg.flow_attach_level_col and cfg.flow_attach_code_col and ts_col):
        print("⚠️ There are IsFlowEvent marked rows but FlowAttachLevel/Code or "
                   "timestamp column is missing; flow events are being skipped.")
        return empty
    frame = df[mask].copy()
    frame = frame.dropna(subset=[ts_col, cfg.flow_attach_level_col, cfg.flow_attach_code_col])
    if frame.empty:
        return empty
    frame["timestamp"] = frame[ts_col].astype("datetime64[ns]").astype("int64")
    frame["node_id"] = frame.apply(
        lambda r: flow_node_key(int(float(r[cfg.flow_attach_level_col])),
                                r[cfg.flow_attach_code_col], r[cfg.activity_code_col]),
        axis=1,
    )
    frame["activity"] = frame[cfg.activity_code_col].astype(str)
    frame["node_label"] = frame["activity"]
    frame["case_id"] = frame[cfg.case_id_col].astype(str)
    attach_info = {}
    for _, r in frame.drop_duplicates("node_id").iterrows():
        attach_info[r["node_id"]] = (int(float(r[cfg.flow_attach_level_col])), r[cfg.flow_attach_code_col])
    return frame[["case_id", "timestamp", "activity", "node_id", "node_label"]], attach_info


def build_event_log(df: pd.DataFrame, cfg: EventLogConfig, agg_level):
    """start/finish/point/flow events is combined to produce a single event_log sorted by time."""
    empty = pd.DataFrame(columns=["case_id", "timestamp", "activity", "node_id", "node_label"])
    if cfg.show_start_finish_box:
        start_events = make_start_events(df, cfg, agg_level)
        finish_events = make_finish_events(df, cfg, agg_level)
        point_events = empty.copy()
    else:
        start_events = empty.copy()
        finish_events = empty.copy()
        point_events = make_point_events(df, cfg, agg_level)
    flow_events, flow_attach_info = make_flow_events(df, cfg)

    event_log = pd.concat(
        [e for e in [start_events, finish_events, point_events, flow_events] if not e.empty],
        ignore_index=True,
    )
    event_log = event_log.dropna(subset=["timestamp"])
    event_log["timestamp"] = pd.to_datetime(event_log["timestamp"])
    event_log = safe_sort(event_log, ["case_id", "timestamp"])
    return event_log, flow_attach_info


# ============================================================================
# 6) DFG DISCOVERY
# ============================================================================
def discover_dfg_full_vectorized(df, case_col="case_id", act_col="node_id", time_col="timestamp"):
    d = safe_sort(df[[case_col, act_col, time_col]], [case_col, time_col])
    nxt_act = d.groupby(case_col, sort=False)[act_col].shift(-1)
    pairs = pd.DataFrame({"src": d[act_col].to_numpy(), "dst": nxt_act.to_numpy()})
    pairs = pairs.dropna(subset=["dst"])
    dfg = {}
    for key, c in pairs.value_counts(["src", "dst"]).items():
        if not isinstance(key, tuple):
            continue
        s, t = key
        dfg[(s, t)] = int(c)
    g = d.groupby(case_col, sort=False)[act_col]
    start_activities = g.first().value_counts().to_dict()
    end_activities = g.last().value_counts().to_dict()
    return dfg, start_activities, end_activities


def discover_performance_dfg_vectorized(df, case_col="case_id", act_col="node_id", time_col="timestamp"):
    d = safe_sort(df[[case_col, act_col, time_col]], [case_col, time_col])
    d["next_act"] = d.groupby(case_col, sort=False)[act_col].shift(-1)
    d["next_time"] = d.groupby(case_col, sort=False)[time_col].shift(-1)
    d = d.dropna(subset=["next_act"])
    d["duration"] = (d["next_time"] - d[time_col]).dt.total_seconds()
    stats = d.groupby([act_col, "next_act"])["duration"].agg(["mean", "min", "max"])
    result = {}
    for idx, row in stats.iterrows():
        if not isinstance(idx, tuple) or len(idx) != 2:
            continue
        s, t = idx
        result[(s, t)] = {"mean": row["mean"], "min": row["min"], "max": row["max"]}
    return result


def bfs_reachable(start, adj):
    from collections import deque
    visited = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in visited:
                visited.add(v); q.append(v)
    return visited


def filter_dfg_on_paths_percentage_faithful(dfg0, start_activities0, end_activities0, activities_count0,
                                            percentage, keep_all_activities=False, keep_nodes=()):
    dfg = dict(dfg0)
    start_activities = dict(start_activities0)
    end_activities = dict(end_activities0)
    activities_count = dict(activities_count0)
    if len(activities_count) <= 1 or len(dfg) <= 1:
        return dfg, start_activities, end_activities, activities_count
    START, END = "_S_START_", "_S_END_"
    nodes = set(activities_count) | {a for e in dfg for a in e} | set(start_activities) | set(end_activities)
    adj = {n: set() for n in nodes}
    rev = {n: set() for n in nodes}
    adj[START] = set(start_activities)
    rev[START] = set()
    adj[END] = set()
    rev[END] = set(end_activities)
    for (a, b) in dfg:
        adj[a].add(b); rev[b].add(a)
    for a in start_activities:
        rev.setdefault(a, set()).add(START)
    for a in end_activities:
        adj.setdefault(a, set()).add(END)
    all_edges = (list(dfg.items())
                 + [((START, a), f) for a, f in start_activities.items()]
                 + [((a, END), f) for a, f in end_activities.items()])
    all_edges.sort(key=lambda x: (x[1], x[0]), reverse=True)
    cut_idx = math.ceil((len(all_edges) - 1) * percentage) + 1
    non_discardable = [e for e, _ in all_edges[:cut_idx]]
    discardable = [e for e, _ in all_edges[cut_idx:]]
    discardable.reverse()
    if keep_all_activities:
        must_keep = set(activities_count) | set(start_activities) | set(end_activities)
    else:
        must_keep = {u for (u, v) in non_discardable if u != START} | \
                    {v for (u, v) in non_discardable if v != END}
    # Flow events (Place Order, etc.) may be infrequent; they are always preserved
    # (added to the reachability condition) so that the path filter does not silently discard them.
    must_keep |= (set(keep_nodes) & nodes)
    for (u, v) in discardable:
        if v in adj.get(u, ()):
            adj[u].discard(v); rev[v].discard(u)
            if must_keep.issubset(bfs_reachable(START, adj)) and \
               must_keep.issubset(bfs_reachable(END, rev)):
                if (u, v) in dfg:
                    del dfg[(u, v)]
                elif u == START and v in start_activities:
                    del start_activities[v]
                elif v == END and u in end_activities:
                    del end_activities[u]
            else:
                adj[u].add(v); rev[v].add(u)
    reach_fwd = bfs_reachable(START, adj)
    reach_bwd = bfs_reachable(END, rev)
    keep = (reach_fwd & reach_bwd) - {START, END}
    activities_count = {a: c for a, c in activities_count.items() if a in keep}
    start_activities = {a: c for a, c in start_activities.items() if a in keep}
    end_activities = {a: c for a, c in end_activities.items() if a in keep}
    dfg = {(a, b): c for (a, b), c in dfg.items() if a in keep and b in keep}
    return dfg, start_activities, end_activities, activities_count


# ============================================================================
# 7) Derived data for the SVG engine (Section 4)
# Section 4 no longer calculates *any* data—it simply looks up values ​​in these dictionaries
# and performs the layout/drawing. This leaves a *single* source of truth (this file)
# instead of two parallel event/DFG implementations.
# ============================================================================
def build_base_paths(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> dict:
    """Maps base-key strings (e.g., 'lvl1_00900901') to the tuple of codes (code0, code1, ..., [activity code])
    required by Section 4 to build the box tree. The node_id string is not split — it is read from this dictionary."""
    idx = agg_level_index(cfg, agg_level)
    key_cols = (list(cfg.hier_code_cols) + [cfg.activity_code_col])[: idx + 1]
    ops = _non_flow_df(df, cfg)
    sub = ops[key_cols].drop_duplicates()
    out = {}
    for _, row in sub.iterrows():
        parts = tuple(str(row[c]) for c in key_cols)
        out[_base_from_parts(parts, idx)] = parts
    return out


def compute_node_stats(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> dict:
    """Computes event counts and (if applicable) duration statistics for each base-key. Section 4
    uses these to display 'n=...' / 'avg=...' in the box titles."""
    idx = agg_level_index(cfg, agg_level)
    key_cols = (list(cfg.hier_code_cols) + [cfg.activity_code_col])[: idx + 1]
    ops = _non_flow_df(df, cfg).copy()
    if cfg.show_start_finish_box:
        ops["_dur"] = (ops[cfg.end_timestamp_col] - ops[cfg.start_timestamp_col]).dt.total_seconds()
    g = ops.groupby(key_cols, observed=True)
    sizes = g.size()
    means = g["_dur"].mean() if cfg.show_start_finish_box else None
    mins = g["_dur"].min() if cfg.show_start_finish_box else None
    maxs = g["_dur"].max() if cfg.show_start_finish_box else None
    def _lookup_group_stat(stat: pd.Series | None, key_vals):
        if stat is None:
            return None
        vals = stat.loc[[key_vals]]
        if isinstance(vals, pd.Series):
            return float(vals.iloc[0])
        return float(vals)

    out = {}
    for key_vals, n in sizes.items():
        parts = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        base = _base_from_parts([str(p) for p in parts], idx)
        dur_mean = None
        dur_min = None
        dur_max = None
        if cfg.show_start_finish_box:
            dur_mean = _lookup_group_stat(means, key_vals)
            dur_min = _lookup_group_stat(mins, key_vals)
            dur_max = _lookup_group_stat(maxs, key_vals)
        out[base] = {
            "n": int(n),
            "dur_mean": dur_mean,
            "dur_min": dur_min,
            "dur_max": dur_max,
        }
    return out


def build_names(df: pd.DataFrame, cfg: EventLogConfig) -> dict:
    """Maps (code0, ..., codeN) tuples to readable names (Work Center name, Operation name, ...)."""
    ops = _non_flow_df(df, cfg)
    names = {}
    for i, lvl in enumerate(cfg.hierarchy_levels):
        if lvl.name_col and lvl.name_col in ops.columns:
            cc = list(cfg.hier_code_cols[: i + 1])
            for r in ops[cc + [lvl.name_col]].astype(str).drop_duplicates(cc).itertuples(index=False):
                names[tuple(r[:-1])] = r[-1]
    if cfg.activity_name_col and cfg.activity_name_col in ops.columns:
        cc = list(cfg.hier_code_cols) + [cfg.activity_code_col]
        for r in ops[cc + [cfg.activity_name_col]].astype(str).drop_duplicates(cc).itertuples(index=False):
            names[tuple(r[:-1])] = r[-1]
    return names


def compute_node_scores(event_log: pd.DataFrame) -> dict:
    """The event_log must be sorted by case_id (build_event_log performs this sorting).
    For each node_id, it returns the average relative position within a case (0 = start of case, 1 = end of case). 
    This value is used for positioning Section 4's flow boxes before/after the process and for the SVG_ORDER_BY='flow' sorting order."""
    case_arr = event_log["case_id"].to_numpy()
    n = len(case_arr)
    if n == 0:
        return {}
    starts = np.r_[0, np.flatnonzero(case_arr[1:] != case_arr[:-1]) + 1]
    sizes = np.diff(np.r_[starts, n])
    idx = np.arange(n) - np.repeat(starts, sizes)
    pos = idx / np.maximum(np.repeat(sizes, sizes) - 1, 1)
    return pd.Series(pos).groupby(event_log["node_id"].to_numpy()).mean().to_dict()


def compute_child_rows(df: pd.DataFrame, cfg: EventLogConfig, agg_level) -> dict:
    """It calculates the distribution of the level immediately below AGG_LEVEL for each base: at the hierarchy level, 
    this is the sub-hierarchy code (🏭); at the 'activity' level, 
    it includes ALL resource columns (👤, now multiple) + other dimensions (📎). 
    We do not concatenate the label text here—'n=...' will be displayed in 'freq' mode and 'avg=...' in 'perf' mode; 
    since Section 4 decides which one to show at render time, we return the raw fields (icon/label_base/n/dur_*). 
    Sorting is already done in descending order by count; the "top N rows" truncation is a VIEW setting, 
    so it is handled in Section 4, not here."""
    idx = agg_level_index(cfg, agg_level)
    ops = _non_flow_df(df, cfg).copy()
    if cfg.show_start_finish_box:
        ops["_dur"] = (ops[cfg.end_timestamp_col] - ops[cfg.start_timestamp_col]).dt.total_seconds()
    parent_cols = (list(cfg.hier_code_cols) + [cfg.activity_code_col])[: idx + 1]

    dim_specs = []   # (column, icon, name_column, label_prefix)
    if idx < len(cfg.hierarchy_levels):
        if idx + 1 < len(cfg.hierarchy_levels):
            nxt = cfg.hierarchy_levels[idx + 1]
            dim_specs.append((nxt.code_col, "🏭", nxt.name_col, ""))
        else:
            dim_specs.append((cfg.activity_code_col, "🏭", cfg.activity_name_col, ""))
    else:
        dim_specs += [(c, "👤", None, "") for c in cfg.resource_cols]
        dim_specs += [(c, "📎", None, f"{c}: ") for c in cfg.other_cols]

    out: dict = {}
    for col, icon, name_col, prefix in dim_specs:
        if not col or col not in ops.columns:
            continue
        group_cols = parent_cols + [col]
        g = ops.groupby(group_cols, observed=True)
        sizes = g.size().rename("cnt").reset_index()
        sizes["neg_cnt"] = -sizes["cnt"]
        sizes = safe_sort(sizes, parent_cols + ["neg_cnt"])
        dur = g["_dur"].agg(["mean", "min", "max"]) if cfg.show_start_finish_box else None
        name_lookup = {}
        if name_col and name_col in ops.columns:
            name_lookup = ops.drop_duplicates(col).set_index(col)[name_col].astype(str).to_dict()

        for _, r in sizes.iterrows():
            parent_key = tuple(str(r[c]) for c in parent_cols)
            base = _base_from_parts(parent_key, idx)
            full_key = tuple(r[c] for c in group_cols)
            d = dur.loc[full_key] if dur is not None else None
            label_base = f"{prefix}{r[col]}"
            if r[col] in name_lookup:
                label_base += f" {name_lookup[r[col]]}"
            out.setdefault(base, []).append({
                "icon": icon, "label_base": label_base, "n": int(r["cnt"]),
                "dur_mean": float(d["mean"]) if d is not None else None,
                "dur_min": float(d["min"]) if d is not None else None,
                "dur_max": float(d["max"]) if d is not None else None,
            })
    return out


def compute_node_degrees(dfg_filtered: dict) -> dict:
    """Compute the total connection weight in the filtered DFG — 
    used to separate isolated (unconnected) boxes to the bottom when SVG_ORDER_BY='degree'."""
    deg = {}
    for (s, t), c in dfg_filtered.items():
        deg[s] = deg.get(s, 0) + c
        deg[t] = deg.get(t, 0) + c
    return deg


# ============================================================================
# 8) MULTI-USER ORCHESTRATION — using `st.cache_data`. 
# It automatically recalculates if `cfg`, `agg_level`, or `path_filter_percentage` change; 
# returns the result from the cache if the same user returns with the same file and settings; 
# and creates a separate cache entry if a different user arrives with a different file or configuration.
# ============================================================================
@st.cache_data(show_spinner="Processing event log and extracting DFG...")
def build_pipeline(raw_df: pd.DataFrame, cfg: EventLogConfig,
                   agg_level, path_filter_percentage: float):
    df = prepare_dataframe(raw_df, cfg)
    event_log, flow_attach_info = build_event_log(df, cfg, agg_level)

    dfg_c, sa_c, ea_c = discover_dfg_full_vectorized(event_log)
    perf_c = discover_performance_dfg_vectorized(event_log)
    ac_c = event_log["node_id"].value_counts().to_dict()
    dfg_f, _, _, ac_f = filter_dfg_on_paths_percentage_faithful(
        dfg_c, sa_c, ea_c, ac_c, percentage=path_filter_percentage,
        keep_nodes=set(flow_attach_info)
    )
    node_label_lookup = event_log.drop_duplicates("node_id").set_index("node_id")["node_label"]

    return {
        "df": df, "event_log": event_log,
        "dfg_full": dfg_c, "perf_full": perf_c,
        "dfg_filtered": dfg_f, "activities_filtered": ac_f,
        "flow_attach_info": flow_attach_info,
        "node_label_lookup": node_label_lookup,
        # ---- Section 4 (SVG engine) JUST reads these, doesn't compute them ----
        "base_paths": build_base_paths(df, cfg, agg_level),
        "node_stats": compute_node_stats(df, cfg, agg_level),
        "names": build_names(df, cfg),
        "scores": compute_node_scores(event_log),
        "deg": compute_node_degrees(dfg_f),
        "child_rows": compute_child_rows(df, cfg, agg_level),
    }


# ============================================================================
# 9) DIAGNOSIS — where are the flow events being lost?
# ============================================================================
def diagnose_flow(df: pd.DataFrame, cfg: EventLogConfig, pipeline: dict) -> dict:
    """Returns: {"steps": [(step, count)], "notes": [str], "flag_values": dict}."""
    steps, notes = [], []
    col = cfg.is_flow_event_col
    if not col or col not in df.columns:
        return {"steps": [("IsFlowEvent column is not selected", 0)], 
                "notes": ["Please select the '5. Flow events' checkbox in the sidebar and choose the columns."], 
                "flag_values": {}
                }
    flag_values = df[col].astype(str).value_counts().head(8).to_dict()
    mask = df[col].astype(str).str.lower().isin(["true", "1"])
    fl = df[mask]
    steps.append(("IsFlowEvent = true/1 rows", len(fl)))
    if len(fl) == 0:
        notes.append(f"Flag column values: {flag_values}. Only 'true' / '1' are accepted.")
        return {"steps": steps, "notes": notes, "flag_values": flag_values}
    ts = cfg.flow_event_ts_col
    n1 = int(fl[ts].notna().sum()) if ts in fl.columns else 0
    steps.append((f"... timestamp column '{ts}' is filled", n1))
    if n1 == 0:
        notes.append(f"'{ts}' flow rows are empty. Please select a column with filled values (e.g., OrderDate/timestamp)")
    n2 = int(fl[cfg.flow_attach_level_col].notna().sum())
    n3 = int(fl[cfg.flow_attach_code_col].notna().sum())
    steps.append((f"... '{cfg.flow_attach_level_col}' is filled", n2))
    steps.append((f"... '{cfg.flow_attach_code_col}' is filled", n3))
    attach = pipeline["flow_attach_info"]
    steps.append(("Generated flow nodes", len(attach)))
    steps.append(("In DFG (unfiltered)", len({n for e in pipeline["dfg_full"] for n in e if n in attach})))
    steps.append(("After filtering", len([n for n in pipeline["activities_filtered"] if n in attach])))
    avail = {}
    for parts in pipeline["base_paths"].values():
        for k in range(1, len(parts) + 1):
            avail.setdefault(k, set()).add(norm_code(parts[k - 1]))
    matched, bad = 0, []
    for (lvl, code) in attach.values():
        if norm_code(code) in avail.get(lvl + 1, set()):
            matched += 1
        else:
            bad.append((lvl, code))
    steps.append(("Target boxes found (drawable)", matched))
    if bad:
        samples = {k: sorted(v)[:5] for k, v in avail.items()}
        notes.append(f"Target boxes not found: {bad[:8]}. Available codes (level index+1 -> example): "
                     f"{samples}. FlowAttachLevel 0 = outer hierarchy level.")
    return {"steps": steps, "notes": notes, "flag_values": flag_values}

# Depends on EventLogConfig/HierLevel in section1_cfg_pipeline

# ============================================================================
# VISUAL / LAYOUT SETTINGS — independent of data schema (cfg), only determines
# how the visualization looks. If you want to add sliders in Streamlit later,
# you would connect them to the fields of this dataclass; each user session
# maintains their own RenderSettings() instance (no module-level global).
# ============================================================================
@dataclass
class RenderSettings:
    order_by: str = "code"                 # "degree" | "flow" | "volume" | "code"
    rankdir: str = "LR"                    # "LR" | "TB"
    top_child_rows: int = 0                # subrow limit in box; 0 = all
    edge_min_count: int = 1
    label_top_n: int = 40
    edge_min_w: float = 1.0
    edge_max_w: float = 2.0
    font: str = "Arial, Helvetica, sans-serif"
    target_aspect_top: float = 1.8
    iso_aspect: float = 10.0
    pad: float = 14
    gap_leaf: float = 60
    gap_group: float = 60
    gap_iso: float = 70
    margin: float = 50
    header_h: float = 78
    char_w: float = 0.60
    strip_w: float = 15
    activity_label: str = "Operation"


# ============================================================================
# Box — layout tree node (root / group / leaf / flow)
# ============================================================================
class Box:
    def __init__(self, kind, path, depth):
        self.kind = kind
        self.path = path
        self.depth = depth
        self.base = None 
        self.level_label = ""
        self.code = ""
        self.name = ""
        self.n = 0
        self.dur_mean = None
        self.dur_min = None
        self.dur_max = None
        self.child_rows = []
        self.grid_cols = 1
        self.grid_col_w = []
        self.short = ""
        self.tip = ""
        self.lines = []
        self.children = []
        self.flows = []
        self.flows_before = []
        self.flows_after = []
        self.score = 0.5
        self.side_before: bool | None = None
        self.iso_gap_y: float | None = None
        self.rx = self.ry = 0.0
        self.x = self.y = self.w = self.h = 0.0
        self.core_w = self.core_h = 0.0
        self.core_dx = self.core_dy = 0.0
        self.id: str | None = None


# ============================================================================
# node_id classification — no longer uses SEP-split. Section 1 generates node_ids
# with fixed prefixes ("start_", "finish_", "flow_L", or bare base) so we use simple
# prefix checks to distinguish them; the hierarchy path is read from
# pipeline["base_paths"] (no string parsing).
# ============================================================================
def _classify_node_id(nid: str):
    if nid.startswith("flow_L"):
        return "flow", nid
    if nid.startswith("start_"):
        return "start", nid[len("start_"):]
    if nid.startswith("finish_"):
        return "finish", nid[len("finish_"):]
    return "point", nid   # bare "lvl..." -> point event, node_id itself is the base


def fmt_dur(sec):
    sec = abs(float(sec))
    if sec < 60:
        return f"{sec:.0f} sec"
    if sec < 3600:
        return f"{sec / 60:.0f} min"
    if sec < 86400:
        return f"{sec / 3600:.1f} hr"
    return f"{sec / 86400:.1f} day"


def duration_to_color(value, vmin, vmax):
    if vmax <= vmin:
        t = 0.5
    else:
        t = min(max((value - vmin) / (vmax - vmin), 0.0), 1.0)
    green, yellow, red = (46, 204, 113), (241, 196, 15), (231, 76, 60)
    if t < 0.5:
        t2 = t / 0.5
        r = int(green[0] + (yellow[0] - green[0]) * t2)
        g = int(green[1] + (yellow[1] - green[1]) * t2)
        b = int(green[2] + (yellow[2] - green[2]) * t2)
    else:
        t2 = (t - 0.5) / 0.5
        r = int(yellow[0] + (red[0] - yellow[0]) * t2)
        g = int(yellow[1] + (red[1] - yellow[1]) * t2)
        b = int(yellow[2] + (red[2] - yellow[2]) * t2)
    return f"#{r:02x}{g:02x}{b:02x}"


# ============================================================================
# Build box tree from pipeline output
# ============================================================================
def _collect_ends(b: Box, cfg) -> list:
    """Returns the actual node_ids of a leaf/group box (start_.../finish_.../bare base).
    No longer uses string split — the b.base written during build_view is used."""
    if b.kind == "leaf":
        base = b.base
        return [f"start_{base}", f"finish_{base}"] if cfg.show_start_finish_box else [base]
    if b.kind == "group":
        ends = []
        for c in b.children:
            ends += _collect_ends(c, cfg)
        return ends
    return []


def _flow_side_before(nid, target, dfg_full: dict, scores: dict, cfg) -> bool:
    """It determines whether a flow event (Place Order, etc.) occurs *before* or *after* 
    a specific box in the process by examining the *actual* DFG weights of the flows entering or leaving that box."""
    ends = _collect_ends(target, cfg)
    into_t = sum(dfg_full.get((nid, e), 0) for e in ends)
    out_t = sum(dfg_full.get((e, nid), 0) for e in ends)
    if into_t == out_t:
        return scores.get(nid, 0.5) < 0.5
    return into_t > out_t


def total_degree(b: Box, deg: dict, cfg) -> int:
    """Returns the total degree of a box (and its descendants) in the filtered DFG.
    A degree of 0 is considered 'isolated', and layout() separates such nodes."""
    d = sum(deg.get(f.path[0], 0) for f in b.flows)
    if b.kind == "leaf":
        d += sum(deg.get(e, 0) for e in _collect_ends(b, cfg))
    elif b.kind == "group":
        for c in b.children:
            d += total_degree(c, deg, cfg)
    return d


def build_view(pipeline: dict, cfg, agg_level, settings: RenderSettings, only_codes=None):
    """pipeline: dictionary returned by section1_cfg_pipeline.build_pipeline().
    only_codes: filter based on the outermost (0th level) hierarchy code
    (e.g., {"009", "017"}); None = all."""

    ac_f = pipeline["activities_filtered"]
    base_paths = pipeline["base_paths"]
    node_stats = pipeline["node_stats"]
    names = pipeline["names"]
    child_rows_all = pipeline["child_rows"]
    scores = pipeline["scores"]

    root = Box("root", (), 0)
    index = {(): root}
    leaf_box, flow_box, by_last = {}, {}, {}

    def get_group(path):
        if path in index:
            return index[path]
        parent = get_group(path[:-1])
        b = Box("group", path, len(path))
        lvl = cfg.hierarchy_levels[len(path) - 1]
        b.level_label = lvl.label
        b.code = path[-1]
        b.name = names.get(path, "")
        b.short = f"{lvl.label} {path[-1]}"
        parent.children.append(b)
        index[path] = b
        by_last[(len(path), path[-1])] = path
        return b

    idx = agg_level_index(cfg, agg_level)

    bases = set()
    for nid in ac_f:
        kind, key = _classify_node_id(nid)
        if kind != "flow":
            bases.add(key)
    bases = sorted(bases)

    for base in bases:
        parts = base_paths.get(base)
        if parts is None:
            continue
        if only_codes and parts[0] not in only_codes:
            continue
        parent = get_group(parts[:-1]) if len(parts) > 1 else root
        stt = node_stats.get(base, {})
        n = int(stt.get("n", 1))
        code = parts[-1]
        nm = names.get(parts, "")
        b = Box("leaf", parts, len(parts))
        b.base = base
        b.n = n
        b.code = code
        b.name = nm
        b.level_label = (settings.activity_label if agg_level == "activity"
                         else cfg.hierarchy_levels[idx].label)
        b.dur_mean = stt.get("dur_mean")
        b.dur_min = stt.get("dur_min")
        b.dur_max = stt.get("dur_max")
        rows = child_rows_all.get(base, [])
        b.child_rows = rows[: settings.top_child_rows] if settings.top_child_rows else rows
        b.short = f"{code} · {nm}" if nm else code
        b.tip = b.short + f"\nn={n:,}" + (f"\navg={fmt_dur(b.dur_mean)}" if b.dur_mean is not None else "")
        sc = [scores[k] for k in (f"start_{base}", f"finish_{base}", base) if k in scores]
        b.score = float(np.mean(sc)) if sc else 0.5
        parent.children.append(b)
        leaf_box[base] = b
        index[parts] = b
        by_last[(len(parts), parts[-1])] = parts

    for nid in ac_f:
        kind, _ = _classify_node_id(nid)
        if kind != "flow":
            continue
        attach = pipeline["flow_attach_info"].get(nid)
        if attach is None:
            continue
        lvl, code = attach
        if only_codes and lvl == 0 and norm_code(code) not in {norm_code(c) for c in only_codes}:
            continue
        path = by_last.get((lvl + 1, str(code)))
        if path is None:   # '9' / '009' / '9.0' gibi biçim farklarına toleranslı eşleme
            path = next((p for (ln, c), p in by_last.items()
                         if ln == lvl + 1 and norm_code(c) == norm_code(code)), None)
        if path is None:
            continue
        target = index[path]
        label = pipeline["node_label_lookup"].get(nid, nid)
        b = Box("flow", (nid,), target.depth + 1)
        b.short = str(label)
        b.lines = [(str(label), 9, True, "#6E2C00")]
        b.tip = f"{label}  ({target.short})"
        b.score = scores.get(nid, 0.5)
        b.side_before = _flow_side_before(nid, target, pipeline["dfg_full"], scores, cfg)
        target.flows.append(b)
        flow_box[nid] = b

    def agg(b):
        if b.kind == "group":
            for c in b.children:
                agg(c)
            b.n = sum(c.n for c in b.children)
            b.score = (sum(c.score * c.n for c in b.children) / b.n) if b.n else 0.5
            means = [(c.dur_mean, c.n) for c in b.children if c.dur_mean is not None and c.n]
            b.dur_mean = (sum(d * n for d, n in means) / sum(n for _, n in means)) if means else None
            mins_ = [c.dur_min for c in b.children if c.dur_min is not None]
            maxs_ = [c.dur_max for c in b.children if c.dur_max is not None]
            b.dur_min = min(mins_) if mins_ else None
            b.dur_max = max(maxs_) if maxs_ else None
        elif b.kind == "root":
            for c in b.children:
                agg(c)
    agg(root)

    def prune(b):
        b.children = [c for c in b.children if c.kind != "group" or (prune(c) or c.children or c.flows)]
    prune(root)
    return root, leaf_box, flow_box


# ============================================================================
# Layout engine compeletely geometric
# ============================================================================
def tw(text, size, settings: RenderSettings, bold=False):
    return len(text) * size * (settings.char_w + (0.06 if bold else 0.0))


def shelf_pack(sizes, gap, target_aspect=1.6, max_width=None):
    if not sizes:
        return [], 0.0, 0.0

    def run(W):
        pos, x, y, rowh, usedw = [], 0.0, 0.0, 0.0, 0.0
        for w, h in sizes:
            if x > 0 and x + w > W + 1e-6:
                y += rowh + gap
                x, rowh = 0.0, 0.0
            pos.append((x, y))
            x += w + gap
            rowh = max(rowh, h)
            usedw = max(usedw, x - gap)
        return pos, usedw, y + rowh

    if max_width is not None:
        return run(max_width)
    maxw = max(w for w, _ in sizes)
    total = sum(w for w, _ in sizes) + gap * (len(sizes) - 1)
    best = None
    for W in np.linspace(maxw, max(total, maxw), 48):
        pos, uw, uh = run(W)
        score = uw * uh * (1 + abs(math.log((uw / uh) / target_aspect)))
        if best is None or score < best[0]:
            best = (score, pos, uw, uh)
    if best is None:
        pos, uw, uh = run(maxw)
        return pos, uw, uh
    return best[1], best[2], best[3]


def shelf_pack_by_columns(sizes, gap, n_cols):
    if not sizes:
        return [], 0.0, 0.0
    n_cols = max(1, min(n_cols, len(sizes)))
    col_w = [0.0] * n_cols
    for i, (w, h) in enumerate(sizes):
        c = i % n_cols
        col_w[c] = max(col_w[c], w)
    col_x = [0.0] * n_cols
    for c in range(1, n_cols):
        col_x[c] = col_x[c - 1] + col_w[c - 1] + gap
    total_w = col_x[-1] + col_w[-1]
    pos, row_y, row_h = [], 0.0, 0.0
    for i, (w, h) in enumerate(sizes):
        c = i % n_cols
        if c == 0 and i > 0:
            row_y += row_h + gap
            row_h = 0.0
        pos.append((col_x[c], row_y))
        row_h = max(row_h, h)
    total_h = row_y + row_h
    return pos, total_w, total_h


def _order(items, settings: RenderSettings):
    if settings.order_by == "volume":
        return sorted(items, key=lambda b: (-b.n, b.path))
    if settings.order_by == "code":
        return sorted(items, key=lambda b: b.path)
    return sorted(items, key=lambda b: (b.score, b.path))


def _hdr(depth):
    return {1: 13, 2: 11}.get(depth, 10)


def _hdr2(depth):
    return max(8, _hdr(depth) - 3)


def _hdr3(depth):
    return max(7, _hdr(depth) - 5)


def _fmt_metric_line(n, dur_mean, dur_min, dur_max, mode):
    if mode == "perf":
        if dur_mean is None:
            return "(no data)"
        dmin = fmt_dur(dur_min) if dur_min is not None else "-"
        dmax = fmt_dur(dur_max) if dur_max is not None else "-"
        return f"(avg={fmt_dur(dur_mean)}, min={dmin}, max={dmax})"
    return f"(n={n:,})"


def _header_lines(level_label, code, name, n, dur_mean, dur_min, dur_max, mode, fs1, fs2, fs3):
    lines = [(f"{level_label} Code: {code}", fs1, True)]
    if name:
        lines.append((name, fs2, False))
    lines.append((_fmt_metric_line(n, dur_mean, dur_min, dur_max, mode), fs3, False))
    return lines


def _header_height(lines):
    return sum(fsz + 3.5 for _, fsz, _ in lines)


def _grid_pack_rows(n):
    if n <= 0:
        return 1, 1
    n_rows = math.ceil(math.sqrt(n))
    n_cols = math.ceil(n / n_rows)
    return n_rows, n_cols


def _child_row_text(row: dict, mode: str) -> str:
    metric = _fmt_metric_line(row["n"], row["dur_mean"], row["dur_min"], row["dur_max"], mode)
    return f'{row["icon"]} {row["label_base"]} {metric}'


def stack_side(items, gap, vertical, max_extent=None):
    if not items:
        return [], 0.0, 0.0
    sizes = [(it.w, it.h) for it in items]
    if max_extent is None:
        max_extent = (sum(h for _, h in sizes) if vertical else sum(w for w, _ in sizes)) \
            + gap * (len(sizes) - 1)
    pos = []
    if vertical:
        cx = cy = col_w = total_w = total_h = 0.0
        for w, h in sizes:
            if cy > 0 and cy + h > max_extent + 1e-6:
                cx += col_w + gap
                cy, col_w = 0.0, 0.0
            pos.append((cx, cy))
            cy += h + gap
            col_w = max(col_w, w)
            total_w = max(total_w, cx + col_w)
            total_h = max(total_h, cy - gap)
    else:
        cx = cy = row_h = total_w = total_h = 0.0
        for w, h in sizes:
            if cx > 0 and cx + w > max_extent + 1e-6:
                cy += row_h + gap
                cx, row_h = 0.0, 0.0
            pos.append((cx, cy))
            cx += w + gap
            row_h = max(row_h, h)
            total_w = max(total_w, cx - gap)
            total_h = max(total_h, cy + row_h)
    return pos, total_w, total_h


def _finalize_box(b: Box, mode, core_w, core_h, cfg, settings: RenderSettings):
    b.core_w, b.core_h = core_w, core_h
    fl = _order(b.flows, settings)
    if not fl:
        b.w, b.h = core_w, core_h
        b.core_dx = b.core_dy = 0.0
        b.flows_before, b.flows_after = [], []
        return
    for f in fl:
        layout(f, mode, cfg, settings, None)
    before = [f for f in fl if (f.side_before if f.side_before is not None else f.score < 0.5)]
    after = [f for f in fl if not (f.side_before if f.side_before is not None else f.score < 0.5)]
    vertical = (settings.rankdir != "TB")
    gap = 6.0
    max_extent = max(core_h if vertical else core_w, 1.0) * 1.4
    bpos, bw, bh = stack_side(before, gap, vertical, max_extent)
    apos, aw, ah = stack_side(after, gap, vertical, max_extent)
    for f, (x, y) in zip(before, bpos):
        f.rx, f.ry = x, y
    for f, (x, y) in zip(after, apos):
        f.rx, f.ry = x, y
    if vertical:
        core_dx = (bw + gap) if before else 0.0
        for f in before:
            f.ry += max(0.0, (max(core_h, bh) - bh) / 2)
        for f in after:
            f.rx += core_dx + core_w + gap
            f.ry += max(0.0, (max(core_h, ah) - ah) / 2)
        b.core_dx, b.core_dy = core_dx, 0.0
        b.w = core_dx + core_w + ((gap + aw) if after else 0.0)
        b.h = max(core_h, bh, ah)
    else:
        core_dy = (bh + gap) if before else 0.0
        for f in before:
            f.rx += max(0.0, (max(core_w, bw) - bw) / 2)
        for f in after:
            f.ry += core_dy + core_h + gap
            f.rx += max(0.0, (max(core_w, aw) - aw) / 2)
        b.core_dx, b.core_dy = 0.0, core_dy
        b.w = max(core_w, bw, aw)
        b.h = core_dy + core_h + ((gap + ah) if after else 0.0)
    b.flows_before, b.flows_after = before, after


def layout(b: Box, mode, cfg, settings: RenderSettings, deg: dict | None):
    if b.kind == "leaf":
        fs1, fs2, fs3 = _hdr(b.depth), _hdr2(b.depth), _hdr3(b.depth)
        header_dummy = _header_lines(b.level_label, b.code, b.name, b.n, b.dur_mean, b.dur_min, b.dur_max,
                                     mode, fs1, fs2, fs3)
        wtxt = max(tw(txt, fsz, settings, bold) for txt, fsz, bold in header_dummy)
        header_h = _header_height(header_dummy)

        n_rows, n_cols = _grid_pack_rows(len(b.child_rows))
        ch_child = 0.0
        fsc = 8.5
        if b.child_rows:
            col_w = [0.0] * n_cols
            for i, row in enumerate(b.child_rows):
                c = i % n_cols
                col_w[c] = max(col_w[c], tw(_child_row_text(row, mode), fsc, settings, False))
            b.grid_cols, b.grid_col_w = n_cols, col_w
            grid_w = sum(col_w) + 8 * (n_cols - 1)
            ch_child = n_rows * (fsc + 4) + 6
            wtxt = max(wtxt, grid_w)

        core_w = max(96.0, wtxt + 14 + (2 * settings.strip_w if cfg.show_start_finish_box else 0))
        core_h = 8 + header_h + ch_child
        _finalize_box(b, mode, core_w, core_h, cfg, settings)
        return
    if b.kind == "flow":
        b.w, b.h = tw(b.lines[0][0], 9, settings, True) + 18, 22.0
        b.core_w, b.core_h = b.w, b.h
        b.core_dx = b.core_dy = 0.0
        return

    isolated = []
    if settings.order_by == "degree" and b.children:
        deg = deg or {}
        deg_of = {k: total_degree(k, deg, cfg) for k in b.children}
        connected = sorted([k for k in b.children if deg_of[k] > 0],
                           key=lambda k: (-deg_of[k], k.path))
        isolated = sorted([k for k in b.children if deg_of[k] == 0], key=lambda k: k.path)
        centered, left = deque(), True
        for k in connected:
            (centered.appendleft if left else centered.append)(k)
            left = not left
        kids = list(centered)
    else:
        kids = _order(b.children, settings)

    for k in kids + isolated:
        layout(k, mode, cfg, settings, deg)

    b.iso_gap_y = None
    cw = ch = 0.0
    if kids:
        gap = settings.gap_group if any(k.kind == "group" for k in kids) else settings.gap_leaf
        if b.depth == 0:
            pos, cw, ch = shelf_pack([(k.w, k.h) for k in kids], gap, settings.target_aspect_top)
        else:
            n_rows = math.ceil(math.sqrt(len(kids)))
            n_cols = math.ceil(len(kids) / n_rows)
            pos, cw, ch = shelf_pack_by_columns([(k.w, k.h) for k in kids], gap, n_cols)
        for k, (x, y) in zip(kids, pos):
            k.rx, k.ry = x, y
    if isolated:
        igap = settings.gap_group if any(k.kind == "group" for k in isolated) else settings.gap_leaf
        ipos, icw, ich = shelf_pack([(k.w, k.h) for k in isolated], igap,
                                    settings.iso_aspect, max_width=(cw if kids else None))
        y_off = ch + (settings.gap_iso if kids else 0.0)
        for k, (x, y) in zip(isolated, ipos):
            k.rx, k.ry = x, y + y_off
        b.iso_gap_y = ch + settings.gap_iso / 2 if kids else None
        cw, ch = max(cw, icw), y_off + ich

    pad = settings.pad if b.depth > 0 else 0
    if b.depth > 0:
        fs1, fs2, fs3 = _hdr(b.depth), _hdr2(b.depth), _hdr3(b.depth)
        header_dummy = _header_lines(b.level_label, b.code, b.name, b.n, b.dur_mean, b.dur_min, b.dur_max,
                                     mode, fs1, fs2, fs3)
        hdr = _header_height(header_dummy) + 8
        title_w = max(tw(txt, fsz, settings, bold) for txt, fsz, bold in header_dummy)
    else:
        title_w = 0.0
        hdr = 0
    for k in kids + isolated:
        k.rx += pad
        k.ry += hdr + pad
    if b.iso_gap_y is not None:
        b.iso_gap_y += hdr + pad
    core_w = max(cw, title_w) + 2 * pad
    core_h = hdr + pad + ch + pad
    b.children = kids + isolated
    _finalize_box(b, mode, core_w, core_h, cfg, settings)


def place(b: Box, ox, oy):
    ext_x, ext_y = ox + b.rx, oy + b.ry
    b.x, b.y = ext_x + b.core_dx, ext_y + b.core_dy
    for c in b.children:
        place(c, b.x, b.y)
    for f in b.flows_before + b.flows_after:
        f.x, f.y = ext_x + f.rx, ext_y + f.ry


def _walk(b: Box):
    yield b
    for c in list(b.children) + list(b.flows_before) + list(b.flows_after):
        yield from _walk(c)


def _hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _shorten_last_segment(points, L):
    if len(points) < 2:
        return list(points)
    (x0, y0), (x1, y1) = points[-2], points[-1]
    seg_len = math.hypot(x1 - x0, y1 - y0) or 1.0
    t = 0.0 if seg_len <= L else (seg_len - L) / seg_len
    nx, ny = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
    return list(points[:-1]) + [(nx, ny)]


_JS = """
(function() {
    var svg = document.querySelector('svg'),
        layer = svg.getElementById('edges'),
        hl = svg.getElementById('hl');
    var edges = [].slice.call(svg.querySelectorAll('.e'));
    [].forEach.call(svg.querySelectorAll('.n'), function(n) {
        n.addEventListener('mouseenter', function() {
            var id = n.getAttribute('data-id');
            edges.forEach(function(e) {
                if (e.getAttribute('data-s') === id || e.getAttribute('data-t') === id) {
                    e.classList.add('hl');
                    hl.appendChild(e);
                } else e.classList.add('dim');
            });
        });
        n.addEventListener('mouseleave', function() {
            edges.forEach(function(e) {
                e.classList.remove('hl');
                e.classList.remove('dim');
                layer.appendChild(e);
            });
        });
    });
    var vb = svg.viewBox.baseVal,
        X0 = vb.x,
        Y0 = vb.y,
        W0 = vb.width,
        H0 = vb.height;

    function pt(ev) {
        var p = svg.createSVGPoint();
        p.x = ev.clientX;
        p.y = ev.clientY;
        return p.matrixTransform(svg.getScreenCTM().inverse());
    }
    svg.addEventListener('wheel', function(ev) {
        ev.preventDefault();
        var k = ev.deltaY < 0 ? 0.85 : 1.18,
            p = pt(ev);
        vb.x = p.x - (p.x - vb.x) * k;
        vb.y = p.y - (p.y - vb.y) * k;
        vb.width *= k;
        vb.height *= k;
    }, {
        passive: false
    });
    var drag = false,
        lx = 0,
        ly = 0;
    svg.addEventListener('mousedown', function(ev) {
        drag = true;
        lx = ev.clientX;
        ly = ev.clientY;
    });
    window.addEventListener('mouseup', function() {
        drag = false;
    });
    window.addEventListener('mousemove', function(ev) {
        if (!drag) return;
        var r = svg.getBoundingClientRect(),
            s = Math.max(vb.width / r.width, vb.height / r.height);
        vb.x -= (ev.clientX - lx) * s;
        vb.y -= (ev.clientY - ly) * s;
        lx = ev.clientX;
        ly = ev.clientY;
    });
    svg.addEventListener('dblclick', function() {
        vb.x = X0;
        vb.y = Y0;
        vb.width = W0;
        vb.height = H0;
    });
})();
"""

# ============================================================================
# Hiyerarşik harita için kat planı yönlendirme motoru (A2'yi yeniden kullanır)
# ============================================================================
def _hier_label_pos(pts):
    i = max(range(len(pts) - 1), key=lambda k: math.dist(pts[k], pts[k + 1]))
    (x0, y0), (x1, y1) = pts[i], pts[i + 1]
    return (x0 + x1) / 2, (y0 + y1) / 2


def route_hier_edges(nl, boxes, mode, settings):
    """nl: [(edge_index, source_box, target_box)] (döngü olmayan kenarlar).
    Döner: (route_of {ei: points}, dashed {ei}, info dict)"""
    by_id = {b.id: b for b in boxes}

    # --- ata grupları: kenar sadece uç kutularının ata gruplarının içinden geçebilir
    ancs = {}

    def _anc(b, chain):
        ancs[b.id] = chain
        sub = (chain | {b.id}) if b.kind == "group" else chain
        for c in b.children:
            _anc(c, sub)
        for f in b.flows:                       # Order-event kutusu: hedef grubu da "ata" sayılır
            _anc(f, chain | {b.id})

    _anc(boxes[0], frozenset())

    def rect(b):
        return (b.x, b.y, b.core_w, b.core_h)

    def ctr(b):
        return (b.x + b.core_w / 2, b.y + b.core_h / 2)

    leaves = [b for b in boxes if b.kind in ("leaf", "flow")]
    groups = [b for b in boxes if b.kind == "group" and b.depth > 0]

    def hdr_h(g):
        fs1, fs2, fs3 = _hdr(g.depth), _hdr2(g.depth), _hdr3(g.depth)
        return _header_height(_header_lines(g.level_label, g.code, g.name, g.n, g.dur_mean,
                                            g.dur_min, g.dur_max, mode, fs1, fs2, fs3)) + 8

    box_raw = _arr([rect(b) for b in leaves])
    hdr_raw = _arr([(g.x, g.y, g.core_w, hdr_h(g)) for g in groups])   # başlık metni engel
    cont_raw = _arr([rect(g) for g in groups])
    cont_keys = [g.id for g in groups]

    w_max = settings.edge_max_w
    arrow_len = 5 + 1.1 * w_max
    margins = (6.0, 3.0, 1.5)
    corner = 4.0
    gap = max(2.5 * w_max + 2.0, 6.0)
    min_end = arrow_len * 1.1

    cache = {}

    def obs(mg, allowed):
        k = (mg, allowed)
        if k not in cache:
            parts = [_inf(box_raw, mg), _inf(hdr_raw, 1.0)]
            if allowed is not None and len(cont_raw):
                keep = np.array([c not in allowed for c in cont_keys])
                parts.append(_inf(cont_raw[keep], mg))
            parts = [p for p in parts if len(p)]
            cache[k] = np.vstack(parts) if parts else np.zeros((0, 4))
        return cache[k]

    def route_edge(a, sa, ta, b, sb, tb, hanan, use_forb):
        pa, pb = _port_xy(rect(a), sa, ta), _port_xy(rect(b), sb, tb)
        allowed = frozenset(ancs[a.id] | ancs[b.id]) if use_forb else None
        for mg in margins:
            arr = obs(mg, allowed)
            L = max(arrow_len * 1.2, mg + 3.0)
            A, B = _stub_xy(pa, sa, L), _stub_xy(pb, sb, L)
            mid = r4_route_between(A, B, sa, _OPP[sb], arr, hanan)
            if mid:
                full = _simplify_points([pa] + mid + [pb])
                return full, _path_cost(full, sa, _OPP[sb])[0]
        return None

    # ---- 1. geçiş: ucuz (doğrudan) adaylarla kenar seçimi
    items = sorted(nl, key=lambda t: math.dist(ctr(t[1]), ctr(t[2])))
    plan = []
    for ei, a, b in items:
        srcs, dsts = _side_candidates(ctr(a), ctr(b))
        pref = [(s, t) for s in srcs for t in dsts]
        rest = [(s, t) for s in "NSWE" for t in "NSWE" if (s, t) not in pref]
        best = None
        for grp in (pref, rest):
            for use_forb in (True, False):
                for s, t in grp:
                    ta = _ideal_coord(s, rect(a), rect(b), corner)
                    tb = _ideal_coord(t, rect(b), rect(a), corner)
                    res = route_edge(a, s, ta, b, t, tb, False, use_forb)
                    if res and (best is None or res[1] < best["cost"]):
                        best = {"cost": res[1], "s": s, "t": t, "ta": ta, "tb": tb}
                if best:
                    break
            if best:
                break
        if best is None:
            s, t = srcs[0], dsts[0]
            best = {"s": s, "t": t,
                    "ta": _ideal_coord(s, rect(a), rect(b), corner),
                    "tb": _ideal_coord(t, rect(b), rect(a), corner)}
        plan.append({"ei": ei, "a": a, "b": b, **best})

    # ---- port dağıtımı (aynı kenarı paylaşan çizgiler)
    groups_p = defaultdict(list)
    for i, p in enumerate(plan):
        groups_p[(p["a"].id, p["s"])].append((i, "ta", p["b"]))
        groups_p[(p["b"].id, p["t"])].append((i, "tb", p["a"]))
    for (bid, side), lst in groups_p.items():
        x, y, w, h = rect(by_id[bid])
        lo, hi = ((x + corner, x + w - corner) if side in "NS" else (y + corner, y + h - corner))
        ax = 0 if side in "NS" else 1
        vals = [plan[i][k] + 1e-4 * ctr(partner)[ax] for i, k, partner in lst]
        for (i, k, _), t in zip(lst, r4_spread(vals, lo, max(lo, hi), gap)):
            plan[i][k] = t

    # ---- 2. geçiş: nihai yönlendirme (gerekirse Hanan)
    for p in plan:
        a, b = p["a"], p["b"]
        res = None
        for use_forb in (True, False):
            res = route_edge(a, p["s"], p["ta"], b, p["t"], p["tb"], True, use_forb)
            if res:
                break
        if res is None:
            for s in "NSWE":
                for t in "NSWE":
                    ta = _ideal_coord(s, rect(a), rect(b), corner)
                    tb = _ideal_coord(t, rect(b), rect(a), corner)
                    r2 = route_edge(a, s, ta, b, t, tb, True, False)
                    if r2 and (res is None or r2[1] < res[1]):
                        res = r2
                        p.update({"s": s, "t": t, "ta": ta, "tb": tb})
        p["pts"] = res[0] if res else None

    # ---- paralel şerit ayırma
    ok = [p for p in plan if p["pts"] is not None]
    routes = [p["pts"] for p in ok]
    hcache = {}

    def hard_for(ri):
        key = frozenset(ancs[ok[ri]["a"].id] | ancs[ok[ri]["b"].id])
        if key not in hcache:
            parts = [_inf(box_raw, 1.0), _inf(hdr_raw, 1.0)]
            if len(cont_raw):
                keep = np.array([c not in key for c in cont_keys])
                parts.append(_inf(cont_raw[keep], 1.0))
            parts = [q for q in parts if len(q)]
            hcache[key] = np.vstack(parts) if parts else np.zeros((0, 4))
        return hcache[key]

    moved = 0
    if routes:
        moved, _ = r4_separate(routes, hard_for, _arr([]), gap, min_end)
        for p, r in zip(ok, routes):
            p["pts"] = r

    route_of, dashed = {}, set()
    for p in plan:
        if p["pts"] is None:                       # çözülemeyen: kesikli L
            pa = _port_xy(rect(p["a"]), p["s"], p["ta"])
            pb = _port_xy(rect(p["b"]), p["t"], p["tb"])
            route_of[p["ei"]] = _simplify_points(_orthogonalize(manhattan_fallback(pa, pb)))
            dashed.add(p["ei"])
        else:
            route_of[p["ei"]] = p["pts"]
    return route_of, dashed, {"edges": len(plan), "failed": len(dashed), "moved": moved}

# ============================================================================
# SVG producer
# ============================================================================
def render_svg(view, mode, title, out_path, pipeline: dict, cfg, settings: RenderSettings):
    root, leaf_box, flow_box = view
    dfg_f, perf = pipeline["dfg_filtered"], pipeline["perf_full"]
    boxes = list(_walk(root))
    for i, b in enumerate(boxes):
        b.id = f"b{i}"
    top = sorted([g for g in root.children if g.kind == "group"], key=lambda g: g.path)
    hue_of = {g.path: i / max(1, len(top)) for i, g in enumerate(top)}

    def hue_for(b):
        return hue_of.get(b.path[:1], 0.6) if b.kind == "group" else 0.6

    def _ancestor_hue(b):
        return hue_of.get(b.path[:1], 0.6)

    E = []
    for (s, t), c in dfg_f.items():
        if c < settings.edge_min_count:
            continue
        ks, base_s = _classify_node_id(s)
        kt, base_t = _classify_node_id(t)
        sb = leaf_box.get(base_s) if ks in ("start", "finish", "point") else flow_box.get(s)
        tb = leaf_box.get(base_t) if kt in ("start", "finish", "point") else flow_box.get(t)
        if sb is None or tb is None:
            continue
        if sb is tb and not ((ks == "finish" and kt == "start") or (ks == "point" and kt == "point")):
            continue
        E.append({"s": s, "t": t, "sb": sb, "tb": tb, "ks": ks, "kt": kt, "c": c, "loop": (sb is tb),
                      "p": perf.get((s, t), {})})
    if not E:
        raise ValueError("No edges to render (filter may be too strict).")
    E.sort(key=lambda e: e["c"])
    for i, e in enumerate(sorted(E, key=lambda e: -e["c"])[:settings.label_top_n]):
        e["top"] = True
    cmax = max(e["c"] for e in E)
    lmax = math.log1p(cmax)
    vmin = vmax = 0
    if mode == "perf":
        means = np.array([e["p"].get("mean", 0.0) for e in E])
        vmin, vmax = np.percentile(means, [5, 95])

    canvas_w = max(root.core_w + 2 * settings.margin, 760.0)
    canvas_h = root.core_h + settings.header_h + settings.margin
    nl = [(ei, e["sb"], e["tb"]) for ei, e in enumerate(E) if not e["loop"]]
    route_of, dashed, rinfo = route_hier_edges(nl, boxes, mode, settings)
    print(f"Routing (floor-plan engine): {rinfo['edges']} edges | "
          f"{rinfo['failed']} unresolved | {rinfo['moved']} segments separated")

    out = []
    W, H = canvas_w, canvas_h
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
               f'font-family="{settings.font}">')
    out.append("<style>.e .lbl{display:none}.e.top .lbl,.e.hl .lbl{display:inline}"
               ".e.dim{opacity:.05}.e.hl{opacity:1}.n{cursor:pointer}.n:hover rect.box{stroke-width:2.4}"
               ".lbl{font-size:9px;font-weight:bold;fill:#1B2631;text-anchor:middle;"
               "paint-order:stroke;stroke:#fff;stroke-width:3px}</style>")
    out.append('<defs><linearGradient id="gperf" x1="0" x2="1" y1="0" y2="0">'
               '<stop offset="0" stop-color="#2ecc71"/><stop offset=".5" stop-color="#f1c40f"/>'
               '<stop offset="1" stop-color="#e74c3c"/></linearGradient></defs>')
    out.append(f'<rect x="0" y="0" width="{W:.0f}" height="{H:.0f}" fill="#FFFFFF"/>')
    out.append(f'<text x="{settings.margin}" y="26" font-size="17" font-weight="bold" fill="#1B2631">'
               f'{html.escape(title)}</text>')
    ly = 50
    if cfg.show_start_finish_box:
        out.append(f'<polygon points="{settings.margin},{ly - 9} {settings.margin + 10},{ly - 4} {settings.margin},{ly + 1}" fill="#229954"/>'
                   f'<text x="{settings.margin + 15}" y="{ly}" font-size="10">Start</text>'
                   f'<rect x="{settings.margin + 55}" y="{ly - 9}" width="9" height="9" fill="#CB4335"/>'
                   f'<text x="{settings.margin + 69}" y="{ly}" font-size="10">Finish</text>')
    out.append(f'<rect x="{settings.margin + 115}" y="{ly - 10}" width="26" height="11" rx="5" fill="#F5CBA7" stroke="#CA6F1E"/>'
               f'<text x="{settings.margin + 146}" y="{ly}" font-size="10">Order event</text>')
    if mode == "perf":
        out.append(f'<text x="{settings.margin + 225}" y="{ly}" font-size="10">Line color = avg wait:</text>'
                   f'<rect x="{settings.margin + 365}" y="{ly - 9}" width="90" height="9" fill="url(#gperf)"/>'
                   f'<text x="{settings.margin + 365}" y="{ly + 11}" font-size="9">{fmt_dur(vmin)}</text>'
                   f'<text x="{settings.margin + 455}" y="{ly + 11}" font-size="9" text-anchor="end">{fmt_dur(vmax)}</text>'
                   f'<text x="{settings.margin + 470}" y="{ly}" font-size="10">· Width = frequency (log)</text>')
    else:
        out.append(f'<text x="{settings.margin + 225}" y="{ly}" font-size="10">'
                   f'Line width/color = transition frequency (log scale) · hover: highlight links · '
                   f'scroll: zoom · drag: pan · double-click: reset</text>')

    titles = []
    for b in boxes:
        if b.kind != "group" or b.depth == 0:
            continue
        iso_gap_y = getattr(b, "iso_gap_y", None)
        if iso_gap_y is not None:
            yy = float(b.y) + float(iso_gap_y)
            out.append(f'<line x1="{b.x+6:.1f}" y1="{yy:.1f}" x2="{b.x+b.core_w-6:.1f}" y2="{yy:.1f}" '
                       f'stroke="#B2BEC3" stroke-width="1" stroke-dasharray="4 4"/>')
            out.append(f'<text x="{b.x+6:.1f}" y="{yy-3:.1f}" font-size="7" fill="#909497">'
                       f'low connectivity</text>')
        h = hue_for(b)
        if b.depth == 1:
            fill, stroke, sw, dash = _hsv(h, 0.10, 1.0), _hsv(h, 0.55, 0.75), 2.0, ""
        elif b.depth == 2:
            fill, stroke, sw, dash = _hsv(h, 0.04, 1.0), _hsv(h, 0.40, 0.85), 1.2, ' stroke-dasharray="5 3"'
        else:
            fill, stroke, sw, dash = "#FFFFFF", "#B2BEC3", 1.0, ' stroke-dasharray="3 3"'
        out.append(f'<rect x="{b.x:.1f}" y="{b.y:.1f}" width="{b.core_w:.1f}" height="{b.core_h:.1f}" rx="8" '
                   f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>')
        fs1, fs2, fs3 = _hdr(b.depth), _hdr2(b.depth), _hdr3(b.depth)
        header = _header_lines(b.level_label, b.code, b.name, b.n, b.dur_mean, b.dur_min, b.dur_max,
                               mode, fs1, fs2, fs3)
        dark = _hsv(h, 0.7, 0.42)
        cur = b.y + 5
        for txt, fsz, bold in header:
            cur += fsz
            fw = ' font-weight="bold"' if bold else ""
            titles.append(f'<text x="{b.x+8:.1f}" y="{cur:.1f}" font-size="{fsz}"{fw} fill="{fill}" '
                          f'stroke="{fill}" stroke-width="4" stroke-linejoin="round">{html.escape(txt)}</text>')
            titles.append(f'<text x="{b.x+8:.1f}" y="{cur:.1f}" font-size="{fsz}"{fw} '
                          f'fill="{dark}">{html.escape(txt)}</text>')
            cur += 3.5

    out.append('<g id="edges">')
    for ei, e in enumerate(E):
        c, sb, tb = e["c"], e["sb"], e["tb"]
        w = settings.edge_min_w + (math.log1p(c) / lmax) * (settings.edge_max_w - settings.edge_min_w)
        if mode == "perf":
            mean = e["p"].get("mean", 0.0)
            col = duration_to_color(mean, vmin, vmax)
            label = fmt_dur(mean)
            tip = (f"{sb.short} → {tb.short}\n{c:,} transition | avg {fmt_dur(mean)} | "
                   f"min {fmt_dur(e['p'].get('min', 0))} | max {fmt_dur(e['p'].get('max', 0))}")
        else:
            col = _hsv(_ancestor_hue(sb), 0.60, 0.55)
            label = f"{c:,}"
            tip = (f"{sb.short} → {tb.short}\n{c:,} transitions | avg wait "
                   f"{fmt_dur(e['p'].get('mean', 0))}")
        L, hw = 5 + 1.1 * w, 2.5 + 0.9 * w

        if e["loop"]:
            xr, xl, ym = sb.x + sb.core_w, sb.x, sb.y + sb.core_h / 2
            bul = sb.core_h / 2 + 38
            p0, p1, p2, p3 = (xr, ym), (xr + 45, ym + bul), (xl - 45, ym + bul), (xl - L, ym)
            tip_x, tip_y, dirx = xl, ym, 1
            mx = (p0[0] + 3 * p1[0] + 3 * p2[0] + p3[0]) / 8
            my = (p0[1] + 3 * p1[1] + 3 * p2[1] + p3[1]) / 8
            bx_ = tip_x - dirx * L
            poly = f"{tip_x:.1f},{tip_y:.1f} {bx_:.1f},{tip_y - hw:.1f} {bx_:.1f},{tip_y + hw:.1f}"
            d = (f"M{p0[0]:.1f},{p0[1]:.1f} C{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} "
                 f"{p3[0]:.1f},{p3[1]:.1f}")
        else:
            path_world = route_of[ei]
            tip_x, tip_y = path_world[-1]
            trimmed = _shorten_last_segment(path_world, L)
            (lx0, ly0) = trimmed[-2] if len(trimmed) > 1 else trimmed[-1]
            (bx0, by0) = trimmed[-1]
            seg_dx, seg_dy = bx0 - lx0, by0 - ly0
            seg_len = math.hypot(seg_dx, seg_dy) or 1.0
            ux, uy = seg_dx / seg_len, seg_dy / seg_len
            perp_x, perp_y = -uy, ux
            poly = (f"{tip_x:.1f},{tip_y:.1f} "
                    f"{bx0 + perp_x * hw:.1f},{by0 + perp_y * hw:.1f} "
                    f"{bx0 - perp_x * hw:.1f},{by0 - perp_y * hw:.1f}")
            d = f"M{trimmed[0][0]:.1f},{trimmed[0][1]:.1f} " + " ".join(
                f"L{x:.1f},{y:.1f}" for x, y in trimmed[1:]
            )
            mx, my = _hier_label_pos(path_world)       # eski: path_world[len(path_world) // 2]

        dash = ' stroke-dasharray="6 4"' if ei in dashed else ""
        out.append(
            f'<g class="e{" top" if e.get("top") else ""}" data-s="{sb.id}" data-t="{tb.id}">'
            f'<title>{html.escape(tip)}</title>'
            f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{w:.2f}" '
            f'stroke-linecap="butt" stroke-linejoin="miter" opacity=".8"{dash}/>'
            f'<polygon points="{poly}" fill="{col}" opacity=".9"/>'
            f'<text class="lbl" x="{mx:.1f}" y="{my + 3:.1f}">{html.escape(label)}</text></g>'
        )
    out.append("</g>")
    out.extend(titles)

    for b in boxes:
        if b.kind == "flow":
            out.append(f'<g class="n" data-id="{b.id}"><title>{html.escape(b.tip)}</title>'
                       f'<rect class="box" x="{b.x:.1f}" y="{b.y:.1f}" width="{b.core_w:.1f}" height="{b.core_h:.1f}" rx="11" '
                       f'fill="#F5CBA7" stroke="#CA6F1E" stroke-width="1.2"/>'
                       f'<text x="{b.x + b.core_w / 2:.1f}" y="{b.y + 15:.1f}" font-size="9" font-weight="bold" '
                       f'text-anchor="middle" fill="#6E2C00">{html.escape(b.lines[0][0])}</text></g>')
        elif b.kind == "leaf":
            fs1, fs2, fs3 = _hdr(b.depth), _hdr2(b.depth), _hdr3(b.depth)
            header = _header_lines(b.level_label, b.code, b.name, b.n, b.dur_mean, b.dur_min, b.dur_max,
                                   mode, fs1, fs2, fs3)
            g = [f'<g class="n" data-id="{b.id}"><title>{html.escape(b.tip)}</title>',
                 (f'<rect class="box" x="{b.x:.1f}" y="{b.y:.1f}" width="{b.core_w:.1f}" height="{b.core_h:.1f}" rx="5" '
                 f'fill="#FFFFFF" stroke="#2C3E50" stroke-width="1.2"/>')]
            if cfg.show_start_finish_box:
                cy = b.y + b.core_h / 2
                g.append(f'<rect x="{b.x + 0.6:.1f}" y="{b.y + 0.6:.1f}" width="{settings.strip_w}" height="{b.core_h - 1.2:.1f}" '
                         f'rx="4" fill="#D5F5E3"/>'
                         f'<polygon points="{b.x + 4:.1f},{cy - 5:.1f} {b.x + 12:.1f},{cy:.1f} {b.x + 4:.1f},{cy + 5:.1f}" fill="#229954"/>')
                g.append(f'<rect x="{b.x + b.core_w - settings.strip_w - 0.6:.1f}" y="{b.y + 0.6:.1f}" width="{settings.strip_w}" '
                         f'height="{b.core_h - 1.2:.1f}" rx="4" fill="#FADBD8"/>'
                         f'<rect x="{b.x + b.core_w - settings.strip_w + 3:.1f}" y="{cy - 4:.1f}" width="8" height="8" fill="#CB4335"/>')
            cur = b.y + 5
            for txt, fsz, bold in header:
                cur += fsz
                fw = ' font-weight="bold"' if bold else ""
                g.append(f'<text x="{b.x + b.core_w / 2:.1f}" y="{cur:.1f}" font-size="{fsz}"{fw} '
                         f'text-anchor="middle" fill="#1B2631">{html.escape(txt)}</text>')
                cur += 3.5
            if b.child_rows:
                cur += 3
                fsc = 8.5
                n_cols = max(1, b.grid_cols)
                col_w = b.grid_col_w or [0.0]
                col_x = [0.0] * n_cols
                for c in range(1, n_cols):
                    col_x[c] = col_x[c - 1] + col_w[c - 1] + 10
                total_w = col_x[-1] + col_w[-1] if col_w else 0.0
                start_x = b.x + (b.core_w - total_w) / 2
                base_y = cur
                for i, row in enumerate(b.child_rows):
                    c, rr = i % n_cols, i // n_cols
                    txt = _child_row_text(row, mode)
                    yy = base_y + rr * (fsc + 4) + fsc
                    g.append(f'<text x="{start_x + col_x[c]:.1f}" y="{yy:.1f}" font-size="{fsc}" '
                             f'text-anchor="start" fill="#566573">{html.escape(txt)}</text>')
            g.append("</g>")
            out.append("".join(g))
    out.append('<g id="hl"></g>')
    out.append("<script><![CDATA[" + _JS + "]]></script>")
    out.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return out_path, W, H, len(E)


def render_process_map(pipeline: dict, cfg, agg_level, settings: RenderSettings | None = None,
                       only_codes=None, mode: str = "freq", title: str | None = None,
                       out_path: str = "process.svg"):
    settings = settings or RenderSettings()
    view = build_view(pipeline, cfg, agg_level, settings, only_codes)
    layout(view[0], mode, cfg, settings, pipeline["deg"])
    place(view[0], settings.margin, settings.header_h)
    default_title = "Factory Process Map (Frequency)" if mode == "freq" else "Factory Bottleneck Map (Duration)"
    return render_svg(view, mode, title or default_title, out_path, pipeline, cfg, settings)


def build_level_transition_matrix(ops: pd.DataFrame, cfg, code_col: str, name_col: str | None = None):
    """Sequential events with the same level values are counted once; only
    actual movements between different locations are counted (SLP 'From-To' table).
    Returns: (matrix_df, pairs_df[from,to,count(,from_name,to_name)], name_lookup)."""
    ts_col = cfg.op_ts_col
    cols = [cfg.case_id_col, code_col, ts_col] + ([name_col] if name_col else [])
    frame = ops[cols].dropna(subset=[code_col, ts_col]).copy()
    frame[code_col] = frame[code_col].astype(str)
    frame = safe_sort(frame, [cfg.case_id_col, ts_col])
    name_lookup = {}
    if name_col:
        name_lookup = frame.drop_duplicates(code_col).set_index(code_col)[name_col].astype(str).to_dict()
    frame["prev_code"] = frame.groupby(cfg.case_id_col, sort=False)[code_col].shift(1)
    moves = frame[frame["prev_code"].notna() & (frame["prev_code"] != frame[code_col])]
    if moves.empty:
        empty = pd.DataFrame(columns=["from", "to", "count"])
        return pd.DataFrame(), empty, name_lookup
    pairs = (moves.groupby(["prev_code", code_col]).size().rename("count").reset_index()
                  .rename(columns={"prev_code": "from", code_col: "to"}))
    pairs = pairs.iloc[np.argsort(-pairs["count"].to_numpy(), kind="stable")].reset_index(drop=True)
    codes = sorted(set(pairs["from"]) | set(pairs["to"]))
    matrix = pd.DataFrame(0, index=codes, columns=codes, dtype=int)
    for r in pairs.itertuples(index=False):
        matrix.loc[r[0], r[1]] = r[2]
    if name_lookup:
        pairs["from_name"] = pairs["from"].map(name_lookup)
        pairs["to_name"] = pairs["to"].map(name_lookup)
    return matrix, pairs, name_lookup


def _duration_sec(ops: pd.DataFrame, cfg) -> pd.Series | None:
    if not cfg.show_start_finish_box:
        return None
    return (ops[cfg.end_timestamp_col] - ops[cfg.start_timestamp_col]).dt.total_seconds()


def describe_node(nid: str, pipeline: dict) -> str:
    kind, key = _classify_node_id(nid)
    if kind == "flow":
        label = pipeline["node_label_lookup"].get(nid, nid)
        attach = pipeline["flow_attach_info"].get(nid)
        return f"{label} @ {attach[1]}" if attach else str(label)
    parts = pipeline["base_paths"].get(key)
    if parts is None:
        return nid
    nm = pipeline["names"].get(parts, "")
    text = "/".join(parts) + (f" {nm}" if nm else "")
    return {"start": "▶ ", "finish": "■ ", "point": ""}[kind] + text


def level_ranking(ops: pd.DataFrame, cfg, level_idx: int, pairs: pd.DataFrame, names: dict) -> pd.DataFrame:
    lvl = cfg.hierarchy_levels[level_idx]
    code = lvl.code_col
    d = ops.copy()
    d[code] = d[code].astype(str)
    dur = _duration_sec(d, cfg)
    if dur is not None:
        d["_dur"] = dur
    g = d.groupby(code, observed=True)
    out = pd.DataFrame({
        "Operation events": g.size(),
        "Distinct orders": g[cfg.case_id_col].nunique(),
        "Distinct operations": g[cfg.activity_code_col].nunique(),
    })
    if dur is not None:
        out["Avg op. duration (sec)"] = round(g["_dur"].mean(), 3)
        out["Avg op. duration"] = out["Avg op. duration (sec)"].map(
            lambda v: format_duration(v) if pd.notna(v) else "-")
    partners = {}
    for a, b in zip(pairs["from"], pairs["to"]):
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    out[f"Connected to # other {lvl.label}s"] = [len(partners.get(c, ())) for c in out.index]
    out.insert(0, "Name", [names.get(c, "") for c in out.index])
    if level_idx > 0:
        parent = cfg.hierarchy_levels[level_idx - 1].code_col
        d[parent] = d[parent].astype(str)
        out.insert(1, f"Parent {cfg.hierarchy_levels[level_idx - 1].label}",
                   d.groupby(code, observed=True)[parent].agg(lambda s: s.mode().iat[0]))
    out.index.name = lvl.label
    out = out.iloc[np.argsort(-out["Operation events"].to_numpy(), kind="stable")]
    return out.reset_index()


def transfer_time_summary(ops: pd.DataFrame, cfg) -> pd.DataFrame:
    """The gap between consecutive operations (the end of the previous -> the start of the next),
    grouped by the level at which they diverge:
    'Same <inner level>' ... 'Different <outer level>'. The outermost divergence
    (e.g., different work center) is a strong layout signal."""
    levels = cfg.hierarchy_levels
    if not cfg.show_start_finish_box or not levels:
        return pd.DataFrame()
    code_cols = list(cfg.hier_code_cols)
    cols = [cfg.case_id_col, cfg.start_timestamp_col, cfg.end_timestamp_col] + code_cols
    seq = ops[cols].dropna(subset=[cfg.start_timestamp_col, cfg.end_timestamp_col]).copy()
    for c in code_cols:
        seq[c] = seq[c].astype(str)
    seq = safe_sort(seq, [cfg.case_id_col, cfg.start_timestamp_col])
    grp = seq.groupby(cfg.case_id_col, sort=False)
    nxt_start = grp[cfg.start_timestamp_col].shift(-1)
    valid = nxt_start.notna().to_numpy().copy()
    gap = (nxt_start - seq[cfg.end_timestamp_col]).dt.total_seconds().to_numpy()
    valid = valid & (gap >= 0)
    n = len(levels)
    first_diff = np.full(len(seq), n)
    for k in reversed(range(n)):
        nxt_code = grp[code_cols[k]].shift(-1).to_numpy()
        first_diff = np.where(seq[code_cols[k]].to_numpy() != nxt_code, k, first_diff)
    first_diff, gap = first_diff[valid], gap[valid]

    def bucket(k):
        if k == n:
            return f"Same {levels[-1].label}"
        if k == 0:
            return f"Different {levels[0].label}"
        return f"Same {levels[k - 1].label}, different {levels[k].label}"

    tmp = pd.DataFrame({"k": first_diff, "gap": gap})
    agg = tmp.groupby("k")["gap"].agg(["count", "mean", "median"])
    agg.columns = ["Moves", "Mean", "Median"]
    res = agg.copy()
    res.insert(0, "Transfer type", [bucket(k) for k in res.index])
    res["Avg wait"] = res["Mean"].map(format_duration)
    res["Median wait (sec)"] = res["Median"]
    return res.drop(columns=["Mean", "Median"]).reset_index(drop=True)


def level_year_share(ops: pd.DataFrame, cfg, level_idx: int) -> pd.DataFrame:
    """Row: code, Column: year, Value: total activity share in that year."""
    code = cfg.hierarchy_levels[level_idx].code_col
    d = ops.dropna(subset=[cfg.op_ts_col]).copy()
    d["year"] = d[cfg.op_ts_col].dt.year
    ct = pd.crosstab(d[code].astype(str), d["year"])
    return ct.div(ct.sum(axis=0), axis=1)


def share_change(year_share: pd.DataFrame, year_from: int, year_to: int) -> pd.DataFrame:
    a = year_share[year_from] if year_from in year_share.columns else 0.0
    b = year_share[year_to] if year_to in year_share.columns else 0.0
    ya, yb = str(year_from), str(year_to)
    out = pd.DataFrame({ya: a, yb: b}).fillna(0.0)
    out["Change (pp)"] = (out[yb] - out[ya]) * 100
    return out.iloc[np.argsort(-out["Change (pp)"].to_numpy(), kind="stable")].round(3)


# ----------------------------------------------------------------------------
# Helpers for readable node names
# ----------------------------------------------------------------------------
def _base_text(key: str, pipeline: dict) -> str:
    parts = pipeline["base_paths"].get(key)
    return "/".join(parts) if parts else key


def short_node(nid: str, pipeline: dict) -> str:
    kind, key = _classify_node_id(nid)
    if kind == "flow":
        return str(pipeline["node_label_lookup"].get(nid, nid))
    return _base_text(key, pipeline) + {"start": " ▶", "finish": " ■", "point": ""}[kind]


def _compress_steps(nids, pipeline: dict) -> list:
    """A Start immediately followed by the Finish of the same operation is shown as a single step."""
    out, i = [], 0
    while i < len(nids):
        kind, key = _classify_node_id(nids[i])
        if kind == "start" and i + 1 < len(nids):
            k2, key2 = _classify_node_id(nids[i + 1])
            if k2 == "finish" and key2 == key:
                out.append(_base_text(key, pipeline))
                i += 2
                continue
        out.append(short_node(nids[i], pipeline))
        i += 1
    return out


def _desc_order(values) -> np.ndarray:
    return np.argsort(-np.asarray(values, dtype=float), kind="stable")


# ----------------------------------------------------------------------------
# Case table (one row per order, flow events included)
# ----------------------------------------------------------------------------
def build_case_table(event_log: pd.DataFrame) -> pd.DataFrame:
    g = event_log.groupby("case_id", sort=False)
    cases = pd.DataFrame({
        "start": g["timestamp"].min(),
        "end": g["timestamp"].max(),
        "n_events": g.size(),
        "seq": g["node_id"].agg("|".join),
    })
    cases["duration_sec"] = (cases["end"] - cases["start"]).dt.total_seconds()
    return cases


def log_summary(pipeline: dict, cfg, cases: pd.DataFrame) -> pd.DataFrame:
    el = pipeline["event_log"]
    is_flow = el["node_id"].str.startswith("flow_L")
    rows = [
        ("Events (all)", f"{len(el):,}"),
        ("  of which flow events", f"{int(is_flow.sum()):,}"),
        ("  of which operation events" + 
         (" (start + finish)" if cfg.show_start_finish_box else ""), 
         f"{int((~is_flow).sum()):,}"
        ),
        ("Orders (cases)", f"{len(cases):,}"),
        ("Distinct nodes", f"{el['node_id'].nunique():,}"),
        ("DFG edges (unfiltered)", f"{len(pipeline['dfg_full']):,}"),
        ("Trace variants", f"{cases['seq'].nunique():,}"),
        ("First event", str(el["timestamp"].min())),
        ("Last event", str(el["timestamp"].max())),
        ("Avg events per order", f"{cases['n_events'].mean():.1f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ----------------------------------------------------------------------------
# DFG edges: frequency, duration, total time (unfiltered, flow events included)
# ----------------------------------------------------------------------------
def flow_table(pipeline: dict) -> pd.DataFrame:
    rows = []
    for (s, t), cnt in pipeline["dfg_full"].items():
        ks, bs = _classify_node_id(s)
        kt, bt = _classify_node_id(t)
        if ks == "flow" or kt == "flow":
            typ = "Flow event"
        elif ks == "start" and kt == "finish" and bs == bt:
            typ = "Processing"
        else:
            typ = "Transfer / wait"
        p = pipeline["perf_full"].get((s, t), {})
        mean = p.get("mean")
        rows.append({
            "From": describe_node(s, pipeline), 
            "To": describe_node(t, pipeline), 
            "Type": typ,
            "Frequency": cnt,
            "Avg duration": format_duration(mean) if mean is not None else "-",
            "Avg duration (sec)": round(mean, 3) if mean is not None else np.nan,
            "Min duration (sec)": p.get("min", np.nan),
            "Max duration (sec)": p.get("max", np.nan),
            "Total time (hours)": round(mean * cnt / 3600, 3) if mean is not None else np.nan,
        })
    cols = ["From", "To", "Type", "Frequency", "Avg duration", "Avg duration (sec)",
            "Min duration (sec)", "Max duration (sec)", "Total time (hours)"]
    out = pd.DataFrame(rows, columns=cols)
    return out.iloc[_desc_order(out["Frequency"])].reset_index(drop=True)


def busiest_flows(pipeline: dict) -> pd.DataFrame:
    df = flow_table(pipeline)
    return df.iloc[_desc_order(df["Avg duration (sec)"])].reset_index(drop=True)


def bottleneck_flows(flows: pd.DataFrame, min_count: int = 30) -> pd.DataFrame:
    """Non-processing edges (waiting/transfer and flow-event edges) ranked by total time spent."""
    d = flows[(flows["Type"] != "Processing") & (flows["Frequency"] >= min_count)]
    d = d.dropna(subset=["Total time (hours)"])
    return d.iloc[_desc_order(d["Total time (hours)"])].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Nodes, start/end nodes
# ----------------------------------------------------------------------------
def node_stats(pipeline: dict) -> pd.DataFrame:
    el = pipeline["event_log"]
    g = el.groupby("node_id")
    out = pd.DataFrame({"Events": g.size(), "Orders": g["case_id"].nunique()})
    out["Avg position in order (0=start, 1=end)"] = [pipeline["scores"].get(n, np.nan) for n in out.index]
    out.insert(0, "Type", [{"start": "Start", "finish": "Finish", "point": "Event", "flow": "Flow event"}
                           [_classify_node_id(n)[0]] for n in out.index])
    out.insert(0, "Node", [describe_node(n, pipeline) for n in out.index])
    out = out.iloc[_desc_order(out["Events"])]
    return out.reset_index(drop=True)


def start_end_nodes(pipeline: dict) -> pd.DataFrame:
    g = pipeline["event_log"].groupby("case_id", sort=False)["node_id"]
    s, e = g.first().value_counts(), g.last().value_counts()
    nodes = s.index.union(e.index)
    out = pd.DataFrame({
        "Node": [describe_node(n, pipeline) for n in nodes],
        "Case starts": s.reindex(nodes).fillna(0).astype(int).to_numpy(),
        "Case ends": e.reindex(nodes).fillna(0).astype(int).to_numpy(),
    })
    out = out.iloc[_desc_order(out["Case starts"] + out["Case ends"])]
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Order lead time (throughput time) and variants
# ----------------------------------------------------------------------------
def case_duration_summary(cases: pd.DataFrame) -> pd.DataFrame:
    d = cases.assign(year=cases["start"].dt.year)
    groups = [("All", d)] + [(int(y), x) for y, x in d.groupby("year")] # type: ignore
    rows = []
    for label, x in groups:
        v = x["duration_sec"].dropna()
        if v.empty:
            continue
        rows.append({
            "Order start year": label, "Orders": len(x),
            "Mean": format_duration(v.mean()), "Median": format_duration(v.median()),
            "P90": format_duration(v.quantile(0.90)), "P95": format_duration(v.quantile(0.95)),
            "Max": format_duration(v.max()), "Median (sec)": v.median(),
        })
    return pd.DataFrame(rows)


def variant_table(cases: pd.DataFrame, pipeline: dict, top_n: int = 20, max_steps: int = 14) -> pd.DataFrame:
    v = cases.groupby("seq", sort=False)["duration_sec"].agg(["size", "mean", "median"]).reset_index()
    v = v.iloc[_desc_order(v["size"])].head(top_n).reset_index(drop=True)
    total = len(cases)
    cum = np.cumsum(v["size"].to_numpy()) / total * 100
    texts, nsteps = [], []
    for key in v["seq"]:
        steps = _compress_steps(key.split("|"), pipeline)
        nsteps.append(len(steps))
        texts.append(" → ".join(steps[:max_steps]) + (f" → … (+{len(steps) - max_steps})"
                                                       if len(steps) > max_steps else ""))
    return pd.DataFrame({
        "Rank": np.arange(1, len(v) + 1),
        "Orders": v["size"].to_numpy(),
        "Share (%)": (v["size"].to_numpy() / total * 100).round(2),
        "Cumulative (%)": cum.round(2),
        "Avg lead time": v["mean"].map(lambda x: format_duration(x) if pd.notna(x) else "-"),
        "Median lead time": v["median"].map(lambda x: format_duration(x) if pd.notna(x) else "-"),
        "Steps": nsteps,
        "Variant": texts,
    })


# ----------------------------------------------------------------------------
# Flow events (Set Order, Update Order, Request Order Complete ...)
# ----------------------------------------------------------------------------
def flow_rows(df: pd.DataFrame, cfg) -> pd.DataFrame:
    if not (cfg.is_flow_event_col and cfg.is_flow_event_col in df.columns):
        return df.iloc[0:0]
    return df[df[cfg.is_flow_event_col].astype(str).str.lower().isin(["true", "1"])]


def flow_event_summary(df: pd.DataFrame, cfg) -> pd.DataFrame:
    fr = flow_rows(df, cfg)
    if fr.empty:
        return pd.DataFrame()
    ts = cfg.flow_event_ts_col
    g = fr.groupby(cfg.activity_code_col, observed=True)
    out = pd.DataFrame({
        "Events": g.size(),
        "Distinct orders": g[cfg.case_id_col].nunique(),
        "First": g[ts].min(),
        "Last": g[ts].max(),
    })
    out.index.name = "Flow event"
    return out.iloc[_desc_order(out["Events"])].reset_index()


def flow_event_counts(df: pd.DataFrame, cfg, level_idx: int) -> dict:
    """Number of flow events attached to each code of the given hierarchy level."""
    fr = flow_rows(df, cfg)
    if fr.empty or not (cfg.flow_attach_level_col and cfg.flow_attach_code_col):
        return {}
    lv = pd.to_numeric(fr[cfg.flow_attach_level_col], errors="coerce")
    sel = fr[lv == level_idx]
    return sel[cfg.flow_attach_code_col].map(norm_code).value_counts().to_dict()


# ----------------------------------------------------------------------------
# Yearly overview (whole event log)
# ----------------------------------------------------------------------------
def yearly_overview(pipeline: dict, cfg, cases: pd.DataFrame) -> pd.DataFrame:
    el = pipeline["event_log"].copy()
    el["year"] = el["timestamp"].dt.year
    el["_flow"] = el["node_id"].str.startswith("flow_L")
    out = el.groupby("year").agg(**{
        "Events (all)": ("case_id", "size"),
        "Flow events": ("_flow", "sum"),
        "Distinct orders": ("case_id", "nunique"),
    })
    ops = _non_flow_df(pipeline["df"], cfg)
    d = ops.dropna(subset=[cfg.op_ts_col]).copy()
    d["year"] = d[cfg.op_ts_col].dt.year
    out["Operations"] = d.groupby("year").size()
    for lvl in cfg.hierarchy_levels:
        out[f"Distinct {lvl.label}"] = d.groupby("year")[lvl.code_col].nunique()
    dur = _duration_sec(d, cfg)
    if dur is not None:
        d["_dur"] = dur
        out["Avg. op duration"] = d.groupby("year")["_dur"].mean().map(format_duration)
    lead = cases.assign(year=cases["start"].dt.year).groupby("year")["duration_sec"].mean()
    out["Avg order lead time"] = lead.map(format_duration)
    return out


# ----------------------------------------------------------------------------
# ONLY ONE ENTRY POINT
# ----------------------------------------------------------------------------
def run_analysis(pipeline: dict, cfg) -> dict:
    df = pipeline["df"]
    ops = _non_flow_df(df, cfg)
    cases = build_case_table(pipeline["event_log"])
    flows = flow_table(pipeline)
    levels = []
    for i, lvl in enumerate(cfg.hierarchy_levels):
        _, pairs, names = build_level_transition_matrix(ops, cfg, lvl.code_col, lvl.name_col)
        ranking = level_ranking(ops, cfg, i, pairs, names)
        fe = flow_event_counts(df, cfg, i)
        if fe:
            ranking["Flow events attached"] = ranking[lvl.label].map(lambda c: fe.get(norm_code(c), 0))  # noqa: B023
        levels.append({"label": lvl.label, 
                       "level_idx": i, 
                       "ranking": ranking,
                       "year_share": level_year_share(ops, cfg, i)
                       })
    return {
        "summary": log_summary(pipeline, cfg, cases),
        "cases": cases,
        "case_durations": case_duration_summary(cases),
        "variants": variant_table(cases, pipeline),
        "flows": flows,
        "busiest_flows": busiest_flows(pipeline),
        "flow_events": flow_event_summary(df, cfg),
        "bottlenecks": bottleneck_flows(flows),
        "nodes": node_stats(pipeline),
        "start_end": start_end_nodes(pipeline),
        "transfer": transfer_time_summary(ops, cfg),
        "yearly": yearly_overview(pipeline, cfg, cases),
        "levels": levels,
    }


def show_svg(svg: str, height: int = 800):
    """Displays the SVG in an iframe (so that zoom/pan/hover JS works)."""
    doc = "<html><body style='margin:0;background:#fff'>" + svg + "</body></html>"
    display(HTML(f'<iframe srcdoc="{html.escape(doc, quote=True)}" width="100%" height="{height}" '
                 f'style="border:1px solid #ddd"></iframe>'))


def render_svg_to_string(pipeline, cfg, agg_level, settings, only_codes, title, mode) -> str:
    path = f"_tmp_{mode}.svg"
    try:
        render_process_map(pipeline, cfg, agg_level, settings, only_codes, mode, title=title, out_path=path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        if os.path.exists(path):
            os.remove(path)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved:", os.path.abspath(path))

