import argparse
import math
import os
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from slider import Beatmap, Circle, Slider, Spinner


# Configuration
TARGET_DIR = "maps"
DEFAULT_FILE = "test.osu"

# Scales are expressed in musical beats.
SCALE_WEIGHTS = {
    0.25: 0.25,
    0.50: 0.50,
    1.00: 1.00,
    2.00: 1.50,
    4.00: 1.75,
}

# RBF bandwidth in osu! pixels.
SIGMA = 6

# Temporal half-life in milliseconds.
MEMORY_HALFLIFE_MS = 7_000.0

# Relative weight of one beat in the distance metric.
TIME_WEIGHT = 2048.0

# Weight applied to successive local scores.
DECAY_FACTOR = 0.90

# Ignore interactions beyond this many half-lives.
MEMORY_CUTOFF_HALFLIVES = 12.0

# Optional debug output.
SHOW_TOP_PATTERNS_PER_MAP = 0

ACTION_EMPTY = "0"
ACTION_CLICK = "1"
ACTION_HOLD = "2"

# Candidate rhythm denominators.
CANDIDATE_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
SNAP_ERROR_THRESHOLD = 0.08
SNAP_MAX_ERROR_FRACTION = 0.20

# Minimum number of occurrences needed for comparison.
MIN_PATTERN_OCCURRENCES = 2


# Timing
@dataclass(frozen=True)
class TimingModel:
    """Piecewise-constant beat clock built from uninherited timing points."""

    times_ms: np.ndarray
    ms_per_beat: np.ndarray
    beat_starts: np.ndarray

    @classmethod
    def from_beatmap(cls, beatmap):
        entries = []

        for tp in beatmap.timing_points:
            offset = getattr(tp, "offset", getattr(tp, "time", None))
            mpb = getattr(tp, "ms_per_beat", None)

            if offset is None or mpb is None:
                continue
            if getattr(tp, "parent", None) is not None or mpb <= 0:
                continue

            t = (
                offset.total_seconds() * 1000.0
                if hasattr(offset, "total_seconds")
                else float(offset)
            )
            entries.append((t, float(mpb)))

        if not entries:
            entries = [(0.0, 500.0)]

        entries.sort(key=lambda item: item[0])

        # Keep the last red line at each timestamp.
        dedup = []
        for t, mpb in entries:
            if dedup and t == dedup[-1][0]:
                dedup[-1] = (t, mpb)
            else:
                dedup.append((t, mpb))

        times = np.asarray([item[0] for item in dedup], dtype=np.float64)
        mpb = np.asarray([item[1] for item in dedup], dtype=np.float64)

        beat_starts = np.zeros_like(times)
        if len(times) > 1:
            beat_starts[1:] = np.cumsum(np.diff(times) / mpb[:-1])

        return cls(times, mpb, beat_starts)

    def _index_for_time(self, t_ms: float) -> int:
        return max(0, int(np.searchsorted(self.times_ms, t_ms, side="right") - 1))

    def ms_per_beat_at(self, t_ms: float) -> float:
        return float(self.ms_per_beat[self._index_for_time(t_ms)])

    def beat_at(self, t_ms: float) -> float:
        idx = self._index_for_time(t_ms)
        return float(
            self.beat_starts[idx]
            + (t_ms - self.times_ms[idx]) / self.ms_per_beat[idx]
        )

    def beats_at(self, times_ms: np.ndarray) -> np.ndarray:
        times_ms = np.asarray(times_ms, dtype=np.float64)
        idx = np.searchsorted(self.times_ms, times_ms, side="right") - 1
        idx = np.maximum(idx, 0)
        return (
            self.beat_starts[idx]
            + (times_ms - self.times_ms[idx]) / self.ms_per_beat[idx]
        )

    def time_at_beat(self, beat: float) -> float:
        idx = max(
            0,
            int(np.searchsorted(self.beat_starts, beat, side="right") - 1),
        )
        return float(
            self.times_ms[idx]
            + (beat - self.beat_starts[idx]) * self.ms_per_beat[idx]
        )


# Slider points
def extract_slider_judgment_points(slider_obj, beatmap, timing: TimingModel):
    """Return slider head/ticks/repeats/tail as discrete analysis points."""

    start_t = slider_obj.time.total_seconds() * 1000.0
    end_t = slider_obj.end_time.total_seconds() * 1000.0
    duration = end_t - start_t

    if duration <= 0.0:
        return []

    repeats = max(1, int(slider_obj.repeat))
    span_duration = duration / repeats

    tick_rate = float(getattr(beatmap, "slider_tick_rate", 1.0) or 1.0)
    tick_rate = max(tick_rate, 1e-9)

    # Tick spacing is based on the beat length at the slider start.
    beat_span = span_duration / timing.ms_per_beat_at(start_t)
    tick_step_beats = 1.0 / tick_rate

    points = []

    for span_index in range(repeats):
        span_start_t = start_t + span_index * span_duration
        span_end_t = span_start_t + span_duration
        reverse = bool(span_index & 1)

        if span_index == 0:
            p = 0.0
            action = ACTION_CLICK
        else:
            p = 1.0 if reverse else 0.0
            action = ACTION_HOLD

        pos = slider_obj.curve(p)
        points.append((pos.x, pos.y, span_start_t, action))

        # Ticks are strictly inside a span.
        tick_count = max(
            0,
            int(math.ceil(beat_span * tick_rate - 1e-10)) - 1,
        )

        for k in range(1, tick_count + 1):
            frac = (k * tick_step_beats) / beat_span
            p = 1.0 - frac if reverse else frac
            t = span_start_t + frac * span_duration

            pos = slider_obj.curve(p)
            points.append((pos.x, pos.y, t, ACTION_HOLD))

        if span_index == repeats - 1:
            p = 0.0 if reverse else 1.0
            pos = slider_obj.curve(p)
            points.append((pos.x, pos.y, span_end_t, ACTION_HOLD))

    return points


def extract_points(beatmap, timing: TimingModel):
    points = []

    for obj in beatmap.hit_objects():
        t = obj.time.total_seconds() * 1000.0

        if isinstance(obj, Circle):
            points.append((obj.position.x, obj.position.y, t, ACTION_CLICK))

        elif isinstance(obj, Slider):
            points.extend(
                extract_slider_judgment_points(obj, beatmap, timing)
            )

        elif isinstance(obj, Spinner):
            end_t = obj.end_time.total_seconds() * 1000.0
            points.append((obj.position.x, obj.position.y, t, ACTION_CLICK))
            points.append(
                (obj.position.x, obj.position.y, end_t, ACTION_HOLD)
            )

    points.sort(key=lambda point: point[2])
    return points


# Pattern quantisation
def detect_adaptive_tick_resolution(point_beats):
    """Choose the simplest denominator that explains most beat intervals."""

    if len(point_beats) < 2:
        return 4

    deltas = np.diff(point_beats)
    deltas = deltas[deltas > 1e-6]

    if deltas.size == 0:
        return 4

    # Exclude extreme intervals when estimating snap density.
    lo, hi = np.quantile(deltas, (0.02, 0.98))
    deltas = deltas[(deltas >= lo) & (deltas <= hi)]

    if deltas.size == 0:
        return 4

    for denominator in CANDIDATE_DENOMINATORS:
        frac_error = np.abs(
            deltas * denominator - np.rint(deltas * denominator)
        )
        mean_error = float(np.mean(frac_error))
        clean_fraction = float(
            np.mean(frac_error < SNAP_MAX_ERROR_FRACTION)
        )

        if (
            mean_error < SNAP_ERROR_THRESHOLD
            and clean_fraction >= 0.90
        ):
            return denominator

    return 8


# Pattern encoding
def encode_pattern(point_slice, beat_start, beat_end, denominator):
    """Encode a slice as ternary equal-duration musical subcells."""

    span = max(beat_end - beat_start, 1e-12)
    num_subs = max(1, int(round(span * denominator)))
    sub_len = span / num_subs

    pattern = [ACTION_EMPTY] * num_subs

    for _, _, beat, action in point_slice:
        index = int((beat - beat_start) / sub_len)
        index = min(max(index, 0), num_subs - 1)

        if action == ACTION_CLICK:
            # Click takes precedence over hold.
            pattern[index] = ACTION_CLICK
        elif pattern[index] == ACTION_EMPTY:
            pattern[index] = ACTION_HOLD

    return "".join(pattern)


# Spatial and temporal signatures
def build_distance_signature(
    points,
    beat_positions,
    slice_beat_start,
    time_weight=TIME_WEIGHT,
):
    """
    Return only the strict upper triangle of the pairwise Euclidean distance
    matrix in (x, y, beat*time) space.

    Keeping the condensed upper triangle removes duplicated symmetric entries
    and the diagonal, which never contributes to a pairwise MSE.
    """

    n = len(points)
    if n <= 1:
        return np.empty(0, dtype=np.float64)

    xy = np.asarray(
        [(point[0], point[1]) for point in points],
        dtype=np.float64,
    )
    rel_beats = np.asarray(beat_positions, dtype=np.float64) - slice_beat_start

    row, col = np.triu_indices(n, k=1)
    dx = xy[row, 0] - xy[col, 0]
    dy = xy[row, 1] - xy[col, 1]
    dz = (rel_beats[row] - rel_beats[col]) * time_weight

    return np.sqrt(dx * dx + dy * dy + dz * dz)


# Continuous complexity
def compute_continuous_complexity(
    signatures,
    time_list,
    sigma=SIGMA,
    half_life_ms=MEMORY_HALFLIFE_MS,
):
    """Return the mean temporally weighted novelty in [0, 1]."""
    m = len(signatures)
    if m < 2:
        return 0.0

    feature_count = len(signatures[0])
    if feature_count == 0 or any(len(s) != feature_count for s in signatures[1:]):
        return 0.0

    # The condensed signature is the mean over C(n, 2) pairwise distances.
    # The original full n x n matrix MSE contains each off-diagonal pair twice,
    # hence MSE_full = (n - 1) / n * MSE_condensed.
    discriminant = 1.0 + 8.0 * feature_count
    point_count = int(round((1.0 + math.sqrt(discriminant)) * 0.5))
    if point_count < 2 or point_count * (point_count - 1) // 2 != feature_count:
        return 0.0
    mse_scale = (point_count - 1.0) / point_count

    x = np.asarray(signatures, dtype=np.float64)
    times = np.asarray(time_list, dtype=np.float64)

    # Callers already provide chronological occurrences.  Sorting here only
    # if necessary keeps the function robust without paying the normal cost.
    if np.any(times[1:] < times[:-1]):
        order = np.argsort(times, kind="stable")
        times = times[order]
        x = x[order]

    # Row-wise mean square is enough to evaluate pairwise MSE:
    # mean((x_i-x_j)^2) = mean(x_i^2) + mean(x_j^2) - 2 mean(x_i*x_j).
    norm2 = np.einsum("ij,ij->i", x, x) / feature_count

    inv_two_sigma2 = 0.5 / (sigma * sigma)
    cutoff_ms = MEMORY_CUTOFF_HALFLIVES * half_life_ms

    novelty_mass = np.zeros(m, dtype=np.float64)
    kernel_mass = np.ones(m, dtype=np.float64)

    for i in range(m - 1):
        stop = int(np.searchsorted(times, times[i] + cutoff_ms, side="right"))
        if stop <= i + 1:
            continue

        xj = x[i + 1:stop]
        mse = mse_scale * (
            norm2[i] + norm2[i + 1:stop]
            - 2.0 * (xj @ x[i]) / feature_count
        )
        # Guard against floating-point cancellation.
        np.maximum(mse, 0.0, out=mse)

        temporal_weight = np.exp2(-(times[i + 1:stop] - times[i]) / half_life_ms)
        pair_novelty = temporal_weight * (-np.expm1(-mse * inv_two_sigma2))

        novelty_mass[i] += pair_novelty.sum()
        novelty_mass[i + 1:stop] += pair_novelty
        kernel_mass[i] += temporal_weight.sum()
        kernel_mass[i + 1:stop] += temporal_weight

    return float(np.clip(np.mean(novelty_mass / kernel_mass), 0.0, 1.0))


# Analysis
def calculate_unpredictability(file_path):
    if not os.path.exists(file_path):
        return None

    try:
        beatmap = Beatmap.from_path(file_path)
    except Exception as exc:
        print(f"   [Error] Failed to parse beatmap: {exc}")
        return None

    timing = TimingModel.from_beatmap(beatmap)
    all_points = extract_points(beatmap, timing)

    if not all_points:
        return None

    point_times = np.asarray(
        [point[2] for point in all_points],
        dtype=np.float64,
    )
    point_beats = timing.beats_at(point_times)

    points_with_beats = [
        (point[0], point[1], beat, point[3])
        for point, beat in zip(all_points, point_beats)
    ]

    d_map = detect_adaptive_tick_resolution(point_beats)
    beat_start = float(point_beats[0])
    beat_end = float(point_beats[-1])

    total_soft_c = 0.0
    all_slice_scores = []
    all_scale_details = []

    for scale, weight in SCALE_WEIGHTS.items():
        span = beat_end - beat_start
        slice_count = max(1, int(math.ceil(span / scale)))

        slice_ids = np.floor(
            (point_beats - beat_start) / scale + 1e-12
        ).astype(np.int64)
        slice_ids = np.clip(slice_ids, 0, slice_count - 1)

        starts = np.r_[0, np.flatnonzero(np.diff(slice_ids)) + 1]
        unique_ids = slice_ids[starts]
        ends = np.r_[starts[1:], len(point_beats)]

        pattern_signatures = defaultdict(list)
        pattern_times = defaultdict(list)

        for slice_id, left, right in zip(unique_ids, starts, ends):
            slice_beat_start = beat_start + float(slice_id) * scale
            slice_beat_end = min(
                slice_beat_start + scale,
                beat_end,
            )

            if right <= left:
                continue

            pts = points_with_beats[left:right]
            pattern = encode_pattern(
                pts,
                slice_beat_start,
                slice_beat_end,
                d_map,
            )

            signature = build_distance_signature(
                pts,
                point_beats[left:right],
                slice_beat_start,
            )

            key = (pattern, len(signature))
            pattern_signatures[key].append(signature)
            pattern_times[key].append(timing.time_at_beat(slice_beat_start))

        for (pattern, point_signature_size), signatures in pattern_signatures.items():
            occurrence_count = len(signatures)

            if occurrence_count < MIN_PATTERN_OCCURRENCES:
                continue

            c_soft = compute_continuous_complexity(
                signatures,
                pattern_times[(pattern, point_signature_size)],
            )

            score = weight * c_soft

            if score > 0.0:
                all_slice_scores.append(score)

            total_soft_c += c_soft
            all_scale_details.append(
                {
                    "scale": scale,
                    "pattern": pattern,
                    "count": occurrence_count,
                    "c_soft": c_soft,
                    "score": score,
                }
            )

    all_slice_scores.sort(reverse=True)

    if all_slice_scores:
        scores = np.asarray(all_slice_scores, dtype=np.float64)
        n = scores.size
        if DECAY_FACTOR == 1.0:
            final_score = float(scores.mean())
        else:
            weighted_sum = 0.0
            for score in scores[::-1]:
                weighted_sum = float(score) + DECAY_FACTOR * weighted_sum
            normalization = (1.0 - DECAY_FACTOR) / (1.0 - DECAY_FACTOR ** n)
            final_score = weighted_sum * normalization
    else:
        final_score = 0.0

    all_scale_details.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    start_time = float(point_times[0])
    end_time = float(point_times[-1])
    drain_mins = max(
        (end_time - start_time) / 60000.0,
        0.1,
    )

    return {
        "title": f"{beatmap.title} [{beatmap.version}]",
        "c_soft": total_soft_c,
        "final_score": final_score,
        "drain_mins": drain_mins,
        "d_map": d_map,
        "top_patterns": all_scale_details[:SHOW_TOP_PATTERNS_PER_MAP],
    }


# File handling
def collect_beatmap_files(target_path):
    file_queue = []

    if os.path.isfile(target_path):
        lower = target_path.lower()

        if lower.endswith(".osu"):
            file_queue.append(
                {
                    "path": target_path,
                    "source": os.path.basename(target_path),
                    "is_temp": False,
                }
            )
        elif lower.endswith(".osz"):
            file_queue.extend(extract_osz(target_path))

    elif os.path.isdir(target_path):
        for root, _, files in os.walk(target_path):
            for name in files:
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, target_path)
                lower = name.lower()

                if lower.endswith(".osu"):
                    file_queue.append(
                        {
                            "path": full_path,
                            "source": rel_path,
                            "is_temp": False,
                        }
                    )
                elif lower.endswith(".osz"):
                    file_queue.extend(
                        extract_osz(
                            full_path,
                            rel_base=target_path,
                        )
                    )

    return file_queue


def extract_osz(osz_path, rel_base=None):
    extracted = []

    source_label = (
        os.path.basename(osz_path)
        if rel_base is None
        else os.path.relpath(osz_path, rel_base)
    )

    try:
        with zipfile.ZipFile(osz_path, "r") as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".osu"):
                    continue

                tmp = tempfile.NamedTemporaryFile(
                    suffix=".osu",
                    delete=False,
                )

                with tmp:
                    tmp.write(zf.read(member))

                extracted.append(
                    {
                        "path": tmp.name,
                        "source": os.path.basename(member),
                        "is_temp": True,
                    }
                )

    except Exception as exc:
        print(
            f"[Warning] Could not read .osz file "
            f"'{source_label}': {exc}"
        )

    return extracted


# CLI
def main():
    parser = argparse.ArgumentParser(
        description=(
            "osu! beatmap unpredictability analysis"
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="Directory or file path (.osu / .osz)",
    )
    args = parser.parse_args()

    if args.path and os.path.exists(args.path):
        input_path = args.path
    elif os.path.exists(TARGET_DIR):
        input_path = TARGET_DIR
    elif os.path.exists(DEFAULT_FILE):
        input_path = DEFAULT_FILE
    else:
        candidates = [
            name
            for name in os.listdir(".")
            if name.lower().endswith((".osu", ".osz"))
        ]
        input_path = candidates[0] if candidates else None

    if not input_path:
        print("[Error] No input directory or .osu / .osz file found.")
        return

    file_queue = collect_beatmap_files(input_path)

    if not file_queue:
        print(f"[Error] No valid beatmaps found in target: {input_path}")
        return

    results = []
    total = len(file_queue)

    for idx, item in enumerate(file_queue, 1):
        f_path = item["path"]
        print(
            f"[{idx}/{total}] {item['source']}",
            end="",
            flush=True,
        )

        res = None
        failed = False
        try:
            res = calculate_unpredictability(f_path)
        except Exception as exc:
            failed = True
            print(f"  ERROR ({exc})")
        finally:
            if item["is_temp"] and os.path.exists(f_path):
                os.remove(f_path)

        if res:
            res["source"] = item["source"]
            results.append(res)
            print(f"  {res['final_score'] * 100:.1f}")
        elif not failed:
            print("  ERROR")

        if (
            SHOW_TOP_PATTERNS_PER_MAP
            and res
            and res["top_patterns"]
        ):
            top = res["top_patterns"][0]
            print(
                f"    Top: {top['scale']}b [{top['pattern']}] | "
                f"C_soft {top['c_soft']:.2f} | "
                f"Local {top['score']:.2f}"
            )

    if not results:
        return

    results.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    print()
    print(
        "════════════════════════════════════════════════════════════════════"
    )
    print("                         LEADERBOARD")
    print(
        "════════════════════════════════════════════════════════════════════"
    )

    for rank, res in enumerate(results, 1):
        source = os.path.splitext(
            os.path.basename(res["source"])
        )[0]
        score = res["final_score"] * 100.0
        print(
            f"{rank:>3}  {score:>6.1f}   {source}"
        )

    print(
        "════════════════════════════════════════════════════════════════════"
    )


if __name__ == "__main__":
    main()
