import cv2
import imagehash
from PIL import Image

import argparse
import configparser
import math
import os
import sys
import time
import subprocess
import numpy as np
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, freeze_support
from queue import Empty


DEFAULT_CONFIG_NAME = "frameMatcher.config"


def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - hours * 3600 - minutes * 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes}:{secs:04.1f}"


def safe_time(seconds: float) -> str:
    tenths = int(round(seconds * 10))
    minutes, rem = divmod(tenths, 600)
    sec, tenth = divmod(rem, 10)
    return f"{minutes}m{sec:02d}.{tenth}s"


def similarity_from_distance(distance: int, hash_size: int) -> float:
    total_bits = hash_size * hash_size
    return max(0.0, 100.0 * (1.0 - distance / total_bits))


def add_best(candidates, candidate, limit: int):
    candidates.append(candidate)
    candidates.sort(key=lambda x: x[0])
    if len(candidates) > limit:
        candidates.pop()


def load_config(path: str):
    cfg = configparser.ConfigParser()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    cfg.read(path)

    def getint(name, fallback=None):
        if fallback is None:
            return cfg.getint("matching", name)
        return cfg.getint("matching", name, fallback=fallback)

    min_gap = cfg.getfloat("matching", "min_gap_seconds")
    max_gap = cfg.getfloat("matching", "max_gap_seconds")
    pass1_fps = cfg.getfloat("matching", "pass1_fps")
    pass2_fps = cfg.getfloat("matching", "pass2_fps")
    refine_window = cfg.getfloat("matching", "refine_window_seconds")
    top_candidates = cfg.getint("matching", "top_candidates")
    pass1_workers = getint("pass1_workers", 1)
    pass2_workers = getint("pass2_workers", None)
    overlap_setting = cfg.get("matching", "chunk_overlap_seconds", fallback="auto").strip()
    hash_size = cfg.getint("matching", "hash_size")
    progress_pct = cfg.getfloat("matching", "progress_interval_percent", fallback=10.0)
    pass1_mode = cfg.get("matching", "pass1_mode", fallback="ffmpeg").strip().lower()
    ffmpeg_path = cfg.get("matching", "ffmpeg_path", fallback="ffmpeg").strip() or "ffmpeg"
    ffmpeg_threads = cfg.getint("matching", "ffmpeg_threads", fallback=0)
    pass1_hash_width = cfg.getint("matching", "pass1_hash_width", fallback=256)

    output_dir = cfg.get("output", "output_directory", fallback="similar_frame_results")
    save_top = cfg.getint("output", "save_top_results", fallback=5)

    if pass2_workers is None:
        pass2_workers = pass1_workers

    if min_gap < 0 or max_gap < min_gap:
        raise ValueError("Require 0 <= min_gap_seconds <= max_gap_seconds")
    if pass1_fps <= 0 or pass2_fps <= 0:
        raise ValueError("pass1_fps and pass2_fps must be > 0")
    if refine_window <= 0:
        raise ValueError("refine_window_seconds must be > 0")
    if top_candidates <= 0 or pass1_workers <= 0 or pass2_workers <= 0:
        raise ValueError("top_candidates and worker counts must be > 0")
    if hash_size <= 0:
        raise ValueError("hash_size must be > 0")
    if not (0 < progress_pct <= 100):
        raise ValueError("progress_interval_percent must be > 0 and <= 100")
    if pass1_mode not in {"ffmpeg", "chunked", "opencv_sequential"}:
        raise ValueError("pass1_mode must be 'ffmpeg', 'opencv_sequential' or 'chunked'")
    if ffmpeg_threads < 0:
        raise ValueError("ffmpeg_threads must be >= 0 (0 = auto)")
    if pass1_hash_width <= 0:
        raise ValueError("pass1_hash_width must be > 0")

    required_overlap = max_gap + 1.0 / pass1_fps
    if overlap_setting.lower() == "auto":
        overlap = required_overlap
    else:
        overlap = float(overlap_setting)
        if overlap < required_overlap - 1e-9:
            raise ValueError(
                f"chunk_overlap_seconds={overlap} is too small. "
                f"Use at least {required_overlap:.3f} seconds, or use 'auto'."
            )

    return {
        "min_gap": min_gap,
        "max_gap": max_gap,
        "pass1_fps": pass1_fps,
        "pass2_fps": pass2_fps,
        "refine_window": refine_window,
        "top_candidates": top_candidates,
        "pass1_workers": pass1_workers,
        "pass2_workers": pass2_workers,
        "overlap": overlap,
        "hash_size": hash_size,
        "progress_pct": progress_pct,
        "pass1_mode": pass1_mode,
        "ffmpeg_path": ffmpeg_path,
        "ffmpeg_threads": ffmpeg_threads,
        "pass1_hash_width": pass1_hash_width,
        "output_dir": output_dir,
        "save_top": save_top,
    }


def log_event(tag, message):
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {message}", flush=True)


def put_progress(q, **payload):
    if q is None:
        return
    try:
        q.put(payload)
    except Exception:
        pass


def frame_to_hash(frame_bgr, hash_size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    return imagehash.phash(image, hash_size=hash_size)



def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError("Could not determine video dimensions/FPS/frame count")
    return width, height, fps, frame_count


def decode_sequential_samples(
    video_path,
    start_time,
    end_time,
    sample_times,
    native_fps,
    hash_size,
    progress_queue=None,
    progress_prefix="",
    progress_percent=25.0,
):
    """OpenCV sequential decoder used by Pass 2.
    Pass 2 is intentionally kept close to the tested/fast version.
    """
    if not sample_times:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    start_frame = max(0, int(math.floor(start_time * native_fps)))
    end_frame = max(start_frame, int(math.ceil(end_time * native_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    targets = []
    seen = set()
    for ts in sample_times:
        idx = int(round(ts * native_fps))
        if idx < start_frame or idx > end_frame:
            continue
        if idx not in seen:
            targets.append((idx, ts))
            seen.add(idx)
    targets.sort()

    target_pos = 0
    results = []
    total_decode_frames = max(1, end_frame - start_frame + 1)
    next_progress = progress_percent
    decoded = 0
    current_frame = start_frame - 1

    while current_frame < end_frame and target_pos < len(targets):
        ok, frame = cap.read()
        if not ok:
            break
        current_frame += 1
        decoded += 1

        while target_pos < len(targets) and targets[target_pos][0] <= current_frame:
            ph = frame_to_hash(frame, hash_size)
            actual_ts = current_frame / native_fps
            results.append((actual_ts, ph))
            target_pos += 1

        pct = 100.0 * decoded / total_decode_frames
        if progress_queue is not None and (pct >= next_progress or decoded == total_decode_frames):
            put_progress(
                progress_queue,
                kind="decode_progress",
                prefix=progress_prefix,
                percent=min(100.0, pct),
            )
            while next_progress <= pct + 1e-9:
                next_progress += progress_percent

    cap.release()
    return results


def ffmpeg_pass1_scan(
    video_path,
    duration,
    native_fps,
    video_width,
    video_height,
    pass1_fps,
    min_gap,
    max_gap,
    hash_size,
    top_candidates,
    progress_queue,
    progress_pct,
    ffmpeg_path,
    hash_width,
    ffmpeg_threads,
):
    """Decode the entire video exactly once with FFmpeg.

    FFmpeg performs the compressed-video decoding sequentially and uses its
    own internal decoder threads. This avoids N independent VideoCapture
    instances fighting over the same compressed file.

    Frames are downscaled before being sent to Python because perceptual
    hashing does not require full-resolution images.
    """
    if hash_width <= 0:
        raise ValueError("pass1_hash_width must be > 0")

    # Keep aspect ratio and make height even for common codecs.
    out_width = min(int(hash_width), max(2, int(video_width)))
    out_height = max(
        2,
        int(round(video_height * out_width / video_width))
    )
    if out_height % 2:
        out_height += 1

    frame_bytes = out_width * out_height * 3
    filter_expr = f"fps={pass1_fps:g},scale={out_width}:{out_height}:flags=fast_bilinear"

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "error",
        "-threads", str(ffmpeg_threads),
        "-i", video_path,
        "-vf", filter_expr,
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "pipe:1",
    ]

    log_event(
        "PASS1",
        f"FFmpeg sequential decode started: {pass1_fps:g} fps -> "
        f"{out_width}x{out_height} RGB, internal threads={ffmpeg_threads}"
    )

    started = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_bytes * 4,
    )

    candidates = []
    frames = []
    frame_index = 0
    next_progress = progress_pct
    next_sample_time = 0.0

    # The fps filter produces a regular output timeline at pass1_fps.
    while True:
        raw = process.stdout.read(frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            # A truncated final frame is not useful.
            break

        timestamp = frame_index / pass1_fps
        if timestamp > duration + (1.0 / pass1_fps):
            break

        arr = memoryview(raw)
        # frombuffer avoids an intermediate Python list copy.
        frame_rgb = np.frombuffer(arr, dtype=np.uint8).reshape((out_height, out_width, 3))
        image = Image.fromarray(frame_rgb, mode="RGB")
        ph = imagehash.phash(image, hash_size=hash_size)
        frames.append((timestamp, ph))
        frame_index += 1

        pct = 100.0 * min(timestamp, duration) / max(duration, 1e-9)
        if progress_queue is not None and pct >= next_progress:
            put_progress(
                progress_queue,
                kind="ffmpeg_progress",
                percent=min(100.0, pct),
                frame=frame_index,
            )
            while next_progress <= pct + 1e-9:
                next_progress += progress_pct

    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    return_code = process.wait()

    if return_code != 0:
        message = stderr[-2000:] if stderr else f"FFmpeg exited with code {return_code}"
        raise RuntimeError(f"Pass 1 FFmpeg failed: {message}")

    # If fps filtering is used, the sampled timestamps are regular. Compare
    # every frame pair satisfying the requested time gap.
    times = [t for t, _ in frames]
    comparisons = 0
    for i, (time1, hash1) in enumerate(frames):
        lo = bisect_left(times, time1 + min_gap - 1e-9, i + 1)
        hi = bisect_right(times, time1 + max_gap + 1e-9, lo)
        for j in range(lo, hi):
            time2, hash2 = frames[j]
            distance = hash1 - hash2
            comparisons += 1
            add_best(candidates, (distance, time1, time2), top_candidates)

    elapsed = time.perf_counter() - started
    best_distance = candidates[0][0] if candidates else None

    rate = len(frames) / max(elapsed, 1e-9)
    best_text = "n/a" if best_distance is None else f"{best_distance} bits"
    log_event(
        "PASS1",
        f"COMPLETE: {len(frames)} sampled frames, {comparisons} comparisons, "
        f"best hash distance {best_text}; {elapsed:.2f}s "
        f"({rate:.1f} sampled fps)"
    )
    return candidates



def opencv_sequential_pass1_scan(
    video_path,
    duration,
    native_fps,
    frame_count,
    pass1_fps,
    min_gap,
    max_gap,
    hash_size,
    top_candidates,
    progress_queue,
    progress_pct,
):
    """Fast Pass 1 that preserves the old OpenCV/pHash semantics.

    The important differences from the FFmpeg pass are:
      1. The video is decoded sequentially exactly once.
      2. Frames are NOT resized before hashing.
      3. We target the same timestamp grid as the original implementation.
      4. We keep the requested sample timestamp for candidate reporting,
         matching the old implementation.

    We use grab() for non-target frames, which avoids copying the frame into
    Python until a sample is actually needed.
    """
    log_event(
        "PASS1",
        "Starting compatibility-optimized OpenCV first pass: one sequential decode, "
        "full-resolution pHash, no random seeks."
    )

    sample_times = make_sample_times(
        0.0,
        duration,
        pass1_fps,
    )

    if not sample_times:
        return []

    # Match the frame selection used elsewhere in this program's sequential
    # decoder (and the natural interpretation of a timestamp in a CFR video).
    targets = []
    seen_indices = set()
    for ts in sample_times:
        frame_index = int(round(ts * native_fps))
        if frame_index < 0 or frame_index >= frame_count:
            continue
        if frame_index in seen_indices:
            continue
        seen_indices.add(frame_index)
        targets.append((frame_index, ts))

    targets.sort(key=lambda x: x[0])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    target_pos = 0
    current_index = -1
    total_targets = len(targets)
    next_progress = progress_pct
    started = time.perf_counter()

    while target_pos < total_targets:
        ok = cap.grab()
        if not ok:
            break

        current_index += 1
        target_index, requested_timestamp = targets[target_pos]

        if current_index < target_index:
            # We only need the decoder to advance; don't retrieve/copy the frame.
            continue

        if current_index > target_index:
            # This should not normally occur for a CFR stream, but don't lose
            # the sample if a backend advances unexpectedly. Associate the
            # decoded frame with the requested sample timestamp.
            ok_retrieve, frame = cap.retrieve()
            if not ok_retrieve:
                break
            ph = frame_to_hash(frame, hash_size)
            frames.append((requested_timestamp, ph))
            target_pos += 1
        else:
            ok_retrieve, frame = cap.retrieve()
            if not ok_retrieve:
                break
            ph = frame_to_hash(frame, hash_size)
            frames.append((requested_timestamp, ph))
            target_pos += 1

        pct = 100.0 * min(current_index + 1, frame_count) / max(frame_count, 1)
        if pct >= next_progress:
            # This pass runs in the main process, so log directly.
            log_event(
                "PROGRESS",
                f"PASS1 sequential decode: {min(100.0, pct):.0f}% "
                f"({len(frames)}/{total_targets} samples hashed)"
            )
            while next_progress <= pct + 1e-9:
                next_progress += progress_pct

    cap.release()

    if target_pos < total_targets:
        log_event(
            "PASS1",
            f"Warning: decoded {target_pos}/{total_targets} requested samples before EOF."
        )

    # Same candidate comparison model as the original pass: compare sample
    # timestamps, not the actual frame presentation timestamp.
    candidates = []
    times = [t for t, _ in frames]
    comparisons = 0

    for i, (time1, hash1) in enumerate(frames):
        lo = bisect_left(
            times,
            time1 + min_gap - 1e-9,
            i + 1,
        )
        hi = bisect_right(
            times,
            time1 + max_gap + 1e-9,
            lo,
        )
        for j in range(lo, hi):
            time2, hash2 = frames[j]
            distance = hash1 - hash2
            comparisons += 1
            add_best(
                candidates,
                (distance, time1, time2),
                top_candidates,
            )

    elapsed = time.perf_counter() - started
    best_text = (
        "n/a"
        if not candidates
        else f"{candidates[0][0]} bits"
    )

    log_event(
        "PASS1",
        f"COMPLETE: {len(frames)} sampled frames, {comparisons} comparisons, "
        f"best hash distance {best_text}; {elapsed:.2f}s"
    )

    return candidates

def pass2_worker(task):
    (
        video_path, candidate_index, candidate_count,
        center1, center2, min_gap, max_gap,
        pass2_fps, refine_window, native_fps,
        hash_size, progress_queue,
    ) = task

    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    region1_start = max(0.0, center1 - refine_window)
    region1_end = center1 + refine_window
    region2_start = max(0.0, center2 - refine_window)
    region2_end = center2 + refine_window

    region1_times = make_sample_times(region1_start, region1_end, pass2_fps)
    region2_times = make_sample_times(region2_start, region2_end, pass2_fps)

    put_progress(
        progress_queue,
        kind="candidate_start",
        candidate=candidate_index,
        total=candidate_count,
        center1=center1,
        center2=center2,
        samples=len(region1_times) + len(region2_times),
    )

    hashes1 = decode_sequential_samples(
        video_path,
        region1_start,
        region1_end,
        region1_times,
        native_fps,
        hash_size,
        progress_queue,
        progress_prefix=f"candidate {candidate_index}/{candidate_count} A",
        progress_percent=50.0,
    )

    hashes2 = decode_sequential_samples(
        video_path,
        region2_start,
        region2_end,
        region2_times,
        native_fps,
        hash_size,
        progress_queue,
        progress_prefix=f"candidate {candidate_index}/{candidate_count} B",
        progress_percent=50.0,
    )

    best = None
    for time1, hash1 in hashes1:
        for time2, hash2 in hashes2:
            gap = time2 - time1
            if gap < min_gap or gap > max_gap:
                continue
            distance = hash1 - hash2
            if best is None or distance < best[0]:
                best = (distance, time1, time2)

    put_progress(
        progress_queue,
        kind="candidate_done",
        candidate=candidate_index,
        total=candidate_count,
        result=best,
    )
    return best


def drain_progress(q, settings):
    while True:
        try:
            ev = q.get_nowait()
        except Empty:
            return

        kind = ev.get("kind")
        if kind == "chunk_start":
            log_event("PASS1", f"Chunk {ev['chunk']}/{ev['total']} started; core {fmt_time(ev['core_start'])} - {fmt_time(ev['core_end'])}; "
                      f"decode range {fmt_time(ev['scan_start'])} - {fmt_time(ev['scan_end'])} ({ev['samples']} target frames)")
        elif kind == "decode_progress":
            log_event("PROGRESS", f"{ev['prefix']}: {ev['percent']:.0f}% decoded")
        elif kind == "opencv_pass1_progress":
            log_event(
                "PROGRESS",
                f"PASS1 sequential decode: {ev['percent']:.0f}% "
                f"({ev['samples']}/{ev['total_samples']} samples hashed)"
            )
        elif kind == "candidate_start":
            log_event("PASS2", f"Candidate {ev['candidate']}/{ev['total']} started: "
                      f"{fmt_time(ev['center1'])} -> {fmt_time(ev['center2'])} ({ev['samples']} target frames)")
        elif kind == "candidate_done":
            result = ev.get("result")
            if result:
                d, t1, t2 = result
                sim = similarity_from_distance(d, settings["hash_size"])
                log_event("PASS2", f"Candidate {ev['candidate']}/{ev['total']} COMPLETE: "
                          f"{fmt_time(t1)} -> {fmt_time(t2)}, gap {t2-t1:.1f}s, similarity {sim:.2f}%")
            else:
                log_event("PASS2", f"Candidate {ev['candidate']}/{ev['total']} COMPLETE: no valid pair")


def run_parallel(tasks, worker_fn, workers, progress_queue, settings):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(worker_fn, task): i for i, task in enumerate(tasks)}
        while future_map:
            drain_progress(progress_queue, settings)
            done_futures = []
            for future in list(future_map):
                if future.done():
                    done_futures.append(future)
            for future in done_futures:
                idx = future_map.pop(future)
                results.append((idx, future.result()))
            time.sleep(0.15)
        drain_progress(progress_queue, settings)
    results.sort(key=lambda x: x[0])
    return [x[1] for x in results]


def find_ffmpeg(executable):
    """Return an executable path for FFmpeg."""
    if executable and os.path.isfile(executable):
        return executable
    if executable:
        return executable  # Let subprocess produce a useful error.
    return "ffmpeg"


def run_pass1_ffmpeg(
    video_path,
    duration,
    native_fps,
    video_width,
    video_height,
    settings,
    progress_queue,
):
    log_event("PASS1", "Starting optimized FFmpeg first pass: one sequential video decode, no chunk seeks.")
    ffmpeg_path = find_ffmpeg(settings["ffmpeg_path"])
    log_event("PASS1", f"FFmpeg executable: {ffmpeg_path}")
    if settings["ffmpeg_threads"] == 0:
        log_event("PASS1", "FFmpeg decoder threads: auto (recommended)")
    else:
        log_event("PASS1", f"FFmpeg decoder threads: {settings['ffmpeg_threads']}")
    log_event("PASS1", f"Pass 1 workers are not used in FFmpeg mode; FFmpeg parallelizes decoding internally.")

    try:
        return ffmpeg_pass1_scan(
            video_path,
            duration,
            native_fps,
            video_width,
            video_height,
            settings["pass1_fps"],
            settings["min_gap"],
            settings["max_gap"],
            settings["hash_size"],
            settings["top_candidates"],
            progress_queue,
            settings["progress_pct"],
            ffmpeg_path,
            settings["pass1_hash_width"],
            settings["ffmpeg_threads"],
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg could not be started. Install FFmpeg and ensure 'ffmpeg' is on PATH, "
            "or set ffmpeg_path in frameMatcher.config."
        ) from exc


def make_pass1_tasks(video_path, duration, native_fps, settings, progress_queue):
    workers = min(settings["pass1_workers"], max(1, math.ceil(duration / max(settings["max_gap"], 1.0))))
    tasks = []
    for i in range(workers):
        core_start = duration * i / workers
        core_end = duration * (i + 1) / workers
        tasks.append((
            video_path, duration, native_fps,
            i + 1, workers,
            core_start, core_end, i == workers - 1,
            settings["overlap"], settings["pass1_fps"],
            settings["min_gap"], settings["max_gap"],
            settings["hash_size"], settings["top_candidates"],
            progress_queue, settings["progress_pct"],
        ))
    return tasks, workers


def merge_candidates(results, limit):
    merged = []
    for result in results:
        for candidate in result["candidates"]:
            add_best(merged, candidate, limit)
    return merged

def make_sample_times(start, end, fps):
    interval = 1.0 / fps
    first = max(0, math.ceil(start / interval - 1e-9))
    last = math.floor(end / interval + 1e-9)
    return [i * interval for i in range(first, last + 1)]

def dedup_results(results, tolerance=0.05):
    out = []
    for result in sorted(results, key=lambda x: x[0]):
        _, t1, t2 = result
        if any(abs(t1-a) <= tolerance and abs(t2-b) <= tolerance for _, a, b in out):
            continue
        out.append(result)
    return out


def save_frame(video, timestamp, filename):
    # For final results only; random seeking here is fine because it is only a few frames.
    video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
    ok, frame = video.read()
    if not ok:
        return False
    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def main():
    parser = argparse.ArgumentParser(description="Fast two-pass similar-frame finder")
    parser.add_argument("video", help="Video file")
    parser.add_argument("--config", default=None, help="Config path")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(args.config) if args.config else os.path.join(script_dir, DEFAULT_CONFIG_NAME)
    settings = load_config(config_path)

    video_width, video_height, native_fps, frame_count = get_video_dimensions(video_path)
    duration = frame_count / native_fps

    print("=" * 70)
    print(" Video Frame Matcher - Sequential Decode + Parallel Refinement")
    print("=" * 70)
    print(f"Video:           {video_path}")
    print(f"Duration:        {fmt_time(duration)}")
    print(f"Native FPS:      {native_fps:.3f}")
    print(f"Gap:             {settings['min_gap']:g} - {settings['max_gap']:g}s")
    print(f"Pass 1:          {settings['pass1_fps']:g} fps")
    print(f"Pass 1 workers:  {settings['pass1_workers']}")
    print(f"Pass 1 mode:     {settings['pass1_mode']}")
    if settings["pass1_mode"] == "ffmpeg":
        print(f"Pass 1 hash width:{settings['pass1_hash_width']}")
        print(f"FFmpeg threads:   {settings['ffmpeg_threads']} (0=auto)")
    elif settings["pass1_mode"] == "opencv_sequential":
        print("Pass 1 hashing:   full-resolution OpenCV pHash (compatibility mode)")
    print(f"Pass 2:          {settings['pass2_fps']:g} fps")
    print(f"Pass 2 workers:  {settings['pass2_workers']}")
    print(f"Refine window:   +/- {settings['refine_window']:g}s")
    print(f"Chunk overlap:   {settings['overlap']:.2f}s")
    print(f"Hash size:       {settings['hash_size']}")
    print("=" * 70)

    with Manager() as manager:
        progress_queue = manager.Queue()

        # ----------------------------------------------------
        # PASS 1
        # ----------------------------------------------------
        if settings["pass1_mode"] == "ffmpeg":
            candidates = run_pass1_ffmpeg(
                video_path,
                duration,
                native_fps,
                video_width,
                video_height,
                settings,
                progress_queue,
            )
        elif settings["pass1_mode"] == "opencv_sequential":
            candidates = opencv_sequential_pass1_scan(
                video_path,
                duration,
                native_fps,
                int(round(frame_count)),
                settings["pass1_fps"],
                settings["min_gap"],
                settings["max_gap"],
                settings["hash_size"],
                settings["top_candidates"],
                progress_queue,
                settings["progress_pct"],
            )
        else:
            log_event("PASS1", "Starting chunked first pass. Each chunk seeks once and then decodes sequentially.")
            p1_tasks, actual_p1_workers = make_pass1_tasks(
                video_path, duration, native_fps, settings, progress_queue
            )
            log_event("PASS1", f"Using {actual_p1_workers} worker process(es)")
            p1_results = run_parallel(
                p1_tasks, pass1_worker, actual_p1_workers,
                progress_queue, settings
            )
            candidates = merge_candidates(p1_results, settings["top_candidates"])
        if not candidates:
            log_event("PASS1", "No valid candidates found")
            return

        print("\nPASS 1 BEST CANDIDATES")
        for i, (d, t1, t2) in enumerate(candidates, 1):
            sim = similarity_from_distance(d, settings["hash_size"])
            print(f"  {i:2d}. {fmt_time(t1)} -> {fmt_time(t2)} | gap {t2-t1:.1f}s | similarity {sim:.2f}%")

        # ----------------------------------------------------
        # PASS 2
        # ----------------------------------------------------
        log_event("PASS2", "Starting high-resolution refinement.")
        count = len(candidates)
        p2_tasks = []
        for i, (_, t1, t2) in enumerate(candidates, 1):
            p2_tasks.append((
                video_path, i, count, t1, t2,
                settings["min_gap"], settings["max_gap"],
                settings["pass2_fps"], settings["refine_window"],
                native_fps, settings["hash_size"], progress_queue,
            ))

        p2_workers = min(settings["pass2_workers"], count)
        log_event("PASS2", f"Using {p2_workers} worker process(es)")
        p2_results = run_parallel(
            p2_tasks, pass2_worker, p2_workers,
            progress_queue, settings
        )

    final = [r for r in p2_results if r is not None]
    final = dedup_results(final)
    final.sort(key=lambda x: x[0])

    if not final:
        log_event("RESULT", "No valid Pass 2 results")
        return

    outdir = os.path.join(os.path.dirname(video_path), settings["output_dir"])
    os.makedirs(outdir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not reopen video to save result frames")

    print("\n" + "=" * 70)
    print(" FINAL RESULTS")
    print("=" * 70)

    for idx, (distance, t1, t2) in enumerate(final[:settings["save_top"]], 1):
        gap = t2 - t1
        sim = similarity_from_distance(distance, settings["hash_size"])
        print()
        print(f"Result {idx}")
        print(f"Most similar frames:   {fmt_time(t1)}    {fmt_time(t2)}")
        print(f"Difference:            {gap:.1f} seconds")
        print(f"Similarity:            {sim:.1f}%")

        f1 = os.path.join(outdir, f"result_{idx}_frame1_{safe_time(t1)}.jpg")
        f2 = os.path.join(outdir, f"result_{idx}_frame2_{safe_time(t2)}.jpg")
        if save_frame(cap, t1, f1) and save_frame(cap, t2, f2):
            print(f"Saved:                 {f1}")
            print(f"                        {f2}")
        else:
            print("WARNING: Could not save one or both result frames")

    cap.release()
    print()
    log_event("DONE", "Finished")


if __name__ == "__main__":
    freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
        sys.exit(130)
    except Exception as exc:
        print("\nERROR:", exc, file=sys.stderr, flush=True)
        sys.exit(1)
