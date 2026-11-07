#!/usr/bin/env python3
"""Deduplicate WAV recordings with BEATs and save reports for audio review.

Example (from the repository root):
    .venv/bin/python -u deduplicate.py --input-dir /mnt/greek_music_dataset/raw_wavs \
        --device cpu --cpu-threads 8

The 0.95 threshold applies to (mean aligned cosine similarity + 1) / 2,
not a calibrated probability. No source audio is changed or deleted.
"""

import argparse
from bisect import bisect_right
import csv
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import faiss
import librosa
import numpy as np
import soundfile as sf
import torch
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from generation.generate_fingerprints import FileDataset, batch_waveforms_and_extract_fbanks
from models.beats.beatswrapper import BEATsWrapper


PROJECT_ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("deduplication")
SAMPLE_RATE = 8_000
SEGMENT_SAMPLES = SAMPLE_RATE
HOP_SAMPLES = SAMPLE_RATE // 2
QUERY_SECONDS = 5.0
QUERY_SAMPLES = 5 * SAMPLE_RATE
EMBEDDING_DIM = 128
N_QUERY_FINGERPRINTS = 9


def discover_wavs(input_dirs):
    if isinstance(input_dirs, (str, os.PathLike)):
        input_dirs = [input_dirs]
    found = set()
    for folder in input_dirs:
        folder = Path(folder).expanduser()
        if not folder.is_absolute():
            folder = PROJECT_ROOT / folder
        if not folder.is_dir():
            raise NotADirectoryError(folder)
        found.update(p.resolve() for p in folder.rglob("*")
                     if p.suffix.lower() == ".wav" and p.is_file())
    return sorted(found, key=lambda p: str(p).casefold())


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    candidates = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                free_bytes, _ = torch.cuda.mem_get_info(i)
                candidates.append((free_bytes, i))
            except RuntimeError:
                continue
    if candidates:
        free_bytes, i = max(candidates)
        if free_bytes >= 2 * 1024**3:
            return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def sample_query_segments(audio, rng):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono audio.")
    if len(audio) < QUERY_SAMPLES:
        raise ValueError("A query requires at least 5 seconds of audio.")
    last_start_frame = (len(audio) - QUERY_SAMPLES) // HOP_SAMPLES
    low, high = math.ceil(last_start_frame * 0.4), math.floor(last_start_frame * 0.6)
    if low > high:
        low = high = last_start_frame // 2
    start = int(rng.integers(low, high + 1)) * HOP_SAMPLES
    query = audio[start:start + QUERY_SAMPLES]
    segments = [query[i:i + SEGMENT_SAMPLES]
                for i in range(0, QUERY_SAMPLES - SEGMENT_SAMPLES + 1, HOP_SAMPLES)]
    return segments, start / SAMPLE_RATE


def normalize_fingerprints(vectors):
    vectors = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM or not len(vectors):
        raise ValueError(f"Expected nonempty (N, {EMBEDDING_DIM}) fingerprints.")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) < 1e-8):
        raise ValueError("Fingerprints must be finite and nonzero.")
    faiss.normalize_L2(vectors)
    return vectors


def cosine_to_score(cosine):
    return float((np.clip(cosine, -1.0, 1.0) + 1.0) / 2.0)


def find_best_match(query_fingerprints, index, indexed_songs, top_k=5):
    """Retrieve candidate offsets, then score nine aligned frames within one song."""
    if index.ntotal == 0:
        return None
    query = normalize_fingerprints(query_fingerprints)
    n_query = len(query)
    starts = [song["start_id"] for song in indexed_songs]
    _, neighbors = index.search(query, min(top_k, index.ntotal))
    candidates = set()
    for query_position, row in enumerate(neighbors):
        for neighbor_id in row:
            sequence_start = int(neighbor_id) - query_position
            owner = bisect_right(starts, sequence_start) - 1
            if neighbor_id < 0 or owner < 0:
                continue
            if sequence_start + n_query <= indexed_songs[owner]["stop_id"]:
                candidates.add((owner, sequence_start))
    best = None
    for owner, sequence_start in sorted(candidates):
        sequence = index.reconstruct_n(sequence_start, n_query)
        cosines = np.clip(np.sum(query * sequence, axis=1), -1.0, 1.0)
        mean_cosine = float(cosines.mean())
        score = cosine_to_score(mean_cosine)
        if best is None or score > best["match_score"]:
            song = indexed_songs[owner]
            best = {
                "matched_file": song["path"], "match_score": score,
                "mean_cosine_similarity": mean_cosine,
                "matched_start_seconds": (sequence_start - song["start_id"]) * 0.5,
                "segment_cosines": cosines.tolist(),
            }
    return best


class Deduplicator:
    def __init__(self, weights, batch_size=8, device="auto"):
        self.batch_size = batch_size
        self.device = choose_device(device)
        LOGGER.info("Selected device: %s (requested: %s)", self.device, device)
        self.model = BEATsWrapper(div_encoder_layer=True)
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        del state_dict
        try:
            self.model.to(self.device)
        except RuntimeError as exc:
            if device != "auto" or "out of memory" not in str(exc).lower():
                raise
            self.model.cpu()
            torch.cuda.empty_cache()
            self.device = torch.device("cpu")
            LOGGER.warning("CUDA memory unavailable; using CPU.")
        self.model.eval()
        if self.device.type == "cuda":
            LOGGER.info("Running on GPU: %s (%s)", self.device, torch.cuda.get_device_name(self.device))
        elif self.device.type == "cpu":
            LOGGER.info("Running on CPU (%d compute threads)", torch.get_num_threads())
        else:
            LOGGER.info("Running on device: %s", self.device)
        LOGGER.info("Loaded checkpoint: %s", weights)

    @torch.inference_mode()
    def extract_fingerprints(self, segments):
        loader = DataLoader(
            segments, batch_size=self.batch_size, shuffle=False, drop_last=False,
            num_workers=0, collate_fn=batch_waveforms_and_extract_fbanks,
        )
        batches = [self.model(features.to(self.device)).cpu().numpy() for features in loader]
        if not batches:
            raise ValueError("No complete 1-second segments were available.")
        return normalize_fingerprints(np.concatenate(batches, axis=0))

    def run(self, paths, threshold=0.95, top_k=5, seed=42, on_progress=None):
        paths = list(dict.fromkeys(str(Path(p).resolve()) for p in paths))
        rng = np.random.default_rng(seed)
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        result = {
            "index": index, "indexed_songs": [], "unique_songs": [], "duplicates": [],
            "skipped_files": [], "errors": [], "audit": [], "processed_files": 0,
            "total_files": len(paths), "elapsed_seconds": 0.0, "status": "running",
        }
        started = time.perf_counter()
        try:
            for number, file_path in enumerate(paths, 1):
                LOGGER.info("[%d/%d] Processing %s", number, len(paths), file_path)
                try:
                    dataset = FileDataset(file=file_path, sr=SAMPLE_RATE, hop_size=HOP_SAMPLES)
                    duration = len(dataset.y) / SAMPLE_RATE
                    reason = None
                    if duration < QUERY_SECONDS:
                        reason = "shorter than 5 seconds"
                    else:
                        segments, query_start = sample_query_segments(dataset.y, rng)
                        if max(float(np.max(np.abs(s))) for s in segments) < 1e-6:
                            reason = "silent query window"
                    if reason:
                        result["skipped_files"].append({"path": file_path, "reason": reason,
                                                        "duration_seconds": duration})
                        LOGGER.info("SKIPPED: %s (%s)", file_path, reason)
                    else:
                        query = self.extract_fingerprints(segments)
                        if query.shape != (N_QUERY_FINGERPRINTS, EMBEDDING_DIM):
                            raise ValueError(f"Unexpected query shape: {query.shape}")
                        match = find_best_match(query, index, result["indexed_songs"], top_k)
                        duplicate = match is not None and match["match_score"] >= threshold
                        if duplicate:
                            result["duplicates"].append(
                                {"duplicate": file_path, "query_start_seconds": query_start, **match})
                            LOGGER.info("DUPLICATE: %s -> %s (score=%.4f)",
                                        file_path, match["matched_file"], match["match_score"])
                        else:
                            fingerprints = self.extract_fingerprints(dataset)
                            start_id = int(index.ntotal)
                            index.add(fingerprints)
                            result["indexed_songs"].append({
                                "path": file_path, "start_id": start_id,
                                "stop_id": int(index.ntotal), "duration_seconds": duration,
                            })
                            result["unique_songs"].append(file_path)
                            LOGGER.info("UNIQUE: %s (%d fingerprints)", file_path, len(fingerprints))
                        result["audit"].append({
                            "path": file_path, "status": "duplicate" if duplicate else "unique",
                            "duration_seconds": duration, "query_start_seconds": query_start,
                            "query_fingerprints": len(query),
                            "best_candidate": match["matched_file"] if match else None,
                            "match_score": match["match_score"] if match else None,
                            "mean_cosine_similarity": match["mean_cosine_similarity"] if match else None,
                        })
                except Exception as exc:
                    if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                        raise
                    result["errors"].append({"path": file_path, "error": f"{type(exc).__name__}: {exc}"})
                    LOGGER.error("ERROR: %s: %s: %s", file_path, type(exc).__name__, exc)
                result["processed_files"] = number
                result["elapsed_seconds"] = time.perf_counter() - started
                if on_progress:
                    on_progress(result)
        except BaseException:
            result["status"] = "interrupted"
            result["elapsed_seconds"] = time.perf_counter() - started
            if on_progress:
                on_progress(result)
            raise
        result["status"] = "complete"
        return result


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_results(result, output_dir, settings, save_index=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in result.items() if key not in {"index", "indexed_songs"}}
    report.update(settings=settings, indexed_fingerprints=int(result["index"].ntotal),
                  updated_at=datetime.now(timezone.utc).isoformat())
    atomic_json(output_dir / "results.json", report)
    write_csv(output_dir / "unique_songs.csv", ({"path": p} for p in result["unique_songs"]), ["path"])
    write_csv(output_dir / "duplicates.csv", result["duplicates"],
              ["duplicate", "matched_file", "match_score", "mean_cosine_similarity",
               "query_start_seconds", "matched_start_seconds"])
    write_csv(output_dir / "skipped_files.csv", result["skipped_files"], ["path", "reason", "duration_seconds"])
    write_csv(output_dir / "errors.csv", result["errors"], ["path", "error"])
    write_csv(output_dir / "audit.csv", result["audit"],
              ["path", "status", "duration_seconds", "query_start_seconds", "query_fingerprints",
               "best_candidate", "match_score", "mean_cosine_similarity"])
    if save_index:
        atomic_json(output_dir / "indexed_songs.json", result["indexed_songs"])
        faiss.write_index(result["index"], str(output_dir / "deduplication.index"))


def check_search_logic():
    vectors = np.eye(EMBEDDING_DIM, dtype=np.float32)[:24]
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    assert find_best_match(vectors[:9], index, []) is None
    index.add(vectors)
    songs = [{"path": "first.wav", "start_id": 0, "stop_id": 12},
             {"path": "second.wav", "start_id": 12, "stop_id": 24}]
    match = find_best_match(vectors[2:11], index, songs)
    assert match["matched_file"] == "first.wav" and match["match_score"] > 0.9999
    assert match["matched_start_seconds"] == 1.0
    unrelated = np.eye(EMBEDDING_DIM, dtype=np.float32)[40:49]
    assert find_best_match(unrelated, index, songs)["match_score"] < 0.95
    crossing = find_best_match(vectors[7:16], index, songs)
    assert crossing is None or crossing["match_score"] < 0.95
    for seconds in (5.0, 5.1, 5.5, 6.0, 30.0):
        segments, offset = sample_query_segments(np.ones(round(seconds * SAMPLE_RATE)), np.random.default_rng(42))
        assert len(segments) == 9 and 0 <= offset <= seconds - 5 + 1e-6
    LOGGER.info("Window and search checks passed.")


def smoke_test(deduplicator, paths, threshold):
    excerpt = None
    for path in paths:
        try:
            duration = sf.info(str(path)).duration
            if duration < QUERY_SECONDS:
                continue
            audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, offset=max(0, duration / 2 - 3), duration=6)
            if len(audio) >= QUERY_SAMPLES and np.max(np.abs(audio)) >= 1e-6:
                excerpt = audio
                break
        except (OSError, RuntimeError, ValueError):
            continue
    if excerpt is None:
        raise ValueError("Smoke test needs a readable, non-silent WAV of at least 5 seconds.")
    with tempfile.TemporaryDirectory(prefix="audio-dedup-smoke-") as temporary:
        folder = Path(temporary)
        original, copied = folder / "01_original.wav", folder / "02_copy.WAV"
        sf.write(original, excerpt, SAMPLE_RATE, subtype="FLOAT")
        shutil.copyfile(original, copied)
        sf.write(folder / "03_short.wav", excerpt[:2 * SAMPLE_RATE], SAMPLE_RATE, subtype="FLOAT")
        (folder / "04_broken.wav").write_bytes(b"Not WAV audio")
        result = deduplicator.run(discover_wavs(folder), threshold=threshold)
        assert result["unique_songs"] == [str(original)]
        assert len(result["duplicates"]) == 1
        assert result["duplicates"][0]["matched_file"] == str(original)
        assert result["duplicates"][0]["match_score"] > 0.999
        assert len(result["skipped_files"]) == 1 and len(result["errors"]) == 1
    LOGGER.info("Real-checkpoint smoke test passed.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", action="append", type=Path,
                        help="WAV directory; repeat for multiple directories. Default: /mnt/greek_music_dataset/raw_wavs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/deduplication")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "pretrained_models/transformer_phase1.pt")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or cuda:N; use cpu to force CPU execution")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.95, help="Normalized similarity threshold on [0, 1]")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, help="Limit the run for a quick trial; default processes all WAVs")
    parser.add_argument("--save-every", type=int, default=25, help="Save report snapshots every N processed files")
    parser.add_argument("--smoke-test", action="store_true", help="Verify a known duplicate before the main run")
    args = parser.parse_args(argv)
    args.input_dir = args.input_dir or [Path("/mnt/greek_music_dataset/raw_wavs")]
    for name in ("cpu_threads", "batch_size", "top_k", "save_every", "max_files"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    LOGGER.info("Deduplication PID: %d", os.getpid())
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    faiss.omp_set_num_threads(args.cpu_threads)
    with threadpool_limits(limits=args.cpu_threads):
        LOGGER.info("Python: %s; CPU threads: %d; CPU affinity: %s", sys.executable,
                    torch.get_num_threads(), sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else "unavailable")
        if not args.weights.is_file():
            raise FileNotFoundError(args.weights)
        paths = discover_wavs(args.input_dir)
        if args.max_files is not None:
            paths = paths[:args.max_files]
        if not paths:
            raise FileNotFoundError("No WAV files found in the input directories.")
        LOGGER.info("Found %d WAV files; threshold %.4f; saving to %s", len(paths), args.threshold, args.output_dir)
        check_search_logic()
        deduplicator = Deduplicator(args.weights, args.batch_size, args.device)
        if args.smoke_test:
            smoke_test(deduplicator, paths, args.threshold)
        settings = {
            "weights": str(args.weights.resolve()), "input_dirs": [str(p.resolve()) for p in args.input_dir],
            "sample_rate": SAMPLE_RATE, "segment_seconds": 1.0, "hop_seconds": 0.5,
            "query_seconds": QUERY_SECONDS, "duplicate_threshold": args.threshold,
            "top_k": args.top_k, "seed": args.seed, "batch_size": args.batch_size,
            "cpu_threads": args.cpu_threads, "device": str(deduplicator.device),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "score_definition": "(mean_aligned_cosine_similarity + 1) / 2; not a calibrated probability",
        }

        def save_progress(result):
            if result["processed_files"] % args.save_every == 0 or result["status"] == "interrupted":
                save_results(result, args.output_dir, settings, save_index=result["status"] == "interrupted")
                LOGGER.info("Saved report snapshot after %d/%d files", result["processed_files"], len(paths))

        result = deduplicator.run(paths, args.threshold, args.top_k, args.seed, save_progress)
        save_results(result, args.output_dir, settings, save_index=True)
        LOGGER.info("COMPLETE: unique=%d duplicates=%d skipped=%d errors=%d fingerprints=%d elapsed=%.1fs",
                    len(result["unique_songs"]), len(result["duplicates"]), len(result["skipped_files"]),
                    len(result["errors"]), result["index"].ntotal, result["elapsed_seconds"])
        LOGGER.info("Results: %s", args.output_dir / "results.json")
        if len(result["errors"]) == len(paths):
            raise RuntimeError("Every input file failed; see errors.csv and the log.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted; partial results were saved.")
        raise SystemExit(130)
    except Exception:
        LOGGER.exception("Deduplication failed.")
        raise SystemExit(1)
