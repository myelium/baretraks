"""FastAPI web server for the karaoke pipeline."""

import warnings
warnings.filterwarnings("ignore", message=".*torchaudio.*torchcodec.*")

import json
import re
import secrets
import subprocess
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from karaoke.compose import compose
from karaoke.download import download, fetch_metadata
from karaoke.separate import separate
from karaoke.subtitles import build_ass
from karaoke.transcribe import transcribe

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
JOBS_DIR = Path("output/jobs")
STATS_PATH = JOBS_DIR / "stats.json"

# Default multipliers (used when no history exists)
DEFAULT_MULTIPLIERS = {
    1: 0.1,   # download: ~0.1x audio duration (network-bound)
    2: 2.5,   # demucs separation
    3: 2.0,   # whisper transcription
    4: 0.0,   # subtitles: instant
    5: 3.0,   # ffmpeg compose
}

app = FastAPI()

# ---------------------------------------------------------------------------
# Global job state & learned stats
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_job: dict | None = None
# Learned multipliers: step -> ratio of step_duration / audio_duration
_learned_multipliers: dict[int, float] = {}


def _load_stats() -> None:
    """Load historical stats and compute learned multipliers."""
    global _learned_multipliers
    if not STATS_PATH.exists():
        return
    try:
        stats = json.loads(STATS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return
    history = stats.get("history", [])
    if not history:
        return
    # Compute average multiplier per step
    step_ratios: dict[int, list[float]] = {}
    for entry in history:
        dur = entry.get("audio_duration", 0)
        if dur <= 0:
            continue
        for step_str, elapsed in entry.get("step_durations", {}).items():
            step = int(step_str)
            if step == 4:
                continue  # subtitles are instant, skip
            if elapsed > 0:
                step_ratios.setdefault(step, []).append(elapsed / dur)
    _learned_multipliers = {
        step: sum(ratios) / len(ratios)
        for step, ratios in step_ratios.items()
        if ratios
    }


def _save_stats(audio_duration: float, step_durations: dict) -> None:
    """Append this job's timing data to the stats file."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    stats = {"history": []}
    if STATS_PATH.exists():
        try:
            stats = json.loads(STATS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    stats["history"].append({
        "audio_duration": audio_duration,
        "step_durations": step_durations,
        "timestamp": _now_iso(),
    })
    # Keep last 50 entries to prevent unbounded growth
    stats["history"] = stats["history"][-50:]
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    # Reload multipliers with new data
    _load_stats()

STEP_NAMES = {
    1: "Downloading",
    2: "Separating vocals",
    3: "Transcribing lyrics",
    4: "Building subtitles",
    5: "Composing video",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_audio_duration(audio_path: Path) -> float:
    """Probe audio duration in seconds using ffprobe."""
    # Use ffprobe from PATH (installed via homebrew / system package)
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _convert_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    """Convert a WAV file to MP3."""
    subprocess.run(
        [FFMPEG, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame",
         "-b:a", "192k", str(mp3_path)],
        check=True,
        capture_output=True,
    )


def _compute_estimates(audio_duration: float) -> list[float]:
    """Return per-step estimated durations in seconds (indices 0-4 for steps 1-5)."""
    def _mult(step: int) -> float:
        return _learned_multipliers.get(step, DEFAULT_MULTIPLIERS[step])
    return [
        audio_duration * _mult(1),   # step 1: download
        audio_duration * _mult(2),   # step 2: demucs
        audio_duration * _mult(3),   # step 3: whisper
        1.0,                         # step 4: subtitles (always instant)
        audio_duration * _mult(5),   # step 5: compose
    ]


def _update_job(**kwargs) -> None:
    with _lock:
        if _job is not None:
            _job.update(kwargs)


def _save_job_json() -> None:
    """Persist job metadata to disk for crash recovery."""
    with _lock:
        if _job is None:
            return
        snapshot = dict(_job)
    job_json = Path(snapshot["output_dir"]) / "job.json"
    job_json.write_text(json.dumps(snapshot, default=str))


def _detect_resume_step(output_dir: Path) -> int:
    """Detect which step to resume from based on existing files on disk."""
    work_dir = output_dir / "work"
    if (output_dir / "karaoke.mp4").exists():
        return 6  # all done
    subtitles_path = work_dir / "karaoke.ass"
    instrumental = output_dir / "instrumental.mp3"
    vocals = output_dir / "vocals.mp3"
    if subtitles_path.exists() and instrumental.exists():
        return 5  # resume from compose
    if (output_dir / "lyrics.json").exists():
        return 4  # resume from subtitles
    if instrumental.exists() and vocals.exists():
        return 3  # resume from transcribe
    if (work_dir / "video.mp4").exists() and (work_dir / "audio.wav").exists():
        return 2  # resume from separate
    return 1  # start from beginning


def _run_pipeline(job_id: str, url: str, output_dir: Path,
                  resume_from: int = 1) -> None:
    """Run the full karaoke pipeline in a background thread."""
    try:
        device = "cpu"  # MPS lacks sparse tensor support needed by Whisper
        work_dir = output_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        video_path = work_dir / "video.mp4"
        audio_path = work_dir / "audio.wav"
        instrumental_mp3 = output_dir / "instrumental.mp3"
        vocals_mp3 = output_dir / "vocals.mp3"
        subtitles_path = work_dir / "karaoke.ass"
        output_path = output_dir / "karaoke.mp4"

        step_durations: dict[str, float] = {}  # step -> actual seconds

        # --- Step 1: Download ---
        if resume_from <= 1:
            t0 = time.monotonic()
            _update_job(step=1, step_name=STEP_NAMES[1], step_progress=0.0,
                         step_started_at=_now_iso())
            _save_job_json()

            def _dl_progress(pct: float) -> None:
                _update_job(step_progress=pct)

            video_path, audio_path = download(url, work_dir, progress_callback=_dl_progress)
            step_durations["1"] = time.monotonic() - t0
            _update_job(step_progress=1.0, artifacts={
                1: [{"name": "video.mp4", "path": "work/video.mp4"}],
            })
        else:
            _update_job(step=1, step_name=STEP_NAMES[1], step_progress=1.0,
                         artifacts={
                             1: [{"name": "video.mp4", "path": "work/video.mp4"}],
                         })

        # Compute time estimates from audio duration
        audio_duration = _get_audio_duration(audio_path)
        estimates = _compute_estimates(audio_duration)
        if resume_from <= 1:
            # Replace estimate with actual for step 1
            estimates[0] = step_durations.get("1", estimates[0])
        else:
            estimates[0] = 0
        for i in range(1, resume_from - 1):
            estimates[i] = 0
        _update_job(
            audio_duration=audio_duration,
            step_estimates=estimates,
            estimated_total=sum(estimates),
        )

        # --- Step 2: Separate ---
        if resume_from <= 2:
            t0 = time.monotonic()
            _update_job(step=2, step_name=STEP_NAMES[2], step_progress=0.0,
                         step_started_at=_now_iso())
            _save_job_json()
            instrumental_wav, vocals_wav = separate(audio_path, work_dir / "demucs")
            _convert_to_mp3(instrumental_wav, instrumental_mp3)
            _convert_to_mp3(vocals_wav, vocals_mp3)
            instrumental_wav.unlink(missing_ok=True)
            vocals_wav.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
            step_durations["2"] = time.monotonic() - t0
            with _lock:
                artifacts = dict(_job.get("artifacts", {}))
            artifacts[2] = [
                {"name": "instrumental.mp3", "path": "instrumental.mp3"},
                {"name": "vocals.mp3", "path": "vocals.mp3"},
            ]
            _update_job(step_progress=1.0, artifacts=artifacts)
        else:
            with _lock:
                artifacts = dict(_job.get("artifacts", {}))
            artifacts[2] = [
                {"name": "instrumental.mp3", "path": "instrumental.mp3"},
                {"name": "vocals.mp3", "path": "vocals.mp3"},
            ]
            _update_job(step=2, step_name=STEP_NAMES[2], step_progress=1.0,
                         artifacts=artifacts)

        # --- Step 3: Transcribe ---
        segments = None
        if resume_from <= 3:
            t0 = time.monotonic()
            _update_job(step=3, step_name=STEP_NAMES[3], step_progress=0.0,
                         step_started_at=_now_iso())
            _save_job_json()
            segments = transcribe(vocals_mp3, device=device)
            words_list = []
            for seg in segments:
                for w in seg.words:
                    words_list.append({"text": w.text, "start": w.start, "end": w.end})
            (output_dir / "lyrics.json").write_text(json.dumps(words_list))
            step_durations["3"] = time.monotonic() - t0
            with _lock:
                artifacts = dict(_job.get("artifacts", {}))
            artifacts[3] = [{"name": "lyrics.json", "path": "lyrics.json"}]
            _update_job(step_progress=1.0, artifacts=artifacts)
        else:
            with _lock:
                artifacts = dict(_job.get("artifacts", {}))
            artifacts[3] = [{"name": "lyrics.json", "path": "lyrics.json"}]
            _update_job(step=3, step_name=STEP_NAMES[3], step_progress=1.0,
                         artifacts=artifacts)

        # --- Step 4: Subtitles ---
        if resume_from <= 4:
            t0 = time.monotonic()
            _update_job(step=4, step_name=STEP_NAMES[4], step_progress=0.0,
                         step_started_at=_now_iso())
            if segments is None:
                from karaoke.transcribe import Segment, Word as TWord
                lyrics = json.loads((output_dir / "lyrics.json").read_text())
                words = [TWord(text=w["text"], start=w["start"], end=w["end"]) for w in lyrics]
                if words:
                    segments = [Segment(start=words[0].start, end=words[-1].end, words=words)]
                else:
                    segments = []
            build_ass(segments, subtitles_path)
            step_durations["4"] = time.monotonic() - t0
            with _lock:
                artifacts = dict(_job.get("artifacts", {}))
            artifacts[4] = []
            _update_job(step_progress=1.0, artifacts=artifacts)
        else:
            with _lock:
                artifacts = dict(_job.get("artifacts", {}))
            artifacts[4] = []
            _update_job(step=4, step_name=STEP_NAMES[4], step_progress=1.0,
                         artifacts=artifacts)

        # --- Step 5: Compose ---
        if resume_from <= 5:
            t0 = time.monotonic()
            _update_job(step=5, step_name=STEP_NAMES[5], step_progress=0.0,
                         step_started_at=_now_iso())
            _save_job_json()
            if output_path.exists():
                output_path.unlink()

            def _compose_progress(pct: float) -> None:
                _update_job(step_progress=pct)

            compose(video_path, instrumental_mp3, subtitles_path, output_path,
                    duration=audio_duration, progress_callback=_compose_progress)
            step_durations["5"] = time.monotonic() - t0
            _update_job(step_progress=1.0)

        with _lock:
            artifacts = dict(_job.get("artifacts", {}))
        artifacts[5] = [{"name": "karaoke.mp4", "path": "karaoke.mp4"}]
        _update_job(artifacts=artifacts)

        # Save timing stats for future estimation
        if audio_duration > 0 and step_durations:
            _save_stats(audio_duration, step_durations)

        _update_job(status="done", finished_at=_now_iso(),
                    step_durations=step_durations)
        _save_job_json()

    except Exception as e:
        _update_job(status="failed", error=str(e), finished_at=_now_iso())
        _save_job_json()


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _recover_jobs() -> None:
    global _job
    # Load historical timing data for better estimates
    _load_stats()
    if not JOBS_DIR.exists():
        return
    # Find the most recent job directory
    job_dirs = sorted(JOBS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in job_dirs:
        job_json = d / "job.json"
        if not job_json.exists():
            continue
        try:
            data = json.loads(job_json.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # If it was running when server died, mark as failed
        if data.get("status") == "running":
            data["status"] = "failed"
            data["error"] = "Server restarted during processing"
            data["finished_at"] = _now_iso()
            job_json.write_text(json.dumps(data, default=str))
        _job = data
        break  # only restore the most recent


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class JobRequest(BaseModel):
    url: str


def _slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a filesystem-safe slug."""
    # Normalize unicode and convert to ASCII
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text).strip("-")
    return text[:max_len] or "untitled"


@app.post("/api/jobs")
def create_job(req: JobRequest):
    global _job
    with _lock:
        if _job is not None and _job["status"] == "running":
            raise HTTPException(409, "A job is already running")

    # Fetch metadata first so we can use the title as folder name
    meta = {}
    try:
        meta = fetch_metadata(req.url)
    except Exception:
        pass
    title = meta.get("title", "Unknown")
    thumbnail = meta.get("thumbnail")

    slug = _slugify(title)
    job_id = f"{slug}-{secrets.token_hex(2)}"
    output_dir = JOBS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _job = {
        "id": job_id,
        "url": req.url,
        "title": title,
        "thumbnail": thumbnail,
        "channel": meta.get("channel"),
        "upload_date": meta.get("upload_date"),
        "categories": meta.get("categories", []),
        "tags": meta.get("tags", []),
        "status": "running",
        "step": 0,
        "step_name": "Starting",
        "step_progress": 0.0,
        "started_at": _now_iso(),
        "step_started_at": _now_iso(),
        "finished_at": None,
        "audio_duration": None,
        "estimated_total": None,
        "step_estimates": None,
        "error": None,
        "artifacts": {},
        "output_dir": str(output_dir),
    }
    _save_job_json()

    thread = threading.Thread(target=_run_pipeline, args=(job_id, req.url, output_dir),
                              daemon=True)
    thread.start()

    return {"job": _job}


@app.get("/api/jobs/current")
def get_current_job():
    with _lock:
        return {"job": dict(_job) if _job else None}


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    global _job
    with _lock:
        if _job is not None and _job["status"] == "running":
            raise HTTPException(409, "A job is already running")

    output_dir = JOBS_DIR / job_id
    job_json = output_dir / "job.json"
    if not job_json.exists():
        raise HTTPException(404, "Job not found")

    try:
        data = json.loads(job_json.read_text())
    except (json.JSONDecodeError, OSError):
        raise HTTPException(500, "Could not read job data")

    if data.get("status") == "done":
        raise HTTPException(400, "Job already completed")

    resume_step = _detect_resume_step(output_dir)
    if resume_step > 5:
        # All pipeline steps done — just re-run post-processing
        resume_step = 5

    _job = {
        "id": job_id,
        "url": data.get("url", ""),
        "title": data.get("title", "Unknown"),
        "thumbnail": data.get("thumbnail"),
        "status": "running",
        "step": 0,
        "step_name": f"Resuming from step {resume_step}",
        "step_progress": 0.0,
        "started_at": _now_iso(),
        "step_started_at": _now_iso(),
        "finished_at": None,
        "audio_duration": data.get("audio_duration"),
        "estimated_total": None,
        "step_estimates": None,
        "error": None,
        "artifacts": data.get("artifacts", {}),
        "output_dir": str(output_dir),
    }
    _save_job_json()

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, data.get("url", ""), output_dir),
        kwargs={"resume_from": resume_step},
        daemon=True,
    )
    thread.start()

    return {"job": _job, "resume_from": resume_step}


@app.get("/api/library")
def get_library():
    """Return all completed jobs as a list, sorted newest first."""
    items = []
    if not JOBS_DIR.exists():
        return {"items": items}
    for d in JOBS_DIR.iterdir():
        if not d.is_dir():
            continue
        job_json = d / "job.json"
        if not job_json.exists():
            continue
        try:
            data = json.loads(job_json.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("status") != "done":
            continue
        items.append({
            "id": data.get("id", d.name),
            "title": data.get("title", "Unknown"),
            "url": data.get("url"),
            "thumbnail": data.get("thumbnail"),
            "channel": data.get("channel"),
            "upload_date": data.get("upload_date"),
            "categories": data.get("categories", []),
            "tags": data.get("tags", []),
            "finished_at": data.get("finished_at"),
            "audio_duration": data.get("audio_duration"),
        })
    items.sort(key=lambda x: x.get("finished_at") or "", reverse=True)
    return {"items": items}


@app.get("/api/jobs/{job_id}/video")
def stream_video(job_id: str):
    path = JOBS_DIR / job_id / "karaoke.mp4"
    if not path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/instrumental")
def stream_instrumental(job_id: str):
    path = JOBS_DIR / job_id / "instrumental.mp3"
    if not path.exists():
        raise HTTPException(404, "Instrumental not found")
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/jobs/{job_id}/vocals")
def stream_vocals(job_id: str):
    path = JOBS_DIR / job_id / "vocals.mp3"
    if not path.exists():
        raise HTTPException(404, "Vocals not found")
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/jobs/{job_id}/lyrics")
def get_lyrics(job_id: str):
    path = JOBS_DIR / job_id / "lyrics.json"
    if not path.exists():
        raise HTTPException(404, "Lyrics not found")
    return json.loads(path.read_text())


@app.get("/api/jobs/{job_id}/download/{filename}")
def download_file(job_id: str, filename: str):
    # Only allow specific files
    allowed = {"karaoke.mp4", "instrumental.mp3", "vocals.mp3"}
    if filename not in allowed:
        raise HTTPException(400, f"Invalid file. Allowed: {', '.join(allowed)}")
    path = JOBS_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    media_types = {
        "karaoke.mp4": "video/mp4",
        "instrumental.mp3": "audio/mpeg",
        "vocals.mp3": "audio/mpeg",
    }
    return FileResponse(
        path,
        media_type=media_types[filename],
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/jobs/{job_id}/artifact/{filepath:path}")
def download_artifact(job_id: str, filepath: str):
    """Download an intermediate artifact from a job's output directory."""
    # Prevent path traversal
    if ".." in filepath:
        raise HTTPException(400, "Invalid path")
    path = JOBS_DIR / job_id / filepath
    if not path.exists() or not path.is_file():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(
        path,
        filename=path.name,
        headers={"Content-Disposition": f"attachment; filename={path.name}"},
    )


# ---------------------------------------------------------------------------
# Static files & index
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
