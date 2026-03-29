"""FastAPI web server for the karaoke pipeline."""

import warnings
warnings.filterwarnings("ignore", message=".*torchaudio.*torchcodec.*")

from dotenv import load_dotenv
load_dotenv()

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
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import (
    create_token, create_user_with_permissions, get_current_user,
    get_optional_user, hash_password, require_admin, verify_password,
)
from database import get_db
from models import Feedback, User, UserPermissions, Vote

import shutil

from karaoke.compose import compose
from karaoke.download import download, download_audio, fetch_metadata
from karaoke.separate import separate
from karaoke.subtitles import build_ass, build_srt
from karaoke.transcribe import transcribe
from karaoke.translate import translate_srt

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
_queue: list[dict] = []  # production queue [{id, url, mode, languages, title, thumbnail, channel, status}]
_pause_requested = False
_cancel_requested = False
QUEUE_PATH = JOBS_DIR / "queue.json"
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

def _save_queue() -> None:
    """Persist queue to disk."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.write_text(json.dumps(_queue, default=str))


def _load_queue() -> None:
    """Load queue from disk."""
    global _queue
    if QUEUE_PATH.exists():
        try:
            _queue = json.loads(QUEUE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            _queue = []


def _process_next_in_queue() -> None:
    """Start processing the next queued item if nothing is running."""
    global _job, _pause_requested
    with _lock:
        if _job is not None and _job.get("status") == "running":
            return  # something already running
        # Find the first "queued" item
        next_item = None
        for item in _queue:
            if item.get("status") == "queued":
                next_item = item
                break
        if not next_item:
            return
        next_item["status"] = "processing"
    _pause_requested = False
    _cancel_requested = False
    _save_queue()

    # Create the job from queue item
    url = next_item["url"]
    mode = next_item.get("mode", "karaoke")
    languages = next_item.get("languages", [])
    job_id = next_item["id"]
    output_dir = JOBS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _job = {
        "id": job_id,
        "url": url,
        "mode": mode,
        "languages": languages,
        "title": next_item.get("title", "Unknown"),
        "thumbnail": next_item.get("thumbnail"),
        "channel": next_item.get("channel"),
        "upload_date": next_item.get("upload_date"),
        "categories": next_item.get("categories", []),
        "tags": next_item.get("tags", []),
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

    if mode == "subtitled":
        thread = threading.Thread(
            target=_run_subtitled_pipeline,
            args=(job_id, url, output_dir, languages),
            daemon=True)
    else:
        thread = threading.Thread(
            target=_run_pipeline,
            args=(job_id, url, output_dir),
            daemon=True)
    thread.start()


def _on_job_finished() -> None:
    """Called when a job finishes — remove from queue and start next."""
    with _lock:
        job_id = _job["id"] if _job else None
        # Remove the finished item from queue
        _queue[:] = [item for item in _queue if item.get("id") != job_id]
    _save_queue()
    _process_next_in_queue()


STEP_NAMES = {
    1: "Downloading",
    2: "Separating vocals",
    3: "Transcribing lyrics",
    4: "Building subtitles",
    5: "Composing video",
}

STEP_NAMES_SUBTITLED = {
    1: "Downloading audio",
    2: "Transcribing",
    3: "Translating",
}

# Language code → full name for Claude translation prompts
LANG_FULL_NAMES = {
    "af": "Afrikaans", "sq": "Albanian", "am": "Amharic", "ar": "Arabic",
    "hy": "Armenian", "as": "Assamese", "az": "Azerbaijani", "ba": "Bashkir",
    "eu": "Basque", "be": "Belarusian", "bn": "Bengali", "bs": "Bosnian",
    "br": "Breton", "bg": "Bulgarian", "my": "Burmese", "yue": "Cantonese",
    "ca": "Catalan", "hr": "Croatian", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "en": "English", "et": "Estonian", "fo": "Faroese",
    "fi": "Finnish", "fr": "French", "gl": "Galician", "ka": "Georgian",
    "de": "German", "el": "Greek", "gu": "Gujarati", "ht": "Haitian",
    "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew", "hi": "Hindi",
    "hu": "Hungarian", "is": "Icelandic", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "jw": "Javanese", "kn": "Kannada", "kk": "Kazakh",
    "km": "Khmer", "ko": "Korean", "lo": "Lao", "la": "Latin",
    "lv": "Latvian", "ln": "Lingala", "lt": "Lithuanian", "lb": "Luxembourgish",
    "mk": "Macedonian", "mg": "Malagasy", "ms": "Malay", "ml": "Malayalam",
    "mt": "Maltese", "zh": "Mandarin Chinese", "mi": "Maori", "mr": "Marathi",
    "mn": "Mongolian", "ne": "Nepali", "no": "Norwegian", "nn": "Norwegian Nynorsk",
    "oc": "Occitan", "ps": "Pashto", "fa": "Persian", "pl": "Polish",
    "pt": "Portuguese", "pa": "Punjabi", "ro": "Romanian", "ru": "Russian",
    "sa": "Sanskrit", "sr": "Serbian", "sn": "Shona", "sd": "Sindhi",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "so": "Somali",
    "es": "Spanish", "su": "Sundanese", "sw": "Swahili", "sv": "Swedish",
    "tl": "Tagalog", "tg": "Tajik", "ta": "Tamil", "tt": "Tatar",
    "te": "Telugu", "th": "Thai", "bo": "Tibetan", "tr": "Turkish",
    "tk": "Turkmen", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "cy": "Welsh", "yi": "Yiddish", "yo": "Yoruba",
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


def _check_pause() -> bool:
    """Check if pause or cancel was requested. Returns True to stop the pipeline."""
    if _cancel_requested:
        _update_job(status="failed", error="Cancelled by user", finished_at=_now_iso())
        _save_job_json()
        # Remove from queue and clean up work dir
        with _lock:
            job_id = _job["id"] if _job else None
            _queue[:] = [i for i in _queue if i.get("id") != job_id]
        _save_queue()
        if _job:
            work_dir = Path(_job["output_dir"]) / "work"
            _cleanup_work_dir(work_dir)
            # Remove the entire job directory since it was cancelled
            job_dir = Path(_job["output_dir"])
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
        _process_next_in_queue()
        return True
    if _pause_requested:
        _update_job(status="paused", finished_at=_now_iso())
        _save_job_json()
        with _lock:
            job_id = _job["id"] if _job else None
            for item in _queue:
                if item.get("id") == job_id:
                    item["status"] = "paused"
                    break
        _save_queue()
        return True
    return False


def _cleanup_work_dir(work_dir: Path) -> None:
    """Remove the work directory to free disk space after a successful job."""
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


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
        if _check_pause(): return
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
        if _check_pause(): return
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
        if _check_pause(): return
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
        if _check_pause(): return
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

        # Clean up work directory to save disk space
        _cleanup_work_dir(work_dir)

        _update_job(status="done", finished_at=_now_iso(),
                    step_durations=step_durations)
        _save_job_json()
        _on_job_finished()

    except Exception as e:
        _update_job(status="failed", error=str(e), finished_at=_now_iso())
        _save_job_json()
        _on_job_finished()


def _run_subtitled_pipeline(job_id: str, url: str, output_dir: Path,
                            languages: list[str] | None = None) -> None:
    """Run the subtitled video pipeline (audio-only download + multi-language transcription)."""
    if not languages:
        languages = []
    try:
        device = "cpu"
        work_dir = output_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        audio_path = work_dir / "audio.wav"
        step_durations: dict[str, float] = {}
        num_langs = len(languages)

        # --- Step 1: Download audio only ---
        t0 = time.monotonic()
        _update_job(step=1, step_name=STEP_NAMES_SUBTITLED[1], step_progress=0.0,
                     step_started_at=_now_iso())
        _save_job_json()

        def _dl_progress(pct: float) -> None:
            _update_job(step_progress=pct)

        audio_path = download_audio(url, work_dir, progress_callback=_dl_progress)
        step_durations["1"] = time.monotonic() - t0

        audio_duration = _get_audio_duration(audio_path)
        # Rough estimate: transcription takes ~2x audio duration per language
        est_transcribe = audio_duration * 2.0 * max(num_langs, 1)
        _update_job(
            step_progress=1.0,
            audio_duration=audio_duration,
            step_estimates=[step_durations["1"], est_transcribe],
            estimated_total=step_durations["1"] + est_transcribe,
            artifacts={1: []},
        )

        # --- Step 2: Transcribe in source language (Whisper auto-detect) ---
        if _check_pause(): return
        t0 = time.monotonic()
        _update_job(step=2, step_name=STEP_NAMES_SUBTITLED[2], step_progress=0.0,
                     step_started_at=_now_iso())
        _save_job_json()

        # Whisper transcribes in the original language (what it does best)
        segments = transcribe(audio_path, device=device)

        # Save lyrics.json from the source-language transcription
        words_list = []
        for seg in segments:
            for w in seg.words:
                words_list.append({"text": w.text, "start": w.start, "end": w.end})
        (output_dir / "lyrics.json").write_text(json.dumps(words_list))

        # Generate source-language SRT (used as base for Claude translations)
        source_srt_path = output_dir / "subtitles_source.srt"
        build_srt(segments, source_srt_path)

        step_durations["2"] = time.monotonic() - t0
        srt_artifacts = []
        _update_job(step_progress=1.0, artifacts={
            1: [],
            2: [{"name": "lyrics.json", "path": "lyrics.json"}],
        })

        # --- Step 3: Translate to all target languages via Claude API ---
        if _check_pause(): return
        if languages:
            t0 = time.monotonic()
            source_srt_text = source_srt_path.read_text()

            for i, lang in enumerate(languages):
                target_name = LANG_FULL_NAMES.get(lang, lang)
                label = (f"Translating to {target_name}"
                         f" ({i+1} of {len(languages)})" if len(languages) > 1
                         else f"Translating to {target_name}")
                _update_job(step=3, step_name=label,
                             step_progress=i / len(languages),
                             step_started_at=_now_iso())
                _save_job_json()

                with _lock:
                    job_title = _job.get("title") if _job else None
                    job_channel = _job.get("channel") if _job else None
                translated_srt = translate_srt(source_srt_text, target_name,
                                               title=job_title, artist=job_channel)
                srt_name = f"subtitles_{lang}.srt"
                (output_dir / srt_name).write_text(translated_srt)
                srt_artifacts.append({"name": srt_name, "path": srt_name})

            step_durations["3"] = time.monotonic() - t0
            _update_job(step_progress=1.0, artifacts={
                1: [],
                2: srt_artifacts + [{"name": "lyrics.json", "path": "lyrics.json"}],
            })

        # Clean up intermediate source SRT (keep only the per-language ones)
        source_srt_path.unlink(missing_ok=True)

        # Clean up work directory
        _cleanup_work_dir(work_dir)

        _update_job(status="done", finished_at=_now_iso(),
                    step_durations=step_durations)
        _save_job_json()
        _on_job_finished()

    except Exception as e:
        _update_job(status="failed", error=str(e), finished_at=_now_iso())
        _save_job_json()
        _on_job_finished()


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _recover_jobs() -> None:
    global _job
    # Load historical timing data and queue
    _load_stats()
    _load_queue()
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

    # Reset any "processing" queue items to "queued" (server restarted mid-job)
    for item in _queue:
        if item.get("status") == "processing":
            item["status"] = "queued"
    _save_queue()

    # Auto-start queued items
    _process_next_in_queue()

    # Clean up work directories for all completed jobs
    for d in JOBS_DIR.iterdir():
        if not d.is_dir():
            continue
        work_dir = d / "work"
        if not work_dir.exists():
            continue
        job_json = d / "job.json"
        if job_json.exists():
            try:
                data = json.loads(job_json.read_text())
                if data.get("status") == "done":
                    _cleanup_work_dir(work_dir)
            except (json.JSONDecodeError, OSError):
                pass


# ---------------------------------------------------------------------------
# Auth API
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class SettingsRequest(BaseModel):
    theme: str | None = None
    dark_mode: str | None = None


def _set_auth_cookie(response: Response, user: User) -> None:
    token = create_token(str(user.id))
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=72 * 3600,
        path="/",
    )


@app.post("/api/auth/register")
def register(req: RegisterRequest, response: Response, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email.lower()).first():
        raise HTTPException(400, "Email already registered")
    user = create_user_with_permissions(db, email=req.email, name=req.name, password=req.password)
    _set_auth_cookie(response, user)
    return {"user": user.to_dict()}


@app.post("/api/auth/login")
def login(req: LoginRequest, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email.lower()).first()
    if not user or not user.password_hash or not verify_password(req.password, user.password_hash):
        raise HTTPException(401, "Invalid email or password")
    from datetime import datetime, timezone as tz
    user.last_login = datetime.now(tz.utc)
    db.commit()
    _set_auth_cookie(response, user)
    return {"user": user.to_dict()}


@app.get("/api/auth/google")
def google_login():
    """Redirect to Google OAuth. Requires GOOGLE_CLIENT_ID to be set."""
    import os
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    if not client_id:
        raise HTTPException(501, "Google OAuth not configured")
    redirect_uri = "/api/auth/google/callback"
    scope = "openid email profile"
    url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={client_id}&response_type=code&scope={scope}"
        f"&redirect_uri={redirect_uri}&access_type=offline"
    )
    return RedirectResponse(url)


@app.get("/api/auth/google/callback")
def google_callback(code: str, response: Response, db: Session = Depends(get_db)):
    """Handle Google OAuth callback."""
    import os
    import httpx
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise HTTPException(501, "Google OAuth not configured")

    # Exchange code for tokens
    token_resp = httpx.post("https://oauth2.googleapis.com/token", data={
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": "/api/auth/google/callback",
        "grant_type": "authorization_code",
    })
    if token_resp.status_code != 200:
        raise HTTPException(400, "Failed to exchange OAuth code")
    tokens = token_resp.json()

    # Get user info
    userinfo_resp = httpx.get("https://www.googleapis.com/oauth2/v2/userinfo",
                               headers={"Authorization": f"Bearer {tokens['access_token']}"})
    if userinfo_resp.status_code != 200:
        raise HTTPException(400, "Failed to get user info")
    info = userinfo_resp.json()

    # Find or create user
    user = db.query(User).filter(User.google_id == info["id"]).first()
    if not user:
        user = db.query(User).filter(User.email == info["email"].lower()).first()
        if user:
            user.google_id = info["id"]
            user.picture_url = info.get("picture")
        else:
            user = create_user_with_permissions(
                db, email=info["email"], name=info.get("name", ""),
                google_id=info["id"], picture_url=info.get("picture"),
            )
    from datetime import datetime, timezone as tz
    user.last_login = datetime.now(tz.utc)
    db.commit()

    resp = RedirectResponse("/")
    _set_auth_cookie(resp, user)
    return resp


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie("access_token", path="/")
    return {"logged_out": True}


@app.get("/api/auth/me")
def get_me(user: User = Depends(get_current_user)):
    data = user.to_dict()
    if user.permissions:
        data["permissions"] = user.permissions.to_dict()
    return {"user": data}


@app.put("/api/auth/settings")
def update_settings(req: SettingsRequest, user: User = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    if req.theme and req.theme in ("retro", "spotify", "disco"):
        user.theme = req.theme
    if req.dark_mode and req.dark_mode in ("dark", "day", "night"):
        user.dark_mode = req.dark_mode
    db.commit()
    return {"user": user.to_dict()}


# ---------------------------------------------------------------------------
# Production API
# ---------------------------------------------------------------------------
class JobRequest(BaseModel):
    url: str
    mode: str = "karaoke"          # "karaoke" or "subtitled"
    language: str | None = None     # DEPRECATED: single language (backward compat)
    languages: list[str] = []      # Whisper language codes (e.g. ["en", "es"]), max 3


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

    mode = req.mode if req.mode in ("karaoke", "subtitled") else "karaoke"
    # Support both old single language and new multi-language
    languages = req.languages[:3] if req.languages else (
        [req.language] if req.language else []
    )
    if mode != "subtitled":
        languages = []

    _job = {
        "id": job_id,
        "url": req.url,
        "mode": mode,
        "languages": languages,
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

    if mode == "subtitled":
        thread = threading.Thread(
            target=_run_subtitled_pipeline,
            args=(job_id, req.url, output_dir, languages),
            daemon=True)
    else:
        thread = threading.Thread(
            target=_run_pipeline,
            args=(job_id, req.url, output_dir),
            daemon=True)
    thread.start()

    return {"job": _job}


@app.get("/api/jobs/current")
def get_current_job():
    with _lock:
        return {"job": dict(_job) if _job else None}


# ---------------------------------------------------------------------------
# Queue API
# ---------------------------------------------------------------------------

class QueueRequest(BaseModel):
    url: str
    mode: str = "karaoke"
    languages: list[str] = []


@app.post("/api/queue")
def add_to_queue(req: QueueRequest, user: User | None = Depends(get_optional_user)):
    """Add an item to the production queue."""
    mode = req.mode if req.mode in ("karaoke", "subtitled") else "karaoke"
    languages = req.languages[:3] if req.languages else []
    if mode != "subtitled":
        languages = []

    # Fetch metadata
    meta = {}
    try:
        meta = fetch_metadata(req.url)
    except Exception:
        pass
    title = meta.get("title", "Unknown")

    slug = _slugify(title)
    item_id = f"{slug}-{secrets.token_hex(2)}"

    item = {
        "id": item_id,
        "url": req.url,
        "mode": mode,
        "languages": languages,
        "title": title,
        "thumbnail": meta.get("thumbnail"),
        "channel": meta.get("channel"),
        "upload_date": meta.get("upload_date"),
        "categories": meta.get("categories", []),
        "tags": meta.get("tags", []),
        "status": "queued",
        "added_by": user.name.split()[0] if user and user.name else "Anonymous",
    }

    with _lock:
        _queue.append(item)
    _save_queue()

    # Auto-start if nothing is running
    _process_next_in_queue()

    return {"item": item, "queue": _queue}


@app.get("/api/queue")
def get_queue():
    """Return the current queue and active job status."""
    with _lock:
        queue_with_progress = []
        for item in _queue:
            entry = dict(item)
            # If this item is currently processing, merge job progress
            if _job and _job.get("id") == item.get("id") and _job.get("status") == "running":
                entry["step"] = _job.get("step", 0)
                entry["step_name"] = _job.get("step_name", "")
                entry["step_progress"] = _job.get("step_progress", 0)
                entry["status"] = "processing"
            queue_with_progress.append(entry)
        return {"queue": queue_with_progress, "job": dict(_job) if _job else None}


@app.delete("/api/queue/{item_id}")
def remove_from_queue(item_id: str):
    """Remove an item from the queue. If processing, request cancellation."""
    global _cancel_requested
    with _lock:
        item = next((i for i in _queue if i["id"] == item_id), None)
        if not item:
            raise HTTPException(404, "Item not found in queue")

        is_processing = (_job and _job.get("id") == item_id and _job.get("status") == "running")

    if is_processing:
        # Signal the pipeline thread to cancel between steps
        _cancel_requested = True
        # Don't remove from queue here — _check_pause will handle cleanup
        return {"deleted": True, "cancelling": True, "queue": _queue}

    # Not processing — just remove immediately
    with _lock:
        _queue[:] = [i for i in _queue if i["id"] != item_id]
    _save_queue()
    return {"deleted": True, "queue": _queue}


class ReorderRequest(BaseModel):
    order: list[str]


@app.post("/api/queue/reorder")
def reorder_queue(req: ReorderRequest):
    """Reorder the queue based on the provided ID order."""
    with _lock:
        id_map = {item["id"]: item for item in _queue}
        new_queue = []
        for item_id in req.order:
            if item_id in id_map:
                new_queue.append(id_map.pop(item_id))
        # Append any items not in the order list (shouldn't happen but be safe)
        for item in id_map.values():
            new_queue.append(item)
        _queue[:] = new_queue
    _save_queue()
    return {"queue": _queue}


@app.post("/api/queue/start")
def start_queue():
    """Manually trigger processing of the next queued item."""
    _process_next_in_queue()
    return {"started": True, "queue": _queue}


@app.post("/api/queue/{item_id}/pause")
def pause_queue_item(item_id: str):
    """Request pause of the currently processing item."""
    global _pause_requested
    with _lock:
        if not _job or _job.get("id") != item_id or _job.get("status") != "running":
            raise HTTPException(400, "Item is not currently processing")
    _pause_requested = True
    return {"paused": True}


@app.post("/api/queue/{item_id}/resume")
def resume_queue_item(item_id: str):
    """Resume a paused queue item."""
    global _pause_requested
    with _lock:
        item = next((i for i in _queue if i["id"] == item_id), None)
        if not item:
            raise HTTPException(404, "Item not found")
    _pause_requested = False

    # Find the job data and resume it
    output_dir = JOBS_DIR / item_id
    job_json = output_dir / "job.json"
    if not job_json.exists():
        # Never started — just re-queue and process
        with _lock:
            item["status"] = "queued"
        _save_queue()
        _process_next_in_queue()
        return {"resumed": True}

    data = json.loads(job_json.read_text())
    resume_step = _detect_resume_step(output_dir)
    mode = data.get("mode", "karaoke")
    languages = data.get("languages", [])

    with _lock:
        item["status"] = "processing"
    _save_queue()

    global _job
    _job = {
        "id": item_id,
        "url": data.get("url", ""),
        "mode": mode,
        "languages": languages,
        "title": data.get("title", "Unknown"),
        "thumbnail": data.get("thumbnail"),
        "channel": data.get("channel"),
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

    if mode == "subtitled":
        thread = threading.Thread(
            target=_run_subtitled_pipeline,
            args=(item_id, data.get("url", ""), output_dir, languages),
            daemon=True)
    else:
        thread = threading.Thread(
            target=_run_pipeline,
            args=(item_id, data.get("url", ""), output_dir),
            kwargs={"resume_from": resume_step},
            daemon=True)
    thread.start()

    return {"resumed": True}


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
def get_library(user: User | None = Depends(get_optional_user),
                db: Session = Depends(get_db)):
    """Return all completed jobs as a list, sorted newest first, with vote data."""
    from sqlalchemy import func

    items = []
    if not JOBS_DIR.exists():
        return {"items": items}

    # Preload all vote counts in one query
    vote_rows = db.query(
        Vote.job_id, Vote.value, func.count()
    ).group_by(Vote.job_id, Vote.value).all()
    vote_map: dict[str, dict] = {}
    for job_id, value, count in vote_rows:
        if job_id not in vote_map:
            vote_map[job_id] = {"upvotes": 0, "downvotes": 0}
        if value == 1:
            vote_map[job_id]["upvotes"] = count
        elif value == -1:
            vote_map[job_id]["downvotes"] = count

    # Preload current user's votes
    user_votes = {}
    if user:
        for v in db.query(Vote).filter(Vote.user_id == user.id).all():
            user_votes[v.job_id] = v.value

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
        job_id = data.get("id", d.name)
        votes = vote_map.get(job_id, {"upvotes": 0, "downvotes": 0})
        items.append({
            "id": job_id,
            "title": data.get("title", "Unknown"),
            "url": data.get("url"),
            "mode": data.get("mode", "karaoke"),
            "languages": data.get("languages") or ([data["language"]] if data.get("language") else []),
            "thumbnail": data.get("thumbnail"),
            "channel": data.get("channel"),
            "upload_date": data.get("upload_date"),
            "categories": data.get("categories", []),
            "tags": data.get("tags", []),
            "finished_at": data.get("finished_at"),
            "audio_duration": data.get("audio_duration"),
            "upvotes": votes["upvotes"],
            "downvotes": votes["downvotes"],
            "user_vote": user_votes.get(job_id, 0),
        })
    items.sort(key=lambda x: x.get("finished_at") or "", reverse=True)
    return {"items": items}


def _extract_video_id(url: str) -> str | None:
    """Extract the YouTube video ID from various URL formats."""
    m = re.search(r"(?:v=|youtu\.be/|/embed/|/v/|/shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


@app.get("/api/library/check-url")
def check_url_in_library(url: str, mode: str = "karaoke"):
    """Check if a YouTube URL has already been generated in the library for the given mode."""
    video_id = _extract_video_id(url)
    if not video_id:
        return {"found": False}

    if not JOBS_DIR.exists():
        return {"found": False}

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
        if data.get("mode", "karaoke") != mode:
            continue
        existing_id = _extract_video_id(data.get("url", ""))
        if existing_id == video_id:
            return {
                "found": True,
                "item": {
                    "id": data.get("id", d.name),
                    "title": data.get("title", "Unknown"),
                    "url": data.get("url"),
                    "mode": data.get("mode", "karaoke"),
                    "thumbnail": data.get("thumbnail"),
                    "channel": data.get("channel"),
                    "finished_at": data.get("finished_at"),
                },
            }

    return {"found": False}


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


@app.get("/api/jobs/{job_id}/subtitles/{lang_code}")
def get_subtitles_lang(job_id: str, lang_code: str):
    """Serve per-language SRT file."""
    # Sanitize lang_code to prevent path traversal
    if not re.match(r"^[a-z]{2,3}$", lang_code):
        raise HTTPException(400, "Invalid language code")
    path = JOBS_DIR / job_id / f"subtitles_{lang_code}.srt"
    if not path.exists():
        raise HTTPException(404, "Subtitles not found")
    filename = f"subtitles_{lang_code}.srt"
    return FileResponse(path, media_type="text/plain",
                        filename=filename,
                        headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.get("/api/jobs/{job_id}/subtitles")
def get_subtitles(job_id: str):
    """Backward compat: serve first available SRT file."""
    job_dir = JOBS_DIR / job_id
    # Try to find any subtitles_*.srt file
    for f in sorted(job_dir.glob("subtitles_*.srt")):
        return FileResponse(f, media_type="text/plain",
                            filename=f.name,
                            headers={"Content-Disposition": f"attachment; filename={f.name}"})
    # Fall back to old single-file format
    path = job_dir / "subtitles.srt"
    if path.exists():
        return FileResponse(path, media_type="text/plain",
                            filename="subtitles.srt",
                            headers={"Content-Disposition": "attachment; filename=subtitles.srt"})
    raise HTTPException(404, "Subtitles not found")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """Delete a job and all its files from disk."""
    global _job
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists() or not job_dir.is_dir():
        raise HTTPException(404, "Job not found")

    # Don't allow deleting a running job
    with _lock:
        if _job is not None and _job.get("id") == job_id and _job.get("status") == "running":
            raise HTTPException(409, "Cannot delete a running job")

    shutil.rmtree(job_dir)

    # Clear current job reference if it was the deleted one
    with _lock:
        if _job is not None and _job.get("id") == job_id:
            _job = None

    return {"deleted": True}


@app.get("/api/jobs/{job_id}/download/{filename}")
def download_file(job_id: str, filename: str):
    # Allow specific files + subtitles_*.srt pattern
    static_allowed = {"karaoke.mp4", "instrumental.mp3", "vocals.mp3", "subtitles.srt"}
    is_subtitle = bool(re.match(r"^subtitles_[a-z]{2,3}\.srt$", filename))
    if filename not in static_allowed and not is_subtitle:
        raise HTTPException(400, "Invalid file")
    path = JOBS_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    media_types = {
        "karaoke.mp4": "video/mp4",
        "instrumental.mp3": "audio/mpeg",
        "vocals.mp3": "audio/mpeg",
    }
    media_type = media_types.get(filename, "text/plain")
    return FileResponse(
        path,
        media_type=media_type,
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
# ---------------------------------------------------------------------------
# Feedback API
# ---------------------------------------------------------------------------

from fastapi import File, Form, UploadFile

@app.post("/api/feedback")
async def submit_feedback(
    subject: str = Form(...),
    description: str = Form(...),
    screenshot: UploadFile | None = File(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    screenshot_path = None
    if screenshot and screenshot.filename:
        feedback_dir = Path("output/feedback")
        feedback_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{secrets.token_hex(8)}_{screenshot.filename}"
        filepath = feedback_dir / filename
        content = await screenshot.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(400, "Screenshot must be under 5MB")
        filepath.write_bytes(content)
        screenshot_path = str(filepath)

    fb = Feedback(
        user_id=user.id,
        subject=subject,
        description=description,
        screenshot_path=screenshot_path,
    )
    db.add(fb)
    db.commit()
    return {"submitted": True}


# ---------------------------------------------------------------------------
# Admin API
# ---------------------------------------------------------------------------

@app.get("/api/admin/users")
def admin_list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    result = []
    for u in users:
        data = u.to_dict()
        data["permissions"] = u.permissions.to_dict() if u.permissions else {}
        result.append(data)
    return {"users": result}


class UpdatePermissionsRequest(BaseModel):
    max_karaoke_per_day: int | None = None
    max_subtitled_per_day: int | None = None
    max_queue_length: int | None = None
    can_download_karaoke: bool | None = None
    can_download_instrumental: bool | None = None
    can_download_vocals: bool | None = None
    can_delete_library: bool | None = None
    can_share_library: bool | None = None


@app.put("/api/admin/users/{user_id}/permissions")
def admin_update_permissions(user_id: str, req: UpdatePermissionsRequest,
                              admin: User = Depends(require_admin),
                              db: Session = Depends(get_db)):
    perms = db.query(UserPermissions).filter(UserPermissions.user_id == user_id).first()
    if not perms:
        raise HTTPException(404, "User not found")
    for field, value in req.model_dump(exclude_none=True).items():
        setattr(perms, field, value)
    db.commit()
    return {"permissions": perms.to_dict()}


@app.put("/api/admin/users/{user_id}/role")
def admin_update_role(user_id: str, role: str,
                       admin: User = Depends(require_admin),
                       db: Session = Depends(get_db)):
    if role not in ("user", "admin"):
        raise HTTPException(400, "Invalid role")
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(404, "User not found")
    target.role = role
    db.commit()
    return {"user": target.to_dict()}


@app.get("/api/admin/feedback")
def admin_list_feedback(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    items = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
    return {"feedback": [fb.to_dict() for fb in items]}


@app.put("/api/admin/feedback/{feedback_id}")
def admin_update_feedback(feedback_id: str, status: str,
                           admin: User = Depends(require_admin),
                           db: Session = Depends(get_db)):
    if status not in ("new", "reviewed", "resolved"):
        raise HTTPException(400, "Invalid status")
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(404, "Feedback not found")
    fb.status = status
    db.commit()
    return {"feedback": fb.to_dict()}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Votes API
# ---------------------------------------------------------------------------

class VoteRequest(BaseModel):
    value: int  # +1 or -1


@app.post("/api/jobs/{job_id}/vote")
def vote_on_job(job_id: str, req: VoteRequest, user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    """Cast or update a vote on a library item."""
    if req.value not in (1, -1):
        raise HTTPException(400, "Vote value must be 1 or -1")

    existing = db.query(Vote).filter(Vote.user_id == user.id, Vote.job_id == job_id).first()
    if existing:
        if existing.value == req.value:
            # Same vote again — remove it (toggle off)
            db.delete(existing)
            db.commit()
            return _get_vote_summary(job_id, user.id, db)
        else:
            # Switch vote
            existing.value = req.value
            db.commit()
            return _get_vote_summary(job_id, user.id, db)
    else:
        vote = Vote(user_id=user.id, job_id=job_id, value=req.value)
        db.add(vote)
        db.commit()
        return _get_vote_summary(job_id, user.id, db)


@app.get("/api/jobs/{job_id}/votes")
def get_job_votes(job_id: str, user: User | None = Depends(get_optional_user),
                  db: Session = Depends(get_db)):
    """Get vote counts for a job."""
    user_id = user.id if user else None
    return _get_vote_summary(job_id, user_id, db)


@app.get("/api/votes/batch")
def get_batch_votes(job_ids: str, user: User | None = Depends(get_optional_user),
                    db: Session = Depends(get_db)):
    """Get vote counts for multiple jobs. job_ids is comma-separated."""
    ids = [jid.strip() for jid in job_ids.split(",") if jid.strip()]
    user_id = user.id if user else None
    result = {}
    for jid in ids:
        result[jid] = _get_vote_summary(jid, user_id, db)
    return {"votes": result}


def _get_vote_summary(job_id: str, user_id, db: Session) -> dict:
    """Get upvotes, downvotes, and current user's vote for a job."""
    from sqlalchemy import func
    rows = db.query(Vote.value, func.count()).filter(Vote.job_id == job_id).group_by(Vote.value).all()
    upvotes = 0
    downvotes = 0
    for value, count in rows:
        if value == 1:
            upvotes = count
        elif value == -1:
            downvotes = count
    user_vote = 0
    if user_id:
        existing = db.query(Vote).filter(Vote.user_id == user_id, Vote.job_id == job_id).first()
        if existing:
            user_vote = existing.value
    return {"job_id": job_id, "upvotes": upvotes, "downvotes": downvotes, "user_vote": user_vote}


# ---------------------------------------------------------------------------
# Static files & index
# ---------------------------------------------------------------------------
@app.get("/")
def index(user: User | None = Depends(get_optional_user)):
    # Always serve the app — the frontend handles auth checks
    # This allows shared watch links (/#/watch/id) to work without login
    return FileResponse("static/index.html")


@app.get("/login")
def login_page():
    return FileResponse("static/login.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.png", media_type="image/png")


app.mount("/static", StaticFiles(directory="static"), name="static")
