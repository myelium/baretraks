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
    get_optional_user, hash_password, is_admin, require_admin, verify_password,
)
from database import SessionLocal, get_db
from models import ActivityLog, Comment, Feedback, Invitation, JobMetadata, Playlist, PlaylistItem, User, UserPermissions, Vote, WishlistItem, WishlistVote

import logging
import os
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


def _log_activity(db, user, event_type: str, detail: dict | None = None) -> None:
    """Append an entry to the activity log. Non-blocking — won't raise on failure."""
    try:
        db.add(ActivityLog(
            user_id=user.id if user else None,
            user_name=user.name if user else "Anonymous",
            event_type=event_type,
            detail=json.dumps(detail) if detail else None,
        ))
        db.commit()
    except Exception:
        pass


# --- Compute device (cpu/cuda) ---
def _detect_device() -> str:
    env = os.getenv("DEVICE", "auto").lower()
    if env in ("cpu", "cuda"):
        return env
    return "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = _detect_device()

# --- Email via Resend ---
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@baretraks.com")
logger = logging.getLogger(__name__)


def _invite_email_html(inviter_name: str, link: str) -> str:
    return f"""\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#0e0c09;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0e0c09;padding:40px 20px;">
    <tr><td align="center">
      <table width="480" cellpadding="0" cellspacing="0" style="max-width:480px;width:100%;">
        <!-- Header -->
        <tr><td align="center" style="padding-bottom:32px;">
          <img src="{BASE_URL}/static/logo.png" alt="Baretraks" width="180" style="display:block;margin:0 auto;" />
        </td></tr>
        <!-- Card -->
        <tr><td style="background:#1a1612;border-radius:12px;padding:36px 32px;border:1px solid #2a2420;">
          <p style="margin:0 0 16px;font-size:16px;color:#e8e0d4;line-height:1.5;">
            <strong style="color:#b48c3c;">{inviter_name}</strong> invited you to join Baretraks &mdash; a community for music lovers who want to sing, discover, and share.
          </p>
          <p style="margin:0 0 28px;font-size:14px;color:#8a8072;line-height:1.5;">
            Create karaoke videos, queue up songs with friends, and explore music across languages.
          </p>
          <!-- Button -->
          <table cellpadding="0" cellspacing="0" width="100%"><tr><td align="center">
            <a href="{link}" style="display:inline-block;background:#b48c3c;color:#1a1612;font-size:15px;font-weight:600;padding:12px 32px;border-radius:8px;text-decoration:none;letter-spacing:0.3px;">
              Accept Invitation
            </a>
          </td></tr></table>
        </td></tr>
        <!-- Footer -->
        <tr><td align="center" style="padding-top:24px;">
          <p style="margin:0 0 8px;font-size:12px;color:#5a5248;">
            Or copy this link: <a href="{link}" style="color:#b48c3c;text-decoration:underline;">{link}</a>
          </p>
          <p style="margin:0;font-size:11px;color:#3a3530;">
            This invitation was sent by {inviter_name} via Baretraks. If you didn&rsquo;t expect this, you can ignore it.
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>"""


def send_invite_email(to_email: str, inviter_name: str, token: str) -> bool:
    """Send an invitation email via Resend. Returns True on success."""
    if not RESEND_API_KEY:
        logger.info("RESEND_API_KEY not set — skipping email to %s", to_email)
        return False
    import resend
    resend.api_key = RESEND_API_KEY
    link = f"{BASE_URL}/login?invite={token}"
    try:
        resend.Emails.send({
            "from": EMAIL_FROM,
            "to": [to_email],
            "subject": f"{inviter_name} invited you to Baretraks",
            "html": _invite_email_html(inviter_name, link),
        })
        return True
    except Exception as e:
        logger.error("Failed to send invite email to %s: %s", to_email, e)
        return False

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


_disk_quota_exceeded = False  # set when disk quota blocks production


def _get_disk_usage_mb() -> float:
    """Calculate total disk usage of all job output directories in MB."""
    total = 0
    if JOBS_DIR.exists():
        for d in JOBS_DIR.iterdir():
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.is_file():
                    total += f.stat().st_size
    return total / (1024 * 1024)


def _process_next_in_queue() -> None:
    """Start processing the next queued item if nothing is running.
    Automatically detects whether to resume from prior progress or start fresh."""
    global _job, _pause_requested, _disk_quota_exceeded

    # Check global disk quota before starting
    settings = _load_settings()
    max_disk = settings.get("max_disk_space_mb", 0)
    if max_disk > 0:
        usage = _get_disk_usage_mb()
        if usage >= max_disk:
            _disk_quota_exceeded = True
            return  # don't start new jobs
    _disk_quota_exceeded = False

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

    url = next_item["url"]
    mode = next_item.get("mode", "karaoke")
    languages = next_item.get("languages", [])
    job_id = next_item["id"]
    output_dir = JOBS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if there's prior work on disk to resume from
    resume_from = 1
    prior_data = {}
    job_json = output_dir / "job.json"
    if job_json.exists():
        try:
            prior_data = json.loads(job_json.read_text())
            resume_from = _detect_resume_step(output_dir)
        except (json.JSONDecodeError, OSError):
            pass

    step_name = f"Resuming from step {resume_from}" if resume_from > 1 else "Starting"

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
        "added_by": next_item.get("added_by"),
        "added_by_id": next_item.get("added_by_id"),
        "status": "running",
        "step": 0,
        "step_name": step_name,
        "step_progress": 0.0,
        "started_at": _now_iso(),
        "step_started_at": _now_iso(),
        "finished_at": None,
        "audio_duration": prior_data.get("audio_duration"),
        "estimated_total": None,
        "step_estimates": None,
        "error": None,
        "artifacts": prior_data.get("artifacts", {}),
        "output_dir": str(output_dir),
    }
    _save_job_json()

    if mode == "subtitled":
        thread = threading.Thread(
            target=_run_subtitled_pipeline,
            args=(job_id, url, output_dir, languages),
            daemon=True)
    elif mode == "both":
        thread = threading.Thread(
            target=_run_combined_pipeline,
            args=(job_id, url, output_dir, languages),
            daemon=True)
    else:
        thread = threading.Thread(
            target=_run_pipeline,
            args=(job_id, url, output_dir),
            kwargs={"resume_from": resume_from},
            daemon=True)
    thread.start()


def _on_job_finished() -> None:
    """Called when a job finishes — remove from queue first, then do async work."""
    with _lock:
        job_id = _job["id"] if _job else None
        job_status = _job.get("status") if _job else None
        job_title = _job.get("title") if _job else None
        job_artist = _job.get("artist") or (_job.get("channel") if _job else None)
        job_url = _job.get("url") if _job else None
        output_dir = Path(_job["output_dir"]) if _job else None
        # Remove from queue immediately so the frontend sees a clean state
        _queue[:] = [item for item in _queue if item.get("id") != job_id]
    _save_queue()

    # Start the next job before doing slow post-processing
    _process_next_in_queue()

    # --- Post-completion tasks (run after queue is already updated) ---

    # Pre-generate "Between the Lines" analysis for completed jobs
    if (job_status == "done" and job_id and output_dir
            and _load_settings().get("feature_analysis", True)):
        try:
            lyrics_path = output_dir / "lyrics.json"
            if lyrics_path.exists():
                words = json.loads(lyrics_path.read_text())
                lyrics_text = " ".join(w["text"] for w in words)
                prompts = _load_prompts()
                custom_prompt = prompts.get("analysis_prompt") or None
                from karaoke.analyze_lyrics import analyze_lyrics
                result = analyze_lyrics(lyrics_text, title=job_title,
                                        artist=job_artist,
                                        custom_prompt=custom_prompt)
                _db = SessionLocal()
                _meta = _db.query(JobMetadata).filter(
                    JobMetadata.job_id == job_id).first()
                if not _meta:
                    _meta = JobMetadata(job_id=job_id)
                    _db.add(_meta)
                _meta.analysis_text = result.get("analysis", "")
                _meta.analysis_song_info = result.get("song_info", "")
                if result.get("year"):
                    _meta.year = result["year"]
                if not _meta.artist and result.get("song_info"):
                    info = result["song_info"]
                    if " by " in info:
                        parts = info.rsplit(" by ", 1)
                        if len(parts) == 2:
                            _meta.artist = parts[1].strip().strip('"')
                _db.commit()
                _db.close()
        except Exception:
            pass

    # Mark matching wishlist items as fulfilled
    if job_status == "done" and job_id and job_url:
        try:
            _db = SessionLocal()
            wish_items = _db.query(WishlistItem).filter(
                WishlistItem.url == job_url,
                WishlistItem.status.in_(["open", "queued"])
            ).all()
            for wi in wish_items:
                wi.status = "fulfilled"
                wi.fulfilled_by_job_id = job_id
            _db.commit()
            _db.close()
        except Exception:
            pass


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

STEP_NAMES_BOTH = {
    1: "Downloading",
    2: "Separating vocals",
    3: "Transcribing lyrics",
    4: "Building subtitles",
    5: "Composing video",
    6: "Translating",
}

VALID_MODES = ("karaoke", "subtitled", "both")

# Language code → full name for Claude translation prompts
LANG_FULL_NAMES = {
    "af": "Afrikaans", "sq": "Albanian", "am": "Amharic", "ar": "Arabic",
    "hy": "Armenian", "as": "Assamese", "az": "Azerbaijani", "ba": "Bashkir",
    "eu": "Basque", "be": "Belarusian", "bn": "Bengali", "bs": "Bosnian",
    "br": "Breton", "bg": "Bulgarian", "my": "Burmese",
    "zh-Hans": "Chinese (Simplified)", "zh-Hant": "Chinese (Traditional)",
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
    "mt": "Maltese", "mi": "Maori", "mr": "Marathi",
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


def _words_to_segments(words):
    """Re-segment a flat word list into natural segments based on timing gaps."""
    from karaoke.transcribe import Segment, Word as TWord
    if not words:
        return []
    SEGMENT_GAP = 1.0  # seconds — gap between words that starts a new segment
    segments = []
    current = [words[0]]
    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap >= SEGMENT_GAP:
            segments.append(Segment(
                start=current[0].start, end=current[-1].end, words=current))
            current = [words[i]]
        else:
            current.append(words[i])
    if current:
        segments.append(Segment(
            start=current[0].start, end=current[-1].end, words=current))
    return segments


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
        device = DEVICE
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
            instrumental_wav, vocals_wav = separate(audio_path, work_dir / "demucs", device=device)
            _convert_to_mp3(instrumental_wav, instrumental_mp3)
            _convert_to_mp3(vocals_wav, vocals_mp3)
            instrumental_wav.unlink(missing_ok=True)
            vocals_wav.unlink(missing_ok=True)
            # Keep audio_path — needed for transcription (better quality than separated vocals)
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
            # Transcribe from original audio (full mix) — much more accurate than
            # separated vocals which can have artifacts that confuse Whisper
            transcribe_path = audio_path if audio_path.exists() else vocals_mp3
            segments, detected_lang = transcribe(transcribe_path, device=device)
            # Clean up original audio now that transcription is done
            if audio_path.exists() and audio_path != vocals_mp3:
                audio_path.unlink(missing_ok=True)
            _update_job(language_detected=detected_lang)
            _save_job_json()
            words_list = []
            for seg in segments:
                for w in seg.words:
                    words_list.append({"text": w.text, "start": w.start, "end": w.end})

            # Correct lyrics using LLM knowledge of the song
            if _load_settings().get("feature_lyrics_correction", True):
                _update_job(step_name="Correcting lyrics", step_progress=0.8)
                _save_job_json()
                try:
                    from karaoke.correct_lyrics import correct_lyrics
                    with _lock:
                        job_title = _job.get("title") if _job else None
                        job_channel = _job.get("channel") if _job else None
                    correction = correct_lyrics(words_list, title=job_title,
                                                artist=job_channel)
                    words_list = correction["words"]
                    if correction.get("identified_artist"):
                        _update_job(artist=correction["identified_artist"])
                        _save_job_json()
                        try:
                            _db = SessionLocal()
                            _meta = _db.query(JobMetadata).filter(
                                JobMetadata.job_id == _job["id"]).first()
                            if not _meta:
                                _meta = JobMetadata(job_id=_job["id"])
                                _db.add(_meta)
                            if not _meta.artist:
                                _meta.artist = correction["identified_artist"]
                            _db.commit()
                            _db.close()
                        except Exception:
                            pass
                except Exception:
                    pass  # keep original transcription on failure

            (output_dir / "lyrics.json").write_text(json.dumps(words_list))

            # Rebuild segments from corrected words, using natural timing
            # gaps to create proper segment boundaries (not one giant segment)
            from karaoke.transcribe import Segment, Word as TWord
            corrected_words = [TWord(text=w["text"], start=w["start"], end=w["end"])
                               for w in words_list]
            segments = _words_to_segments(corrected_words)

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
                from karaoke.transcribe import Word as TWord
                lyrics = json.loads((output_dir / "lyrics.json").read_text())
                words = [TWord(text=w["text"], start=w["start"], end=w["end"]) for w in lyrics]
                segments = _words_to_segments(words)
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
        device = DEVICE
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
        segments, detected_lang = transcribe(audio_path, device=device)
        _update_job(language_detected=detected_lang)
        _save_job_json()

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
        if languages and _load_settings().get("feature_translation", True):
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


def _run_combined_pipeline(job_id: str, url: str, output_dir: Path,
                           languages: list[str] | None = None) -> None:
    """Run karaoke + subtitles in a single pipeline (6 steps)."""
    if not languages:
        languages = []
    try:
        device = DEVICE
        work_dir = output_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        video_path = work_dir / "video.mp4"
        audio_path = work_dir / "audio.wav"
        instrumental_mp3 = output_dir / "instrumental.mp3"
        vocals_mp3 = output_dir / "vocals.mp3"
        subtitles_path = work_dir / "karaoke.ass"
        output_path = output_dir / "karaoke.mp4"

        step_durations: dict[str, float] = {}

        # --- Step 1: Download ---
        t0 = time.monotonic()
        _update_job(step=1, step_name=STEP_NAMES_BOTH[1], step_progress=0.0,
                     step_started_at=_now_iso())
        _save_job_json()

        def _dl_progress(pct: float) -> None:
            _update_job(step_progress=pct)

        video_path, audio_path = download(url, work_dir, progress_callback=_dl_progress)
        step_durations["1"] = time.monotonic() - t0
        _update_job(step_progress=1.0, artifacts={
            1: [{"name": "video.mp4", "path": "work/video.mp4"}],
        })

        # Compute time estimates
        audio_duration = _get_audio_duration(audio_path)
        estimates = _compute_estimates(audio_duration)
        # Add estimate for step 6 (translation)
        est_translate = audio_duration * 2.0 * max(len(languages), 1)
        estimates.append(est_translate)
        estimates[0] = step_durations.get("1", estimates[0])
        _update_job(
            audio_duration=audio_duration,
            step_estimates=estimates,
            estimated_total=sum(estimates),
        )

        # --- Step 2: Separate ---
        if _check_pause(): return
        t0 = time.monotonic()
        _update_job(step=2, step_name=STEP_NAMES_BOTH[2], step_progress=0.0,
                     step_started_at=_now_iso())
        _save_job_json()
        instrumental_wav, vocals_wav = separate(audio_path, work_dir / "demucs", device=device)
        _convert_to_mp3(instrumental_wav, instrumental_mp3)
        _convert_to_mp3(vocals_wav, vocals_mp3)
        instrumental_wav.unlink(missing_ok=True)
        vocals_wav.unlink(missing_ok=True)
        # Keep audio_path — needed for transcription
        step_durations["2"] = time.monotonic() - t0
        with _lock:
            artifacts = dict(_job.get("artifacts", {}))
        artifacts[2] = [
            {"name": "instrumental.mp3", "path": "instrumental.mp3"},
            {"name": "vocals.mp3", "path": "vocals.mp3"},
        ]
        _update_job(step_progress=1.0, artifacts=artifacts)

        # --- Step 3: Transcribe ---
        if _check_pause(): return
        t0 = time.monotonic()
        _update_job(step=3, step_name=STEP_NAMES_BOTH[3], step_progress=0.0,
                     step_started_at=_now_iso())
        _save_job_json()
        transcribe_path = audio_path if audio_path.exists() else vocals_mp3
        segments, detected_lang = transcribe(transcribe_path, device=device)
        if audio_path.exists() and audio_path != vocals_mp3:
            audio_path.unlink(missing_ok=True)
        _update_job(language_detected=detected_lang)
        _save_job_json()
        words_list = []
        for seg in segments:
            for w in seg.words:
                words_list.append({"text": w.text, "start": w.start, "end": w.end})

        # Correct lyrics using LLM knowledge of the song
        if _load_settings().get("feature_lyrics_correction", True):
            _update_job(step_name="Correcting lyrics", step_progress=0.8)
            _save_job_json()
            try:
                from karaoke.correct_lyrics import correct_lyrics
                with _lock:
                    job_title = _job.get("title") if _job else None
                    job_channel = _job.get("channel") if _job else None
                correction = correct_lyrics(words_list, title=job_title,
                                            artist=job_channel)
                words_list = correction["words"]
                if correction.get("identified_artist"):
                    _update_job(artist=correction["identified_artist"])
                    _save_job_json()
                    try:
                        _db = SessionLocal()
                        _meta = _db.query(JobMetadata).filter(
                            JobMetadata.job_id == _job["id"]).first()
                        if not _meta:
                            _meta = JobMetadata(job_id=_job["id"])
                            _db.add(_meta)
                        if not _meta.artist:
                            _meta.artist = correction["identified_artist"]
                        _db.commit()
                        _db.close()
                    except Exception:
                        pass
            except Exception:
                pass  # keep original transcription on failure

        (output_dir / "lyrics.json").write_text(json.dumps(words_list))

        # Rebuild segments from corrected words
        from karaoke.transcribe import Segment, Word as TWord
        corrected_words = [TWord(text=w["text"], start=w["start"], end=w["end"])
                           for w in words_list]
        if corrected_words:
            segments = [Segment(start=corrected_words[0].start,
                                end=corrected_words[-1].end,
                                words=corrected_words)]
        else:
            segments = []

        step_durations["3"] = time.monotonic() - t0
        with _lock:
            artifacts = dict(_job.get("artifacts", {}))
        artifacts[3] = [{"name": "lyrics.json", "path": "lyrics.json"}]
        _update_job(step_progress=1.0, artifacts=artifacts)

        # --- Step 4: Build ASS subtitles ---
        if _check_pause(): return
        t0 = time.monotonic()
        _update_job(step=4, step_name=STEP_NAMES_BOTH[4], step_progress=0.0,
                     step_started_at=_now_iso())
        build_ass(segments, subtitles_path)
        step_durations["4"] = time.monotonic() - t0
        with _lock:
            artifacts = dict(_job.get("artifacts", {}))
        artifacts[4] = []
        _update_job(step_progress=1.0, artifacts=artifacts)

        # --- Step 5: Compose video ---
        if _check_pause(): return
        t0 = time.monotonic()
        _update_job(step=5, step_name=STEP_NAMES_BOTH[5], step_progress=0.0,
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

        # --- Step 6: Translate subtitles ---
        if _check_pause(): return
        srt_artifacts = []
        if languages and _load_settings().get("feature_translation", True):
            t0 = time.monotonic()
            # Build source SRT from transcription for Claude translation
            source_srt_path = output_dir / "subtitles_source.srt"
            build_srt(segments, source_srt_path)
            source_srt_text = source_srt_path.read_text()

            for i, lang in enumerate(languages):
                target_name = LANG_FULL_NAMES.get(lang, lang)
                label = (f"Translating to {target_name}"
                         f" ({i+1} of {len(languages)})" if len(languages) > 1
                         else f"Translating to {target_name}")
                _update_job(step=6, step_name=label,
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

            step_durations["6"] = time.monotonic() - t0
            source_srt_path.unlink(missing_ok=True)

        with _lock:
            artifacts = dict(_job.get("artifacts", {}))
        artifacts[5] = [{"name": "karaoke.mp4", "path": "karaoke.mp4"}]
        if srt_artifacts:
            artifacts[6] = srt_artifacts
        _update_job(step_progress=1.0, artifacts=artifacts)

        # Save timing stats
        if audio_duration > 0 and step_durations:
            _save_stats(audio_duration, step_durations)

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
def _run_migrations() -> None:
    """Add missing columns incrementally."""
    from sqlalchemy import inspect, text
    from database import engine
    insp = inspect(engine)

    # --- Invitation table: token, expires_at, accepted_by_id ---
    inv_cols = {c["name"] for c in insp.get_columns("invitations")}
    needs_invite_backfill = "token" not in inv_cols
    if needs_invite_backfill:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE invitations ADD COLUMN token VARCHAR(64)"))
            conn.execute(text("ALTER TABLE invitations ADD COLUMN expires_at TIMESTAMPTZ"))
            conn.execute(text("ALTER TABLE invitations ADD COLUMN accepted_by_id UUID REFERENCES users(id)"))
        # Backfill in a separate transaction (no lock conflict)
        import secrets as _sec
        from datetime import timedelta
        db = SessionLocal()
        from models import Invitation as InvModel
        for inv in db.query(InvModel).all():
            inv.token = _sec.token_urlsafe(32)
            inv.expires_at = inv.created_at + timedelta(days=7)
        db.commit()
        db.close()
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE invitations ALTER COLUMN token SET NOT NULL"))
            conn.execute(text("ALTER TABLE invitations ALTER COLUMN expires_at SET NOT NULL"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_invitations_token ON invitations (token)"))

    # --- User table: invited_by_id ---
    user_cols = {c["name"] for c in insp.get_columns("users")}
    if "invited_by_id" not in user_cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN invited_by_id UUID REFERENCES users(id)"))

    # --- UserPermissions table ---
    perm_cols = {c["name"] for c in insp.get_columns("user_permissions")}
    with engine.begin() as conn:
        if "max_invitations" not in perm_cols:
            conn.execute(text("ALTER TABLE user_permissions ADD COLUMN max_invitations INTEGER NOT NULL DEFAULT 5"))
        if "can_request_songs" not in perm_cols:
            conn.execute(text("ALTER TABLE user_permissions ADD COLUMN can_request_songs BOOLEAN NOT NULL DEFAULT TRUE"))

    # --- JobMetadata: year column ---
    if "job_metadata" in set(insp.get_table_names()):
        jm_cols = {c["name"] for c in insp.get_columns("job_metadata")}
        if "year" not in jm_cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE job_metadata ADD COLUMN year VARCHAR(10)"))
            # Backfill year from existing analysis_song_info, e.g. "Alone by Heart (1987)"
            import re as _re
            _db = SessionLocal()
            for m in _db.query(JobMetadata).filter(JobMetadata.analysis_song_info.isnot(None)).all():
                match = _re.search(r"\((\d{4})\)", m.analysis_song_info or "")
                if match:
                    m.year = match.group(1)
            _db.commit()
            _db.close()

    # --- Wishlist tables ---
    tables = set(insp.get_table_names())
    if "wishlist_items" not in tables:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE wishlist_items (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    url VARCHAR(512),
                    title VARCHAR(512) NOT NULL,
                    artist VARCHAR(255),
                    thumbnail VARCHAR(512),
                    note TEXT,
                    status VARCHAR(20) NOT NULL DEFAULT 'open',
                    fulfilled_by_job_id VARCHAR(255),
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """))
    if "wishlist_votes" not in tables:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE wishlist_votes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    wishlist_item_id UUID NOT NULL REFERENCES wishlist_items(id) ON DELETE CASCADE,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    CONSTRAINT uq_user_wishlist_vote UNIQUE (user_id, wishlist_item_id)
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_wishlist_votes_item ON wishlist_votes (wishlist_item_id)"))

    # --- Activity log table ---
    if "activity_log" not in tables:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE activity_log (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                    user_name VARCHAR(255) NOT NULL DEFAULT 'Anonymous',
                    event_type VARCHAR(30) NOT NULL,
                    detail TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_activity_log_event ON activity_log (event_type)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_activity_log_created ON activity_log (created_at DESC)"))


@app.on_event("startup")
def _recover_jobs() -> None:
    global _job
    # Load historical timing data and queue
    _load_stats()
    _load_queue()
    if not JOBS_DIR.exists():
        return

    # Build set of IDs already in the queue
    queue_ids = {item["id"] for item in _queue}

    # Scan all job directories for failed or interrupted jobs
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

        job_id = data.get("id", d.name)

        # If it was running when server died, mark as failed
        if data.get("status") == "running":
            data["status"] = "failed"
            data["error"] = "Server restarted during processing"
            data["finished_at"] = _now_iso()
            job_json.write_text(json.dumps(data, default=str))

        # Skip jobs that were explicitly removed by user
        if data.get("status") == "removed":
            continue

        # Add any failed job to the queue so it retries automatically
        if data.get("status") == "failed" and job_id not in queue_ids:
            _queue.append({
                "id": job_id,
                "url": data.get("url", ""),
                "mode": data.get("mode", "karaoke"),
                "languages": data.get("languages", []),
                "title": data.get("title", "Unknown"),
                "thumbnail": data.get("thumbnail"),
                "channel": data.get("channel"),
                "upload_date": data.get("upload_date"),
                "categories": data.get("categories", []),
                "tags": data.get("tags", []),
                "status": "queued",
                "added_by": data.get("added_by"),
                "was_failed": True,
            })
            queue_ids.add(job_id)

    # Set the most recent job as current (for status display)
    job_dirs = sorted(JOBS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in job_dirs:
        job_json = d / "job.json"
        if not job_json.exists():
            continue
        try:
            data = json.loads(job_json.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        _job = data
        break

    # Reset any non-queued items back to "queued" (server restarted mid-job)
    for item in _queue:
        if item.get("status") in ("processing", "failed"):
            item["status"] = "queued"
            item["was_failed"] = True
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
    invite_token: str


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


def _validate_invite_token(db: Session, token: str) -> Invitation:
    """Validate an invite token. Returns the Invitation or raises HTTPException."""
    inv = db.query(Invitation).filter(Invitation.token == token).first()
    if not inv:
        raise HTTPException(400, "Invalid invitation token")
    if inv.status != "pending":
        raise HTTPException(400, "Invitation already used")
    if not inv.is_valid():
        raise HTTPException(400, "Invitation has expired")
    return inv


@app.post("/api/auth/register")
def register(req: RegisterRequest, response: Response, db: Session = Depends(get_db)):
    invitation = _validate_invite_token(db, req.invite_token)
    if invitation.email.lower() != req.email.lower():
        raise HTTPException(400, "This invitation was sent to a different email address")
    if db.query(User).filter(User.email == req.email.lower()).first():
        raise HTTPException(400, "Email already registered")
    user = create_user_with_permissions(
        db, email=req.email, name=req.name, password=req.password,
        invited_by_id=str(invitation.inviter_id),
    )
    invitation.status = "accepted"
    invitation.accepted_by_id = user.id
    db.commit()
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
    _log_activity(db, user, "login", {"method": "password"})
    _set_auth_cookie(response, user)
    return {"user": user.to_dict()}


@app.get("/api/auth/google")
def google_login(invite: str | None = None):
    """Redirect to Google OAuth. Requires GOOGLE_CLIENT_ID to be set."""
    import os
    from urllib.parse import quote
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    if not client_id:
        raise HTTPException(501, "Google OAuth not configured")
    redirect_uri = "/api/auth/google/callback"
    scope = "openid email profile"
    state = quote(invite or "", safe="")
    url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={client_id}&response_type=code&scope={scope}"
        f"&redirect_uri={redirect_uri}&access_type=offline"
        f"&state={state}"
    )
    return RedirectResponse(url)


@app.get("/api/auth/google/callback")
def google_callback(code: str, state: str = "", db: Session = Depends(get_db)):
    """Handle Google OAuth callback."""
    import os
    import httpx
    from urllib.parse import unquote
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise HTTPException(501, "Google OAuth not configured")

    invite_token = unquote(state) if state else ""

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
            # Existing user — link Google account, no invite needed
            user.google_id = info["id"]
            user.picture_url = info.get("picture")
        else:
            # New user — require valid invite token
            if not invite_token:
                return RedirectResponse("/login?error=invite_required")
            inv = db.query(Invitation).filter(Invitation.token == invite_token).first()
            if not inv or not inv.is_valid():
                return RedirectResponse("/login?error=invalid_invite")
            if inv.email.lower() != info["email"].lower():
                return RedirectResponse("/login?error=email_mismatch")
            user = create_user_with_permissions(
                db, email=info["email"], name=info.get("name", ""),
                google_id=info["id"], picture_url=info.get("picture"),
                invited_by_id=str(inv.inviter_id),
            )
            inv.status = "accepted"
            inv.accepted_by_id = user.id
    from datetime import datetime, timezone as tz
    user.last_login = datetime.now(tz.utc)
    db.commit()
    _log_activity(db, user, "login", {"method": "google"})

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


class ProfileUpdateRequest(BaseModel):
    name: str | None = None
    email: str | None = None
    current_password: str | None = None
    new_password: str | None = None


@app.put("/api/auth/profile")
def update_profile(req: ProfileUpdateRequest, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Update user profile (name, email, password)."""
    if req.name is not None:
        name = req.name.strip()
        if not name:
            raise HTTPException(400, "Name is required")
        user.name = name

    if req.email is not None:
        email = req.email.strip().lower()
        if not email:
            raise HTTPException(400, "Email is required")
        existing = db.query(User).filter(User.email == email, User.id != user.id).first()
        if existing:
            raise HTTPException(409, "Email already in use")
        user.email = email

    if req.new_password:
        if not req.new_password or len(req.new_password) < 6:
            raise HTTPException(400, "Password must be at least 6 characters")
        # If user has a password, require current password
        if user.password_hash:
            if not req.current_password or not verify_password(req.current_password, user.password_hash):
                raise HTTPException(403, "Current password is incorrect")
        user.password_hash = hash_password(req.new_password)

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
def create_job(req: JobRequest, user: User | None = Depends(get_optional_user)):
    global _job
    with _lock:
        if _job is not None and _job["status"] == "running":
            raise HTTPException(409, "A job is already running")

    settings = _load_settings()
    allowed = settings.get("allowed_modes", list(VALID_MODES))
    mode = req.mode if req.mode in VALID_MODES else "karaoke"
    if mode not in allowed:
        raise HTTPException(403, f"Production mode '{mode}' is not enabled")

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
    # Support both old single language and new multi-language
    languages = req.languages[:3] if req.languages else (
        [req.language] if req.language else []
    )
    if mode not in ("subtitled", "both"):
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
        "added_by": user.name.split()[0] if user and user.name else None,
        "added_by_id": str(user.id) if user else None,
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
    elif mode == "both":
        thread = threading.Thread(
            target=_run_combined_pipeline,
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
def add_to_queue(req: QueueRequest, user: User | None = Depends(get_optional_user),
                 db: Session = Depends(get_db)):
    """Add an item to the production queue."""
    settings = _load_settings()
    allowed = settings.get("allowed_modes", list(VALID_MODES))
    mode = req.mode if req.mode in VALID_MODES else "karaoke"
    if mode not in allowed:
        raise HTTPException(403, f"Production mode '{mode}' is not enabled")

    languages = req.languages[:3] if req.languages else []
    if mode not in ("subtitled", "both"):
        languages = []

    # Enforce per-user limits (admins bypass)
    if user and not is_admin(user) and user.permissions:
        p = user.permissions
        # Queue length limit
        with _lock:
            user_queued = sum(1 for q in _queue if q.get("added_by_id") == str(user.id))
        if user_queued >= p.max_queue_length:
            raise HTTPException(429, f"Queue limit reached ({p.max_queue_length})")
        # Daily production limits
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        user_today_karaoke = 0
        user_today_subtitled = 0
        for d in JOBS_DIR.iterdir():
            if not d.is_dir():
                continue
            jf = d / "job.json"
            if not jf.exists():
                continue
            try:
                jdata = json.loads(jf.read_text())
                if jdata.get("added_by_id") != str(user.id):
                    continue
                if jdata.get("status") != "done":
                    continue
                finished = jdata.get("finished_at", "")
                if not finished.startswith(today_str):
                    continue
                jmode = jdata.get("mode", "karaoke")
                if jmode == "karaoke":
                    user_today_karaoke += 1
                elif jmode in ("subtitled", "both"):
                    user_today_subtitled += 1
            except (json.JSONDecodeError, OSError):
                continue
        if mode == "karaoke" and user_today_karaoke >= p.max_karaoke_per_day:
            raise HTTPException(429, f"Daily karaoke limit reached ({p.max_karaoke_per_day})")
        if mode in ("subtitled", "both") and user_today_subtitled >= p.max_subtitled_per_day:
            raise HTTPException(429, f"Daily subtitled limit reached ({p.max_subtitled_per_day})")

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
        "added_by_id": str(user.id) if user else None,
    }

    with _lock:
        _queue.append(item)
    _save_queue()
    _log_activity(db, user, "queue", {"title": title, "url": req.url, "mode": mode})

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
                entry["pause_requested"] = _pause_requested
            queue_with_progress.append(entry)
        return {
            "queue": queue_with_progress,
            "job": dict(_job) if _job else None,
            "disk_quota_exceeded": _disk_quota_exceeded,
        }


@app.delete("/api/queue/{item_id}")
def remove_from_queue(item_id: str):
    """Remove an item from the queue. If processing, request cancellation.
    For failed jobs, also cleans up intermediate files on disk."""
    global _cancel_requested, _job
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

    # Not processing — remove from queue and clean up files if failed
    with _lock:
        _queue[:] = [i for i in _queue if i["id"] != item_id]
    _save_queue()

    # Clean up job directory for failed/incomplete jobs
    job_dir = JOBS_DIR / item_id
    if job_dir.exists():
        job_json = job_dir / "job.json"
        if job_json.exists():
            try:
                data = json.loads(job_json.read_text())
            except (json.JSONDecodeError, OSError):
                data = {}
            # Only clean up if not a completed job (those live in the library)
            if data.get("status") != "done":
                shutil.rmtree(job_dir, ignore_errors=True)
                # If directory couldn't be deleted, mark as removed so recovery skips it
                if job_dir.exists() and job_json.exists():
                    data["status"] = "removed"
                    job_json.write_text(json.dumps(data, default=str))
                # Clear current job reference if it was the deleted one
                with _lock:
                    if _job is not None and _job.get("id") == item_id:
                        _job = None

    # Clean up any DB metadata
    try:
        _db = SessionLocal()
        _db.query(JobMetadata).filter(JobMetadata.job_id == item_id).delete()
        _db.commit()
        _db.close()
    except Exception:
        pass

    return {"deleted": True, "queue": _queue}


@app.post("/api/queue/{item_id}/request")
def convert_queue_to_wishlist(item_id: str, user: User = Depends(get_current_user),
                              db: Session = Depends(get_db)):
    """Move a queue item to the wishlist (song request). Only the user who added it can do this."""
    with _lock:
        item = next((i for i in _queue if i["id"] == item_id), None)
        if not item:
            raise HTTPException(404, "Item not found in queue")
        # Only the user who queued it (or admin) can convert it
        if item.get("added_by_id") != str(user.id) and not is_admin(user):
            raise HTTPException(403, "You can only move your own queue items to requests")
        # Don't allow converting a processing item
        if _job and _job.get("id") == item_id and _job.get("status") == "running":
            raise HTTPException(409, "Cannot move a processing item")
        _queue[:] = [i for i in _queue if i["id"] != item_id]
    _save_queue()

    # Create wishlist item from queue data
    wish = WishlistItem(
        user_id=user.id,
        url=item.get("url"),
        title=item.get("title", "Unknown"),
        artist=item.get("channel"),
        thumbnail=item.get("thumbnail"),
    )
    db.add(wish)
    db.commit()

    # Clean up job directory if it exists
    job_dir = JOBS_DIR / item_id
    if job_dir.exists():
        job_json = job_dir / "job.json"
        if job_json.exists():
            try:
                data = json.loads(job_json.read_text())
                if data.get("status") != "done":
                    shutil.rmtree(job_dir, ignore_errors=True)
            except (json.JSONDecodeError, OSError):
                pass

    return {"moved": True, "queue": _queue}


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
    """Toggle pause request for the currently processing item."""
    global _pause_requested
    with _lock:
        if not _job or _job.get("id") != item_id or _job.get("status") != "running":
            raise HTTPException(400, "Item is not currently processing")
    # Toggle: if already requesting pause, cancel it
    _pause_requested = not _pause_requested
    return {"pause_requested": _pause_requested}


@app.post("/api/queue/{item_id}/resume")
def resume_queue_item(item_id: str):
    """Resume a paused or failed queue item."""
    global _pause_requested, _job
    with _lock:
        item = next((i for i in _queue if i["id"] == item_id), None)
        if not item:
            raise HTTPException(404, "Item not found")
        if _job is not None and _job.get("status") == "running":
            raise HTTPException(409, "A job is already running")
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

    # Preload comment counts
    comment_rows = db.query(
        Comment.job_id, func.count()
    ).group_by(Comment.job_id).all()
    comment_counts: dict[str, int] = {job_id: count for job_id, count in comment_rows}

    # Preload job metadata from DB
    meta_rows = db.query(JobMetadata).all()
    meta_map: dict[str, JobMetadata] = {m.job_id: m for m in meta_rows}

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
        meta = meta_map.get(job_id)
        items.append({
            "id": job_id,
            "title": (meta.title if meta and meta.title else None) or data.get("title", "Unknown"),
            "artist": (meta.artist if meta and meta.artist else None) or data.get("artist", ""),
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
            "year": (meta.year if meta and meta.year else None),
            "language": data.get("language_detected"),
            "output_dir": str(d),
            "added_by": data.get("added_by"),
            "added_by_id": data.get("added_by_id"),
            "view_count": meta.view_count if meta else 0,
            "upvotes": votes["upvotes"],
            "downvotes": votes["downvotes"],
            "user_vote": user_votes.get(job_id, 0),
            "comment_count": comment_counts.get(job_id, 0),
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
        existing_mode = data.get("mode", "karaoke")
        # "both" covers both karaoke and subtitled
        if existing_mode != mode and not (existing_mode == "both" and mode in ("karaoke", "subtitled")):
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


@app.post("/api/jobs/{job_id}/view")
def record_view(job_id: str, user: User | None = Depends(get_optional_user),
                db: Session = Depends(get_db)):
    """Increment the view count for a job."""
    meta = db.query(JobMetadata).filter(JobMetadata.job_id == job_id).first()
    if not meta:
        meta = JobMetadata(job_id=job_id, view_count=1)
        db.add(meta)
    else:
        meta.view_count = (meta.view_count or 0) + 1
    db.commit()
    _log_activity(db, user, "view", {"job_id": job_id})
    return {"view_count": meta.view_count}


@app.get("/api/jobs/{job_id}/lyrics")
def get_lyrics(job_id: str):
    path = JOBS_DIR / job_id / "lyrics.json"
    if not path.exists():
        raise HTTPException(404, "Lyrics not found")
    return json.loads(path.read_text())


PROMPTS_PATH = JOBS_DIR / "prompts.json"
SETTINGS_PATH = JOBS_DIR / "settings.json"

# Default settings
_DEFAULT_SETTINGS = {
    "default_max_karaoke_per_day": 5,
    "default_max_subtitled_per_day": 15,
    "default_max_queue_length": 10,
    "default_can_download_karaoke": True,
    "default_can_download_instrumental": True,
    "default_can_download_vocals": True,
    "default_can_delete_library": False,
    "default_can_share_library": True,
    "feature_lyrics_correction": True,
    "feature_analysis": True,
    "feature_translation": True,
    "allowed_modes": ["karaoke", "subtitled", "both"],
    "feature_wishlist": True,
    "default_can_request_songs": True,
    "max_disk_space_mb": 0,
}


def _load_settings() -> dict:
    """Load admin settings from disk."""
    if SETTINGS_PATH.exists():
        try:
            stored = json.loads(SETTINGS_PATH.read_text())
            return {**_DEFAULT_SETTINGS, **stored}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULT_SETTINGS)


def _save_settings(settings: dict) -> None:
    """Save admin settings to disk."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def _load_prompts() -> dict:
    """Load custom prompts from disk."""
    if PROMPTS_PATH.exists():
        try:
            return json.loads(PROMPTS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_prompts(prompts: dict) -> None:
    """Save custom prompts to disk."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_PATH.write_text(json.dumps(prompts, indent=2))


@app.get("/api/admin/prompts")
def get_prompts(user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(403, "Admin only")
    from karaoke.analyze_lyrics import DEFAULT_ANALYSIS_PROMPT
    prompts = _load_prompts()
    return {
        "analysis_prompt": prompts.get("analysis_prompt", ""),
        "analysis_prompt_default": DEFAULT_ANALYSIS_PROMPT,
    }


@app.get("/api/settings/public")
def get_public_settings():
    """Return non-sensitive settings for the client UI."""
    s = _load_settings()
    return {
        "allowed_modes": s.get("allowed_modes", ["karaoke", "subtitled", "both"]),
        "feature_translation": s.get("feature_translation", True),
        "feature_analysis": s.get("feature_analysis", True),
        "feature_wishlist": s.get("feature_wishlist", True),
    }


@app.get("/api/admin/settings")
def get_settings(user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(403, "Admin only")
    return _load_settings()


@app.post("/api/admin/settings")
def save_settings(req: dict, user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(403, "Admin only")
    settings = _load_settings()
    for key in _DEFAULT_SETTINGS:
        if key in req:
            settings[key] = req[key]
    _save_settings(settings)
    return {"saved": True}


@app.post("/api/admin/prompts")
def save_prompts(req: dict, user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(403, "Admin only")
    prompts = _load_prompts()
    if "analysis_prompt" in req:
        prompts["analysis_prompt"] = req["analysis_prompt"]
    _save_prompts(prompts)
    return {"saved": True}


@app.post("/api/jobs/{job_id}/analysis/rerun")
def rerun_analysis(job_id: str, db: Session = Depends(get_db)):
    """Clear cached analysis so it regenerates on next fetch."""
    meta = db.query(JobMetadata).filter(JobMetadata.job_id == job_id).first()
    if meta:
        meta.analysis_text = None
        meta.analysis_song_info = None
        db.commit()
    return {"cleared": True}


@app.get("/api/jobs/{job_id}/analysis")
def get_analysis(job_id: str, db: Session = Depends(get_db)):
    """Return cached or generate line-by-line lyric analysis."""
    # Check DB cache first
    meta = db.query(JobMetadata).filter(JobMetadata.job_id == job_id).first()
    if meta and meta.analysis_text and len(meta.analysis_text) > 10:
        return {"song_info": meta.analysis_song_info or "", "analysis": meta.analysis_text}

    # Need lyrics to analyze
    job_dir = JOBS_DIR / job_id
    lyrics_path = job_dir / "lyrics.json"
    if not lyrics_path.exists():
        raise HTTPException(404, "Lyrics not found")

    # Get title/artist from DB first, fall back to job.json
    title = meta.title if meta else None
    artist = meta.artist if meta else None
    if not title or not artist:
        job_json = job_dir / "job.json"
        if job_json.exists():
            try:
                job_data = json.loads(job_json.read_text())
                title = title or job_data.get("title")
                artist = artist or job_data.get("artist") or job_data.get("channel")
            except (json.JSONDecodeError, OSError):
                pass

    # Build plain text lyrics from word-level data
    words = json.loads(lyrics_path.read_text())
    lyrics_text = " ".join(w["text"] for w in words)

    # Load custom prompt if set
    prompts = _load_prompts()
    custom_prompt = prompts.get("analysis_prompt") or None

    from karaoke.analyze_lyrics import analyze_lyrics
    result = analyze_lyrics(lyrics_text, title=title, artist=artist,
                            custom_prompt=custom_prompt)

    # Save to DB
    if not meta:
        meta = JobMetadata(job_id=job_id)
        db.add(meta)
    analysis_text = result.get("analysis", "")
    meta.analysis_text = analysis_text if analysis_text else None
    meta.analysis_song_info = result.get("song_info", "") or None

    # Save identified artist if not already set
    if not meta.artist and result.get("song_info"):
        info = result["song_info"]
        if " by " in info:
            parts = info.rsplit(" by ", 1)
            if len(parts) == 2:
                meta.artist = parts[1].strip().strip('"')

    db.commit()
    return result


@app.get("/api/jobs/{job_id}/subtitles/{lang_code}")
def get_subtitles_lang(job_id: str, lang_code: str):
    """Serve per-language SRT file."""
    # Sanitize lang_code to prevent path traversal
    if not re.match(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$", lang_code):
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


class UpdateJobRequest(BaseModel):
    title: str | None = None
    artist: str | None = None
    year: str | None = None


@app.patch("/api/jobs/{job_id}")
def update_job_metadata(job_id: str, req: UpdateJobRequest,
                        user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    """Update editable job metadata (title, artist)."""
    job_dir = JOBS_DIR / job_id
    job_json = job_dir / "job.json"
    if not job_json.exists():
        raise HTTPException(404, "Job not found")

    # Update DB
    meta = db.query(JobMetadata).filter(JobMetadata.job_id == job_id).first()
    if not meta:
        meta = JobMetadata(job_id=job_id)
        db.add(meta)
    if req.title is not None:
        meta.title = req.title
    if req.artist is not None:
        meta.artist = req.artist
    if req.year is not None:
        meta.year = req.year
    db.commit()

    # Also update job.json (pipeline reads it) and in-memory job
    try:
        data = json.loads(job_json.read_text())
        if req.title is not None:
            data["title"] = req.title
        if req.artist is not None:
            data["artist"] = req.artist
        job_json.write_text(json.dumps(data, default=str))
    except (json.JSONDecodeError, OSError):
        pass

    with _lock:
        if _job is not None and _job.get("id") == job_id:
            if req.title is not None:
                _job["title"] = req.title
            if req.artist is not None:
                _job["artist"] = req.artist

    return {"title": meta.title, "artist": meta.artist, "year": meta.year}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, user: User = Depends(get_current_user)):
    """Delete a job and all its files from disk."""
    global _job

    # Enforce delete permission (admins bypass)
    if not is_admin(user) and user.permissions and not user.permissions.can_delete_library:
        raise HTTPException(403, "You don't have permission to delete songs")

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

    # Clean up DB metadata (votes, comments, job metadata)
    try:
        _db = SessionLocal()
        _db.query(JobMetadata).filter(JobMetadata.job_id == job_id).delete()
        _db.query(Vote).filter(Vote.job_id == job_id).delete()
        _db.query(Comment).filter(Comment.job_id == job_id).delete()
        _db.commit()
        _db.close()
    except Exception:
        pass

    return {"deleted": True}


@app.get("/api/jobs/{job_id}/download/{filename}")
def download_file(job_id: str, filename: str,
                  user: User | None = Depends(get_optional_user)):
    # Allow specific files + subtitles_*.srt pattern
    static_allowed = {"karaoke.mp4", "instrumental.mp3", "vocals.mp3", "subtitles.srt"}
    is_subtitle = bool(re.match(r"^subtitles_[a-z]{2,3}(-[A-Za-z]{2,4})?\.srt$", filename))
    if filename not in static_allowed and not is_subtitle:
        raise HTTPException(400, "Invalid file")

    # Enforce download permissions (admins bypass)
    if user and not is_admin(user) and user.permissions:
        p = user.permissions
        if filename == "karaoke.mp4" and not p.can_download_karaoke:
            raise HTTPException(403, "You don't have permission to download karaoke files")
        if filename == "instrumental.mp3" and not p.can_download_instrumental:
            raise HTTPException(403, "You don't have permission to download instrumental files")
        if filename == "vocals.mp3" and not p.can_download_vocals:
            raise HTTPException(403, "You don't have permission to download vocal files")

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
        data["invites_sent"] = db.query(Invitation).filter(Invitation.inviter_id == u.id).count()
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
    max_invitations: int | None = None  # 0 = unlimited
    can_request_songs: bool | None = None


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


@app.delete("/api/admin/feedback/{feedback_id}")
def admin_delete_feedback(feedback_id: str, admin: User = Depends(require_admin),
                           db: Session = Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(404, "Feedback not found")
    # Delete screenshot file if exists
    if fb.screenshot_path:
        p = Path(fb.screenshot_path)
        if p.exists():
            p.unlink(missing_ok=True)
    db.delete(fb)
    db.commit()
    return {"deleted": True}


@app.get("/api/admin/feedback/{feedback_id}/screenshot")
def admin_get_screenshot(feedback_id: str, admin: User = Depends(require_admin),
                          db: Session = Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb or not fb.screenshot_path:
        raise HTTPException(404, "Screenshot not found")
    p = Path(fb.screenshot_path)
    if not p.exists():
        raise HTTPException(404, "Screenshot file missing")
    return FileResponse(p)


class AdminCreateUserRequest(BaseModel):
    email: str
    name: str
    password: str
    role: str = "user"


@app.post("/api/admin/users")
def admin_create_user(req: AdminCreateUserRequest, admin: User = Depends(require_admin),
                       db: Session = Depends(get_db)):
    if req.role not in ("user", "admin"):
        raise HTTPException(400, "Invalid role")
    if not req.email or not req.name or not req.password:
        raise HTTPException(400, "All fields are required")
    if db.query(User).filter(User.email == req.email.lower()).first():
        raise HTTPException(400, "Email already registered")
    user = create_user_with_permissions(db, email=req.email, name=req.name, password=req.password)
    if req.role != "user":
        user.role = req.role
        db.commit()
        db.refresh(user)
    return {"user": user.to_dict()}


@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: str, admin: User = Depends(require_admin),
                       db: Session = Depends(get_db)):
    if str(admin.id) == user_id:
        raise HTTPException(400, "Cannot delete yourself")
    target = db.query(User).filter(User.id == user_id).first()
    if not target:
        raise HTTPException(404, "User not found")
    db.delete(target)
    db.commit()
    return {"deleted": True}


@app.get("/api/admin/activity")
def admin_activity(admin: User = Depends(require_admin),
                   db: Session = Depends(get_db),
                   event_type: str | None = None,
                   limit: int = 100,
                   offset: int = 0):
    """Return recent activity log entries with optional filtering."""
    q = db.query(ActivityLog).order_by(ActivityLog.created_at.desc())
    if event_type:
        q = q.filter(ActivityLog.event_type == event_type)
    total = q.count()
    entries = q.offset(offset).limit(min(limit, 500)).all()
    return {
        "total": total,
        "entries": [
            {
                "id": str(e.id),
                "user_name": e.user_name,
                "event_type": e.event_type,
                "detail": json.loads(e.detail) if e.detail else None,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in entries
        ],
    }


@app.get("/api/admin/stats")
def admin_stats(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    from sqlalchemy import func

    user_count = db.query(func.count(User.id)).scalar()
    comment_count = db.query(func.count(Comment.id)).scalar()
    invitation_count = db.query(func.count(Invitation.id)).scalar()
    feedback_new = db.query(func.count(Feedback.id)).filter(Feedback.status == "new").scalar()

    # Library stats from disk
    total_songs = 0
    total_size_bytes = 0
    if JOBS_DIR.exists():
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
            if data.get("status") == "done":
                total_songs += 1
                for f in d.iterdir():
                    if f.is_file():
                        total_size_bytes += f.stat().st_size

    # Queue status
    with _lock:
        queue_len = len(_queue)
        running = _job is not None and _job.get("status") == "running"

    return {
        "users": user_count,
        "songs": total_songs,
        "storage_mb": round(total_size_bytes / (1024 * 1024), 1),
        "comments": comment_count,
        "invitations": invitation_count,
        "feedback_new": feedback_new,
        "queue_length": queue_len,
        "processing": running,
    }


@app.get("/api/admin/comments")
def admin_list_comments(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    comments = db.query(Comment).order_by(Comment.created_at.desc()).all()
    return {"comments": [c.to_dict() for c in comments]}


@app.delete("/api/admin/comments/{comment_id}")
def admin_delete_comment(comment_id: str, admin: User = Depends(require_admin),
                          db: Session = Depends(get_db)):
    comment = db.query(Comment).filter(Comment.id == comment_id).first()
    if not comment:
        raise HTTPException(404, "Comment not found")
    db.delete(comment)
    db.commit()
    return {"deleted": True}


@app.get("/api/admin/invitations")
def admin_list_invitations(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    invites = db.query(Invitation).order_by(Invitation.created_at.desc()).all()
    return {"invitations": [i.to_dict() for i in invites]}


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
            db.delete(existing)
            db.commit()
        else:
            existing.value = req.value
            db.commit()
    else:
        db.add(Vote(user_id=user.id, job_id=job_id, value=req.value))
        db.commit()
    _log_activity(db, user, "vote", {"job_id": job_id, "value": req.value})
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
# Wishlist API
# ---------------------------------------------------------------------------

class WishlistCreateRequest(BaseModel):
    url: str | None = None
    title: str | None = None
    artist: str | None = None
    note: str | None = None


@app.get("/api/wishlist")
def list_wishlist(user: User | None = Depends(get_optional_user),
                  db: Session = Depends(get_db)):
    """List open wishlist items with vote counts, sorted by votes desc."""
    if not _load_settings().get("feature_wishlist", True):
        raise HTTPException(403, "Wishlist feature is disabled")
    from sqlalchemy import func
    items = db.query(WishlistItem).filter(WishlistItem.status == "open").all()
    result = []
    for item in items:
        vote_count = db.query(func.count(WishlistVote.id)).filter(
            WishlistVote.wishlist_item_id == item.id).scalar() or 0
        user_voted = False
        if user:
            user_voted = db.query(WishlistVote).filter(
                WishlistVote.user_id == user.id,
                WishlistVote.wishlist_item_id == item.id).first() is not None
        d = item.to_dict()
        d["vote_count"] = vote_count
        d["user_voted"] = user_voted
        result.append(d)
    result.sort(key=lambda x: x["vote_count"], reverse=True)
    return {"items": result}


@app.post("/api/wishlist")
def create_wishlist_item(req: WishlistCreateRequest,
                         user: User = Depends(get_current_user),
                         db: Session = Depends(get_db)):
    """Create a new song request."""
    if not _load_settings().get("feature_wishlist", True):
        raise HTTPException(403, "Wishlist feature is disabled")
    if not is_admin(user) and user.permissions and not user.permissions.can_request_songs:
        raise HTTPException(403, "You don't have permission to request songs")

    title = (req.title or "").strip()
    url = (req.url or "").strip() or None
    artist = (req.artist or "").strip() or None
    note = (req.note or "").strip() or None
    thumbnail = None

    if url:
        try:
            meta = fetch_metadata(url)
            title = title or meta.get("title", "Unknown")
            artist = artist or meta.get("channel")
            thumbnail = meta.get("thumbnail")
        except Exception:
            pass

    if not title:
        raise HTTPException(400, "Title is required")

    item = WishlistItem(
        user_id=user.id, url=url, title=title, artist=artist,
        thumbnail=thumbnail, note=note,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    d = item.to_dict()
    d["vote_count"] = 0
    d["user_voted"] = False
    return d


@app.post("/api/wishlist/{item_id}/vote")
def toggle_wishlist_vote(item_id: str, user: User = Depends(get_current_user),
                         db: Session = Depends(get_db)):
    """Toggle upvote on a wishlist item."""
    item = db.query(WishlistItem).filter(WishlistItem.id == item_id).first()
    if not item:
        raise HTTPException(404, "Wishlist item not found")
    existing = db.query(WishlistVote).filter(
        WishlistVote.user_id == user.id,
        WishlistVote.wishlist_item_id == item.id).first()
    if existing:
        db.delete(existing)
        db.commit()
        voted = False
    else:
        db.add(WishlistVote(user_id=user.id, wishlist_item_id=item.id))
        db.commit()
        voted = True
    from sqlalchemy import func
    count = db.query(func.count(WishlistVote.id)).filter(
        WishlistVote.wishlist_item_id == item.id).scalar() or 0
    _log_activity(db, user, "vote", {"wishlist_item_id": str(item.id), "title": item.title, "voted": voted})
    return {"item_id": str(item.id), "vote_count": count, "user_voted": voted}


@app.get("/api/wishlist/preview")
def preview_wishlist_url(url: str, user: User = Depends(get_current_user)):
    """Fetch YouTube metadata for preview in request form."""
    try:
        meta = fetch_metadata(url)
        return {"title": meta.get("title"), "channel": meta.get("channel"),
                "thumbnail": meta.get("thumbnail")}
    except Exception:
        raise HTTPException(400, "Could not fetch metadata")


# --- Admin Wishlist ---

@app.get("/api/admin/wishlist")
def admin_list_wishlist(admin: User = Depends(require_admin),
                        db: Session = Depends(get_db)):
    from sqlalchemy import func
    items = db.query(WishlistItem).order_by(WishlistItem.created_at.desc()).all()
    result = []
    for item in items:
        vote_count = db.query(func.count(WishlistVote.id)).filter(
            WishlistVote.wishlist_item_id == item.id).scalar() or 0
        d = item.to_dict()
        d["vote_count"] = vote_count
        result.append(d)
    return {"items": result}


@app.post("/api/admin/wishlist/{item_id}/queue")
def admin_queue_wishlist(item_id: str, admin: User = Depends(require_admin),
                         db: Session = Depends(get_db)):
    """Convert a wishlist request into a production queue item."""
    item = db.query(WishlistItem).filter(WishlistItem.id == item_id).first()
    if not item:
        raise HTTPException(404, "Wishlist item not found")
    if not item.url:
        raise HTTPException(400, "Item has no URL — cannot queue for production")
    item.status = "queued"
    db.commit()
    req = QueueRequest(url=item.url, mode="karaoke", languages=[])
    return add_to_queue(req, admin, db)


@app.delete("/api/admin/wishlist/{item_id}")
def admin_delete_wishlist(item_id: str, admin: User = Depends(require_admin),
                          db: Session = Depends(get_db)):
    item = db.query(WishlistItem).filter(WishlistItem.id == item_id).first()
    if not item:
        raise HTTPException(404, "Wishlist item not found")
    db.delete(item)
    db.commit()
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Playlists API
# ---------------------------------------------------------------------------

class CreatePlaylistRequest(BaseModel):
    name: str


class AddToPlaylistRequest(BaseModel):
    job_id: str


@app.get("/api/playlists")
def list_playlists(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all playlists for the current user."""
    playlists = db.query(Playlist).filter(Playlist.user_id == user.id).order_by(Playlist.created_at.desc()).all()
    return {"playlists": [p.to_dict(include_items=True) for p in playlists]}


@app.post("/api/playlists")
def create_playlist(req: CreatePlaylistRequest, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    """Create a new playlist."""
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Playlist name is required")
    if len(name) > 100:
        raise HTTPException(400, "Playlist name too long")
    pl = Playlist(user_id=user.id, name=name)
    db.add(pl)
    db.commit()
    db.refresh(pl)
    return {"playlist": pl.to_dict()}


@app.delete("/api/playlists/{playlist_id}")
def delete_playlist(playlist_id: str, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    """Delete a playlist owned by the current user."""
    pl = db.query(Playlist).filter(Playlist.id == playlist_id, Playlist.user_id == user.id).first()
    if not pl:
        raise HTTPException(404, "Playlist not found")
    db.delete(pl)
    db.commit()
    return {"deleted": True}


@app.put("/api/playlists/{playlist_id}")
def rename_playlist(playlist_id: str, req: CreatePlaylistRequest,
                    user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Rename a playlist."""
    pl = db.query(Playlist).filter(Playlist.id == playlist_id, Playlist.user_id == user.id).first()
    if not pl:
        raise HTTPException(404, "Playlist not found")
    name = req.name.strip()
    if not name:
        raise HTTPException(400, "Playlist name is required")
    pl.name = name
    db.commit()
    return {"playlist": pl.to_dict(include_items=True)}


@app.post("/api/playlists/{playlist_id}/items")
def add_to_playlist(playlist_id: str, req: AddToPlaylistRequest,
                    user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Add a song to a playlist."""
    pl = db.query(Playlist).filter(Playlist.id == playlist_id, Playlist.user_id == user.id).first()
    if not pl:
        raise HTTPException(404, "Playlist not found")
    existing = db.query(PlaylistItem).filter(
        PlaylistItem.playlist_id == pl.id, PlaylistItem.job_id == req.job_id
    ).first()
    if existing:
        raise HTTPException(409, "Song already in playlist")
    max_pos = db.query(PlaylistItem.position).filter(
        PlaylistItem.playlist_id == pl.id
    ).order_by(PlaylistItem.position.desc()).first()
    position = (max_pos[0] + 1) if max_pos else 0
    item = PlaylistItem(playlist_id=pl.id, job_id=req.job_id, position=position)
    db.add(item)
    db.commit()
    db.refresh(pl)
    return {"playlist": pl.to_dict(include_items=True)}


@app.delete("/api/playlists/{playlist_id}/items/{job_id}")
def remove_from_playlist(playlist_id: str, job_id: str,
                         user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Remove a song from a playlist."""
    pl = db.query(Playlist).filter(Playlist.id == playlist_id, Playlist.user_id == user.id).first()
    if not pl:
        raise HTTPException(404, "Playlist not found")
    item = db.query(PlaylistItem).filter(
        PlaylistItem.playlist_id == pl.id, PlaylistItem.job_id == job_id
    ).first()
    if not item:
        raise HTTPException(404, "Song not in playlist")
    db.delete(item)
    db.commit()
    db.refresh(pl)
    return {"playlist": pl.to_dict(include_items=True)}


class ReorderPlaylistRequest(BaseModel):
    job_ids: list[str]  # ordered list of job IDs


@app.post("/api/playlists/{playlist_id}/reorder")
def reorder_playlist(playlist_id: str, req: ReorderPlaylistRequest,
                     user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Reorder songs in a playlist."""
    pl = db.query(Playlist).filter(Playlist.id == playlist_id, Playlist.user_id == user.id).first()
    if not pl:
        raise HTTPException(404, "Playlist not found")
    items_by_job = {item.job_id: item for item in pl.items}
    for pos, job_id in enumerate(req.job_ids):
        if job_id in items_by_job:
            items_by_job[job_id].position = pos
    db.commit()
    db.refresh(pl)
    return {"playlist": pl.to_dict(include_items=True)}


# ---------------------------------------------------------------------------
# Comments API
# ---------------------------------------------------------------------------

class CommentRequest(BaseModel):
    text: str


@app.get("/api/jobs/{job_id}/comments")
def get_comments(job_id: str, db: Session = Depends(get_db)):
    """Get all comments for a song."""
    comments = db.query(Comment).filter(Comment.job_id == job_id).order_by(Comment.created_at.asc()).all()
    return {"comments": [c.to_dict() for c in comments]}


@app.post("/api/jobs/{job_id}/comments")
def post_comment(job_id: str, req: CommentRequest, user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    """Post a comment on a song."""
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Comment text is required")
    if len(text) > 2000:
        raise HTTPException(400, "Comment too long (max 2000 characters)")
    comment = Comment(user_id=user.id, job_id=job_id, text=text)
    db.add(comment)
    db.commit()
    db.refresh(comment)
    _log_activity(db, user, "comment", {"job_id": job_id, "text": text[:100]})
    return {"comment": comment.to_dict()}


@app.delete("/api/comments/{comment_id}")
def delete_comment(comment_id: str, user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Delete own comment (or any if admin)."""
    comment = db.query(Comment).filter(Comment.id == comment_id).first()
    if not comment:
        raise HTTPException(404, "Comment not found")
    if str(comment.user_id) != str(user.id) and user.role != "admin":
        raise HTTPException(403, "Cannot delete this comment")
    db.delete(comment)
    db.commit()
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Invitations API
# ---------------------------------------------------------------------------

class InviteRequest(BaseModel):
    emails: list[str]


@app.post("/api/invitations")
def send_invitations(req: InviteRequest, user: User = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    """Send email invitations to friends."""
    import re
    # Enforce invite limit
    max_inv = user.permissions.max_invitations if user.permissions else 5
    if max_inv > 0:  # 0 = unlimited
        used = db.query(Invitation).filter(Invitation.inviter_id == user.id).count()
        remaining = max_inv - used
        if remaining <= 0:
            raise HTTPException(403, "You've used all your invitations")
    else:
        remaining = None  # unlimited

    sent = []
    for email in req.emails[:10]:  # max 10 at a time
        if remaining is not None and len(sent) >= remaining:
            break
        email = email.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            continue
        # Skip if already a user
        if db.query(User).filter(User.email == email).first():
            continue
        # Skip if already invited by this user
        existing = db.query(Invitation).filter(
            Invitation.inviter_id == user.id, Invitation.email == email
        ).first()
        if existing:
            continue
        inv = Invitation(inviter_id=user.id, email=email)
        db.add(inv)
        emailed = send_invite_email(email, user.name or "A friend", inv.token)
        sent.append({"email": email, "token": inv.token, "link": f"/login?invite={inv.token}", "emailed": emailed})
    db.commit()
    for s in sent:
        _log_activity(db, user, "invite", {"email": s["email"]})
    total_used = db.query(Invitation).filter(Invitation.inviter_id == user.id).count()
    return {
        "sent": sent,
        "count": len(sent),
        "remaining": (max_inv - total_used) if max_inv > 0 else None,
    }


@app.get("/api/invitations")
def list_invitations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List invitations sent by the current user."""
    invites = db.query(Invitation).filter(Invitation.inviter_id == user.id).order_by(Invitation.created_at.desc()).all()
    max_inv = user.permissions.max_invitations if user.permissions else 5
    total_used = db.query(Invitation).filter(Invitation.inviter_id == user.id).count()
    return {
        "invitations": [i.to_dict() for i in invites],
        "remaining": (max_inv - total_used) if max_inv > 0 else None,
        "max_invitations": max_inv,
    }


@app.get("/api/invitations/{token}/validate")
def validate_invitation(token: str, db: Session = Depends(get_db)):
    """Public endpoint to check if an invite token is valid."""
    inv = db.query(Invitation).filter(Invitation.token == token).first()
    if not inv or not inv.is_valid():
        return {"valid": False}
    return {
        "valid": True,
        "email": inv.email,
        "inviter_name": inv.inviter.name if inv.inviter else None,
    }


@app.get("/api/community/network")
def community_network(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Return the invitation tree for network map visualization."""
    users = db.query(User).all()
    return {"nodes": [
        {
            "id": str(u.id),
            "name": u.name,
            "invited_by_id": str(u.invited_by_id) if u.invited_by_id else None,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in users
    ]}


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
