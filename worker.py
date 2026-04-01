"""Standalone karaoke production worker.

A stateless HTTP service that accepts jobs from the Baretraks app,
runs the pipeline (download, separate, transcribe, compose), uploads
results to R2, and reports back to the app via callbacks.

Usage:
    python worker.py                           # default port 8001
    WORKER_PORT=8002 python worker.py          # custom port
"""

from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
import platform
import secrets
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

# Pipeline imports — these are the core processing modules
from karaoke.compose import compose
from karaoke.download import download, download_audio
from karaoke.separate import separate
from karaoke.subtitles import build_ass, build_srt
from karaoke.transcribe import transcribe
from karaoke.translate import translate_srt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORKER_NAME = os.getenv("WORKER_NAME", platform.node() or "worker")
WORKER_API_KEY = os.getenv("WORKER_API_KEY", "")
WORKER_PORT = int(os.getenv("WORKER_PORT", "8001"))
DEVICE = os.getenv("DEVICE", "auto").lower()
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEMUCS_MODEL = os.getenv("DEMUCS_MODEL", "htdemucs_ft")
WORK_DIR = Path(os.getenv("WORKER_WORK_DIR", "worker_output"))


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Baretraks Worker")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_current_job: dict | None = None  # {job_id, step, step_name, step_progress, ...}
_cancel_requested = False

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def verify_api_key(request: Request):
    if not WORKER_API_KEY:
        return  # no auth configured
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {WORKER_API_KEY}":
        raise HTTPException(401, "Invalid API key")


# ---------------------------------------------------------------------------
# R2 Upload
# ---------------------------------------------------------------------------

def _upload_via_presigned(local_path: Path, presigned_url: str) -> int:
    """Upload a file using a presigned PUT URL. Returns file size in bytes."""
    content_types = {
        ".mp4": "video/mp4",
        ".mp3": "audio/mpeg",
        ".json": "application/json",
        ".srt": "text/plain; charset=utf-8",
    }
    ct = content_types.get(local_path.suffix, "application/octet-stream")
    size = local_path.stat().st_size
    with open(local_path, "rb") as f:
        resp = httpx.put(presigned_url, content=f, headers={"Content-Type": ct},
                         timeout=300)
    if resp.status_code >= 400:
        logger.error("Presigned upload failed (%d): %s", resp.status_code, resp.text[:200])
        return 0
    logger.info("Uploaded %s (%d bytes)", local_path.name, size)
    return size


def _request_upload_urls(callback_url: str, callback_key: str,
                         job_id: str, filenames: list[str]) -> dict[str, str]:
    """Request presigned upload URLs from the app."""
    headers = {"Content-Type": "application/json"}
    if callback_key:
        headers["Authorization"] = f"Bearer {callback_key}"
    try:
        resp = httpx.post(
            f"{callback_url}/api/worker/upload-urls",
            json={"job_id": job_id, "filenames": filenames},
            headers=headers, timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("urls", {})
        logger.warning("Failed to get upload URLs: %d", resp.status_code)
    except Exception as e:
        logger.error("Failed to request upload URLs: %s", e)
    return {}


# ---------------------------------------------------------------------------
# Callback helpers
# ---------------------------------------------------------------------------

_last_progress_time = 0.0


def _send_callback(callback_url: str, callback_key: str, endpoint: str, data: dict,
                   retries: int = 0) -> bool:
    """POST data to the app's callback endpoint. Returns True on success."""
    url = f"{callback_url}{endpoint}"
    headers = {"Content-Type": "application/json"}
    if callback_key:
        headers["Authorization"] = f"Bearer {callback_key}"
    attempts = 1 + retries
    for attempt in range(attempts):
        try:
            resp = httpx.post(url, json=data, headers=headers, timeout=15)
            if resp.status_code < 400:
                return True
            logger.warning("Callback %s returned %d: %s", endpoint, resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Callback %s failed (attempt %d/%d): %s", endpoint, attempt + 1, attempts, e)
        if attempt < attempts - 1:
            time.sleep(min(10 * (attempt + 1), 60))  # backoff: 10s, 20s, 30s...
    return False


def _report_progress(callback_url: str, callback_key: str, job_id: str, **kwargs):
    """Send a debounced progress update (max 1/sec)."""
    global _last_progress_time
    now = time.monotonic()
    # Always send for step changes, debounce for progress-only updates
    is_step_change = "step" in kwargs or "status" in kwargs
    if not is_step_change and (now - _last_progress_time) < 1.0:
        return
    _last_progress_time = now

    with _lock:
        if _current_job:
            _current_job.update(kwargs)

    data = {"job_id": job_id, **kwargs}
    _send_callback(callback_url, callback_key, "/api/worker/progress", data)


# ---------------------------------------------------------------------------
# Pipeline helpers (extracted from server.py)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_audio_duration(audio_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _convert_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame",
         "-b:a", "192k", str(mp3_path)],
        check=True, capture_output=True,
    )


def _words_to_segments(words):
    from karaoke.transcribe import Segment, Word as TWord
    if not words:
        return []
    SEGMENT_GAP = 1.0
    segments = []
    current = [words[0]]
    for i in range(1, len(words)):
        gap = words[i].start - words[i - 1].end
        if gap >= SEGMENT_GAP:
            segments.append(Segment(start=current[0].start, end=current[-1].end, words=current))
            current = [words[i]]
        else:
            current.append(words[i])
    if current:
        segments.append(Segment(start=current[0].start, end=current[-1].end, words=current))
    return segments


def _check_cancel() -> bool:
    return _cancel_requested


def _cleanup_work_dir(work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _run_karaoke_pipeline(job_id: str, url: str, output_dir: Path,
                          callback_url: str, callback_key: str, settings: dict):
    """Run the karaoke pipeline: download → separate → transcribe → subtitles → compose."""
    device = DEVICE
    demucs_model = settings.get("demucs_model", DEMUCS_MODEL)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path = work_dir / "video.mp4"
    audio_path = work_dir / "audio.wav"
    instrumental_mp3 = output_dir / "instrumental.mp3"
    vocals_mp3 = output_dir / "vocals.mp3"
    subtitles_path = work_dir / "karaoke.ass"
    output_path = output_dir / "karaoke.mp4"
    step_durations = {}

    # Step 1: Download
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=1, step_name="Downloading", step_progress=0.0)

    def _dl_progress(pct):
        _report_progress(callback_url, callback_key, job_id, step_progress=pct)

    video_path, audio_path = download(url, work_dir, progress_callback=_dl_progress)
    step_durations["1"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    audio_duration = _get_audio_duration(audio_path)
    _report_progress(callback_url, callback_key, job_id, audio_duration=audio_duration)

    # Step 2: Separate
    if _check_cancel(): return
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=2, step_name="Separating vocals", step_progress=0.0)
    instrumental_wav, vocals_wav = separate(audio_path, work_dir / "demucs",
                                            device=device, model=demucs_model)
    _convert_to_mp3(instrumental_wav, instrumental_mp3)
    _convert_to_mp3(vocals_wav, vocals_mp3)
    instrumental_wav.unlink(missing_ok=True)
    vocals_wav.unlink(missing_ok=True)
    step_durations["2"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    # Step 3: Transcribe
    if _check_cancel(): return
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=3, step_name="Transcribing lyrics", step_progress=0.0)
    transcribe_path = audio_path if audio_path.exists() else vocals_mp3
    segments, detected_lang = transcribe(transcribe_path, device=device)
    if audio_path.exists() and audio_path != vocals_mp3:
        audio_path.unlink(missing_ok=True)
    _report_progress(callback_url, callback_key, job_id, language_detected=detected_lang)

    words_list = []
    for seg in segments:
        for w in seg.words:
            words_list.append({"text": w.text, "start": w.start, "end": w.end})

    # Correct lyrics
    if settings.get("feature_lyrics_correction", True):
        _report_progress(callback_url, callback_key, job_id,
                         step_name="Correcting lyrics", step_progress=0.8)
        try:
            from karaoke.correct_lyrics import correct_lyrics
            with _lock:
                title = _current_job.get("title") if _current_job else None
                channel = _current_job.get("channel") if _current_job else None
            correction = correct_lyrics(words_list, title=title, artist=channel)
            words_list = correction["words"]
            if correction.get("identified_artist"):
                _report_progress(callback_url, callback_key, job_id,
                                 artist=correction["identified_artist"])
        except Exception:
            pass

    (output_dir / "lyrics.json").write_text(json.dumps(words_list))

    from karaoke.transcribe import Word as TWord
    corrected_words = [TWord(text=w["text"], start=w["start"], end=w["end"]) for w in words_list]
    segments = _words_to_segments(corrected_words)
    step_durations["3"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    # Step 4: Build subtitles
    if _check_cancel(): return
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=4, step_name="Building subtitles", step_progress=0.0)
    build_ass(segments, subtitles_path)
    step_durations["4"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    # Step 5: Compose
    if _check_cancel(): return
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=5, step_name="Composing video", step_progress=0.0)
    if output_path.exists():
        output_path.unlink()

    def _compose_progress(pct):
        _report_progress(callback_url, callback_key, job_id, step_progress=pct)

    compose(video_path, instrumental_mp3, subtitles_path, output_path,
            duration=audio_duration, progress_callback=_compose_progress)
    step_durations["5"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    return {
        "audio_duration": audio_duration,
        "language_detected": detected_lang,
        "step_durations": step_durations,
        "words_list": words_list,
    }


def _run_subtitled_pipeline(job_id: str, url: str, output_dir: Path,
                            languages: list[str], callback_url: str,
                            callback_key: str, settings: dict):
    """Run subtitle-only pipeline: download audio → transcribe → translate."""
    device = DEVICE
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_path = work_dir / "audio.wav"
    step_durations = {}

    # Step 1: Download audio
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=1, step_name="Downloading audio", step_progress=0.0)

    def _dl_progress(pct):
        _report_progress(callback_url, callback_key, job_id, step_progress=pct)

    audio_path = download_audio(url, work_dir, progress_callback=_dl_progress)
    step_durations["1"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    audio_duration = _get_audio_duration(audio_path)
    _report_progress(callback_url, callback_key, job_id, audio_duration=audio_duration)

    # Step 2: Transcribe
    if _check_cancel(): return None
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=2, step_name="Transcribing", step_progress=0.0)
    segments, detected_lang = transcribe(audio_path, device=device)
    _report_progress(callback_url, callback_key, job_id, language_detected=detected_lang)

    words_list = []
    for seg in segments:
        for w in seg.words:
            words_list.append({"text": w.text, "start": w.start, "end": w.end})
    (output_dir / "lyrics.json").write_text(json.dumps(words_list))

    # Build source SRT for translation
    source_srt_path = output_dir / "subtitles_source.srt"
    build_srt(segments, source_srt_path)
    source_srt_text = source_srt_path.read_text()
    source_srt_path.unlink(missing_ok=True)
    step_durations["2"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    # Step 3: Translate
    if _check_cancel(): return None
    srt_data = {}
    if languages and settings.get("feature_translation", True):
        t0 = time.monotonic()
        max_langs = settings.get("max_subtitle_languages", 3)
        target_langs = languages[:max_langs]
        for i, lang in enumerate(target_langs):
            _report_progress(callback_url, callback_key, job_id,
                             step=3, step_name=f"Translating ({lang})",
                             step_progress=i / len(target_langs))
            try:
                translated_srt = translate_srt(source_srt_text, lang,
                                                detected_language=detected_lang)
                srt_name = f"subtitles_{lang}.srt"
                (output_dir / srt_name).write_text(translated_srt)
                srt_data[lang] = translated_srt
            except Exception as e:
                logger.error("Translation to %s failed: %s", lang, e)
        step_durations["3"] = time.monotonic() - t0
        _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    return {
        "audio_duration": audio_duration,
        "language_detected": detected_lang,
        "step_durations": step_durations,
        "words_list": words_list,
        "srt_data": srt_data,
    }


def _run_combined_pipeline(job_id: str, url: str, output_dir: Path,
                           languages: list[str], callback_url: str,
                           callback_key: str, settings: dict):
    """Run combined karaoke + subtitles pipeline."""
    device = DEVICE
    demucs_model = settings.get("demucs_model", DEMUCS_MODEL)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path = work_dir / "video.mp4"
    audio_path = work_dir / "audio.wav"
    instrumental_mp3 = output_dir / "instrumental.mp3"
    vocals_mp3 = output_dir / "vocals.mp3"
    subtitles_path = work_dir / "karaoke.ass"
    output_path = output_dir / "karaoke.mp4"
    step_durations = {}

    # Step 1: Download
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=1, step_name="Downloading", step_progress=0.0)

    def _dl_progress(pct):
        _report_progress(callback_url, callback_key, job_id, step_progress=pct)

    video_path, audio_path = download(url, work_dir, progress_callback=_dl_progress)
    step_durations["1"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    audio_duration = _get_audio_duration(audio_path)
    _report_progress(callback_url, callback_key, job_id, audio_duration=audio_duration)

    # Step 2: Separate
    if _check_cancel(): return None
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=2, step_name="Separating vocals", step_progress=0.0)
    instrumental_wav, vocals_wav = separate(audio_path, work_dir / "demucs",
                                            device=device, model=demucs_model)
    _convert_to_mp3(instrumental_wav, instrumental_mp3)
    _convert_to_mp3(vocals_wav, vocals_mp3)
    instrumental_wav.unlink(missing_ok=True)
    vocals_wav.unlink(missing_ok=True)
    step_durations["2"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    # Step 3: Transcribe
    if _check_cancel(): return None
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=3, step_name="Transcribing lyrics", step_progress=0.0)
    transcribe_path = audio_path if audio_path.exists() else vocals_mp3
    segments, detected_lang = transcribe(transcribe_path, device=device)
    if audio_path.exists() and audio_path != vocals_mp3:
        audio_path.unlink(missing_ok=True)
    _report_progress(callback_url, callback_key, job_id, language_detected=detected_lang)

    words_list = []
    for seg in segments:
        for w in seg.words:
            words_list.append({"text": w.text, "start": w.start, "end": w.end})

    if settings.get("feature_lyrics_correction", True):
        _report_progress(callback_url, callback_key, job_id,
                         step_name="Correcting lyrics", step_progress=0.8)
        try:
            from karaoke.correct_lyrics import correct_lyrics
            with _lock:
                title = _current_job.get("title") if _current_job else None
                channel = _current_job.get("channel") if _current_job else None
            correction = correct_lyrics(words_list, title=title, artist=channel)
            words_list = correction["words"]
            if correction.get("identified_artist"):
                _report_progress(callback_url, callback_key, job_id,
                                 artist=correction["identified_artist"])
        except Exception:
            pass

    (output_dir / "lyrics.json").write_text(json.dumps(words_list))

    from karaoke.transcribe import Word as TWord
    corrected_words = [TWord(text=w["text"], start=w["start"], end=w["end"]) for w in words_list]
    segments = _words_to_segments(corrected_words)
    step_durations["3"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    # Step 4: Build subtitles
    if _check_cancel(): return None
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=4, step_name="Building subtitles", step_progress=0.0)
    build_ass(segments, subtitles_path)
    step_durations["4"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    # Step 5: Compose
    if _check_cancel(): return None
    t0 = time.monotonic()
    _report_progress(callback_url, callback_key, job_id,
                     step=5, step_name="Composing video", step_progress=0.0)
    if output_path.exists():
        output_path.unlink()

    def _compose_progress(pct):
        _report_progress(callback_url, callback_key, job_id, step_progress=pct)

    compose(video_path, instrumental_mp3, subtitles_path, output_path,
            duration=audio_duration, progress_callback=_compose_progress)
    step_durations["5"] = time.monotonic() - t0
    _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    # Step 6: Translate
    srt_data = {}
    if languages and settings.get("feature_translation", True):
        t0 = time.monotonic()
        source_srt_path = output_dir / "subtitles_source.srt"
        build_srt(segments, source_srt_path)
        source_srt_text = source_srt_path.read_text()
        source_srt_path.unlink(missing_ok=True)

        max_langs = settings.get("max_subtitle_languages", 3)
        target_langs = languages[:max_langs]
        for i, lang in enumerate(target_langs):
            if _check_cancel(): return None
            _report_progress(callback_url, callback_key, job_id,
                             step=6, step_name=f"Translating ({lang})",
                             step_progress=i / len(target_langs))
            try:
                translated_srt = translate_srt(source_srt_text, lang,
                                                detected_language=detected_lang)
                srt_name = f"subtitles_{lang}.srt"
                (output_dir / srt_name).write_text(translated_srt)
                srt_data[lang] = translated_srt
            except Exception as e:
                logger.error("Translation to %s failed: %s", lang, e)
        step_durations["6"] = time.monotonic() - t0
        _report_progress(callback_url, callback_key, job_id, step_progress=1.0)

    return {
        "audio_duration": audio_duration,
        "language_detected": detected_lang,
        "step_durations": step_durations,
        "words_list": words_list,
        "srt_data": srt_data,
    }


# ---------------------------------------------------------------------------
# Job execution thread
# ---------------------------------------------------------------------------

def _execute_job(job: dict):
    """Run the pipeline, upload results to R2, and report completion."""
    global _current_job, _cancel_requested
    _cancel_requested = False

    job_id = job["job_id"]
    url = job["url"]
    mode = job["mode"]
    languages = job.get("languages", [])
    callback_url = job["callback_url"]
    callback_key = job.get("callback_key", "")
    r2_prefix = job.get("r2_prefix", f"jobs/{job_id}")
    settings = job.get("settings", {})

    output_dir = WORK_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with _lock:
        _current_job = {
            "job_id": job_id,
            "url": url,
            "mode": mode,
            "title": job.get("title"),
            "channel": job.get("channel"),
            "step": 0,
            "step_name": "Starting",
            "step_progress": 0.0,
            "status": "running",
        }

    try:
        _report_progress(callback_url, callback_key, job_id,
                         status="running", step=0, step_name="Starting")

        if mode == "subtitled":
            result = _run_subtitled_pipeline(
                job_id, url, output_dir, languages,
                callback_url, callback_key, settings)
        elif mode == "both":
            result = _run_combined_pipeline(
                job_id, url, output_dir, languages,
                callback_url, callback_key, settings)
        else:
            result = _run_karaoke_pipeline(
                job_id, url, output_dir,
                callback_url, callback_key, settings)

        if result is None:
            # Cancelled
            _send_callback(callback_url, callback_key, "/api/worker/failed", {
                "job_id": job_id,
                "error": "Cancelled by user",
            })
            return

        # Collect files to upload
        upload_files = {}
        for fname in ["karaoke.mp4", "instrumental.mp3", "vocals.mp3", "lyrics.json"]:
            fpath = output_dir / fname
            if fpath.exists():
                upload_files[fname] = fpath
        for srt in output_dir.glob("subtitles_*.srt"):
            upload_files[srt.name] = srt

        # Request presigned upload URLs from the app
        r2_keys = []
        total_size = 0
        if upload_files:
            urls = _request_upload_urls(callback_url, callback_key,
                                        job_id, list(upload_files.keys()))
            for fname, fpath in upload_files.items():
                presigned_url = urls.get(fname)
                if presigned_url:
                    size = _upload_via_presigned(fpath, presigned_url)
                    total_size += size
                    r2_keys.append(f"{r2_prefix}/{fname}")
                else:
                    logger.warning("No presigned URL for %s — skipping", fname)

        # Build completion payload
        completion_data = {
            "job_id": job_id,
            "mode": mode,
            "audio_duration": result.get("audio_duration"),
            "language_detected": result.get("language_detected"),
            "step_durations": result.get("step_durations"),
            "lyrics": json.dumps(result.get("words_list", [])),
            "subtitles": json.dumps(result.get("srt_data", {})),
            "r2_keys": r2_keys,
            "file_size_bytes": total_size,
            "finished_at": _now_iso(),
        }

        # Save completion manifest locally in case callback fails
        manifest_path = WORK_DIR / f"{job_id}.manifest.json"
        manifest_path.write_text(json.dumps({
            **completion_data,
            "callback_url": callback_url,
            "callback_key": callback_key,
        }))

        # Send completion callback with retries
        delivered = _send_callback(callback_url, callback_key,
                                   "/api/worker/complete", completion_data, retries=5)
        if delivered:
            manifest_path.unlink(missing_ok=True)
            _cleanup_work_dir(output_dir)
            logger.info("Job %s completed and delivered", job_id)
        else:
            logger.error("Job %s completed but callback failed — manifest saved at %s",
                         job_id, manifest_path)
            # Keep output_dir so files can be re-uploaded if needed

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)
        _send_callback(callback_url, callback_key, "/api/worker/failed", {
            "job_id": job_id,
            "error": str(e),
        }, retries=3)
        _cleanup_work_dir(output_dir)
    finally:
        with _lock:
            _current_job = None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    job_id: str
    url: str
    mode: str = "karaoke"
    languages: list[str] = []
    callback_url: str
    callback_key: str = ""
    r2_prefix: str = ""
    settings: dict = {}
    title: str | None = None
    channel: str | None = None


@app.get("/worker/status")
def get_status(_=Depends(verify_api_key)):
    with _lock:
        if _current_job:
            return {
                "name": WORKER_NAME,
                "device": DEVICE,
                "status": "busy",
                "current_job": {
                    "job_id": _current_job.get("job_id"),
                    "step": _current_job.get("step"),
                    "step_name": _current_job.get("step_name"),
                    "step_progress": _current_job.get("step_progress"),
                },
            }
        return {
            "name": WORKER_NAME,
            "device": DEVICE,
            "status": "idle",
            "current_job": None,
        }


@app.post("/worker/jobs")
def accept_job(req: JobRequest, _=Depends(verify_api_key)):
    with _lock:
        if _current_job is not None:
            raise HTTPException(409, "Worker is busy")

    logger.info("Accepted job %s: %s (%s)", req.job_id, req.url, req.mode)
    thread = threading.Thread(target=_execute_job, args=(req.model_dump(),), daemon=True)
    thread.start()
    return {"accepted": True, "worker": WORKER_NAME}


@app.post("/worker/jobs/{job_id}/cancel")
def cancel_job(job_id: str, _=Depends(verify_api_key)):
    global _cancel_requested
    with _lock:
        if not _current_job or _current_job.get("job_id") != job_id:
            raise HTTPException(404, "Job not found on this worker")
    _cancel_requested = True
    return {"cancelled": True}


# ---------------------------------------------------------------------------
# Startup: retry undelivered completions
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _retry_undelivered():
    """On startup, check for manifest files from jobs that completed but whose
    callbacks failed. Retry delivering them to the app."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for manifest_path in WORK_DIR.glob("*.manifest.json"):
        try:
            data = json.loads(manifest_path.read_text())
            callback_url = data.pop("callback_url", "")
            callback_key = data.pop("callback_key", "")
            if not callback_url:
                continue
            logger.info("Retrying delivery for job %s", data.get("job_id"))
            delivered = _send_callback(callback_url, callback_key,
                                       "/api/worker/complete", data, retries=3)
            if delivered:
                manifest_path.unlink(missing_ok=True)
                # Clean up output dir if it still exists
                job_dir = WORK_DIR / data.get("job_id", "")
                if job_dir.exists():
                    _cleanup_work_dir(job_dir)
                logger.info("Successfully delivered job %s on retry", data.get("job_id"))
            else:
                logger.warning("Still cannot deliver job %s — manifest kept", data.get("job_id"))
        except Exception as e:
            logger.error("Error retrying manifest %s: %s", manifest_path.name, e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting worker '%s' on port %d (device=%s)", WORKER_NAME, WORKER_PORT, DEVICE)
    uvicorn.run(app, host="0.0.0.0", port=WORKER_PORT)
