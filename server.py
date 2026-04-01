"""FastAPI web server for the karaoke pipeline."""

from dotenv import load_dotenv
load_dotenv()

import json
import re
import secrets
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import (
    create_token, create_user_with_permissions, get_current_user,
    get_optional_user, hash_password, is_admin, require_admin, verify_password,
)
from database import get_db, get_session
from models import ActivityLog, AppConfig, Comment, Feedback, Invitation, JobMetadata, Playlist, PlaylistItem, User, UserPermissions, Vote, WishlistItem, WishlistVote
from storage import storage

import logging
import os
import shutil

from karaoke.download import fetch_metadata
JOBS_DIR = Path("output/jobs")


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
    try:
        import resend
        resend.api_key = RESEND_API_KEY
        link = f"{BASE_URL}/login?invite={token}"
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

app = FastAPI()

# ---------------------------------------------------------------------------
# Global job state & worker management
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_active_jobs: dict[str, dict] = {}  # job_id -> progress dict (from worker callbacks)
_queue: list[dict] = []  # production queue
_worker_states: dict[str, dict] = {}  # worker_id -> {status, last_seen, current_job, ...}


def _db_config_get(key: str, default: dict | list | None = None):
    """Read a JSON config value from the app_config table."""
    db = get_session()
    try:
        row = db.query(AppConfig).filter(AppConfig.key == key).first()
        if row:
            return json.loads(row.value)
        return default if default is not None else {}
    except Exception:
        return default if default is not None else {}
    finally:
        db.close()


def _db_config_set(key: str, value) -> None:
    """Write a JSON config value to the app_config table."""
    db = get_session()
    try:
        row = db.query(AppConfig).filter(AppConfig.key == key).first()
        serialized = json.dumps(value, default=str)
        if row:
            row.value = serialized
        else:
            db.add(AppConfig(key=key, value=serialized))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def _save_queue() -> None:
    """Persist queue to DB."""
    _db_config_set("queue", _queue)


def _load_queue() -> None:
    """Load queue from DB."""
    global _queue
    _queue = _db_config_get("queue", [])




# ---------------------------------------------------------------------------
# Worker management
# ---------------------------------------------------------------------------

def _load_workers() -> list[dict]:
    """Load configured workers from DB."""
    return _db_config_get("workers", [])


def _save_workers(workers: list[dict]) -> None:
    _db_config_set("workers", workers)


def _dispatch_next_jobs() -> None:
    """Dispatch queued jobs to idle workers."""
    workers = _load_workers()
    if not workers:
        return

    with _lock:
        queued_items = [item for item in _queue if item.get("status") == "queued"]
        if not queued_items:
            return

    settings = _load_settings()

    for item in queued_items:
        # Find an idle, enabled worker (treat unknown/new workers as potentially idle)
        idle_worker = None
        for w in workers:
            if not w.get("enabled", True):
                continue
            wid = w["id"]
            state = _worker_states.get(wid, {})
            status = state.get("status", "unknown")
            if status in ("idle", "unknown"):
                idle_worker = w
                break

        if not idle_worker:
            break  # no idle workers available

        job_id = item["id"]
        callback_url = settings.get("base_url", BASE_URL)

        payload = {
            "job_id": job_id,
            "url": item["url"],
            "mode": item.get("mode", "karaoke"),
            "languages": item.get("languages", []),
            "callback_url": callback_url,
            "callback_key": idle_worker.get("api_key", ""),
            "r2_prefix": f"jobs/{job_id}",
            "title": item.get("title"),
            "channel": item.get("channel"),
            "settings": {
                "feature_lyrics_correction": settings.get("feature_lyrics_correction", True),
                "feature_translation": settings.get("feature_translation", True),
                "feature_analysis": settings.get("feature_analysis", True),
                "demucs_model": settings.get("demucs_model", "htdemucs_ft"),
                "max_subtitle_languages": settings.get("max_subtitle_languages", 3),
            },
        }

        try:
            headers = {"Content-Type": "application/json"}
            if idle_worker.get("api_key"):
                headers["Authorization"] = f"Bearer {idle_worker['api_key']}"
            resp = httpx.post(
                f"{idle_worker['url']}/worker/jobs",
                json=payload, headers=headers, timeout=10,
            )
            if resp.status_code == 200:
                with _lock:
                    item["status"] = "processing"
                    item["worker_id"] = idle_worker["id"]
                    item["dispatched_at"] = _now_iso()
                    _active_jobs[job_id] = {
                        "id": job_id,
                        "status": "running",
                        "step": 0,
                        "step_name": "Starting",
                        "step_progress": 0.0,
                        "worker_id": idle_worker["id"],
                        "worker_name": idle_worker.get("name", ""),
                    }
                _worker_states.setdefault(idle_worker["id"], {})["status"] = "busy"
                _save_queue()
                logger.info("Dispatched job %s to worker %s", job_id, idle_worker.get("name"))
            elif resp.status_code == 409:
                # Worker is busy — mark it so we skip it for next items
                _worker_states.setdefault(idle_worker["id"], {})["status"] = "busy"
                logger.info("Worker %s is busy", idle_worker.get("name"))
            else:
                logger.warning("Worker %s rejected job: %d %s",
                               idle_worker.get("name"), resp.status_code, resp.text[:100])
        except Exception as e:
            logger.warning("Failed to dispatch to worker %s: %s", idle_worker.get("name"), e)


def _on_job_completed(job_id: str, data: dict) -> None:
    """Called when a worker reports job completion. Saves metadata to DB and triggers post-processing."""
    # Find queue item for metadata
    queue_item = None
    with _lock:
        for item in _queue:
            if item.get("id") == job_id:
                queue_item = dict(item)
                break
        # Remove from queue and active jobs
        _queue[:] = [item for item in _queue if item.get("id") != job_id]
        _active_jobs.pop(job_id, None)
    _save_queue()

    # Save to JobMetadata
    try:
        db = get_session()
        meta = db.query(JobMetadata).filter(JobMetadata.job_id == job_id).first()
        if not meta:
            meta = JobMetadata(job_id=job_id)
            db.add(meta)

        meta.title = meta.title or (queue_item or {}).get("title")
        meta.artist = meta.artist or data.get("artist") or (queue_item or {}).get("channel")
        meta.url = (queue_item or {}).get("url")
        meta.mode = data.get("mode") or (queue_item or {}).get("mode", "karaoke")
        meta.languages = json.dumps((queue_item or {}).get("languages", []))
        meta.thumbnail = (queue_item or {}).get("thumbnail")
        meta.channel = (queue_item or {}).get("channel")
        meta.upload_date = (queue_item or {}).get("upload_date")
        meta.categories = json.dumps((queue_item or {}).get("categories", []))
        meta.tags = json.dumps((queue_item or {}).get("tags", []))
        meta.finished_at = data.get("finished_at", _now_iso())
        meta.audio_duration = data.get("audio_duration")
        meta.language_detected = data.get("language_detected")
        meta.status = "done"
        meta.added_by = (queue_item or {}).get("added_by")
        meta.added_by_id = (queue_item or {}).get("added_by_id")
        meta.file_size_bytes = data.get("file_size_bytes")

        # Lyrics and subtitles from worker
        if data.get("lyrics"):
            meta.lyrics = data["lyrics"]
        if data.get("subtitles"):
            meta.subtitles = data["subtitles"]

        db.commit()
        db.close()
    except Exception as e:
        logger.error("Failed to save job metadata for %s: %s", job_id, e)

    # Post-completion: generate analysis
    if _load_settings().get("feature_analysis", True) and data.get("lyrics"):
        try:
            words = json.loads(data["lyrics"])
            lyrics_text = " ".join(w["text"] for w in words)
            prompts = _load_prompts()
            custom_prompt = prompts.get("analysis_prompt") or None
            from karaoke.analyze_lyrics import analyze_lyrics
            title = (queue_item or {}).get("title")
            artist = data.get("artist") or (queue_item or {}).get("channel")
            result = analyze_lyrics(lyrics_text, title=title, artist=artist,
                                    custom_prompt=custom_prompt)
            _db = get_session()
            _meta = _db.query(JobMetadata).filter(JobMetadata.job_id == job_id).first()
            if _meta:
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
    job_url = (queue_item or {}).get("url")
    if job_url:
        try:
            _db = get_session()
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

    # Dispatch next queued jobs
    _dispatch_next_jobs()


def _on_job_failed(job_id: str, error: str) -> None:
    """Called when a worker reports job failure."""
    with _lock:
        for item in _queue:
            if item.get("id") == job_id:
                item["status"] = "failed"
                item["error"] = error
                item.pop("worker_id", None)
                break
        _active_jobs.pop(job_id, None)
    _save_queue()
    _dispatch_next_jobs()


def _worker_health_loop() -> None:
    """Background thread that pings workers every 30 seconds."""
    while True:
        time.sleep(30)
        try:
            workers = _load_workers()
            for w in workers:
                if not w.get("enabled", True):
                    continue
                wid = w["id"]
                try:
                    headers = {}
                    if w.get("api_key"):
                        headers["Authorization"] = f"Bearer {w['api_key']}"
                    resp = httpx.get(f"{w['url']}/worker/status", headers=headers, timeout=10)
                    if resp.status_code == 200:
                        status_data = resp.json()
                        _worker_states[wid] = {
                            "status": status_data.get("status", "idle"),
                            "name": status_data.get("name", w.get("name", "")),
                            "device": status_data.get("device", "unknown"),
                            "current_job": status_data.get("current_job"),
                            "last_seen": _now_iso(),
                            "online": True,
                        }
                    else:
                        _worker_states[wid] = {
                            **_worker_states.get(wid, {}),
                            "online": False,
                            "status": "offline",
                            "last_error": f"HTTP {resp.status_code}",
                        }
                except Exception as e:
                    _worker_states[wid] = {
                        **_worker_states.get(wid, {}),
                        "online": False,
                        "status": "offline",
                        "last_error": str(e),
                    }

            # Re-queue jobs from workers that have been offline > 5 min
            now = datetime.now(timezone.utc)
            with _lock:
                for item in _queue:
                    if item.get("status") != "processing":
                        continue
                    wid = item.get("worker_id")
                    if not wid:
                        continue
                    state = _worker_states.get(wid, {})
                    if state.get("online"):
                        continue
                    last_seen = state.get("last_seen")
                    if last_seen:
                        ls = datetime.fromisoformat(last_seen)
                        if (now - ls).total_seconds() > 300:
                            logger.warning("Worker %s offline > 5min, re-queuing job %s",
                                           wid, item.get("id"))
                            item["status"] = "queued"
                            item.pop("worker_id", None)
                _save_queue()

            # Try dispatching if we have queued items and idle workers
            _dispatch_next_jobs()
        except Exception as e:
            logger.error("Worker health check error: %s", e)

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






# ---------------------------------------------------------------------------
# Worker callback endpoints (called by workers to report progress/completion)
# ---------------------------------------------------------------------------

def _verify_worker_key(request) -> bool:
    """Check if the request comes from a configured worker."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        key = auth[7:]
    else:
        key = ""
    workers = _load_workers()
    if not workers:
        return False
    # If the key matches any worker's api_key (including empty matching empty)
    return any(w.get("api_key", "") == key for w in workers)


@app.post("/api/worker/progress")
async def worker_progress(request: Request):
    """Receive progress update from a worker."""
    if not _verify_worker_key(request):
        raise HTTPException(401, "Invalid worker key")
    data = await request.json()
    job_id = data.get("job_id")
    if not job_id:
        raise HTTPException(400, "job_id required")
    with _lock:
        if job_id in _active_jobs:
            _active_jobs[job_id].update(data)
        else:
            _active_jobs[job_id] = data
    return {"ok": True}


@app.post("/api/worker/complete")
async def worker_complete(request: Request):
    """Receive job completion from a worker."""
    if not _verify_worker_key(request):
        raise HTTPException(401, "Invalid worker key")
    data = await request.json()
    job_id = data.get("job_id")
    if not job_id:
        raise HTTPException(400, "job_id required")
    logger.info("Worker completed job %s", job_id)
    # Run in background to avoid blocking the worker
    threading.Thread(target=_on_job_completed, args=(job_id, data), daemon=True).start()
    return {"ok": True}


@app.post("/api/worker/failed")
async def worker_failed(request: Request):
    """Receive job failure from a worker."""
    if not _verify_worker_key(request):
        raise HTTPException(401, "Invalid worker key")
    data = await request.json()
    job_id = data.get("job_id")
    error = data.get("error", "Unknown error")
    if not job_id:
        raise HTTPException(400, "job_id required")
    logger.warning("Worker reported failure for job %s: %s", job_id, error)
    _on_job_failed(job_id, error)
    return {"ok": True}


@app.post("/api/worker/upload-urls")
async def worker_upload_urls(request: Request):
    """Generate presigned upload URLs for a worker to upload job files to R2."""
    if not _verify_worker_key(request):
        raise HTTPException(401, "Invalid worker key")
    data = await request.json()
    job_id = data.get("job_id")
    filenames = data.get("filenames", [])
    if not job_id or not filenames:
        raise HTTPException(400, "job_id and filenames required")

    if not storage.is_r2():
        raise HTTPException(501, "R2 storage not configured")

    urls = {}
    for fname in filenames:
        # Only allow expected file types
        if not fname.endswith((".mp4", ".mp3", ".json", ".srt")):
            continue
        key = f"jobs/{job_id}/{fname}"
        url = storage.generate_presigned_upload(key)
        if url:
            urls[fname] = url
    return {"urls": urls}


# ---------------------------------------------------------------------------
# Admin worker management endpoints
# ---------------------------------------------------------------------------

@app.get("/api/admin/workers")
def admin_list_workers(admin: User = Depends(require_admin)):
    """List configured workers with live status."""
    workers = _load_workers()
    result = []
    for w in workers:
        state = _worker_states.get(w["id"], {})
        result.append({
            **w,
            "api_key": "***" if w.get("api_key") else "",  # don't expose keys
            "status": state.get("status", "unknown"),
            "device": state.get("device", "unknown"),
            "online": state.get("online", False),
            "last_seen": state.get("last_seen"),
            "current_job": state.get("current_job"),
        })
    return {"workers": result}


class WorkerCreateRequest(BaseModel):
    name: str
    url: str
    api_key: str = ""


@app.post("/api/admin/workers")
def admin_add_worker(req: WorkerCreateRequest, admin: User = Depends(require_admin)):
    workers = _load_workers()
    worker = {
        "id": f"w-{secrets.token_hex(4)}",
        "name": req.name,
        "url": req.url.rstrip("/"),
        "api_key": req.api_key,
        "enabled": True,
    }
    workers.append(worker)
    _save_workers(workers)
    return {"worker": {**worker, "api_key": "***" if worker["api_key"] else ""}}


@app.put("/api/admin/workers/{worker_id}")
def admin_update_worker(worker_id: str, req: WorkerCreateRequest,
                        admin: User = Depends(require_admin)):
    workers = _load_workers()
    for w in workers:
        if w["id"] == worker_id:
            w["name"] = req.name
            w["url"] = req.url.rstrip("/")
            if req.api_key and req.api_key != "***":
                w["api_key"] = req.api_key
            _save_workers(workers)
            return {"updated": True}
    raise HTTPException(404, "Worker not found")


@app.delete("/api/admin/workers/{worker_id}")
def admin_delete_worker(worker_id: str, admin: User = Depends(require_admin)):
    workers = _load_workers()
    workers = [w for w in workers if w["id"] != worker_id]
    _save_workers(workers)
    _worker_states.pop(worker_id, None)
    return {"deleted": True}


@app.post("/api/admin/workers/{worker_id}/test")
def admin_test_worker(worker_id: str, admin: User = Depends(require_admin)):
    """Ping a worker to test connectivity."""
    workers = _load_workers()
    worker = next((w for w in workers if w["id"] == worker_id), None)
    if not worker:
        raise HTTPException(404, "Worker not found")
    try:
        headers = {}
        if worker.get("api_key"):
            headers["Authorization"] = f"Bearer {worker['api_key']}"
        resp = httpx.get(f"{worker['url']}/worker/status", headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # Update worker state immediately so UI reflects it
            _worker_states[worker_id] = {
                "status": data.get("status", "idle"),
                "name": data.get("name", worker.get("name", "")),
                "device": data.get("device", "unknown"),
                "current_job": data.get("current_job"),
                "last_seen": _now_iso(),
                "online": True,
            }
            return {"success": True, "status": data}
        return {"success": False, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/admin/workers/{worker_id}/toggle")
def admin_toggle_worker(worker_id: str, admin: User = Depends(require_admin)):
    """Enable/disable a worker."""
    workers = _load_workers()
    for w in workers:
        if w["id"] == worker_id:
            w["enabled"] = not w.get("enabled", True)
            _save_workers(workers)
            return {"enabled": w["enabled"]}
    raise HTTPException(404, "Worker not found")


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
        db = get_session()
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
            _db = get_session()
            for m in _db.query(JobMetadata).filter(JobMetadata.analysis_song_info.isnot(None)).all():
                match = _re.search(r"\((\d{4})\)", m.analysis_song_info or "")
                if match:
                    m.year = match.group(1)
            _db.commit()
            _db.close()

    # --- JobMetadata: subtitles column ---
    if "job_metadata" in set(insp.get_table_names()):
        jm_cols2 = {c["name"] for c in insp.get_columns("job_metadata")}
        if "subtitles" not in jm_cols2:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE job_metadata ADD COLUMN subtitles TEXT"))
            # Backfill from local SRT files or R2
            _db = get_session()
            for m in _db.query(JobMetadata).filter(JobMetadata.status == "done").all():
                srt_data = {}
                job_dir = JOBS_DIR / m.job_id
                if job_dir.exists():
                    for srt in job_dir.glob("subtitles_*.srt"):
                        lang = srt.stem.replace("subtitles_", "")
                        srt_data[lang] = srt.read_text()
                elif storage.is_r2():
                    # Try to list SRT files from R2
                    try:
                        keys = storage.list_keys(f"jobs/{m.job_id}/subtitles_")
                        for key in keys:
                            fname = key.split("/")[-1]
                            lang = fname.replace("subtitles_", "").replace(".srt", "")
                            content = storage.read_text(key)
                            if content:
                                srt_data[lang] = content
                    except Exception:
                        pass
                if srt_data:
                    m.subtitles = json.dumps(srt_data)
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
    """On startup: load queue, reset stale jobs, start worker health monitor."""
    _load_queue()

    # Reset processing items back to queued (server restarted, workers may have changed)
    changed = False
    for item in _queue:
        if item.get("status") in ("processing",):
            item["status"] = "queued"
            item.pop("worker_id", None)
            changed = True
    if changed:
        _save_queue()

    # Start worker health monitoring thread
    health_thread = threading.Thread(target=_worker_health_loop, daemon=True)
    health_thread.start()

    # Try dispatching queued jobs to any available workers
    _dispatch_next_jobs()


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
    """Add a job to the front of the queue and dispatch immediately."""
    settings = _load_settings()
    allowed = settings.get("allowed_modes", list(VALID_MODES))
    mode = req.mode if req.mode in VALID_MODES else "karaoke"
    if mode not in allowed:
        raise HTTPException(403, f"Production mode '{mode}' is not enabled")

    meta = {}
    try:
        meta = fetch_metadata(req.url)
    except Exception:
        pass
    title = meta.get("title", "Unknown")

    slug = _slugify(title)
    job_id = f"{slug}-{secrets.token_hex(2)}"
    max_langs = settings.get("max_subtitle_languages", 3)
    languages = req.languages[:max_langs] if req.languages else (
        [req.language] if req.language else []
    )
    if mode not in ("subtitled", "both"):
        languages = []

    item = {
        "id": job_id,
        "url": req.url,
        "mode": mode,
        "languages": languages,
        "title": title,
        "thumbnail": meta.get("thumbnail"),
        "channel": meta.get("channel"),
        "upload_date": meta.get("upload_date"),
        "categories": meta.get("categories", []),
        "tags": meta.get("tags", []),
        "added_by": user.name.split()[0] if user and user.name else None,
        "added_by_id": str(user.id) if user else None,
        "status": "queued",
    }
    with _lock:
        _queue.insert(0, item)  # add to front
    _save_queue()
    _dispatch_next_jobs()
    return {"job": item}


@app.get("/api/jobs/current")
def get_current_job():
    """Return the most recently active job (for backward compat)."""
    with _lock:
        if _active_jobs:
            # Return the first active job
            job = next(iter(_active_jobs.values()))
            return {"job": dict(job)}
        return {"job": None}


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

    max_langs = settings.get("max_subtitle_languages", 3)
    languages = req.languages[:max_langs] if req.languages else []
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
        # Daily production limits (query DB instead of local disk)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _db = get_session()
        try:
            today_jobs = _db.query(JobMetadata).filter(
                JobMetadata.added_by_id == str(user.id),
                JobMetadata.status == "done",
                JobMetadata.finished_at >= today_str,
            ).all()
            user_today_karaoke = sum(1 for j in today_jobs if (j.mode or "karaoke") == "karaoke")
            user_today_subtitled = sum(1 for j in today_jobs if (j.mode or "karaoke") in ("subtitled", "both"))
        finally:
            _db.close()
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
    _dispatch_next_jobs()

    return {"item": item, "queue": _queue}


@app.get("/api/queue")
def get_queue():
    """Return the current queue and active job status."""
    with _lock:
        queue_with_progress = []
        for item in _queue:
            entry = dict(item)
            job_id = item.get("id")
            # If this item is being processed by a worker, merge progress
            if job_id in _active_jobs:
                active = _active_jobs[job_id]
                entry["step"] = active.get("step", 0)
                entry["step_name"] = active.get("step_name", "")
                entry["step_progress"] = active.get("step_progress", 0)
                entry["status"] = "processing"
                entry["worker_name"] = active.get("worker_name", "")
            queue_with_progress.append(entry)

        # Return first active job for backward compat
        first_active = next(iter(_active_jobs.values()), None) if _active_jobs else None
        return {
            "queue": queue_with_progress,
            "job": dict(first_active) if first_active else None,
            "workers_online": sum(1 for s in _worker_states.values() if s.get("online")),
        }


@app.delete("/api/queue/{item_id}")
def remove_from_queue(item_id: str):
    """Remove an item from the queue. If processing on a worker, send cancel."""
    with _lock:
        item = next((i for i in _queue if i["id"] == item_id), None)
        if not item:
            raise HTTPException(404, "Item not found in queue")
        is_processing = item.get("status") == "processing"
        worker_id = item.get("worker_id")

    # If being processed by a worker, send cancel request
    if is_processing and worker_id:
        workers = _load_workers()
        worker = next((w for w in workers if w["id"] == worker_id), None)
        if worker:
            try:
                headers = {}
                if worker.get("api_key"):
                    headers["Authorization"] = f"Bearer {worker['api_key']}"
                httpx.post(f"{worker['url']}/worker/jobs/{item_id}/cancel",
                           headers=headers, timeout=5)
            except Exception:
                pass

    # Remove from queue and active jobs
    with _lock:
        _queue[:] = [i for i in _queue if i["id"] != item_id]
        _active_jobs.pop(item_id, None)
    _save_queue()

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
        if item.get("status") == "processing":
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
    _dispatch_next_jobs()
    return {"started": True, "queue": _queue}


@app.post("/api/queue/{item_id}/pause")
def pause_queue_item(item_id: str):
    """Cancel a processing item and re-queue it (pause is not supported with external workers)."""
    with _lock:
        item = next((i for i in _queue if i["id"] == item_id), None)
        if not item or item.get("status") != "processing":
            raise HTTPException(400, "Item is not currently processing")
        worker_id = item.get("worker_id")

    # Send cancel to worker, then re-queue
    if worker_id:
        workers = _load_workers()
        worker = next((w for w in workers if w["id"] == worker_id), None)
        if worker:
            try:
                headers = {}
                if worker.get("api_key"):
                    headers["Authorization"] = f"Bearer {worker['api_key']}"
                httpx.post(f"{worker['url']}/worker/jobs/{item_id}/cancel",
                           headers=headers, timeout=5)
            except Exception:
                pass

    with _lock:
        item["status"] = "queued"
        item.pop("worker_id", None)
        _active_jobs.pop(item_id, None)
    _save_queue()
    return {"paused": True}


@app.post("/api/queue/{item_id}/resume")
def resume_queue_item(item_id: str):
    """Resume a failed queue item by re-queuing it."""
    with _lock:
        item = next((i for i in _queue if i["id"] == item_id), None)
        if not item:
            raise HTTPException(404, "Item not found")
        item["status"] = "queued"
        item.pop("worker_id", None)
        item.pop("error", None)
    _save_queue()
    _dispatch_next_jobs()
    return {"resumed": True}


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    """Re-queue a job for processing."""
    # Find in queue or re-add
    with _lock:
        item = next((i for i in _queue if i["id"] == job_id), None)
        if item:
            item["status"] = "queued"
            item.pop("worker_id", None)
            item.pop("error", None)
        else:
            raise HTTPException(404, "Job not found in queue")
    _save_queue()
    _dispatch_next_jobs()
    return {"resumed": True}


@app.get("/api/library")
def get_library(user: User | None = Depends(get_optional_user),
                db: Session = Depends(get_db)):
    """Return all completed jobs as a list, sorted newest first, with vote data."""
    from sqlalchemy import func

    # Query completed jobs from DB
    meta_rows = db.query(JobMetadata).filter(JobMetadata.status == "done").all()
    if not meta_rows:
        return {"items": []}

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

    items = []
    for meta in meta_rows:
        job_id = meta.job_id
        votes = vote_map.get(job_id, {"upvotes": 0, "downvotes": 0})
        item = meta.to_library_dict()
        item["upvotes"] = votes["upvotes"]
        item["downvotes"] = votes["downvotes"]
        item["user_vote"] = user_votes.get(job_id, 0)
        item["comment_count"] = comment_counts.get(job_id, 0)
        items.append(item)

    items.sort(key=lambda x: x.get("finished_at") or "", reverse=True)
    return {"items": items}


def _extract_video_id(url: str) -> str | None:
    """Extract the YouTube video ID from various URL formats."""
    m = re.search(r"(?:v=|youtu\.be/|/embed/|/v/|/shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


@app.get("/api/library/check-url")
def check_url_in_library(url: str, mode: str = "karaoke",
                         db: Session = Depends(get_db)):
    """Check if a YouTube URL has already been generated in the library for the given mode."""
    video_id = _extract_video_id(url)
    if not video_id:
        return {"found": False}

    rows = db.query(JobMetadata).filter(JobMetadata.status == "done").all()
    for meta in rows:
        existing_mode = meta.mode or "karaoke"
        if existing_mode != mode and not (existing_mode == "both" and mode in ("karaoke", "subtitled")):
            continue
        existing_id = _extract_video_id(meta.url or "")
        if existing_id == video_id:
            return {
                "found": True,
                "item": {
                    "id": meta.job_id,
                    "title": meta.title or "Unknown",
                    "url": meta.url,
                    "mode": meta.mode or "karaoke",
                    "thumbnail": meta.thumbnail,
                    "channel": meta.channel,
                    "finished_at": meta.finished_at,
                },
            }

    return {"found": False}


def _serve_file(job_id: str, filename: str, media_type: str, label: str = "File"):
    """Serve a job file from local disk or redirect to R2."""
    local = JOBS_DIR / job_id / filename
    if local.exists():
        return FileResponse(local, media_type=media_type)
    url = storage.get_url(f"jobs/{job_id}/{filename}")
    if url:
        return RedirectResponse(url, status_code=302)
    raise HTTPException(404, f"{label} not found")


@app.get("/api/jobs/{job_id}/video")
def stream_video(job_id: str):
    return _serve_file(job_id, "karaoke.mp4", "video/mp4", "Video")


@app.get("/api/jobs/{job_id}/instrumental")
def stream_instrumental(job_id: str):
    return _serve_file(job_id, "instrumental.mp3", "audio/mpeg", "Instrumental")


@app.get("/api/jobs/{job_id}/vocals")
def stream_vocals(job_id: str):
    return _serve_file(job_id, "vocals.mp3", "audio/mpeg", "Vocals")


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
def get_lyrics(job_id: str, db: Session = Depends(get_db)):
    # Prefer DB (lyrics travel with the library item)
    meta = db.query(JobMetadata).filter(JobMetadata.job_id == job_id).first()
    if meta and meta.lyrics:
        return json.loads(meta.lyrics)
    # Fallback to local file (during active job processing)
    path = JOBS_DIR / job_id / "lyrics.json"
    if path.exists():
        return json.loads(path.read_text())
    raise HTTPException(404, "Lyrics not found")



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
    "demucs_model": "htdemucs_ft",
    "max_subtitle_languages": 3,
}


def _load_settings() -> dict:
    """Load admin settings from DB."""
    stored = _db_config_get("settings", {})
    return {**_DEFAULT_SETTINGS, **stored}


def _save_settings(settings: dict) -> None:
    """Save admin settings to DB."""
    _db_config_set("settings", settings)


def _load_prompts() -> dict:
    """Load custom prompts from DB."""
    return _db_config_get("prompts", {})


def _save_prompts(prompts: dict) -> None:
    """Save custom prompts to DB."""
    _db_config_set("prompts", prompts)


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

    # Need lyrics to analyze — prefer DB, fallback to file
    lyrics_raw = None
    if meta and meta.lyrics:
        lyrics_raw = meta.lyrics
    else:
        lyrics_path = JOBS_DIR / job_id / "lyrics.json"
        if lyrics_path.exists():
            lyrics_raw = lyrics_path.read_text()
    if not lyrics_raw:
        raise HTTPException(404, "Lyrics not found")

    title = meta.title if meta else None
    artist = meta.artist if meta else None

    # Build plain text lyrics from word-level data
    words = json.loads(lyrics_raw)
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
def get_subtitles_lang(job_id: str, lang_code: str, db: Session = Depends(get_db)):
    """Serve per-language SRT file."""
    if not re.match(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$", lang_code):
        raise HTTPException(400, "Invalid language code")
    # Try DB first
    meta = db.query(JobMetadata).filter(JobMetadata.job_id == job_id).first()
    if meta and meta.subtitles:
        srt_data = json.loads(meta.subtitles)
        if lang_code in srt_data:
            return Response(content=srt_data[lang_code], media_type="text/plain")
    # Fallback to local file
    path = JOBS_DIR / job_id / f"subtitles_{lang_code}.srt"
    if path.exists():
        return FileResponse(path, media_type="text/plain")
    # Fallback to R2
    r2_key = f"jobs/{job_id}/subtitles_{lang_code}.srt"
    url = storage.get_url(r2_key)
    if url:
        return RedirectResponse(url, status_code=302)
    raise HTTPException(404, "Subtitles not found")


@app.get("/api/jobs/{job_id}/subtitles")
def get_subtitles(job_id: str, db: Session = Depends(get_db)):
    """Serve first available SRT file."""
    # Try DB first
    meta = db.query(JobMetadata).filter(JobMetadata.job_id == job_id).first()
    if meta and meta.subtitles:
        srt_data = json.loads(meta.subtitles)
        if srt_data:
            lang = sorted(srt_data.keys())[0]
            return Response(content=srt_data[lang], media_type="text/plain")
    # Fallback to local file
    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        for f in sorted(job_dir.glob("subtitles_*.srt")):
            return FileResponse(f, media_type="text/plain")
    # Fallback to R2
    if meta:
        langs = json.loads(meta.languages or "[]")
        for lang in langs:
            r2_key = f"jobs/{job_id}/subtitles_{lang}.srt"
            url = storage.get_url(r2_key)
            if url:
                return RedirectResponse(url, status_code=302)
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
    # Update DB (source of truth)
    meta = db.query(JobMetadata).filter(JobMetadata.job_id == job_id).first()
    if not meta:
        raise HTTPException(404, "Job not found")
    if req.title is not None:
        meta.title = req.title
    if req.artist is not None:
        meta.artist = req.artist
    if req.year is not None:
        meta.year = req.year
    db.commit()

    # Update in-memory active job if currently processing
    with _lock:
        if job_id in _active_jobs:
            if req.title is not None:
                _active_jobs[job_id]["title"] = req.title
            if req.artist is not None:
                _active_jobs[job_id]["artist"] = req.artist

    return {"title": meta.title, "artist": meta.artist, "year": meta.year}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, user: User = Depends(get_current_user)):
    """Delete a job and all its files."""

    # Enforce delete permission (admins bypass)
    if not is_admin(user) and user.permissions and not user.permissions.can_delete_library:
        raise HTTPException(403, "You don't have permission to delete songs")

    # Check job exists locally or in DB
    job_dir = JOBS_DIR / job_id
    has_local = job_dir.exists() and job_dir.is_dir()
    _db = get_session()
    has_db = _db.query(JobMetadata).filter(JobMetadata.job_id == job_id).first() is not None
    if not has_local and not has_db:
        _db.close()
        raise HTTPException(404, "Job not found")

    # Don't allow deleting a processing job
    with _lock:
        if job_id in _active_jobs:
            _db.close()
            raise HTTPException(409, "Cannot delete a running job")

    # Delete local files
    if has_local:
        shutil.rmtree(job_dir)

    # Delete from R2
    if storage.is_r2():
        storage.delete_prefix(f"jobs/{job_id}/")

    # Clean up DB metadata (votes, comments, job metadata)
    try:
        _db.query(JobMetadata).filter(JobMetadata.job_id == job_id).delete()
        _db.query(Vote).filter(Vote.job_id == job_id).delete()
        _db.query(Comment).filter(Comment.job_id == job_id).delete()
        _db.commit()
    except Exception:
        _db.rollback()
    finally:
        _db.close()

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

    media_types = {
        "karaoke.mp4": "video/mp4",
        "instrumental.mp3": "audio/mpeg",
        "vocals.mp3": "audio/mpeg",
    }
    media_type = media_types.get(filename, "text/plain")
    return _serve_file(job_id, filename, media_type, "File")


@app.get("/api/jobs/{job_id}/artifact/{filepath:path}")
def download_artifact(job_id: str, filepath: str):
    """Download an intermediate artifact from a job's output directory."""
    if ".." in filepath:
        raise HTTPException(400, "Invalid path")
    path = JOBS_DIR / job_id / filepath
    if path.exists() and path.is_file():
        return FileResponse(
            path, filename=path.name,
            headers={"Content-Disposition": f"attachment; filename={path.name}"},
        )
    url = storage.get_url(f"jobs/{job_id}/{filepath}")
    if url:
        return RedirectResponse(url, status_code=302)
    raise HTTPException(404, "Artifact not found")


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
        filename = f"{secrets.token_hex(8)}_{screenshot.filename}"
        content = await screenshot.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(400, "Screenshot must be under 5MB")
        if storage.is_r2():
            # Write temp file, upload to R2, store R2 key as path
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)
            storage.upload(f"feedback/{filename}", tmp_path)
            tmp_path.unlink(missing_ok=True)
            screenshot_path = f"r2:feedback/{filename}"
        else:
            feedback_dir = Path("output/feedback")
            feedback_dir.mkdir(parents=True, exist_ok=True)
            filepath = feedback_dir / filename
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
        if fb.screenshot_path.startswith("r2:"):
            storage.delete(fb.screenshot_path[3:])
        else:
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
    if fb.screenshot_path.startswith("r2:"):
        url = storage.get_url(fb.screenshot_path[3:])
        if url:
            return RedirectResponse(url, status_code=302)
        raise HTTPException(404, "Screenshot file missing")
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
                   since_hours: float | None = None,
                   limit: int = 100,
                   offset: int = 0):
    """Return recent activity log entries with optional filtering."""
    q = db.query(ActivityLog).order_by(ActivityLog.created_at.desc())
    if event_type:
        q = q.filter(ActivityLog.event_type == event_type)
    if since_hours is not None and since_hours > 0:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=since_hours)
        q = q.filter(ActivityLog.created_at >= cutoff)
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

    # Library stats from DB
    total_songs = db.query(func.count(JobMetadata.job_id)).filter(JobMetadata.status == "done").scalar() or 0
    total_size_bytes = db.query(func.coalesce(func.sum(JobMetadata.file_size_bytes), 0)).filter(JobMetadata.status == "done").scalar() or 0

    # Queue status
    with _lock:
        queue_len = len(_queue)
        running = len(_active_jobs) > 0

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


@app.delete("/api/admin/invitations/{invitation_id}")
def admin_delete_invitation(invitation_id: str, admin: User = Depends(require_admin),
                            db: Session = Depends(get_db)):
    """Delete a pending invitation, returning the credit to the inviter."""
    inv = db.query(Invitation).filter(Invitation.id == invitation_id).first()
    if not inv:
        raise HTTPException(404, "Invitation not found")
    if inv.status == "accepted":
        raise HTTPException(400, "Cannot delete an accepted invitation")
    db.delete(inv)
    db.commit()
    return {"deleted": True}


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
    skipped = []
    for email in req.emails[:10]:  # max 10 at a time
        if remaining is not None and len(sent) >= remaining:
            skipped.append({"email": email, "reason": "No invitations remaining"})
            continue
        email = email.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            skipped.append({"email": email, "reason": "Invalid email address"})
            continue
        # Skip if already a user
        if db.query(User).filter(User.email == email).first():
            skipped.append({"email": email, "reason": "Already a member"})
            continue
        # Skip if already invited by this user
        existing = db.query(Invitation).filter(
            Invitation.inviter_id == user.id, Invitation.email == email
        ).first()
        if existing:
            skipped.append({"email": email, "reason": "Already invited"})
            continue
        inv = Invitation(inviter_id=user.id, email=email)
        db.add(inv)
        db.flush()  # generate token from default before sending email
        emailed = send_invite_email(email, user.name or "A friend", inv.token)
        sent.append({"email": email, "token": inv.token, "link": f"/login?invite={inv.token}", "emailed": emailed})
    db.commit()
    for s in sent:
        _log_activity(db, user, "invite", {"email": s["email"]})
    total_used = db.query(Invitation).filter(Invitation.inviter_id == user.id).count()
    return {
        "sent": sent,
        "skipped": skipped,
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
