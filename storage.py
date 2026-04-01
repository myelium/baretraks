"""Storage abstraction for local filesystem and Cloudflare R2."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()
JOBS_DIR = Path("output/jobs")


class LocalStorage:
    """Serves files from the local filesystem."""

    def upload(self, key: str, local_path: Path) -> str:
        """No-op for local storage — files are already on disk."""
        return f"/api/jobs/{'/'.join(key.split('/')[1:])}"

    def get_url(self, key: str) -> str | None:
        """Return local API path if file exists."""
        path = JOBS_DIR.parent / key  # key is like "jobs/{id}/karaoke.mp4"
        if path.exists():
            parts = key.split("/")  # ["jobs", job_id, filename]
            if len(parts) >= 3:
                return f"/api/jobs/{parts[1]}/download/{parts[2]}"
        return None

    def delete(self, key: str) -> None:
        path = JOBS_DIR.parent / key
        if path.exists():
            path.unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        """Delete all files under a prefix (like a directory)."""
        import shutil
        path = JOBS_DIR.parent / prefix
        if path.exists() and path.is_dir():
            shutil.rmtree(path)

    def exists(self, key: str) -> bool:
        return (JOBS_DIR.parent / key).exists()

    def list_keys(self, prefix: str) -> list[str]:
        path = JOBS_DIR.parent / prefix
        if not path.exists() or not path.is_dir():
            return []
        return [f"{prefix}/{f.name}" for f in path.iterdir() if f.is_file()]

    def read_text(self, key: str) -> str | None:
        path = JOBS_DIR.parent / key
        if path.exists():
            return path.read_text()
        return None

    def generate_presigned_upload(self, key: str, expires_in: int = 3600) -> str | None:
        return None

    def is_r2(self) -> bool:
        return False


class R2Storage:
    """Stores and serves files from Cloudflare R2 (S3-compatible)."""

    def __init__(self):
        import boto3
        self._bucket_name = os.getenv("R2_BUCKET_NAME", "baretraks")
        self._public_url = os.getenv("R2_PUBLIC_URL", "").rstrip("/")
        self._client = boto3.client(
            "s3",
            endpoint_url=os.getenv("R2_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
        )

    def upload(self, key: str, local_path: Path) -> str:
        """Upload a file to R2. Returns the public URL."""
        content_types = {
            ".mp4": "video/mp4",
            ".mp3": "audio/mpeg",
            ".json": "application/json",
            ".srt": "text/plain; charset=utf-8",
        }
        content_type = content_types.get(local_path.suffix, "application/octet-stream")
        self._client.upload_file(
            str(local_path), self._bucket_name, key,
            ExtraArgs={"ContentType": content_type},
        )
        logger.info("Uploaded %s to R2 (%s)", key, content_type)
        return f"{self._public_url}/{key}"

    def get_url(self, key: str) -> str | None:
        """Return the public URL for a key, or None if it doesn't exist."""
        if not self._public_url:
            return None
        try:
            self._client.head_object(Bucket=self._bucket_name, Key=key)
            return f"{self._public_url}/{key}"
        except self._client.exceptions.ClientError:
            return None

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket_name, Key=key)
        except Exception as e:
            logger.error("Failed to delete R2 key %s: %s", key, e)

    def delete_prefix(self, prefix: str) -> None:
        """Delete all objects under a prefix."""
        keys = self.list_keys(prefix)
        for key in keys:
            self.delete(key)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket_name, Key=key)
            return True
        except self._client.exceptions.ClientError:
            return False

    def list_keys(self, prefix: str) -> list[str]:
        try:
            resp = self._client.list_objects_v2(
                Bucket=self._bucket_name, Prefix=prefix,
            )
            return [obj["Key"] for obj in resp.get("Contents", [])]
        except Exception as e:
            logger.error("Failed to list R2 keys with prefix %s: %s", prefix, e)
            return []

    def read_text(self, key: str) -> str | None:
        """Read a text file from R2. Returns None if not found."""
        try:
            resp = self._client.get_object(Bucket=self._bucket_name, Key=key)
            return resp["Body"].read().decode("utf-8")
        except Exception:
            return None

    def generate_presigned_upload(self, key: str, expires_in: int = 3600) -> str | None:
        """Generate a presigned PUT URL scoped to a single key. Expires in 1 hour."""
        try:
            return self._client.generate_presigned_url(
                "put_object",
                Params={"Bucket": self._bucket_name, "Key": key},
                ExpiresIn=expires_in,
            )
        except Exception as e:
            logger.error("Failed to generate presigned URL for %s: %s", key, e)
            return None

    def is_r2(self) -> bool:
        return True


def get_storage() -> LocalStorage | R2Storage:
    """Return the configured storage backend."""
    if STORAGE_BACKEND == "r2":
        return R2Storage()
    return LocalStorage()


# Module-level singleton
storage = get_storage()
