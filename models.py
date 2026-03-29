"""SQLAlchemy models for users, permissions, and feedback."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, SmallInteger, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    picture_url: Mapped[str | None] = mapped_column(String(512))
    password_hash: Mapped[str | None] = mapped_column(String(255))  # null for OAuth-only
    google_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    role: Mapped[str] = mapped_column(String(20), default="user")  # "user" or "admin"
    theme: Mapped[str] = mapped_column(String(20), default="retro")  # "retro", "spotify", "disco"
    dark_mode: Mapped[str] = mapped_column(String(20), default="dark")  # "dark", "day", "night"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    permissions: Mapped["UserPermissions"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": str(self.id),
            "email": self.email,
            "name": self.name,
            "picture_url": self.picture_url,
            "role": self.role,
            "theme": self.theme,
            "dark_mode": self.dark_mode,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class UserPermissions(Base):
    __tablename__ = "user_permissions"

    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    max_karaoke_per_day: Mapped[int] = mapped_column(Integer, default=5)
    max_subtitled_per_day: Mapped[int] = mapped_column(Integer, default=15)
    max_queue_length: Mapped[int] = mapped_column(Integer, default=10)
    can_download_karaoke: Mapped[bool] = mapped_column(Boolean, default=True)
    can_download_instrumental: Mapped[bool] = mapped_column(Boolean, default=True)
    can_download_vocals: Mapped[bool] = mapped_column(Boolean, default=True)
    can_delete_library: Mapped[bool] = mapped_column(Boolean, default=False)
    can_share_library: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped["User"] = relationship(back_populates="permissions")

    def to_dict(self):
        return {
            "max_karaoke_per_day": self.max_karaoke_per_day,
            "max_subtitled_per_day": self.max_subtitled_per_day,
            "max_queue_length": self.max_queue_length,
            "can_download_karaoke": self.can_download_karaoke,
            "can_download_instrumental": self.can_download_instrumental,
            "can_download_vocals": self.can_download_vocals,
            "can_delete_library": self.can_delete_library,
            "can_share_library": self.can_share_library,
        }


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    screenshot_path: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(20), default="new")  # "new", "reviewed", "resolved"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped["User"] = relationship()

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "user_name": self.user.name if self.user else None,
            "user_email": self.user.email if self.user else None,
            "subject": self.subject,
            "description": self.description,
            "screenshot_path": self.screenshot_path,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Vote(Base):
    __tablename__ = "votes"
    __table_args__ = (
        UniqueConstraint("user_id", "job_id", name="uq_user_job_vote"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    job_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    value: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # +1 or -1
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
