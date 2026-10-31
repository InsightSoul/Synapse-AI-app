from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DB_PATH = Path(os.getenv("SYNAPSE_DB_PATH", str(BASE_DIR / "synapse.db")))
AUDIO_DIR = Path(os.getenv("SYNAPSE_AUDIO_DIR", str(BASE_DIR / ".data" / "audio")))
PROMPT_PATH = BASE_DIR / "Sys prompt - rdzen analityczny"
COOKIE_NAME = "synapse_session"
SESSION_DAYS = 7
MAX_AUDIO_BYTES = 25 * 1024 * 1024
ANALYSIS_CONSENT_VERSION = "1.0"
SUPPORTED_LANGUAGES = {
    "pl": "Polish",
    "en": "English",
    "fr": "French",
    "es": "Spanish",
}
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    voice_control_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    analysis_consent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    analysis_consent_version: Mapped[str | None] = mapped_column(String, nullable=True)
    preferred_language: Mapped[str | None] = mapped_column(String(2), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class LoginSession(Base):
    __tablename__ = "login_sessions"
    token_hash: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Source(Base):
    __tablename__ = "sources"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    source_type: Mapped[str] = mapped_column(String)
    file_name: Mapped[str | None] = mapped_column(String, nullable=True)
    content_text: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String, default="unknown")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AudioSession(Base):
    __tablename__ = "audio_sessions"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String, default="ready")
    language: Mapped[str] = mapped_column(String, default="unknown")
    save_raw_audio: Mapped[bool] = mapped_column(Boolean, default=False)
    raw_audio_path: Mapped[str | None] = mapped_column(String, nullable=True)
    transcript_source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id", ondelete="CASCADE"), index=True)
    audio_session_id: Mapped[str | None] = mapped_column(ForeignKey("audio_sessions.id", ondelete="SET NULL"), nullable=True)
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class LanguageModelConfig(Base):
    __tablename__ = "language_model_configs"
    language_code: Mapped[str] = mapped_column(String(2), primary_key=True)
    language_name: Mapped[str] = mapped_column(String(40))
    analysis_model: Mapped[str] = mapped_column(String(100))
    transcription_model: Mapped[str] = mapped_column(String(100))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class RegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=10, max_length=256)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class AnalysisConsentRequest(BaseModel):
    accepted: Literal[True]


class LanguagePreferenceRequest(BaseModel):
    language: Literal["pl", "en", "fr", "es"]


class SourceCreate(BaseModel):
    source_type: Literal["text_file", "pasted_text", "transcript"] = "pasted_text"
    file_name: str | None = Field(default=None, max_length=255)
    content_text: str = Field(min_length=1, max_length=2_000_000)
    language: Literal["pl", "fr", "en", "es", "mixed", "unknown"] = "unknown"


class AnalysisCreate(BaseModel):
    source_id: str
    relationship_context: str = Field(default="unspecified", max_length=2000)
    response_language: Literal["pl", "fr", "en", "es"] | None = None


class ParticipantConsent(BaseModel):
    speaker_label: str = Field(min_length=1, max_length=80)
    consented: Literal[True]


class AudioSessionCreate(BaseModel):
    consent_confirmed: Literal[True]
    participants: list[ParticipantConsent] = Field(min_length=1, max_length=20)
    language: Literal["pl", "fr", "en", "es", "mixed", "unknown"] = "unknown"
    save_raw_audio: bool = True


class AudioControl(BaseModel):
    action: Literal["start", "pause", "resume", "stop", "delete_session"]
    issued_by: Literal["ui", "voice"] = "ui"
    start_confirmed: bool = False
    delete_confirmed: bool = False


class VoiceControlRequest(BaseModel):
    enabled: bool


@asynccontextmanager
async def lifespan(_: FastAPI):
    if DATABASE_URL.startswith("sqlite"):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    user_columns = {column["name"] for column in inspect(engine).get_columns("users")}
    audio_columns = {column["name"] for column in inspect(engine).get_columns("audio_sessions")}
    with engine.begin() as connection:
        if "analysis_consent_at" not in user_columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN analysis_consent_at DATETIME"))
        if "analysis_consent_version" not in user_columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN analysis_consent_version VARCHAR(40)"))
        if "preferred_language" not in user_columns:
            connection.execute(text("ALTER TABLE users ADD COLUMN preferred_language VARCHAR(2)"))
        if "consent_json" in audio_columns:
            connection.execute(text("ALTER TABLE audio_sessions DROP COLUMN consent_json"))
    with SessionLocal() as db:
        analysis_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        transcription_model = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
        for language_code, language_name in SUPPORTED_LANGUAGES.items():
            language_config = db.get(LanguageModelConfig, language_code)
            if language_config is None:
                db.add(LanguageModelConfig(
                    language_code=language_code,
                    language_name=language_name,
                    analysis_model=analysis_model,
                    transcription_model=transcription_model,
                    enabled=True,
                ))
            else:
                language_config.language_name = language_name
        db.commit()
    yield


app = FastAPI(title="Synapse AI API", version="1.0.0", description="Private text analysis, user history, and consent-based audio sessions.", lifespan=lifespan)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/assets", StaticFiles(directory=str(BASE_DIR / "assets")), name="assets")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DbSession = Annotated[Session, Depends(get_db)]


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split(":", 1)
        actual = hash_password(password, bytes.fromhex(salt_hex)).split(":", 1)[1]
        return hmac.compare_digest(actual, digest_hex)
    except (ValueError, TypeError):
        return False


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_login_session(db: Session, user_id: str) -> str:
    token = secrets.token_urlsafe(36)
    db.add(LoginSession(token_hash=hash_token(token), user_id=user_id, expires_at=utc_now() + timedelta(days=SESSION_DAYS)))
    db.commit()
    return token


def set_login_cookie(response: Response, token: str) -> None:
    response.set_cookie(COOKIE_NAME, token, httponly=True, secure=os.getenv("COOKIE_SECURE", "false").lower() == "true", samesite="lax", max_age=SESSION_DAYS * 86400, path="/")


def get_current_user(
    db: DbSession,
    authorization: Annotated[str | None, Header()] = None,
    synapse_session: Annotated[str | None, Cookie()] = None,
) -> User:
    token = synapse_session
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Sign in is required.")
    login_session = db.get(LoginSession, hash_token(token))
    if not login_session or login_session.expires_at.replace(tzinfo=timezone.utc) <= utc_now():
        if login_session:
            db.delete(login_session)
            db.commit()
        raise HTTPException(status_code=401, detail="Session is invalid or expired.")
    user = db.get(User, login_session.user_id)
    if not user:
        raise HTTPException(status_code=401, detail="Account not found.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_analysis_consent(user: CurrentUser) -> User:
    if user.analysis_consent_at is None:
        raise HTTPException(status_code=403, detail="Accept the one-time content analysis consent first.")
    return user


ConsentedUser = Annotated[User, Depends(require_analysis_consent)]


def normalize_email(value: str) -> str:
    email = value.strip().lower()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    return email


def source_out(source: Source) -> dict:
    return {"source_id": source.id, "source_type": source.source_type, "file_name": source.file_name, "content_text": source.content_text, "language": source.language, "created_at": source.created_at.isoformat()}


def analysis_out(analysis: Analysis) -> dict:
    return {"analysis_id": analysis.id, "source_id": analysis.source_id, "source_ref": f"/api/v1/me/sources/{analysis.source_id}", "audio_session_id": analysis.audio_session_id, "created_at": analysis.created_at.isoformat(), "analysis_result": json.loads(analysis.result_json)}


def get_system_prompt() -> str:
    try:
        prompt_text = PROMPT_PATH.read_text(encoding="utf-8").lstrip("\ufeff \t\r\n")
        prompt, _ = json.JSONDecoder().raw_decode(prompt_text)
        return json.dumps(prompt, ensure_ascii=False)
    except (OSError, json.JSONDecodeError):
        return "Return valid JSON. Separate observed facts from tentative interpretations, cite evidence, and state limitations."


@app.get("/")
def root(request: Request) -> Response:
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.get("/manifest.webmanifest", include_in_schema=False)
def pwa_manifest() -> FileResponse:
    return FileResponse(BASE_DIR / "manifest.webmanifest", media_type="application/manifest+json", headers={"Cache-Control": "no-cache"})


@app.get("/service-worker.js", include_in_schema=False)
def pwa_service_worker() -> FileResponse:
    return FileResponse(
        BASE_DIR / "service-worker.js",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/auth/register", status_code=201)
def register(request: RegisterRequest, response: Response, db: DbSession) -> dict:
    email = normalize_email(request.email)
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    user = User(email=email, password_hash=hash_password(request.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    set_login_cookie(response, create_login_session(db, user.id))
    return {"user_id": user.id, "email": user.email, "analysis_consent_required": True}


@app.post("/api/v1/auth/login")
def login(request: LoginRequest, response: Response, db: DbSession) -> dict:
    user = db.scalar(select(User).where(User.email == normalize_email(request.email)))
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    set_login_cookie(response, create_login_session(db, user.id))
    return {"user_id": user.id, "email": user.email}


@app.post("/api/v1/auth/logout", status_code=204)
def logout(response: Response, db: DbSession, authorization: Annotated[str | None, Header()] = None, synapse_session: Annotated[str | None, Cookie()] = None) -> Response:
    token = synapse_session or (authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None)
    if token:
        login_session = db.get(LoginSession, hash_token(token))
        if login_session:
            db.delete(login_session)
            db.commit()
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="lax")
    response.status_code = 204
    return response


@app.get("/api/v1/auth/me")
def who_am_i(user: CurrentUser) -> dict:
    return {
        "user_id": user.id,
        "email": user.email,
        "analysis_consent_at": user.analysis_consent_at.isoformat() if user.analysis_consent_at else None,
        "analysis_consent_version": user.analysis_consent_version,
        "preferred_language": user.preferred_language,
    }


@app.post("/api/v1/auth/analysis-consent")
def accept_analysis_consent(request: AnalysisConsentRequest, user: CurrentUser, db: DbSession) -> dict:
    if user.analysis_consent_at is None:
        user.analysis_consent_at = utc_now()
        user.analysis_consent_version = ANALYSIS_CONSENT_VERSION
        db.commit()
        db.refresh(user)
    return {
        "accepted": True,
        "recorded_once": True,
        "consented_at": user.analysis_consent_at.isoformat(),
        "consent_version": user.analysis_consent_version,
    }


@app.get("/api/v1/languages")
def list_languages(db: DbSession) -> dict:
    configs = db.scalars(select(LanguageModelConfig).where(LanguageModelConfig.enabled.is_(True)).order_by(LanguageModelConfig.language_code)).all()
    return {
        "items": [
            {
                "code": config.language_code,
                "name": config.language_name,
                "analysis_model": config.analysis_model,
                "transcription_model": config.transcription_model,
            }
            for config in configs
        ]
    }


@app.put("/api/v1/me/language")
def set_preferred_language(request: LanguagePreferenceRequest, user: CurrentUser, db: DbSession) -> dict:
    config = db.get(LanguageModelConfig, request.language)
    if not config or not config.enabled:
        raise HTTPException(status_code=422, detail="The selected language is unavailable.")
    user.preferred_language = request.language
    db.commit()
    return {"preferred_language": user.preferred_language, "language_name": config.language_name}


@app.post("/api/v1/me/sources", status_code=201)
def create_source(request: SourceCreate, user: ConsentedUser, db: DbSession) -> dict:
    source = Source(user_id=user.id, **request.model_dump())
    db.add(source)
    db.commit()
    db.refresh(source)
    return source_out(source)


@app.get("/api/v1/me/sources")
def list_sources(user: CurrentUser, db: DbSession, limit: int = 25, offset: int = 0) -> dict:
    limit = max(1, min(limit, 100))
    rows = db.scalars(select(Source).where(Source.user_id == user.id).order_by(Source.created_at.desc()).offset(max(0, offset)).limit(limit)).all()
    return {"items": [source_out(row) for row in rows], "limit": limit, "offset": max(0, offset)}


@app.get("/api/v1/me/sources/{source_id}")
def get_source(source_id: str, user: CurrentUser, db: DbSession) -> dict:
    source = db.scalar(select(Source).where(Source.id == source_id, Source.user_id == user.id))
    if not source:
        raise HTTPException(status_code=404, detail="Source not found.")
    return source_out(source)


@app.delete("/api/v1/me/sources/{source_id}", status_code=204)
def delete_source(source_id: str, user: CurrentUser, db: DbSession) -> Response:
    source = db.scalar(select(Source).where(Source.id == source_id, Source.user_id == user.id))
    if not source:
        raise HTTPException(status_code=404, detail="Source not found.")
    db.delete(source)
    db.commit()
    return Response(status_code=204)


@app.post("/api/v1/me/analyses", status_code=201)
async def create_analysis(request: AnalysisCreate, user: ConsentedUser, db: DbSession) -> dict:
    source = db.scalar(select(Source).where(Source.id == request.source_id, Source.user_id == user.id))
    if not source:
        raise HTTPException(status_code=404, detail="Source not found.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Set OPENAI_API_KEY to enable AI analysis.")
    source_language = source.language if source.language in SUPPORTED_LANGUAGES else None
    language = request.response_language or user.preferred_language or source_language or "en"
    model_config = db.get(LanguageModelConfig, language)
    if not model_config or not model_config.enabled:
        raise HTTPException(status_code=503, detail="No language model is configured for the requested language.")
    text = (
        f"Relationship context: {request.relationship_context}\n"
        f"Response language: {language} ({SUPPORTED_LANGUAGES[language]})\n"
        "Write fluent, grammatically correct, natural text in this language and match its appropriate register and style. "
        "Preserve the source's meaning. Do not invent facts; distinguish evidence from uncertainty and say when information is insufficient.\n"
        f"Source type: {source.source_type}\nText to analyze:\n{source.content_text}"
    )
    try:
        client = OpenAI(api_key=api_key)
        completion = await asyncio.to_thread(client.chat.completions.create, model=model_config.analysis_model, messages=[{"role": "system", "content": get_system_prompt()}, {"role": "user", "content": text}], response_format={"type": "json_object"}, temperature=0.2)
        result = json.loads(completion.choices[0].message.content or "{}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail="The AI analysis request failed.") from exc
    record = Analysis(user_id=user.id, source_id=source.id, result_json=json.dumps(result, ensure_ascii=False))
    db.add(record)
    db.commit()
    db.refresh(record)
    return analysis_out(record)


@app.get("/api/v1/me/analyses")
def list_analyses(user: CurrentUser, db: DbSession, limit: int = 25, offset: int = 0) -> dict:
    limit = max(1, min(limit, 100))
    rows = db.scalars(select(Analysis).where(Analysis.user_id == user.id).order_by(Analysis.created_at.desc()).offset(max(0, offset)).limit(limit)).all()
    return {"items": [analysis_out(row) for row in rows], "limit": limit, "offset": max(0, offset)}


@app.get("/api/v1/me/analyses/{analysis_id}")
def get_analysis(analysis_id: str, user: CurrentUser, db: DbSession) -> dict:
    record = db.scalar(select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user.id))
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return analysis_out(record)


@app.delete("/api/v1/me/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str, user: CurrentUser, db: DbSession) -> Response:
    record = db.scalar(select(Analysis).where(Analysis.id == analysis_id, Analysis.user_id == user.id))
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    db.delete(record)
    db.commit()
    return Response(status_code=204)


@app.put("/api/v1/me/voice-control")
def set_voice_control(request: VoiceControlRequest, user: CurrentUser, db: DbSession) -> dict:
    user.voice_control_enabled = request.enabled
    db.commit()
    return {"enabled": user.voice_control_enabled}


@app.post("/api/v1/me/audio-sessions", status_code=201)
def create_audio_session(request: AudioSessionCreate, user: ConsentedUser, db: DbSession) -> dict:
    language = request.language if request.language in SUPPORTED_LANGUAGES else user.preferred_language or "unknown"
    session = AudioSession(user_id=user.id, language=language, save_raw_audio=request.save_raw_audio)
    db.add(session)
    db.commit()
    db.refresh(session)
    return {"audio_session_id": session.id, "status": session.status, "created_at": session.created_at.isoformat()}


@app.get("/api/v1/me/audio-sessions")
def list_audio_sessions(user: CurrentUser, db: DbSession, limit: int = 25, offset: int = 0) -> dict:
    limit = max(1, min(limit, 100))
    sessions = db.scalars(
        select(AudioSession)
        .where(AudioSession.user_id == user.id)
        .order_by(AudioSession.created_at.desc())
        .offset(max(0, offset))
        .limit(limit)
    ).all()
    return {
        "items": [
            {
                "audio_session_id": session.id,
                "status": session.status,
                "language": session.language,
                "created_at": session.created_at.isoformat(),
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "raw_audio_saved": bool(session.raw_audio_path and Path(session.raw_audio_path).is_file()),
                "transcript_source_id": session.transcript_source_id,
                "audio_url": f"/api/v1/me/audio-sessions/{session.id}/audio" if session.raw_audio_path and Path(session.raw_audio_path).is_file() else None,
            }
            for session in sessions
        ],
        "limit": limit,
        "offset": max(0, offset),
    }


@app.get("/api/v1/me/audio-sessions/{session_id}/audio")
def get_audio_file(session_id: str, user: CurrentUser, db: DbSession) -> FileResponse:
    session = db.scalar(select(AudioSession).where(AudioSession.id == session_id, AudioSession.user_id == user.id))
    if not session or not session.raw_audio_path or not Path(session.raw_audio_path).is_file():
        raise HTTPException(status_code=404, detail="Audio recording not found.")
    audio_path = Path(session.raw_audio_path).resolve()
    if not audio_path.is_relative_to(AUDIO_DIR.resolve() / user.id):
        raise HTTPException(status_code=404, detail="Audio recording not found.")
    return FileResponse(audio_path, media_type="audio/webm", filename=f"synapse-{session.id}.webm")


@app.post("/api/v1/me/audio-sessions/{session_id}/controls")
def control_audio_session(session_id: str, request: AudioControl, user: CurrentUser, db: DbSession) -> dict:
    if request.issued_by == "voice" and not user.voice_control_enabled:
        raise HTTPException(status_code=400, detail="Enable voice control in your account first.")
    session = db.scalar(select(AudioSession).where(AudioSession.id == session_id, AudioSession.user_id == user.id))
    if not session:
        raise HTTPException(status_code=404, detail="Audio session not found.")
    if request.action == "delete_session":
        if not request.delete_confirmed:
            raise HTTPException(status_code=400, detail="Confirm deletion first.")
        if session.raw_audio_path:
            Path(session.raw_audio_path).unlink(missing_ok=True)
        db.delete(session)
        db.commit()
        return {"audio_session_id": session_id, "status": "deleted"}
    transitions = {"start": ("ready", "recording"), "pause": ("recording", "paused"), "resume": ("paused", "recording"), "stop": (("recording", "paused"), "completed")}
    allowed, new_status = transitions[request.action]
    if request.action == "start" and not request.start_confirmed:
        raise HTTPException(status_code=400, detail="Confirm recording start first.")
    if session.status not in (allowed if isinstance(allowed, tuple) else (allowed,)):
        raise HTTPException(status_code=409, detail=f"Cannot {request.action} a session in {session.status} state.")
    session.status = new_status
    if request.action == "start":
        session.started_at = utc_now()
    if request.action == "stop":
        session.ended_at = utc_now()
    db.commit()
    return {"audio_session_id": session.id, "status": session.status}


@app.websocket("/api/v1/me/audio-sessions/{session_id}/stream")
async def stream_audio(websocket: WebSocket, session_id: str, format: Literal["webm", "wav", "mp3", "m4a"] = "webm") -> None:
    token = websocket.cookies.get(COOKIE_NAME)
    if not token:
        await websocket.close(code=4401, reason="Sign in is required")
        return
    with SessionLocal() as db:
        login_session = db.get(LoginSession, hash_token(token))
        if not login_session or login_session.expires_at.replace(tzinfo=timezone.utc) <= utc_now():
            await websocket.close(code=4401, reason="Session expired")
            return
        session = db.scalar(select(AudioSession).where(AudioSession.id == session_id, AudioSession.user_id == login_session.user_id))
        if not session or session.status != "recording":
            await websocket.close(code=4409, reason="Audio session is not recording")
            return
        owner_id, language, save_raw_audio = login_session.user_id, session.language, session.save_raw_audio
        user = db.get(User, owner_id)
        preferred_language = user.preferred_language if user else None
        model_language = language if language in SUPPORTED_LANGUAGES else preferred_language or "en"
        language_config = db.get(LanguageModelConfig, model_language)
    await websocket.accept()
    audio = bytearray()
    await websocket.send_json({"type": "session_status", "status": "recording"})
    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                audio.extend(message["bytes"])
                if len(audio) > MAX_AUDIO_BYTES:
                    await websocket.close(code=1009, reason="Audio limit exceeded")
                    return
                continue
            payload = json.loads(message.get("text") or "{}")
            if payload.get("type") == "stop":
                break
            await websocket.send_json({"type": "error", "detail": "Only stop control is accepted on the audio stream."})
    except WebSocketDisconnect:
        return
    except json.JSONDecodeError:
        await websocket.close(code=1003, reason="Invalid stream message")
        return

    raw_path = None
    if audio and save_raw_audio:
        user_dir = AUDIO_DIR / owner_id
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / f"{session_id}.{format}"
        path.write_bytes(audio)
        raw_path = str(path)
    transcript = None
    if audio and os.getenv("OPENAI_API_KEY"):
        audio_file = BytesIO(audio)
        audio_file.name = f"recording.{format}"
        try:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            result = await asyncio.to_thread(client.audio.transcriptions.create, model=language_config.transcription_model if language_config else os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1"), file=audio_file, language=language if language in SUPPORTED_LANGUAGES else None)
            transcript = result.text
        except Exception:
            await websocket.send_json({"type": "error", "detail": "Audio transcription failed."})

    with SessionLocal() as db:
        session = db.scalar(select(AudioSession).where(AudioSession.id == session_id, AudioSession.user_id == owner_id))
        source_id = None
        if transcript:
            source = Source(user_id=owner_id, source_type="transcript", content_text=transcript, language=model_language)
            db.add(source)
            db.flush()
            source_id = source.id
        if session:
            session.status = "completed"
            session.ended_at = utc_now()
            session.raw_audio_path = raw_path
            session.transcript_source_id = source_id
        db.commit()
    await websocket.send_json({"type": "session_status", "status": "completed", "transcript_source_id": source_id, "raw_audio_saved": bool(raw_path)})
    await websocket.close()
