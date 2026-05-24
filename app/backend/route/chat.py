import json
import unicodedata
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from backend.llm import ModelManager
from backend.schema import ChatRequest, PreChatResponse, SessionMessagesResponse, SessionResponse
from database import (
    add_message_rating,
    add_preference,
    check_login,
    create_chat_session,
    delete_chat_session,
    get_chat_session,
    get_message,
    get_message_rating,
    get_session_with_messages,
    get_user_sessions,
)

from .utils import CommonResponse

router = APIRouter()

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_MODES = {"auto", "local", "web", "hybrid"}


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").strip()


def _normalize_source_params(params: dict) -> None:
    """Keep frontend/API source settings unambiguous before forwarding to worker."""
    k_docs = int(params.get("k_docs") or 0)
    k_pages = int(params.get("k_pages") or 0)
    if k_docs <= 0 and k_pages <= 0 and not params.get("use_localdb"):
        return

    raw_mode = str(params.get("source_mode") or "").lower()
    if raw_mode in SOURCE_MODES:
        source_mode = raw_mode
    elif bool(params.get("auto_source", True)):
        source_mode = "auto"
    else:
        use_local = bool(params.get("use_localdb"))
        use_web = bool(params.get("use_websearch"))
        if use_local and use_web:
            source_mode = "hybrid"
        elif use_local:
            source_mode = "local"
        elif use_web:
            source_mode = "web"
        else:
            source_mode = "auto"

    params["source_mode"] = source_mode
    params["auto_source"] = source_mode == "auto"
    if source_mode in {"auto", "hybrid"}:
        params["use_localdb"] = True
        params["use_websearch"] = True
    elif source_mode == "local":
        params["use_localdb"] = True
        params["use_websearch"] = False
    elif source_mode == "web":
        params["use_localdb"] = False
        params["use_websearch"] = True


def _build_school_info(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    return {
        "name": entry.get("name"),
        "acronym": entry.get("acronym") or entry.get("universityCode"),
        "address": entry.get("address"),
        "type": entry.get("type"),
        "phone": entry.get("phone"),
        "website": entry.get("website"),
        "city": entry.get("city"),
    }


@lru_cache(maxsize=1)
def _load_school_names() -> list[dict]:
    school_path = APP_ROOT / "package" / "school_name.json"
    with school_path.open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_school_alias_entries() -> list[dict]:
    names = _load_school_names()
    acronym_lookup = {
        (s.get("acronym") or "").lower(): s for s in names if s.get("acronym")
    }

    alias_path = APP_ROOT / "package" / "school_alias.json"
    with alias_path.open(encoding="utf-8") as f:
        alias_map = json.load(f)

    entries: list[dict] = []
    for key, alias_list in alias_map.items():
        base = acronym_lookup.get(key.lower())
        base_name = base.get("name") if base else None
        base_norm = _normalize_text(base.get("normalized_name")) if base else None

        for alias in alias_list:
            entries.append(
                {
                    "alias": alias,
                    "normalized_alias": _normalize_text(alias),
                    "canonical_name": base_name or alias,
                    "normalized_name": base_norm or _normalize_text(alias),
                    "acronym": base.get("acronym") if base else key,
                }
            )
    return entries


@lru_cache(maxsize=1)
def _load_school_detail_map() -> dict[str, dict]:
    data_path = PROJECT_ROOT / "validation_data" / "uni_data.json"
    if not data_path.exists():
        return {}

    with data_path.open(encoding="utf-8") as f:
        data = json.load(f)

    mapping: dict[str, dict] = {}
    for item in data:
        key = _normalize_text(item.get("name"))
        if not key or key in mapping:
            continue
        info = _build_school_info(item)
        if info:
            mapping[key] = info
    return mapping
    
@router.post("/chat", name="chat")
async def chat(request: Request, data: ChatRequest) -> PreChatResponse:
    """
    Chat route.\n
    This would send a `PreChatResponse` first, which contain `WebSource`, `RagSource`, ...\n
    Then send access answer through `result_url` field.\n
    Result would be stored with a call from worker (See `worker_router`).\n
    Would need server to call `pre_inference` in worker, so we could prevent direct `pre_inference` request from unauthozied user, create new `ChatSession`,  provide chat history, ...
    """
    user = await check_login(request)
    session_id = data.session_id
    if session_id is None:
        session_id = await create_chat_session(user.id)
        if session_id is None:
            raise HTTPException(status_code=500, detail=f"Failed to create new chat session")
    _normalize_source_params(data.params)
    model_output = await ModelManager.pre_inference(session_id, user.id, data.text, data.params)
    if model_output == None:
        raise HTTPException(status_code=500, detail="Failed to inference model")    
    response: PreChatResponse = {
        "session_id": session_id,
        "role": "bot",
        "web_sources": model_output["web_sources"],
        "rag_sources": model_output["rag_sources"],
        "extra_data": model_output["extra_data"],
        "result_url": model_output["result_url"]
    }    
    return response


@router.get("/schools")
async def list_schools(request: Request):
    """Provide school catalog and alias data for client-side detection."""
    await check_login(request)
    schools = _load_school_names()
    aliases = _load_school_alias_entries()

    payload = [
        {
            "name": item.get("name"),
            "normalized_name": item.get("normalized_name")
            or _normalize_text(item.get("name")),
            "acronym": item.get("acronym"),
            "website": item.get("website"),
        }
        for item in schools
    ]
    return {"schools": payload, "aliases": aliases}


@router.get("/schools/info")
async def get_school_info(request: Request, q: str = Query(..., min_length=1)):
    """Return basic school details used by the suggestion popup."""
    await check_login(request)

    target = _normalize_text(q)
    detail_map = _load_school_detail_map()
    info = detail_map.get(target)

    if not info:
        for alias in _load_school_alias_entries():
            if alias["normalized_alias"] == target:
                canonical = alias.get("normalized_name") or target
                info = detail_map.get(canonical)
                if info:
                    break

    if not info:
        matched = next(
            (
                s
                for s in _load_school_names()
                if _normalize_text(s.get("normalized_name")) == target
                or _normalize_text(s.get("name")) == target
                or _normalize_text(s.get("acronym")) == target
            ),
            None,
        )
        info = _build_school_info(matched)

    if not info:
        raise HTTPException(status_code=404, detail="School not found")

    return info


@router.get("/sessions")
async def sessions(request: Request) -> list[SessionResponse]:
    """Get list of `ChatSession`"""
    user = await check_login(request)
    sessions = await get_user_sessions(user.id)
    return [session.to_dict() for session in sessions] #type:ignore

@router.get("/session/{session_id}/messages")
async def session_messages(request: Request, session_id: str) -> SessionMessagesResponse:
    """Get all `ChatMessage` inside a `ChatSession`"""
    user = await check_login(request)
    chat_session = await get_session_with_messages(session_id)
    if chat_session and chat_session.user_id == user.id:
        # Fix web_sources in messages for backward compatibility
        messages = [msg.to_dict()for msg in chat_session.messages]
        result: SessionMessagesResponse = {
            "session": chat_session.to_dict(), #type:ignore
            "messages": messages
        }
        return result
    raise HTTPException(status_code=404, detail=f"Not found session with id: {session_id}")

@router.delete("/session/{session_id}")
async def delete_session(request: Request, session_id: str):
    """Delete a `ChatSession`"""
    user = await check_login(request)
    chat_session = await get_chat_session(session_id)
    if chat_session and chat_session.user_id == user.id:
        chat_session = await delete_chat_session(chat_session)
        return CommonResponse(200, True, "Ok")
    else:
        raise HTTPException(status_code=404, detail=f"Not found session with id: {session_id}")

class RatingRequest(BaseModel):
    rating: int

@router.post("/message/{message_id}/rate")
async def rate_message(request: Request, message_id: str, data: RatingRequest):
    """Rate a bot message (1-5 stars)"""
    user = await check_login(request)
    
    # Validate rating
    if data.rating < 1 or data.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    # Add/update rating
    rating = await add_message_rating(message_id, user.id, data.rating)
    
    if rating is None:
        raise HTTPException(status_code=404, detail=f"Message not found: {message_id}")
    
    return CommonResponse(200, True, "Rating saved successfully")

@router.get("/message/{message_id}/rating")
async def get_rating(request: Request, message_id: str):
    """Get rating for a message"""
    user = await check_login(request)
    rating = await get_message_rating(message_id)
    
    if rating:
        return rating.to_dict()
    else:
        return None

@router.get("/session/{session_id}/latest_bot_message")
async def get_latest_bot_message(request: Request, session_id: str):
    """Get latest bot message ID for rating"""
    user = await check_login(request)
    chat_session = await get_session_with_messages(session_id)
    
    if not chat_session or chat_session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Find latest bot message
    bot_messages = [msg for msg in chat_session.messages if msg.role == "bot"]
    if bot_messages:
        latest = bot_messages[-1]
        return {"message_id": latest.id}
    
    return {"message_id": None}

@router.post("/message/{message_id}/regenerate")
async def regenerate_response(request: Request, message_id: str):
    """
    Regenerate response with different temperature
    Triggered when user rates <= 3 stars
    """
    user = await check_login(request)
    
    # Get original message
    original_msg = await get_message(message_id)
    if not original_msg:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Get session
    chat_session = await get_session_with_messages(original_msg.session_id)
    if not chat_session or chat_session.user_id != user.id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Find user query (message before bot response)
    messages = chat_session.messages
    msg_index = next((i for i, m in enumerate(messages) if m.id == message_id), -1)
    
    if msg_index <= 0:
        raise HTTPException(status_code=400, detail="Cannot find user query")
    
    user_query = messages[msg_index - 1]
    
    # Change temperature for variety
    new_params = original_msg.generation_params.copy()
    old_temp = new_params.get("temperature", 0.7)
    new_params["temperature"] = 0.8 if old_temp < 0.7 else 0.5
    
    # Generate new response
    model_output = await ModelManager.pre_inference(
        session_id=original_msg.session_id,
        user_id=user.id,
        text=user_query.text,
        params=new_params
    )
    
    if not model_output:
        raise HTTPException(status_code=500, detail="Failed to regenerate")
    
    return {
        "original_message_id": message_id,
        "result_url": model_output["result_url"],
        "query_text": user_query.text,
        "variation": {
            "old_temperature": old_temp,
            "new_temperature": new_params["temperature"]
        }
    }

class PreferenceRequest(BaseModel):
    query_text: str
    original_message_id: str
    regenerated_message_id: str
    preferred_message_id: str

@router.post("/preference/submit")
async def submit_preference(request: Request, data: PreferenceRequest):
    """Submit A/B preference after comparison"""
    user = await check_login(request)
    
    # Validate
    if data.preferred_message_id not in [data.original_message_id, data.regenerated_message_id]:
        raise HTTPException(status_code=400, detail="preferred_message_id must be one of the compared messages")
    
    # Save preference
    preference = await add_preference(
        user_id=user.id,
        query_text=data.query_text,
        original_message_id=data.original_message_id,
        regenerated_message_id=data.regenerated_message_id,
        preferred_message_id=data.preferred_message_id,
        trigger_type="low_rating"
    )
    
    if not preference:
        raise HTTPException(status_code=404, detail="Messages not found")
    
    return CommonResponse(200, True, "Preference saved successfully")
