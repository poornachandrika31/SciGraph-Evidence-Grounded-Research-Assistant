from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime, timezone
from typing import List
from bson import ObjectId
from app.schemas.session import SessionCreate, SessionUpdate, SessionResponse
from app.services.auth import get_current_user
from app.config.database import get_db

router = APIRouter(prefix="/api/sessions", tags=["Research Sessions"])

@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(session_in: SessionCreate, current_user: dict = Depends(get_current_user)):
    db = get_db()
    user_id = current_user["id"]
    
    title = session_in.title or session_in.research_question[:60]
    now = datetime.now(timezone.utc)
    
    doc = {
        "user_id": user_id,
        "title": title,
        "research_question": session_in.research_question,
        "created_at": now,
        "updated_at": now
    }
    result = await db.research_sessions.insert_one(doc)
    session_id = str(result.inserted_id)

    # Automatically add initial welcome message from AI assistant
    welcome_msg = {
        "user_id": user_id,
        "session_id": session_id,
        "role": "assistant",
        "content": (
            f"Welcome to your research workspace! We are investigating:\n"
            f"> **\"{session_in.research_question}\"**\n\n"
            f"You can prompt me to:\n"
            f"- Search scientific literature (`Semantic Scholar` & `PubMed`)\n"
            f"- Formulate multiple testable candidate hypotheses\n"
            f"- Evaluate novelty, testability, and falsifiability\n"
            f"- Design controlled laboratory experiments"
        ),
        "timestamp": now
    }
    await db.messages.insert_one(welcome_msg)

    return SessionResponse(
        id=session_id,
        user_id=user_id,
        title=title,
        research_question=session_in.research_question,
        created_at=now,
        updated_at=now,
        hypotheses_count=0,
        papers_count=0,
        experiments_count=0
    )

@router.get("", response_model=List[SessionResponse])
async def list_sessions(current_user: dict = Depends(get_current_user)):
    db = get_db()
    user_id = current_user["id"]
    cursor = db.research_sessions.find({"user_id": user_id}).sort("updated_at", -1)
    
    sessions = []
    async for s in cursor:
        s_id = str(s["_id"])
        # Aggregate counts
        h_count = await db.hypotheses.count_documents({"session_id": s_id})
        p_count = await db.papers.count_documents({"session_id": s_id})
        e_count = await db.experiments.count_documents({"session_id": s_id})
        
        sessions.append(SessionResponse(
            id=s_id,
            user_id=user_id,
            title=s.get("title", "Untitled Session"),
            research_question=s.get("research_question", ""),
            created_at=s.get("created_at", datetime.now(timezone.utc)),
            updated_at=s.get("updated_at", datetime.now(timezone.utc)),
            hypotheses_count=h_count,
            papers_count=p_count,
            experiments_count=e_count
        ))
    return sessions

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    user_id = current_user["id"]
    try:
        query = {"_id": ObjectId(session_id), "user_id": user_id}
    except Exception:
        query = {"_id": session_id, "user_id": user_id}
        
    s = await db.research_sessions.find_one(query)
    if not s:
        raise HTTPException(status_code=404, detail="Research session not found")

    s_id = str(s["_id"])
    h_count = await db.hypotheses.count_documents({"session_id": s_id})
    p_count = await db.papers.count_documents({"session_id": s_id})
    e_count = await db.experiments.count_documents({"session_id": s_id})

    return SessionResponse(
        id=s_id,
        user_id=user_id,
        title=s.get("title", "Untitled Session"),
        research_question=s.get("research_question", ""),
        created_at=s.get("created_at"),
        updated_at=s.get("updated_at"),
        hypotheses_count=h_count,
        papers_count=p_count,
        experiments_count=e_count
    )

@router.patch("/{session_id}", response_model=SessionResponse)
async def update_session(session_id: str, session_update: SessionUpdate, current_user: dict = Depends(get_current_user)):
    db = get_db()
    user_id = current_user["id"]
    try:
        query = {"_id": ObjectId(session_id), "user_id": user_id}
    except Exception:
        query = {"_id": session_id, "user_id": user_id}

    update_fields = {}
    if session_update.title is not None:
        update_fields["title"] = session_update.title
    if session_update.research_question is not None:
        update_fields["research_question"] = session_update.research_question

    if update_fields:
        update_fields["updated_at"] = datetime.now(timezone.utc)
        await db.research_sessions.update_one(query, {"$set": update_fields})

    return await get_session(session_id, current_user)

@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, current_user: dict = Depends(get_current_user)):
    db = get_db()
    user_id = current_user["id"]
    try:
        query = {"_id": ObjectId(session_id), "user_id": user_id}
    except Exception:
        query = {"_id": session_id, "user_id": user_id}

    s = await db.research_sessions.find_one(query)
    if not s:
        raise HTTPException(status_code=404, detail="Research session not found")

    # Cascading delete of all session resources
    await db.research_sessions.delete_one(query)
    await db.messages.delete_many({"session_id": session_id})
    await db.papers.delete_many({"session_id": session_id})
    await db.hypotheses.delete_many({"session_id": session_id})
    await db.experiments.delete_many({"session_id": session_id})
    return None
