"""
Manual Validation Routes
Allows users/reviewers to manually score, annotate, and approve/reject hypotheses.
This satisfies the "manual validation" academic review requirement.
"""
import logging
import hashlib
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field
from bson import ObjectId

from app.services.auth import get_current_user
from app.config.database import get_db
from app.agent.knowledge_graph import build_full_knowledge_graph
from app.services.validation_report import build_validation_report_pdf
from app.services.llm import LLMService
from app.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Manual Validation"])


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ValidationCreate(BaseModel):
    status: str = Field(
        default="pending",
        description="One of: pending, validated, rejected, needs_revision"
    )
    manual_novelty: Optional[float] = Field(None, ge=0, le=10)
    manual_testability: Optional[float] = Field(None, ge=0, le=10)
    manual_plausibility: Optional[float] = Field(None, ge=0, le=10)
    manual_falsifiability: Optional[float] = Field(None, ge=0, le=10)
    reviewer_notes: Optional[str] = Field(None, max_length=2000)


class ValidationResponse(BaseModel):
    id: str
    hypothesis_id: str
    session_id: str
    user_id: str
    status: str
    manual_novelty: Optional[float] = None
    manual_testability: Optional[float] = None
    manual_plausibility: Optional[float] = None
    manual_falsifiability: Optional[float] = None
    manual_overall: Optional[float] = None
    reviewer_notes: Optional[str] = None
    validated_at: str
    updated_at: Optional[str] = None


class EdgeValidationCreate(BaseModel):
    source: str = Field(..., min_length=1, max_length=300)
    predicate: str = Field(..., min_length=1, max_length=150)
    target: str = Field(..., min_length=1, max_length=300)
    paper_index: int = Field(..., ge=0)
    paper_title: str = Field(default="", max_length=500)
    evidence: str = Field(default="", max_length=2000)
    evidence_verified: bool = False
    decision: str = Field(..., description="supported, unsupported, or needs_revision")
    reviewer_notes: Optional[str] = Field(None, max_length=2000)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

VALID_STATUSES = {"pending", "validated", "rejected", "needs_revision"}
EDGE_DECISIONS = {"supported", "unsupported", "needs_revision"}


def _edge_validation_key(data: EdgeValidationCreate) -> str:
    raw = f"{data.paper_index}|{data.source.casefold()}|{data.predicate.casefold()}|{data.target.casefold()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _edge_validation_response(doc: dict) -> dict:
    return {
        "id": str(doc.get("_id", "")),
        "edge_key": doc["edge_key"],
        "session_id": doc["session_id"],
        "source": doc["source"],
        "predicate": doc["predicate"],
        "target": doc["target"],
        "paper_index": doc["paper_index"],
        "paper_title": doc.get("paper_title", ""),
        "evidence": doc.get("evidence", ""),
        "evidence_verified": doc.get("evidence_verified", False),
        "decision": doc["decision"],
        "reviewer_notes": doc.get("reviewer_notes"),
        "reviewer_id": doc["reviewer_id"],
        "reviewed_at": doc["reviewed_at"].isoformat() if hasattr(doc.get("reviewed_at"), "isoformat") else str(doc.get("reviewed_at", "")),
    }


def compute_manual_overall(data: ValidationCreate) -> Optional[float]:
    scores = [
        s for s in [
            data.manual_novelty,
            data.manual_testability,
            data.manual_plausibility,
            data.manual_falsifiability,
        ]
        if s is not None
    ]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def doc_to_response(v: dict) -> ValidationResponse:
    return ValidationResponse(
        id=str(v["_id"]),
        hypothesis_id=v["hypothesis_id"],
        session_id=v["session_id"],
        user_id=v["user_id"],
        status=v.get("status", "pending"),
        manual_novelty=v.get("manual_novelty"),
        manual_testability=v.get("manual_testability"),
        manual_plausibility=v.get("manual_plausibility"),
        manual_falsifiability=v.get("manual_falsifiability"),
        manual_overall=v.get("manual_overall"),
        reviewer_notes=v.get("reviewer_notes"),
        validated_at=v["validated_at"].isoformat() if hasattr(v.get("validated_at"), "isoformat") else str(v.get("validated_at", "")),
        updated_at=v["updated_at"].isoformat() if hasattr(v.get("updated_at"), "isoformat") else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CREATE / UPSERT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/hypotheses/{hypothesis_id}/validate", response_model=ValidationResponse)
async def create_or_update_validation(
    hypothesis_id: str,
    validation_in: ValidationCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Create or update a manual validation record for a hypothesis.
    Uses upsert so calling this endpoint multiple times updates existing record.
    """
    db = get_db()
    user_id = current_user["id"]

    if validation_in.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Must be one of: {', '.join(VALID_STATUSES)}"
        )

    # Fetch hypothesis to get session_id
    try:
        h = await db.hypotheses.find_one({"_id": ObjectId(hypothesis_id)})
    except Exception:
        h = await db.hypotheses.find_one({"_id": hypothesis_id})

    if not h:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    session_id = str(h.get("session_id", ""))
    now = datetime.now(timezone.utc)
    overall = compute_manual_overall(validation_in)

    # Check for existing validation
    try:
        existing = await db.validations.find_one({
            "hypothesis_id": hypothesis_id, "user_id": user_id
        })
    except Exception:
        existing = None

    if existing:
        # Update
        update_fields = {
            "status": validation_in.status,
            "manual_novelty": validation_in.manual_novelty,
            "manual_testability": validation_in.manual_testability,
            "manual_plausibility": validation_in.manual_plausibility,
            "manual_falsifiability": validation_in.manual_falsifiability,
            "manual_overall": overall,
            "reviewer_notes": validation_in.reviewer_notes,
            "updated_at": now,
        }
        await db.validations.update_one({"_id": existing["_id"]}, {"$set": update_fields})
        existing.update(update_fields)
        return doc_to_response(existing)
    else:
        # Insert
        doc = {
            "hypothesis_id": hypothesis_id,
            "session_id": session_id,
            "user_id": user_id,
            "status": validation_in.status,
            "manual_novelty": validation_in.manual_novelty,
            "manual_testability": validation_in.manual_testability,
            "manual_plausibility": validation_in.manual_plausibility,
            "manual_falsifiability": validation_in.manual_falsifiability,
            "manual_overall": overall,
            "reviewer_notes": validation_in.reviewer_notes,
            "validated_at": now,
            "updated_at": None,
        }
        res = await db.validations.insert_one(doc)
        doc["_id"] = res.inserted_id
        return doc_to_response(doc)


# ─────────────────────────────────────────────────────────────────────────────
# GET ALL VALIDATIONS FOR SESSION
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/validations", response_model=List[ValidationResponse])
async def get_session_validations(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all manual validation records for a session."""
    db = get_db()
    cursor = db.validations.find({
        "session_id": session_id,
        "user_id": current_user["id"]
    }).sort("validated_at", -1)

    results = []
    async for v in cursor:
        results.append(doc_to_response(v))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# GET SINGLE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/hypotheses/{hypothesis_id}/validation", response_model=Optional[ValidationResponse])
async def get_hypothesis_validation(
    hypothesis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get the manual validation record for a specific hypothesis."""
    db = get_db()
    v = await db.validations.find_one({
        "hypothesis_id": hypothesis_id,
        "user_id": current_user["id"]
    })
    if not v:
        return None
    return doc_to_response(v)


# ─────────────────────────────────────────────────────────────────────────────
# DELETE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

@router.delete("/hypotheses/{hypothesis_id}/validate", status_code=status.HTTP_204_NO_CONTENT)
async def delete_validation(
    hypothesis_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Remove a manual validation record."""
    db = get_db()
    result = await db.validations.delete_one({
        "hypothesis_id": hypothesis_id,
        "user_id": current_user["id"]
    })
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Validation record not found")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SESSION VALIDATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/validation-summary")
async def get_validation_summary(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Returns aggregate validation statistics for a session:
    - total hypotheses
    - validated / rejected / pending / needs_revision counts
    - average manual scores per dimension
    """
    db = get_db()
    user_id = current_user["id"]

    total_hyp = await db.hypotheses.count_documents({"session_id": session_id})
    cursor = db.validations.find({"session_id": session_id, "user_id": user_id})

    validations = []
    async for v in cursor:
        validations.append(v)

    status_counts = {"validated": 0, "rejected": 0, "pending": 0, "needs_revision": 0}
    scores = {"novelty": [], "testability": [], "plausibility": [], "falsifiability": [], "overall": []}

    for v in validations:
        s = v.get("status", "pending")
        if s in status_counts:
            status_counts[s] += 1
        for dim in ["manual_novelty", "manual_testability", "manual_plausibility", "manual_falsifiability", "manual_overall"]:
            val = v.get(dim)
            if val is not None:
                key = dim.replace("manual_", "")
                scores[key].append(val)

    avg_scores = {
        dim: round(sum(vals) / len(vals), 2) if vals else None
        for dim, vals in scores.items()
    }

    return {
        "session_id": session_id,
        "total_hypotheses": total_hyp,
        "validated_count": len(validations),
        "status_breakdown": status_counts,
        "average_scores": avg_scores,
        "validation_rate": round(len(validations) / total_hyp * 100, 1) if total_hyp > 0 else 0,
    }


@router.get("/sessions/{session_id}/edge-validations")
async def get_session_edge_validations(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    db = get_db()
    try:
        query = {"_id": ObjectId(session_id), "user_id": current_user["id"]}
    except Exception:
        query = {"_id": session_id, "user_id": current_user["id"]}
    if not await db.research_sessions.find_one(query):
        raise HTTPException(status_code=404, detail="Research session not found")
    cursor = db.edge_validations.find({"session_id": session_id, "reviewer_id": current_user["id"]}).sort("reviewed_at", -1)
    return [_edge_validation_response(item) async for item in cursor]


@router.post("/sessions/{session_id}/edge-validations")
async def save_edge_validation(
    session_id: str,
    review: EdgeValidationCreate,
    current_user: dict = Depends(get_current_user),
):
    if review.decision not in EDGE_DECISIONS:
        raise HTTPException(status_code=400, detail=f"Decision must be one of: {', '.join(sorted(EDGE_DECISIONS))}")
    db = get_db()
    try:
        query = {"_id": ObjectId(session_id), "user_id": current_user["id"]}
    except Exception:
        query = {"_id": session_id, "user_id": current_user["id"]}
    if not await db.research_sessions.find_one(query):
        raise HTTPException(status_code=404, detail="Research session not found")

    edge_key = _edge_validation_key(review)
    key_query = {"session_id": session_id, "reviewer_id": current_user["id"], "edge_key": edge_key}
    now = datetime.now(timezone.utc)
    fields = review.model_dump(exclude={"decision"})
    fields.update({"decision": review.decision, "reviewed_at": now})
    await db.edge_validations.update_one(key_query, {"$set": fields, "$setOnInsert": key_query}, upsert=True)
    saved = await db.edge_validations.find_one(key_query)
    return _edge_validation_response(saved)


@router.get("/sessions/{session_id}/validation-report", response_class=Response)
async def download_validation_report(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Generate a downloadable PDF report of paper-edge traceability and
    human hypothesis review. Quote matching is explicitly not treated as
    scientific validation.
    """
    db = get_db()
    try:
        session_query = {"_id": ObjectId(session_id), "user_id": current_user["id"]}
    except Exception:
        session_query = {"_id": session_id, "user_id": current_user["id"]}
    session = await db.research_sessions.find_one(session_query)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    hypotheses = []
    async for item in db.hypotheses.find({"session_id": session_id}).sort("created_at", 1):
        hypotheses.append(item)
    validations = []
    async for item in db.validations.find({"session_id": session_id, "user_id": current_user["id"]}):
        validations.append(item)
    edge_reviews = []
    async for item in db.edge_validations.find({"session_id": session_id, "reviewer_id": current_user["id"]}):
        edge_reviews.append(item)

    papers = []
    async for paper in db.papers.find({"session_id": session_id, "source": "uploaded_pdf"}).sort("paper_index", 1):
        papers.append(paper)
    try:
        graph = session.get("knowledge_graph_snapshot")
        if graph is None:
            graph = await build_full_knowledge_graph(papers) if papers else {"edges": [], "bridges": []}
            graph_model = f"Gemini / {settings.GEMINI_MODEL}" if LLMService.is_configured() else "offline heuristic fallback"
            await db.research_sessions.update_one(
                {"_id": session["_id"]},
                {"$set": {"knowledge_graph_snapshot": graph, "knowledge_graph_updated_at": datetime.now(timezone.utc), "knowledge_graph_model": graph_model}},
            )
            session["knowledge_graph_model"] = graph_model
    except Exception as exc:
        logger.exception("Unable to build graph for validation report")
        graph = {"edges": [], "bridges": [], "report_note": f"Graph extraction was unavailable while generating this report ({str(exc)[:250]})."}

    pdf_bytes = build_validation_report_pdf(
        session=session,
        papers=papers,
        graph=graph,
        hypotheses=hypotheses,
        validations=validations,
        edge_reviews=edge_reviews,
        extraction_model=session.get("knowledge_graph_model", "Not recorded for saved graph"),
    )
    safe_id = str(session.get("_id", session_id))
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="validation-report-{safe_id}.pdf"'},
    )
