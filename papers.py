"""
Paper Upload Routes
Handles PDF upload (up to 3 files per session), text extraction, 
concept analysis via Gemini, and knowledge graph endpoint.
"""
import logging
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends, status
from bson import ObjectId

from app.services.auth import get_current_user
from app.services.pdf_extractor import extract_text_from_pdf, extract_metadata_heuristic, chunk_text
from app.services.llm import LLMService
from app.config.settings import settings
from app.agent.knowledge_graph import build_full_knowledge_graph
from app.config.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Papers & Knowledge Graph"])

# ─────────────────────────────────────────────────────────────────────────────
# PAPER ANALYSIS PROMPT
# ─────────────────────────────────────────────────────────────────────────────

PAPER_ANALYSIS_PROMPT = """You are a scientific research analyst.
Analyze this research paper excerpt and extract structured information.

Return ONLY valid JSON:
{
  "title": "<paper title if found, else infer from content>",
  "abstract_summary": "<2-3 sentence summary of the paper's core contribution>",
  "domain": "<primary domain: neuroscience|biology|chemistry|physics|psychology|medicine|ecology|other>",
  "key_concepts": ["concept1", "concept2", ...],
  "key_findings": [
    "Finding 1: specific, quantitative finding if available",
    "Finding 2: ...",
    "Finding 3: ..."
  ],
  "methodology": "<brief description of experimental approach>",
  "independent_variable": "<main variable manipulated>",
  "dependent_variable": "<main outcome measured>"
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/sessions/{session_id}/upload-papers", status_code=status.HTTP_201_CREATED)
async def upload_papers(
    session_id: str,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload up to 3 PDF research papers for a session.
    Extracts text, analyzes with Gemini, stores in MongoDB.
    """
    db = get_db()
    user_id = current_user["id"]

    # Validate session ownership
    try:
        session = await db.research_sessions.find_one({
            "_id": ObjectId(session_id), "user_id": user_id
        })
    except Exception:
        session = await db.research_sessions.find_one({
            "_id": session_id, "user_id": user_id
        })

    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 PDF files allowed per upload")

    # Check existing uploaded papers count
    existing_count = await db.papers.count_documents({
        "session_id": session_id, "source": "uploaded_pdf"
    })
    if existing_count + len(files) > 3:
        raise HTTPException(
            status_code=400,
            detail=f"Session already has {existing_count} uploaded paper(s). Maximum is 3 total."
        )

    results = []
    for idx, file in enumerate(files):
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF. Only .pdf files are accepted."
            )

        # Read file bytes
        file_bytes = await file.read()
        if len(file_bytes) > 20 * 1024 * 1024:  # 20MB limit
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' exceeds 20MB limit")

        try:
            # Step 1: Extract raw text
            logger.info(f"Extracting text from: {file.filename}")
            raw_text = extract_text_from_pdf(file_bytes)

            # Step 2: Heuristic metadata
            meta = extract_metadata_heuristic(raw_text, file.filename)

            # Step 3: AI analysis via Gemini
            text_chunk = chunk_text(raw_text, max_chars=5000)
            prompt = f"Filename: {file.filename}\n\nContent:\n{text_chunk}"

            try:
                analysis = await LLMService.generate_structured_json(PAPER_ANALYSIS_PROMPT, prompt)
            except Exception as e:
                logger.warning(f"Gemini analysis failed for {file.filename}: {e}")
                analysis = {
                    "title": meta.get("title", file.filename),
                    "abstract_summary": meta.get("abstract", "")[:500],
                    "domain": "unknown",
                    "key_concepts": [],
                    "key_findings": [],
                    "methodology": "",
                    "independent_variable": "",
                    "dependent_variable": ""
                }

            # Step 4: Merge and store
            title = analysis.get("title") or meta.get("title") or file.filename
            abstract = analysis.get("abstract_summary") or meta.get("abstract", "")

            doc = {
                "user_id": user_id,
                "session_id": session_id,
                "title": title,
                "filename": file.filename,
                "authors": [],
                "abstract": abstract,
                "year": None,
                "source": "uploaded_pdf",
                "url": None,
                "doi": None,
                "citation_count": 0,
                "relevance_score": 1.0,
                "domain": analysis.get("domain", "unknown"),
                "key_concepts": analysis.get("key_concepts", []),
                "key_findings": analysis.get("key_findings", []),
                "methodology": analysis.get("methodology", ""),
                "independent_variable": analysis.get("independent_variable", ""),
                "dependent_variable": analysis.get("dependent_variable", ""),
                "extracted_text": raw_text[:15000],  # Store first 15k chars
                "paper_index": existing_count + idx,
                "created_at": datetime.now(timezone.utc)
            }

            res = await db.papers.insert_one(doc)
            doc["id"] = str(res.inserted_id)
            doc.pop("_id", None)
            doc.pop("extracted_text", None)  # Don't return full text in response

            results.append(doc)
            logger.info(f"Successfully processed paper: {title}")

        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.exception(f"Unexpected error processing {file.filename}")
            raise HTTPException(status_code=500, detail=f"Failed to process {file.filename}: {str(e)}")

    # Update session timestamp
    await db.research_sessions.update_one(
        {"_id": ObjectId(session_id) if len(session_id) == 24 else session_id},
        {"$set": {"updated_at": datetime.now(timezone.utc)}, "$unset": {"knowledge_graph_snapshot": "", "knowledge_graph_updated_at": "", "knowledge_graph_model": ""}}
    )

    return {
        "uploaded": len(results),
        "papers": results,
        "message": f"Successfully analyzed {len(results)} paper(s). Knowledge graph will be available in the Graph tab."
    }


# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE GRAPH ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/sessions/{session_id}/knowledge-graph")
async def get_knowledge_graph(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Build and return the knowledge graph for a session's uploaded papers.
    Finds multi-hop cross-paper bridges (ABC connections).
    """
    db = get_db()
    user_id = current_user["id"]

    # Validate session
    try:
        session = await db.research_sessions.find_one({
            "_id": ObjectId(session_id), "user_id": user_id
        })
    except Exception:
        session = await db.research_sessions.find_one({
            "_id": session_id, "user_id": user_id
        })

    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    # Fetch uploaded papers with extracted text
    cursor = db.papers.find(
        {"session_id": session_id, "source": "uploaded_pdf"},
        sort=[("paper_index", 1)]
    )
    papers = []
    async for p in cursor:
        p["id"] = str(p["_id"])
        papers.append(p)

    if not papers:
        return {
            "nodes": [],
            "edges": [],
            "bridges": [],
            "papers_triples": [],
            "stats": {"total_nodes": 0, "total_edges": 0, "total_bridges": 0, "papers_analyzed": 0},
            "message": "No uploaded papers found. Please upload PDF papers first."
        }

    try:
        graph_data = await build_full_knowledge_graph(papers)
        graph_model = f"Gemini / {settings.GEMINI_MODEL}" if LLMService.is_configured() else "offline heuristic fallback"
        await db.research_sessions.update_one(
            {"_id": session["_id"]},
            {"$set": {"knowledge_graph_snapshot": graph_data, "knowledge_graph_updated_at": datetime.now(timezone.utc), "knowledge_graph_model": graph_model}},
        )
        return graph_data
    except Exception as e:
        logger.exception("Knowledge graph build failed")
        raise HTTPException(status_code=500, detail=f"Failed to build knowledge graph: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# DELETE UPLOADED PAPER
# ─────────────────────────────────────────────────────────────────────────────

@router.delete("/papers/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_paper(paper_id: str, current_user: dict = Depends(get_current_user)):
    """Delete an uploaded paper from a session."""
    db = get_db()
    try:
        paper = await db.papers.find_one({"_id": ObjectId(paper_id), "user_id": current_user["id"]})
        result = await db.papers.delete_one({"_id": ObjectId(paper_id), "user_id": current_user["id"]})
    except Exception:
        paper = await db.papers.find_one({"_id": paper_id, "user_id": current_user["id"]})
        result = await db.papers.delete_one({"_id": paper_id, "user_id": current_user["id"]})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Paper not found")
    if paper and paper.get("session_id"):
        sid = paper["session_id"]
        try:
            session_oid = ObjectId(sid)
        except Exception:
            session_oid = sid
        await db.research_sessions.update_one(
            {"_id": session_oid, "user_id": current_user["id"]},
            {"$unset": {"knowledge_graph_snapshot": "", "knowledge_graph_updated_at": "", "knowledge_graph_model": ""}},
        )
    return None
