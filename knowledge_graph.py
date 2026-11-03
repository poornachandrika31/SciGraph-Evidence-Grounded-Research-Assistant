"""
Knowledge Graph Builder
Extracts Subject-Predicate-Object triples from research papers using Gemini,
builds a NetworkX graph, finds multi-hop cross-paper bridges (ABC connections),
and serializes graph data for React Flow visualization.
"""
import logging
import json
import hashlib
from typing import List, Dict, Any, Optional
from app.services.llm import LLMService
from app.services.pdf_extractor import normalize_pdf_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

KG_EXTRACTION_SYSTEM_PROMPT = """You are a scientific knowledge graph extraction specialist. Extract scientific concept relationships from the supplied research paper excerpt.

For each relationship identify:
- subject: a specific scientific concept or entity
- predicate: the relationship verb
- object: the concept acted upon
- evidence: an exact, short quotation from the excerpt that supports the relationship
- relation_type: causal, associative, mechanistic, or comparative

Rules:
- Prefer specific, measurable entities; keep entity names concise (2-5 words).
- Extract only relationships explicitly stated in the excerpt. Never complete a mechanism using outside knowledge.
- Extract up to 15 high-quality relationships; return fewer rather than infer unsupported edges.
- Copy evidence verbatim, not paraphrased.
- Return raw JSON only with this schema:
{"domain":"neuroscience|biology|chemistry|physics|psychology|medicine|ecology|other","key_concepts":["..."],"triples":[{"subject":"...","predicate":"...","object":"...","evidence":"...","relation_type":"causal|associative|mechanistic|comparative"}],"key_findings":["..."]}
"""

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

async def extract_triples_from_paper(paper_text: str, paper_index: int, paper_title: str) -> Dict[str, Any]:
    """
    Call Gemini to extract SPO triples from a paper's text chunk.
    Returns dict with domain, key_concepts, triples, key_findings.
    """
    prompt = (
        f"Paper Title: {paper_title}\n"
        f"Paper Index: {paper_index + 1}\n\n"
        f"Paper Content (excerpt):\n{paper_text[:5000]}\n\n"
        "Extract the knowledge graph triples from this paper."
    )
    if not LLMService.is_configured():
        logger.info("Gemini is not configured; extracting only explicit text relationships")
        return _heuristic_triple_extractor(paper_text, paper_index, paper_title)

    try:
        result = await LLMService.generate_structured_json(KG_EXTRACTION_SYSTEM_PROMPT, prompt)
        # A quote check is a traceability signal, not proof that the scientific
        # interpretation is correct. Keep unsupported model claims out of graph edges.
        source = paper_text[:5000]
        verified = []
        for triple in result.get("triples", []):
            quote = str(triple.get("evidence", "")).strip()
            triple["evidence_verified"] = bool(quote and quote.casefold() in source.casefold())
            quote_folded = quote.casefold()
            source_folded = source.casefold()
            subject = str(triple.get("subject", "")).strip()
            obj = str(triple.get("object", "")).strip()
            triple["subject_mentioned"] = bool(subject and subject.casefold() in source_folded)
            triple["object_mentioned"] = bool(obj and obj.casefold() in source_folded)
            triple["subject_in_quote"] = bool(subject and subject.casefold() in quote_folded)
            triple["object_in_quote"] = bool(obj and obj.casefold() in quote_folded)
            triple["review_status"] = "needs_review" if triple["evidence_verified"] else "unsupported"
            if triple["evidence_verified"]:
                verified.append(triple)
        result["triples"] = verified
        result["paper_index"] = paper_index
        result["paper_title"] = paper_title
        return result
    except Exception as e:
        logger.warning(f"Gemini triple extraction failed for paper {paper_index}, falling back to heuristic: {e}")
        return _heuristic_triple_extractor(paper_text, paper_index, paper_title)


def _heuristic_triple_extractor(paper_text: str, paper_index: int, paper_title: str) -> Dict[str, Any]:
    """Extract simple explicit relationships and retain their source sentence.

    This conservative fallback intentionally does not use domain-specific
    presets: a relationship must be found in the paper text to become an edge.
    """
    import re

    triples = []
    key_concepts = []
    key_findings = []
    domain = "unknown"

    # Keep the fallback deliberately modest. Sentence-local matching avoids
    # joining unrelated phrases across the paper and stores the matched sentence.
    relation_patterns = [
        (r"\b(suppresses|inhibits|blocks|attenuates|reduces|decreases)\b", "causal"),
        (r"\b(activates|stimulates|promotes|induces|increases|enhances)\b", "causal"),
        (r"\b(regulates|modulates|controls|mediates)\b", "mechanistic"),
        (r"\b(correlates with|associated with|associates with)\b", "associative"),
        (r"\b(compared with|relative to)\b", "comparative"),
    ]
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", paper_text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for pattern, relation_type in relation_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if not match:
                continue
            subject = sentence[:match.start()].strip(" ,;:-")
            obj = sentence[match.end():].strip(" ,;:-")
            # Avoid treating whole long clauses as entity names in a fallback.
            subject = " ".join(subject.split()[-5:]).strip(" ,;:-")
            obj = " ".join(obj.split()[:7]).strip(" ,;:-")
            if len(subject) < 3 or len(obj) < 3 or subject.casefold() == obj.casefold():
                continue
            triple = {
                "subject": subject[:100], "predicate": match.group(0).lower(),
                "object": obj[:120], "evidence": sentence[:500],
                "relation_type": relation_type, "evidence_verified": True,
                "subject_mentioned": subject.casefold() in paper_text[:5000].casefold(),
                "object_mentioned": obj.casefold() in paper_text[:5000].casefold(),
                "subject_in_quote": subject.casefold() in sentence.casefold(),
                "object_in_quote": obj.casefold() in sentence.casefold(),
                "review_status": "needs_review",
            }
            if triple not in triples:
                triples.append(triple)
                key_concepts.extend([triple["subject"], triple["object"]])
            break
        if len(triples) >= 15:
            break

    # Clean concept list
    seen = set()
    unique_concepts = []
    for c in key_concepts:
        if c.lower() not in seen and len(c) > 2:
            seen.add(c.lower())
            unique_concepts.append(c)

    return {
        "paper_index": paper_index,
        "paper_title": paper_title,
        "domain": domain,
        "key_concepts": unique_concepts[:10],
        "triples": triples[:12],
        "key_findings": key_findings,
    }


def build_networkx_graph(papers_triples: List[Dict[str, Any]]):
    """
    Build a NetworkX DiGraph from extracted triples across multiple papers.
    Each node stores which paper(s) it came from.
    Returns the graph object.
    """
    try:
        import networkx as nx
    except ImportError:
        raise RuntimeError("networkx is not installed. Run: pip install networkx>=3.2")

    G = nx.DiGraph()

    for paper_data in papers_triples:
        paper_idx = paper_data.get("paper_index", 0)
        paper_title = paper_data.get("paper_title", f"Paper {paper_idx + 1}")
        triples = paper_data.get("triples", [])

        for triple in triples:
            subj = triple.get("subject", "").strip()
            pred = triple.get("predicate", "").strip()
            obj = triple.get("object", "").strip()

            if not subj or not obj:
                continue

            # Add/update nodes with paper metadata
            if G.has_node(subj):
                if paper_idx not in G.nodes[subj].get("papers", []):
                    G.nodes[subj]["papers"].append(paper_idx)
            else:
                G.add_node(subj, papers=[paper_idx], label=subj)

            if G.has_node(obj):
                if paper_idx not in G.nodes[obj].get("papers", []):
                    G.nodes[obj]["papers"].append(paper_idx)
            else:
                G.add_node(obj, papers=[paper_idx], label=obj)

            # Add edge
            G.add_edge(
                subj, obj, predicate=pred, paper_index=paper_idx,
                paper_title=paper_title, evidence=triple.get("evidence", ""),
                evidence_verified=triple.get("evidence_verified", False),
                subject_mentioned=triple.get("subject_mentioned", False),
                object_mentioned=triple.get("object_mentioned", False),
                subject_in_quote=triple.get("subject_in_quote", False),
                object_in_quote=triple.get("object_in_quote", False),
                relation_type=triple.get("relation_type", "unknown"),
                review_status=triple.get("review_status", "needs_review"),
            )

    return G


def find_cross_paper_bridges(G, papers_triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find multi-hop paths (A→B→C) where A originates from a different paper than C.
    These are the "ABC connections" — the core scientific discovery mechanism.
    """
    try:
        import networkx as nx
    except ImportError:
        return []

    bridges = []
    # Index: concept → set of paper indices
    concept_papers: Dict[str, set] = {}
    for node, data in G.nodes(data=True):
        concept_papers[node] = set(data.get("papers", []))

    nodes_list = list(G.nodes())

    for start_node in nodes_list:
        start_papers = concept_papers.get(start_node, set())
        # Find all nodes reachable in 2-3 hops
        try:
            paths = nx.single_source_shortest_path(G, start_node, cutoff=3)
        except Exception:
            continue

        for end_node, path in paths.items():
            if len(path) < 3:  # Need at least A→B→C (3 nodes, 2 edges)
                continue
            end_papers = concept_papers.get(end_node, set())
            # Cross-paper condition: start and end come from DIFFERENT papers
            if start_papers and end_papers and start_papers.isdisjoint(end_papers):
                # Build bridge description
                bridge_steps = []
                for i in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[i], path[i+1]) or {}
                    bridge_steps.append({
                        "from": path[i],
                        "predicate": edge_data.get("predicate", "→"),
                        "to": path[i+1],
                        "paper": edge_data.get("paper_title", "")
                    })

                bridge = {
                    "start_concept": start_node,
                    "end_concept": end_node,
                    "path": path,
                    "path_length": len(path) - 1,
                    "steps": bridge_steps,
                    "start_paper_indices": list(start_papers),
                    "end_paper_indices": list(end_papers),
                    "significance": f"Cross-disciplinary link: '{start_node}' from Paper {list(start_papers)[0]+1} connects through {len(path)-2} intermediate concept(s) to '{end_node}' in Paper {list(end_papers)[0]+1}"
                }
                bridges.append(bridge)

    # Sort by path length (shorter = more direct, higher priority)
    bridges.sort(key=lambda b: b["path_length"])
    # Deduplicate and cap at 10 best bridges
    seen = set()
    unique_bridges = []
    for b in bridges:
        key = (b["start_concept"], b["end_concept"])
        if key not in seen:
            seen.add(key)
            unique_bridges.append(b)
        if len(unique_bridges) >= 10:
            break

    return unique_bridges


# Color palette per paper index
PAPER_COLORS = {
    0: {"bg": "#0d9488", "border": "#14b8a6", "text": "#f0fdfa"},  # Teal – Paper 1
    1: {"bg": "#d97706", "border": "#f59e0b", "text": "#fffbeb"},  # Amber – Paper 2
    2: {"bg": "#7c3aed", "border": "#a78bfa", "text": "#f5f3ff"},  # Purple – Paper 3
}
BRIDGE_COLOR = {"bg": "#1e293b", "border": "#94a3b8", "text": "#f1f5f9"}  # Slate – Bridge/Shared


def serialize_for_frontend(
    G,
    papers_triples: List[Dict[str, Any]],
    bridges: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convert NetworkX graph to React Flow compatible nodes/edges format.
    """
    rf_nodes = []
    rf_edges = []
    bridge_node_ids = set()

    # Mark nodes that appear in bridges as highlighted
    for bridge in bridges:
        for node in bridge["path"]:
            bridge_node_ids.add(node)

    # ── Nodes ──
    node_positions: Dict[str, Dict] = {}
    import math
    all_nodes = list(G.nodes(data=True))
    n = len(all_nodes)

    for i, (node_id, data) in enumerate(all_nodes):
        papers = data.get("papers", [0])
        primary_paper = papers[0] if papers else 0

        # Spiral layout for visual spread
        angle = (i / max(n, 1)) * 2 * math.pi * 2
        radius = 200 + (i % 4) * 80
        x = 500 + radius * math.cos(angle)
        y = 400 + radius * math.sin(angle)

        # Multi-paper nodes get bridge color
        color_info = BRIDGE_COLOR if len(papers) > 1 else PAPER_COLORS.get(primary_paper, PAPER_COLORS[0])
        is_bridge = node_id in bridge_node_ids

        rf_nodes.append({
            "id": node_id,
            "type": "default",
            "position": {"x": round(x), "y": round(y)},
            "data": {
                "label": node_id,
                "papers": papers,
                "paperTitles": [
                    papers_triples[p]["paper_title"]
                    for p in papers
                    if p < len(papers_triples)
                ],
                "isBridge": is_bridge,
                "color": color_info,
            },
            "style": {
                "background": color_info["bg"],
                "border": f"2px solid {color_info['border']}",
                "borderRadius": "12px",
                "color": color_info["text"],
                "padding": "8px 12px",
                "fontSize": "11px",
                "fontWeight": "600",
                "boxShadow": f"0 0 12px {color_info['border']}55" if is_bridge else "none",
                "minWidth": "80px",
                "textAlign": "center",
            }
        })

    # ── Edges ──
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        paper_idx = data.get("paper_index", 0)
        color_info = PAPER_COLORS.get(paper_idx, PAPER_COLORS[0])

        rf_edges.append({
            "id": f"e-{i}-{u}-{v}",
            "source": u,
            "target": v,
            "label": data.get("predicate", "→"),
            "animated": False,
            "style": {"stroke": color_info["border"], "strokeWidth": 1.5},
            "labelStyle": {"fill": "#94a3b8", "fontSize": "9px", "fontWeight": "500"},
            "labelBgStyle": {"fill": "#0f172a", "fillOpacity": 0.8},
            "markerEnd": {"type": "arrowclosed", "color": color_info["border"]},
            "data": {
                "predicate": data.get("predicate"),
                "paper_index": paper_idx,
                "paper_title": data.get("paper_title", ""),
                "evidence": data.get("evidence", ""),
                "evidence_verified": data.get("evidence_verified", False),
                "subject_mentioned": data.get("subject_mentioned", False),
                "object_mentioned": data.get("object_mentioned", False),
                "subject_in_quote": data.get("subject_in_quote", False),
                "object_in_quote": data.get("object_in_quote", False),
                "relation_type": data.get("relation_type", "unknown"),
                "review_status": data.get("review_status", "needs_review"),
                "edge_key": hashlib.sha256(
                    f"{paper_idx}|{u.casefold()}|{data.get('predicate', '').casefold()}|{v.casefold()}".encode("utf-8")
                ).hexdigest(),
            }
        })

    return {
        "nodes": rf_nodes,
        "edges": rf_edges,
        "bridges": bridges,
        "stats": {
            "total_nodes": len(rf_nodes),
            "total_edges": len(rf_edges),
            "total_bridges": len(bridges),
            "papers_analyzed": len(papers_triples),
        }
    }


async def build_full_knowledge_graph(papers_db_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main entry point: given a list of paper documents from MongoDB,
    extract triples, build graph, find bridges, serialize for frontend.
    """
    if not papers_db_docs:
        return {"nodes": [], "edges": [], "bridges": [], "stats": {}}

    # Only process papers that have extracted_text
    valid_papers = [
        p for p in papers_db_docs
        if p.get("extracted_text") or p.get("abstract")
    ][:3]  # Max 3 papers

    # Extract triples from each paper concurrently (sequential to avoid rate limits)
    papers_triples = []
    for i, paper in enumerate(valid_papers):
        text = normalize_pdf_text(paper.get("extracted_text") or paper.get("abstract", ""))
        title = paper.get("title", f"Paper {i + 1}")
        triples_data = await extract_triples_from_paper(text, i, title)
        papers_triples.append(triples_data)

    # Build NetworkX graph
    G = build_networkx_graph(papers_triples)

    # Find cross-paper bridges
    bridges = find_cross_paper_bridges(G, papers_triples)

    # Serialize
    graph_data = serialize_for_frontend(G, papers_triples, bridges)
    graph_data["papers_triples"] = [
        {
            "paper_index": pt["paper_index"],
            "paper_title": pt["paper_title"],
            "domain": pt.get("domain", "unknown"),
            "key_concepts": pt.get("key_concepts", []),
            "candidate_edges": len(pt.get("triples", [])),
            "key_findings": pt.get("key_findings", []),
        }
        for pt in papers_triples
    ]

    return graph_data
