"""Create a polished, printable PDF validation report."""
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    PageBreak,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#17233B")
TEAL = colors.HexColor("#087F8C")
PALE = colors.HexColor("#EAF5F6")
MUTED = colors.HexColor("#5D6B7B")
LINE = colors.HexColor("#D9E1E8")
AMBER = colors.HexColor("#A85D08")
RED = colors.HexColor("#A33434")


def _font_names() -> tuple[str, str]:
    font_dir = Path("C:/Windows/Fonts")
    regular = font_dir / "arial.ttf"
    bold = font_dir / "arialbd.ttf"
    if regular.exists() and bold.exists():
        if "ValidationArial" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("ValidationArial", str(regular)))
            pdfmetrics.registerFont(TTFont("ValidationArial-Bold", str(bold)))
        return "ValidationArial", "ValidationArial-Bold"
    return "Helvetica", "Helvetica-Bold"


def _clean(value: Any) -> str:
    text = str(value if value is not None else "")
    replacements = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2192": "->", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2022": "*", "\u2264": "<=", "\u2265": ">=", "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(_clean(text)).replace("\n", "<br/>"), style)


def build_validation_report_pdf(
    *, session: dict, papers: list[dict], graph: dict, hypotheses: list[dict],
    validations: list[dict], edge_reviews: list[dict], extraction_model: str = "Not recorded",
) -> bytes:
    regular, bold = _font_names()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle", fontName=bold, fontSize=23, leading=28,
        textColor=INK, alignment=TA_LEFT, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Subtitle", fontName=regular, fontSize=10, leading=15,
        textColor=MUTED, spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="Section", fontName=bold, fontSize=14, leading=18,
        textColor=INK, spaceBefore=12, spaceAfter=7, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="Subsection", fontName=bold, fontSize=10, leading=14,
        textColor=TEAL, spaceBefore=7, spaceAfter=4, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall", fontName=regular, fontSize=8.5, leading=12,
        textColor=INK, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="MutedSmall", fontName=regular, fontSize=7.7, leading=10.5,
        textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        name="Quote", fontName=regular, fontSize=8, leading=11,
        textColor=INK, leftIndent=7, rightIndent=5,
    ))
    styles.add(ParagraphStyle(
        name="KpiValue", fontName=bold, fontSize=18, leading=22,
        textColor=TEAL, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="KpiLabel", fontName=regular, fontSize=7.5, leading=10,
        textColor=MUTED, alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="TableHead", fontName=bold, fontSize=8, leading=10,
        textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="TableCell", fontName=regular, fontSize=7.7, leading=10,
        textColor=INK,
    ))

    session_title = _clean(session.get("title", "Research session"))
    research_question = _clean(session.get("research_question", "Not specified"))
    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    edges = graph.get("edges", [])
    quote_checked = sum(1 for edge in edges if (edge.get("data") or {}).get("evidence_verified"))
    review_map = {record.get("edge_key"): record for record in edge_reviews}
    hypothesis_map = {record.get("hypothesis_id"): record for record in validations}
    edge_counts = {"supported": 0, "unsupported": 0, "needs_revision": 0}
    for record in edge_reviews:
        decision = record.get("decision", "needs_revision")
        edge_counts[decision] = edge_counts.get(decision, 0) + 1
    status_counts = {"validated": 0, "rejected": 0, "needs_revision": 0, "pending": 0}
    for record in validations:
        status = record.get("status", "pending")
        status_counts[status] = status_counts.get(status, 0) + 1

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title=f"Validation Report - {session_title}",
        author="Evidence-Grounded Scientific Hypothesis Generator",
    )
    story: list[Any] = []
    story.append(_p("Scientific Validation Report", styles["ReportTitle"]))
    story.append(_p("Paper-grounded concept and relationship audit with human review", styles["Subtitle"]))
    meta = Table([
        [_p("RESEARCH SESSION", styles["MutedSmall"]), _p("GENERATED", styles["MutedSmall"])],
        [_p(session_title, styles["BodySmall"]), _p(generated, styles["BodySmall"])],
        [_p("RESEARCH QUESTION", styles["MutedSmall"]), ""],
        [_p(research_question, styles["BodySmall"]), ""],
        [_p("GRAPH EXTRACTION PROVIDER / MODEL", styles["MutedSmall"]), ""],
        [_p(extraction_model, styles["BodySmall"]), ""],
    ], colWidths=[112 * mm, 60 * mm])
    meta.setStyle(TableStyle([
        ("SPAN", (0, 2), (1, 2)), ("SPAN", (0, 3), (1, 3)),
        ("SPAN", (0, 4), (1, 4)), ("SPAN", (0, 5), (1, 5)),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F8FA")),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([meta, Spacer(1, 10)])

    story.append(_p("At a glance", styles["Section"]))
    kpis = [
        (len(papers), "PAPERS"), (len(edges), "GRAPH EDGES"),
        (f"{quote_checked}/{len(edges)}" if edges else "0/0", "QUOTES FOUND"),
        (f"{len(edge_reviews)}/{len(edges)}" if edges else "0/0", "EDGES REVIEWED"),
        (len(hypotheses), "HYPOTHESES"), (f"{len(validations)}/{len(hypotheses)}" if hypotheses else "0/0", "HYPOTHESES REVIEWED"),
    ]
    kpi_cells = []
    for value, label in kpis:
        kpi_cells.append([_p(value, styles["KpiValue"]), _p(label, styles["KpiLabel"])])
    kpi_table = Table([
        [Table([[cell[0]], [cell[1]]], colWidths=[53 * mm]) for cell in kpi_cells[:3]],
        [Table([[cell[0]], [cell[1]]], colWidths=[53 * mm]) for cell in kpi_cells[3:]],
    ], colWidths=[57.3 * mm] * 3)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(kpi_table)
    story.append(_p("What these numbers mean", styles["Section"]))
    story.append(_p(
        "A quote-found result means an extracted quotation matched text supplied to the extractor. Exact subject/object phrase checks do not recognize synonyms or prove scientific meaning. Neither check proves that the relation is supported, its direction is right, or a causal interpretation is justified; human decisions are reported separately.",
        styles["BodySmall"],
    ))

    story.append(_p("Validation protocol", styles["Section"]))
    protocol = [
        "1. Extract candidate concepts and subject - predicate - object relationships from uploaded paper text.",
        "2. Check whether the returned source quotation occurs in the text used for extraction; separately check whether the exact subject and object phrases occur in the paper text and in the quotation.",
        "3. Ask a reviewer to assess concept presence, relation meaning and direction, and causal versus associative strength.",
        "4. Record each edge as supported, unsupported, or needs revision with a written rationale.",
        "5. Rate each generated hypothesis for novelty, testability, plausibility, and falsifiability.",
    ]
    for item in protocol:
        story.append(_p(item, styles["BodySmall"]))

    story.append(_p("Paper inventory", styles["Section"]))
    if papers:
        rows = [[_p("#", styles["TableHead"]), _p("Paper title", styles["TableHead"]), _p("Uploaded file", styles["TableHead"])]]
        for i, paper in enumerate(papers, 1):
            rows.append([
                _p(i, styles["TableCell"]),
                _p(paper.get("title", paper.get("filename", f"Paper {i}")), styles["TableCell"]),
                _p(paper.get("filename", "Unknown"), styles["TableCell"]),
            ])
        table = Table(rows, colWidths=[12 * mm, 97 * mm, 63 * mm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
            ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
    else:
        story.append(_p("No uploaded papers were attached to this session.", styles["BodySmall"]))

    story.append(PageBreak())
    story.append(_p("Relationship audit", styles["Section"]))
    if graph.get("report_note"):
        story.append(_p(f"Report note: {graph['report_note']}", styles["BodySmall"]))
    story.append(_p(
        f"Human edge decisions: {edge_counts['supported']} supported, {edge_counts['unsupported']} unsupported, {edge_counts['needs_revision']} needs revision. Unreviewed edge count: {max(0, len(edges) - len(edge_reviews))}.",
        styles["BodySmall"],
    ))
    if edges:
        for i, edge in enumerate(edges, 1):
            data = edge.get("data") or {}
            review = review_map.get(data.get("edge_key"))
            decision = review.get("decision", "Not reviewed") if review else "Not reviewed"
            quote_status = "Quote found" if data.get("evidence_verified") else "No matching quote"
            decision_color = TEAL if decision == "supported" else RED if decision == "unsupported" else AMBER
            block = [
                _p(f"Edge {i}. {edge.get('source', '?')} -{edge.get('label', '?')}-> {edge.get('target', '?')}", styles["Subsection"]),
                _p(f"Paper: {data.get('paper_title', 'Unknown')} | Relation: {data.get('relation_type', 'unknown')} | Traceability: {quote_status}", styles["MutedSmall"]),
                _p(f"Exact phrase in paper: subject {'found' if data.get('subject_mentioned') else 'not found'}; object {'found' if data.get('object_mentioned') else 'not found'} | In quote: subject {'found' if data.get('subject_in_quote') else 'not found'}; object {'found' if data.get('object_in_quote') else 'not found'} (synonyms are not checked automatically)", styles["MutedSmall"]),
                _p(f"Human decision: {_clean(decision).replace('_', ' ').title()}", ParagraphStyle(name=f"Decision{i}", parent=styles["BodySmall"], textColor=decision_color)),
            ]
            if review and review.get("reviewer_notes"):
                block.append(_p(f"Reviewer rationale: {review['reviewer_notes']}", styles["BodySmall"]))
            block.append(_p(f"Source quotation: {data.get('evidence') or 'No quotation returned.'}", styles["Quote"]))
            block.append(Spacer(1, 5))
            story.append(KeepTogether(block))
    else:
        story.append(_p("No relationships were extracted from the available text.", styles["BodySmall"]))

    story.append(_p("Hypothesis review", styles["Section"]))
    story.append(_p(
        f"Reviewed status totals: {status_counts['validated']} validated, {status_counts['rejected']} rejected, {status_counts['needs_revision']} needs revision, {max(0, len(hypotheses) - len(validations)) + status_counts['pending']} pending.",
        styles["BodySmall"],
    ))
    if hypotheses:
        for i, hypothesis in enumerate(hypotheses, 1):
            validation = hypothesis_map.get(str(hypothesis.get("_id"))) or hypothesis_map.get(hypothesis.get("id"))
            status = validation.get("status", "pending") if validation else "pending"
            block = [
                _p(f"Hypothesis {i} - {status.replace('_', ' ').title()}", styles["Subsection"]),
                _p(hypothesis.get("hypothesis", "No hypothesis text recorded."), styles["BodySmall"]),
            ]
            for label, field in [
                ("Novelty", "manual_novelty"), ("Testability", "manual_testability"),
                ("Plausibility", "manual_plausibility"), ("Falsifiability", "manual_falsifiability"),
            ]:
                value = validation.get(field) if validation else None
                block.append(_p(f"{label}: {value if value is not None else 'Not scored'} / 10", styles["MutedSmall"]))
            if validation and validation.get("reviewer_notes"):
                block.append(_p(f"Reviewer notes: {validation['reviewer_notes']}", styles["BodySmall"]))
            story.append(KeepTogether(block + [Spacer(1, 5)]))
    else:
        story.append(_p("No hypotheses have been generated in this session.", styles["BodySmall"]))

    story.append(_p("Recommended model comparison", styles["Section"]))
    comparison = [
        "Select a fixed paper set. Run Gemini, GPT, and Claude on the same first 5,000 text characters, prompt/schema, and settings; log exact model/version and run date.",
        "Before viewing model outputs, ask two domain reviewers to independently create a reference graph from the papers. Adjudicate disagreements and freeze the gold labels.",
        "Randomize and anonymize outputs, keeping the model-to-code key private. Have both reviewers independently label concept presence, relation support, direction, and causal wording.",
        "Match predicted edges to the reference graph using pre-defined normalization rules. Count reference edges no model predicted when calculating recall.",
        "Compare precision, recall, F1, unsupported-edge rate, raw reviewer agreement, and Cohen's kappa. Include confidence intervals and representative errors.",
        "Current app graph provider: Gemini when configured, otherwise offline heuristics. GPT and Claude runs are collected separately; this app does not calculate comparison metrics yet.",
    ]
    for i, item in enumerate(comparison, 1):
        story.append(_p(f"{i}. {item}", styles["BodySmall"]))
    story.append(_p("Limitations", styles["Section"]))
    limitations = [
        "PDF text extraction can miss tables, figures, equations, footnotes, and scanned pages.",
        "A verbatim quote may not entail the extracted edge; reviewers must read the sentence in context and check the study design.",
        "This report summarizes one session and reviewer records; it is not evidence of causal truth or experimental replication.",
    ]
    for item in limitations:
        story.append(_p(f"- {item}", styles["BodySmall"]))

    def draw_page(canvas, document):
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(TEAL)
        canvas.rect(0, height - 5 * mm, width, 5 * mm, stroke=0, fill=1)
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
        canvas.setFont(bold, 7.5)
        canvas.setFillColor(INK)
        canvas.drawString(18 * mm, 9 * mm, "EVIDENCE-GROUNDED RESEARCH REVIEW")
        canvas.setFont(regular, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(width - 18 * mm, 9 * mm, f"{document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return buffer.getvalue()
