# SciGraph — Evidence-Grounded Research Assistant

SciGraph is a full-stack research-support application for exploring scientific papers, mapping relationships between concepts, generating paper-grounded hypotheses, and documenting human review.

> **Research prototype:** Automated quotation and phrase checks help reviewers inspect evidence. They do not prove that a paper supports a claim. Review extracted relationships and generated hypotheses in context before relying on them.

## Features

- **Research sessions:** Create authenticated sessions for research questions.
- **PDF uploads:** Upload up to three selectable-text research papers per session.
- **Paper analysis:** Extract text and metadata from PDFs and search scientific literature.
- **Knowledge graphs:** Extract subject–predicate–object relationships from papers. Build directed graphs with NetworkX, identify multi-hop cross-paper connections, and explore the results in an interactive React Flow view.
- **Hypothesis generation:** Generate hypotheses grounded in available papers, including variable breakdowns and supporting evidence.
- **Experimental protocol drafts:** Generate draft experimental designs for hypotheses.
- **Human review:** Inspect extracted quotations and exact concept phrases, record decisions and rationales for graph relationships, and score hypotheses.
- **Blind scoring:** Record reviewer scores before revealing AI-generated scores.
- **PDF reports:** Export a session report containing paper details, graph evidence checks, human reviews, and hypothesis scores.
- **Model-comparison materials:** Copy the shared extraction prompt and download CSV templates for a manual, blinded model comparison.

## Technology

| Area | Technologies |
| --- | --- |
| Frontend | React, TypeScript, Vite, Tailwind CSS, React Flow |
| Backend | Python, FastAPI, Pydantic |
| Database | MongoDB, Motor, PyMongo |
| AI workflow | Google Gemini, LangGraph, heuristic fallbacks |
| Graph processing | NetworkX |
| PDF tools | pypdf, ReportLab |

## Research workflow

```text
Research question
        ↓
Create a research session
        ↓
Upload 1–3 papers
        ↓
Extract paper text and relationships
        ↓
Explore the knowledge graph and cross-paper connections
        ↓
Review evidence and record decisions
        ↓
Generate hypotheses and experiment drafts
        ↓
Download a PDF validation report
```

## Requirements

- Python 3.10 or later
- Node.js 18 or later
- MongoDB local instance or MongoDB Atlas
- Gemini API key for model-powered analysis (optional; some workflows have heuristic fallbacks)

## Run locally on Windows

### 1. Configure and start the backend

From the repository root, open PowerShell and run:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `backend/.env` and set your configuration:

```dotenv
MONGODB_URI=mongodb://localhost:27017
DATABASE_NAME=scientific_hypothesis_db
GEMINI_API_KEY=your_gemini_api_key
JWT_SECRET=replace_with_a_long_random_secret
```

For MongoDB Atlas, use your Atlas connection string as `MONGODB_URI`. Keep `.env` private. Never commit API keys, passwords, or production secrets.

Start the API from the `backend` directory:

```powershell
uvicorn app.main:app --reload --port 8000
```

API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Start the frontend

Open a second PowerShell terminal in the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173), register, and create a research session.

## Try the sample workflow

Sample PDFs are available in `backend/sample_papers/`. They are demo fixtures, not a scientific gold standard.

1. Register and create a session. Example research question: *How might nocturnal light exposure affect sleep and working memory?*
2. Upload the sample papers.
3. Open the **Graph** view to inspect extracted relationships and cross-paper connections.
4. Select a graph relationship to inspect its quoted evidence and record a human review.
5. Generate hypotheses, enter your independent scores and review notes, then reveal AI scores if desired.
6. Download the PDF validation report.

## Main API routes

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/api/sessions` | Create a research session |
| `POST` | `/api/sessions/{session_id}/upload-papers` | Upload papers to a session |
| `GET` | `/api/sessions/{session_id}/knowledge-graph` | Build or retrieve the session knowledge graph |
| `POST` | `/api/sessions/{session_id}/edge-validations` | Save a human graph-edge review |
| `POST` | `/api/hypotheses/{hypothesis_id}/validate` | Save a human hypothesis review |
| `GET` | `/api/sessions/{session_id}/validation-report` | Download the PDF validation report |

## Evidence checks and limitations

- Quote traceability checks whether an extracted quotation appears in the text excerpt sent to the graph extractor.
- Concept checks look for exact subject and object phrases in supplied paper text and quotations. They do not recognize synonyms or abbreviations.
- A matching quotation does not prove that the relationship is supported, correctly directed, or causal. Human review is needed.
- PDF text extraction may miss content in scanned pages, figures, tables, equations, and complex layouts.
- Gemini is used for model-powered graph extraction when configured. Heuristic fallback behavior is not equivalent to model analysis.
- GPT and Claude are not called by the application. The comparison tab provides materials for a manual comparison; it does not calculate precision, recall, F1, or reviewer agreement.
- Sample papers are for demonstrating the UI and are not a validation dataset.

## Repository layout

```text
backend/
  app/
    agent/       Research workflow and knowledge-graph construction
    config/      Application and database configuration
    routes/      FastAPI endpoints
    schemas/     API data models
    services/    LLM, literature, PDF, auth, and report services
  sample_papers/ Demo PDFs
frontend/
  src/
    components/  Graph, upload, review, and chat components
    pages/       Application screens
```

## License

No license file is currently included. Add a `LICENSE` file before redistributing the project or accepting external contributions under specific terms.
