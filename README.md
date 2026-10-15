# DataCleanAI

DataCleanAI is a local, rule-based data profiling and preprocessing dashboard. It does **not** call an AI model or external service: objectives are interpreted using a transparent keyword/alias parser and every transformation is chosen and confirmed by the user.

## What works

- Upload CSV, XLS/XLSX, and JSON-record datasets with type/content validation and a size limit.
- Profile columns, missing/duplicate values, suspicious text values, and paginated previews.
- Interpret supported plain-English objectives, with explicit uncertainty and unsupported-request feedback.
- Review and run allowlisted cleaning operations: missing-value treatment, duplicate removal, outlier flags/capping/removal, conversions, category trimming/case normalization, and scaling.
- Compare opt-in strategy simulations without mutating original data; select a candidate and apply it only after approval.
- Persist session/history metadata in SQLite, undo to a saved safe snapshot, and export CSV, XLSX, or JSON reports.

## Architecture

`backend/app` contains FastAPI routes, services, schemas, and SQLite history. Files live under `backend/data` (intentionally ignored). `frontend` is a React + TypeScript + Tailwind Vite client. The API docs are at `http://127.0.0.1:8000/docs`.

## Windows PowerShell setup

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). Copy `.env.example` to `.env` only if you need to adjust limits or API URL.

## Intent parser vocabulary

Supported aliases include: `missing`, `null`, `nan`, `fill`, `impute`; `duplicate`, `dedupe`; `outlier`, `anomaly`; `numeric`, `date`, `datetime`, `convert`; `standardize`, `trim`, `categorical`; `scale`, `normalize`, `machine learning`. Column names surrounded by a matching phrase are recognized. Unsupported wording is reported instead of being treated as an instruction.

## Local NLP and planning

Submitting the objective automatically uses a local TF-IDF matcher (word unigrams/bigrams and cosine similarity) against the maintained catalog in backend/app/intent_catalog.py. Deterministic phrase rules validate candidates, segment conjunctions into clauses, handle supported negations, and reject unknown wording. Similarity is a lexical score, not a probability.

The planner selects columns independently for each clause: missing-value operations prefer columns with missing values, categorical operations prefer text columns, and outlier/scaling operations prefer numerical non-identifier columns. Objective recommendations, profile recommendations, and manual selections remain separate until the user explicitly adds recommendations and runs cleaning.

Run the held-out lightweight evaluation from backend with: .\\.venv\\Scripts\\python.exe -m scripts.evaluate_nlp. It writes backend/scripts/nlp_evaluation.json. This small labeled set is a regression check only and does not demonstrate broad natural-language understanding.

The evaluation uses 25 independently authored held-out examples and reports keyword-only, TF-IDF-only, and combined TF-IDF-plus-rule results, including micro precision/recall/F1, macro-F1, per-intent scores, exact multi-intent matches, and unknown-request rejection. The combined validator may match the conservative rule baseline; it is designed to reject unsafe unsupported predictions rather than guarantee a higher score.

Browser automation is not configured in this lightweight local project. API integration tests exercise the upload, plan, simulation, cleaning, export, history, and undo lifecycle; the React production build verifies the client type-checks and bundles.

## Outlier flags and exports

The default outlier strategy is detection-only. It adds an _outlier_flag boolean column to the cleaned working dataset, preview, CSV/XLSX exports, and report; it does not remove source rows. Undo restores the preceding snapshot. CSV downloads prefix spreadsheet-formula-like text values (=, +, -, @) with an apostrophe to mitigate formula injection.

## Simulator and quality scores

Simulation candidates run on independent copies. Its transparent advisory score is `0.45 × completeness + 0.25 × record retention + 0.15 × uniqueness + 0.15 × (1 - changed-cell ratio)`. It is a comparison aid, not a claim that one strategy is objectively best; destructive changes receive warnings. Completeness is nonmissing cells / all cells, uniqueness is unique rows / rows, and retention is output rows / input rows.

## Safety and limitations

The original upload is never edited. Simulations are opt-in, destructive plans need confirmation, filenames are server-generated, and the backend only runs registered operations. This is a trusted-user local-development application, not a hardened multi-tenant deployment. JSON support is records/array or a top-level object containing records. XLS support depends on `xlrd`; very large datasets are rejected. Data constraints and semantic category merging require user-defined rules and are not inferred.

## Tests

```powershell
cd backend
pytest -q
cd ..\frontend
npm run build
```
