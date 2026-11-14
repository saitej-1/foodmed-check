# FoodMed Check: Explainable Drug–Food Interaction Prediction

A full-stack platform that predicts whether a drug and a food compound are likely to interact, using molecular descriptors and an XGBoost classifier, and explains each prediction with SHAP. The app's interface and API use the working name **Aegis DFI**.

> **Research and educational project.** This is not a medical device and is not medical advice.

## Project status

The pipeline is implemented end to end: database seeding, feature extraction, feature selection, model training, a REST API and a web interface.

One part is not finished. `database/seed.py` currently runs in offline mode (`IS_OFFLINE = True`), which assigns placeholder structures to compounds in place of real PubChem structures. A model trained in this mode exercises the whole pipeline, but its scores do not reflect real chemistry. Before reporting results, switch to live PubChem lookups, re-seed the database and retrain.

## Architecture

```
FooDrugs SQL dump ──► database/parse_zenodo.py ──► database/seed.py ──► PostgreSQL
                                                                           │
                                 scripts/train_pipeline.py ◄───────────────┤
                                   │  RDKit descriptors                    │
                                   │  4-stage feature selection            │
                                   ▼                                       ▼
                           XGBoost model (models/) ──────────► FastAPI backend (backend/)
                                                                           │
                                                                           ▼
                                                              Next.js frontend (frontend/)
```

## How it works

**Data.** `database/parse_zenodo.py` reads drug, food and interaction records from the FooDrugs SQL dump (`FinalFooDrugs_v4.sql`), and `database/seed.py` loads them into PostgreSQL. If the dump is missing, a small built-in fallback set is used. The schema covers drugs, foods, their compounds, interaction labels, prediction history, model metadata and user feedback.

**Features.** `pipeline/feature_extractor.py` turns each SMILES string into RDKit 2D descriptors, 192 AUTOCORR2D values, QED properties and a 1,024-bit Morgan fingerprint. For each drug and food pair it keeps the drug features (`D_`), the food features (`F_`) and their absolute differences (`DIFF_`).

**Feature selection.** `pipeline/feature_selector.py` reduces the feature set in four stages: variance threshold, Pearson correlation filter, LASSO, then recursive feature elimination with cross-validation.

**Model.** `pipeline/model_trainer.py` trains a 3-class XGBoost classifier (severe interaction, moderate interaction, no known interaction) with class weighting and 5-fold stratified cross-validation. Pairs with no known interaction are sampled at random from unlabelled drug and food pairs.

**Explainability.** `pipeline/shap_analyzer.py` and the `/api/explain` endpoint compute SHAP values to show which features drove a prediction.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/search/drugs?q=` | Drug name autocomplete |
| GET | `/api/search/foods?q=` | Food name autocomplete |
| POST | `/api/predict` | Predict the interaction class for a drug and food (by name or SMILES) |
| GET | `/api/explain` | SHAP explanation for a drug and food SMILES pair |
| POST | `/api/admin/retrain` | Start retraining in the background |
| GET | `/api/admin/retrain/status` | Retraining status |
| GET | `/api/admin/stats` | Platform statistics |
| GET | `/api/admin/model-info` | Active model details |
| GET | `/health` | API and database health check |

Interactive API docs are served at `/docs` when the backend is running.

## Tech stack

| Area | Tools |
| --- | --- |
| Backend | Python, FastAPI, SQLAlchemy |
| Database | PostgreSQL |
| Machine learning | XGBoost, scikit-learn, RDKit, SHAP |
| Frontend | Next.js, React, TypeScript, Tailwind CSS |

## Run locally

Requirements: Python 3.10 or newer, Node.js 20 or newer, PostgreSQL 14 or newer.

```bash
# 1. Python environment (the retrain endpoint expects it to be named "venv")
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# 2. Configuration
cp .env.example .env        # then edit .env and set your own SECRET_KEY and passwords

# 3. Data: place the FooDrugs dump at datasets/raw/FinalFooDrugs_v4.sql
#    (or set ZENODO_SQL_PATH in .env), then seed and clean the database
python database/seed.py
python scripts/preprocess_pipeline.py

# 4. Train the model (writes to models/)
python scripts/train_pipeline.py

# 5. Start the API at http://localhost:8080
python backend/main.py
```

In a second terminal, start the frontend at http://localhost:3000:

```bash
cd frontend
npm install
npm run dev
```

The frontend calls the API at `http://localhost:8080` by default. Set `NEXT_PUBLIC_API_URL` to change it.

## Known limitations

- Compound structures are placeholders while seeding runs in offline mode (see Project status).
- Feature selection is fitted on the full dataset before the train and test split, which makes test scores optimistic. It should be fitted on the training folds only.
- The "no known interaction" class is sampled from unlabelled pairs, so some of those pairs may be real interactions that are simply not recorded.
- The admin endpoints have no authentication yet.
- The frontend's feedback form posts to `/api/predict/feedback`, which the backend does not implement yet.
- The Docker image currently copies only `backend/`, so it cannot import `database/` and `pipeline/` yet.

## Project structure

```
.
├── backend/            # FastAPI app: main.py, config.py, api/ routers
├── database/           # schema.sql, seeding scripts, DB connection
├── pipeline/           # Feature extraction, selection, training, SHAP
├── scripts/            # Dataset download, preprocessing, training entry point
├── frontend/           # Next.js web interface
├── models/             # Trained model files (generated, not tracked)
├── datasets/           # Data caches and raw files (generated, not tracked)
├── docker-compose.yml
└── .env.example
```

## Author

**Sai Tej**, B.Tech Computer Science and Engineering, Keshav Memorial College of Engineering, Hyderabad.
