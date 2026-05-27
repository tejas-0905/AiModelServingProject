# AI Model Serving Project

FastAPI serves Hugging Face model endpoints and the Vite React frontend provides a professional console for running each workflow.

## Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The API runs at `http://localhost:8000`.

For Render deployment, use:

- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt`
- Start Command: `python app.py`
- Health Check Path: `/health`
- Python version: `3.11.11` from `backend/.python-version`
- Environment:
  - `MODEL_RUNTIME=light` for low-memory Render instances.
  - `MODEL_RUNTIME=transformers` only if your instance has enough memory for Hugging Face models.
  - `ALLOWED_ORIGINS=https://your-frontend-domain.onrender.com,http://localhost:5173`

The default backend runtime is `light` to avoid Render memory-limit restarts. In
`transformers` mode, the backend keeps only one Hugging Face pipeline in memory
at a time, but model loading can still require a larger instance.

The default `backend/requirements.txt` installs only the web API dependencies.
For local transformer experiments, install the optional ML packages too:

```bash
pip install -r requirements.txt -r requirements-ml.txt
```

Available endpoints:

- `GET /health`
- `POST /generate-text`
- `POST /summarize`
- `POST /sentiment`
- `POST /question-answering`

Default task models:

- Text generation: `sshleifer/tiny-gpt2`
- Summarization: `sshleifer/distilbart-cnn-12-6`
- Sentiment: `distilbert-base-uncased-finetuned-sst-2-english`
- Question answering: `distilbert-base-cased-distilled-squad`

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:5173` and proxies `/api` requests to the FastAPI backend.

To point the frontend at another API URL, create `frontend/.env`:

```bash
VITE_API_BASE_URL=http://localhost:8000
```
