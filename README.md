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
- Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Python version: `3.11.11` from `backend/.python-version`

Available endpoints:

- `GET /health`
- `POST /generate-text`
- `POST /summarize`
- `POST /sentiment`
- `POST /question-answering`

Default task models:

- Text generation: `Qwen/Qwen2.5-0.5B-Instruct`
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
