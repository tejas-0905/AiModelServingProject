import gc
import logging
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


logger = logging.getLogger("ai-model-serving")

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

TEXT_GENERATION_MODEL = os.getenv(
    "TEXT_GENERATION_MODEL", "sshleifer/tiny-gpt2"
)
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6")
SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
)
QUESTION_ANSWERING_MODEL = os.getenv(
    "QUESTION_ANSWERING_MODEL", "distilbert-base-cased-distilled-squad"
)
HF_CACHE_DIR = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
MODEL_RUNTIME = os.getenv("MODEL_RUNTIME", "light").strip().lower()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


app = FastAPI(
    title="AI Model Serving API",
    description="FastAPI backend serving multiple Hugging Face models.",
    version="1.0.0",
)

DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173,"
    "http://localhost:4173,"
    "http://127.0.0.1:4173"
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
]
allow_all_origins = "*" in allowed_origins
allowed_origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else allowed_origins,
    allow_origin_regex=allowed_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def log_startup_config():
    logger.warning(
        "Starting AI Model Serving API on port=%s runtime=%s allowed_origins=%s",
        os.getenv("PORT", "8000"),
        MODEL_RUNTIME,
        ",".join(allowed_origins),
    )


def get_cors_origin(origin: str | None) -> str:
    if allow_all_origins:
        return "*"

    if origin and origin in allowed_origins:
        return origin

    if origin and allowed_origin_regex and re.fullmatch(allowed_origin_regex, origin):
        return origin

    return "null"


@app.options("/{full_path:path}", include_in_schema=False)
def preflight(full_path: str, request: Request):
    requested_headers = request.headers.get("access-control-request-headers", "*")
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": get_cors_origin(request.headers.get("origin")),
            "Access-Control-Allow-Methods": "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
            "Access-Control-Allow-Headers": requested_headers,
            "Access-Control-Max-Age": "600",
        },
    )


class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    max_new_tokens: int = Field(70, ge=10, le=120)
    temperature: float = Field(0.8, ge=0.1, le=1.5)
    top_p: float = Field(0.95, ge=0.1, le=1.0)


class TextGenerationResponse(BaseModel):
    task: Literal["text-generation"]
    model: str
    prompt: str
    generated_text: str


class SummarizationRequest(BaseModel):
    text: str = Field(..., min_length=30, max_length=10000)
    max_length: int = Field(150, ge=30, le=300)
    min_length: int = Field(20, ge=8, le=120)


class SummarizationResponse(BaseModel):
    task: Literal["summarization"]
    model: str
    summary: str


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=3000)


class SentimentResponse(BaseModel):
    task: Literal["sentiment-analysis"]
    model: str
    label: str
    score: float
    sentiment: Literal["positive", "negative", "neutral"]


class QuestionAnsweringRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    context: str = Field(..., min_length=20, max_length=10000)
    max_answer_length: Optional[int] = Field(80, ge=5, le=200)


class QuestionAnsweringResponse(BaseModel):
    task: Literal["question-answering"]
    model: str
    question: str
    answer: str
    score: float
    start: int
    end: int


@lru_cache(maxsize=1)
def resolve_cached_model(model_name: str) -> str:
    cache_name = f"models--{model_name.replace('/', '--')}"
    model_cache = HF_CACHE_DIR / cache_name
    refs_main = model_cache / "refs" / "main"

    if refs_main.exists():
        commit = refs_main.read_text(encoding="utf-8").strip()
        snapshot_path = model_cache / "snapshots" / commit
        if snapshot_path.exists():
            return str(snapshot_path)

    snapshots_dir = model_cache / "snapshots"
    if snapshots_dir.exists():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            return str(max(snapshots, key=lambda path: path.stat().st_mtime))

    return model_name


def load_pipeline():
    try:
        import torch
        import transformers.utils.import_utils as transformers_import_utils

        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
        transformers_import_utils._torchvision_available = False

        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Could not import transformers.pipeline. Rebuild the backend environment "
            "with the pinned dependencies in requirements.txt."
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not import transformers.pipeline. This is usually caused by an "
            "incompatible torchvision package in the runtime; remove torchvision or "
            "rebuild the environment from requirements.txt."
        ) from exc

    return pipeline


_pipeline_lock = threading.Lock()
_active_pipeline = {"key": None, "pipeline": None}


def unload_active_pipeline():
    _active_pipeline["key"] = None
    _active_pipeline["pipeline"] = None
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_single_pipeline(key: str, task_name: str, model_name: str):
    with _pipeline_lock:
        if _active_pipeline["key"] == key and _active_pipeline["pipeline"] is not None:
            return _active_pipeline["pipeline"]

        unload_active_pipeline()
        pipeline = load_pipeline()

        try:
            loaded_pipeline = pipeline(
                task_name,
                model=resolve_cached_model(model_name),
            )
        except Exception:
            unload_active_pipeline()
            raise

        _active_pipeline["key"] = key
        _active_pipeline["pipeline"] = loaded_pipeline
        return loaded_pipeline


def get_text_generator():
    generator = get_single_pipeline(
        "text-generation",
        "text-generation",
        TEXT_GENERATION_MODEL,
    )
    generator.model.generation_config.do_sample = False
    generator.model.generation_config.temperature = None
    generator.model.generation_config.top_p = None
    generator.model.generation_config.top_k = None
    return generator


def get_summarizer():
    return get_single_pipeline(
        "summarization",
        "summarization",
        SUMMARIZATION_MODEL,
    )


def get_sentiment_analyzer():
    return get_single_pipeline(
        "sentiment-analysis",
        "sentiment-analysis",
        SENTIMENT_MODEL,
    )


def get_question_answerer():
    return get_single_pipeline(
        "question-answering",
        "question-answering",
        QUESTION_ANSWERING_MODEL,
    )


def build_generation_prompt(user_prompt: str) -> str:
    cleaned_prompt = user_prompt.strip()
    return (
        "<|im_start|>system\n"
        "You are a helpful AI assistant. Answer the user's request directly, clearly, "
        "and in simple language. If the user gives an incomplete phrase, complete it "
        "with a useful explanation instead of asking unrelated questions.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{cleaned_prompt}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def clean_generated_text(text: str) -> str:
    cleaned = text.strip()
    stop_markers = [
        "<|im_end|>",
        "<|im_start|>",
        "\nUser:",
        "\nuser:",
        "\nAssistant:",
        "\nassistant:",
        "\nQuestion:",
    ]

    for marker in stop_markers:
        marker_index = cleaned.find(marker)
        if marker_index != -1:
            cleaned = cleaned[:marker_index].strip()

    return cleaned


def normalized_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def fallback_summary(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]

    if len(sentences) >= 2:
        words = " ".join(sentences[:2]).split()
    else:
        words = cleaned.split()

    target_words = max(10, min(35, len(words) // 2 if len(words) > 24 else len(words)))
    summary = " ".join(words[:target_words]).strip()

    if len(words) > target_words and not summary.endswith("..."):
        summary = f"{summary}..."

    return summary or cleaned


def lightweight_generation(prompt: str) -> str:
    cleaned = re.sub(r"\s+", " ", prompt).strip()
    if not cleaned:
        return "Please provide a prompt so I can generate a response."

    return (
        "Here is a concise response based on your prompt: "
        f"{cleaned[:220]}"
        f"{'...' if len(cleaned) > 220 else ''}"
    )


def lightweight_sentiment(text: str) -> tuple[str, float, Literal["positive", "negative", "neutral"]]:
    positive_words = {
        "amazing",
        "clear",
        "easy",
        "excellent",
        "fast",
        "good",
        "great",
        "happy",
        "helpful",
        "love",
        "positive",
        "ready",
        "smooth",
        "useful",
        "works",
    }
    negative_words = {
        "bad",
        "broken",
        "confusing",
        "crash",
        "error",
        "fail",
        "failed",
        "hate",
        "issue",
        "negative",
        "problem",
        "slow",
        "unavailable",
        "wrong",
    }
    words = set(normalized_text(text).split())
    positive_score = len(words & positive_words)
    negative_score = len(words & negative_words)

    if positive_score > negative_score:
        return "POSITIVE", min(0.98, 0.65 + positive_score * 0.08), "positive"

    if negative_score > positive_score:
        return "NEGATIVE", min(0.98, 0.65 + negative_score * 0.08), "negative"

    return "NEUTRAL", 0.55, "neutral"


def lightweight_answer(question: str, context: str) -> tuple[str, float, int, int]:
    question_terms = {
        word
        for word in normalized_text(question).split()
        if len(word) > 3 and word not in {"what", "when", "where", "which", "with", "from", "does"}
    }
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", context.strip()) if sentence.strip()]

    if not sentences:
        return "", 0.0, 0, 0

    best_sentence = max(
        sentences,
        key=lambda sentence: len(question_terms & set(normalized_text(sentence).split())),
    )
    start = context.find(best_sentence)
    end = start + len(best_sentence) if start >= 0 else len(best_sentence)
    overlap = len(question_terms & set(normalized_text(best_sentence).split()))
    score = 0.45 if not question_terms else min(0.95, 0.45 + overlap / max(len(question_terms), 1) * 0.45)

    return best_sentence, score, max(start, 0), max(end, 0)


def is_too_similar_summary(source: str, summary: str) -> bool:
    normalized_source = normalized_text(source)
    normalized_summary = normalized_text(summary)

    if not normalized_summary:
        return True

    return (
        normalized_source == normalized_summary
        or normalized_source.startswith(normalized_summary) and len(normalized_summary) > len(normalized_source) * 0.45
        or normalized_summary in normalized_source and len(normalized_summary) > len(normalized_source) * 0.75
        or len(summary.split()) > len(source.split()) * 0.85
        or summary.rstrip().endswith((",", ";", ":"))
    )


def generate_instruction_summary(text: str) -> str:
    generator = get_text_generator()
    prompt = build_generation_prompt(
        "Summarize the following text in one clear sentence. "
        "Do not copy the text word for word.\n\n"
        f"{text.strip()}"
    )
    result = generator(
        prompt,
        max_new_tokens=65,
        do_sample=False,
        pad_token_id=generator.tokenizer.eos_token_id,
        eos_token_id=generator.tokenizer.eos_token_id,
    )[0]
    return clean_generated_text(result["generated_text"].replace(prompt, "", 1))


@app.get("/")
def root():
    return {
        "message": "AI Model Serving API is running.",
        "runtime": MODEL_RUNTIME,
        "docs": "/docs",
        "endpoints": [
            "/generate-text",
            "/summarize",
            "/sentiment",
            "/question-answering",
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "runtime": MODEL_RUNTIME}


@app.post("/generate-text", response_model=TextGenerationResponse)
def generate_text(payload: TextGenerationRequest):
    if MODEL_RUNTIME != "transformers":
        return TextGenerationResponse(
            task="text-generation",
            model="local-lightweight-generator",
            prompt=payload.prompt,
            generated_text=lightweight_generation(payload.prompt),
        )

    try:
        generator = get_text_generator()
        prompt = build_generation_prompt(payload.prompt)
        result = generator(
            prompt,
            max_new_tokens=min(payload.max_new_tokens, 70),
            do_sample=False,
            pad_token_id=generator.tokenizer.eos_token_id,
            eos_token_id=generator.tokenizer.eos_token_id,
        )[0]
        generated_text = clean_generated_text(
            result["generated_text"].replace(prompt, "", 1)
        )

        return TextGenerationResponse(
            task="text-generation",
            model=TEXT_GENERATION_MODEL,
            prompt=payload.prompt,
            generated_text=generated_text or result["generated_text"].strip(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {exc}")


@app.post("/summarize", response_model=SummarizationResponse)
def summarize(payload: SummarizationRequest):
    if payload.min_length >= payload.max_length:
        raise HTTPException(
            status_code=400,
            detail="min_length must be smaller than max_length.",
        )

    if MODEL_RUNTIME != "transformers":
        return SummarizationResponse(
            task="summarization",
            model="local-lightweight-summarizer",
            summary=fallback_summary(payload.text),
        )

    try:
        word_count = len(payload.text.split())
        summarizer = get_summarizer()
        max_length = min(payload.max_length, max(45, int(word_count * 0.8)))
        min_length = min(payload.min_length, max(12, max_length - 20))

        if min_length >= max_length:
            min_length = max(8, max_length // 2)

        result = summarizer(
            payload.text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True,
            no_repeat_ngram_size=3,
        )[0]

        summary = result["summary_text"].strip()
        if is_too_similar_summary(payload.text, summary):
            instruction_summary = generate_instruction_summary(payload.text)
            summary = (
                instruction_summary
                if instruction_summary
                and normalized_text(instruction_summary) != normalized_text(payload.text)
                and len(instruction_summary.split()) < len(payload.text.split())
                else fallback_summary(payload.text)
            )

        return SummarizationResponse(
            task="summarization",
            model=SUMMARIZATION_MODEL,
            summary=summary,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {exc}")


@app.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(payload: SentimentRequest):
    if MODEL_RUNTIME != "transformers":
        label, score, sentiment = lightweight_sentiment(payload.text)
        return SentimentResponse(
            task="sentiment-analysis",
            model="local-lightweight-sentiment",
            label=label,
            score=score,
            sentiment=sentiment,
        )

    try:
        analyzer = get_sentiment_analyzer()
        result = analyzer(payload.text, truncation=True)[0]
        label = result["label"].lower()
        score = float(result["score"])

        if label in {"positive", "label_2"} and score >= 0.55:
            sentiment = "positive"
        elif label in {"negative", "label_0"} and score >= 0.55:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return SentimentResponse(
            task="sentiment-analysis",
            model=SENTIMENT_MODEL,
            label=result["label"],
            score=score,
            sentiment=sentiment,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {exc}")


@app.post("/question-answering", response_model=QuestionAnsweringResponse)
def answer_question(payload: QuestionAnsweringRequest):
    if MODEL_RUNTIME != "transformers":
        answer, score, start, end = lightweight_answer(payload.question, payload.context)
        return QuestionAnsweringResponse(
            task="question-answering",
            model="local-lightweight-qa",
            question=payload.question,
            answer=answer,
            score=score,
            start=start,
            end=end,
        )

    try:
        question_answerer = get_question_answerer()
        result = question_answerer(
            question=payload.question,
            context=payload.context,
            max_answer_len=payload.max_answer_length,
        )

        return QuestionAnsweringResponse(
            task="question-answering",
            model=QUESTION_ANSWERING_MODEL,
            question=payload.question,
            answer=result["answer"],
            score=float(result["score"]),
            start=int(result["start"]),
            end=int(result["end"]),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("APP_RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=reload_enabled)
