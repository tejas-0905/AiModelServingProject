import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

TEXT_GENERATION_MODEL = os.getenv(
    "TEXT_GENERATION_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
)
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6")
SENTIMENT_MODEL = os.getenv(
    "SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
)
QUESTION_ANSWERING_MODEL = os.getenv(
    "QUESTION_ANSWERING_MODEL", "distilbert-base-cased-distilled-squad"
)
HF_CACHE_DIR = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


app = FastAPI(
    title="AI Model Serving API",
    description="FastAPI backend serving multiple Hugging Face models.",
    version="1.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
allow_all_origins = "*" in allowed_origins
allowed_origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX", ".*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else allowed_origins,
    allow_origin_regex=allowed_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
        import transformers.utils.import_utils as transformers_import_utils

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


@lru_cache(maxsize=1)
def get_text_generator():
    pipeline = load_pipeline()

    generator = pipeline(
        "text-generation",
        model=resolve_cached_model(TEXT_GENERATION_MODEL),
    )
    generator.model.generation_config.do_sample = False
    generator.model.generation_config.temperature = None
    generator.model.generation_config.top_p = None
    generator.model.generation_config.top_k = None
    return generator


@lru_cache(maxsize=1)
def get_summarizer():
    pipeline = load_pipeline()

    return pipeline(
        "summarization",
        model=resolve_cached_model(SUMMARIZATION_MODEL),
    )


@lru_cache(maxsize=1)
def get_sentiment_analyzer():
    pipeline = load_pipeline()

    return pipeline(
        "sentiment-analysis",
        model=resolve_cached_model(SENTIMENT_MODEL),
    )


@lru_cache(maxsize=1)
def get_question_answerer():
    pipeline = load_pipeline()

    return pipeline(
        "question-answering",
        model=resolve_cached_model(QUESTION_ANSWERING_MODEL),
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
    return {"status": "healthy"}


@app.post("/generate-text", response_model=TextGenerationResponse)
def generate_text(payload: TextGenerationRequest):
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
