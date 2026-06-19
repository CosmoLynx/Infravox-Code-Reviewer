import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.agents import check_groq_connectivity
from app.models import (
    HealthResponse,
    ReviewReport,
    ReviewRequest,
    ReviewSummary,
    StoredReview,
)
from app.pipeline import run_review
from app.store import get_review, list_reviews, save_review

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
    )


app = FastAPI(
    title="Infravox PR Review API",
    description="Multi-agent LangGraph pipeline for structured GitHub PR code reviews",
    version="1.0.0",
)


@app.post("/review", response_model=ReviewReport)
def create_review(request: ReviewRequest) -> ReviewReport:
    try:
        report = run_review(
            diff=request.diff,
            language=request.language,
            context=request.context,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Review pipeline failed: {exc}") from exc

    stored = StoredReview(
        review_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        report=report,
    )
    save_review(stored)
    return report


@app.get("/review/{review_id}", response_model=ReviewReport)
def fetch_review(review_id: str) -> ReviewReport:
    stored = get_review(review_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    return stored.report


@app.get("/reviews", response_model=list[ReviewSummary])
def fetch_reviews() -> list[ReviewSummary]:
    return list_reviews()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        connected, message = check_groq_connectivity()
    except Exception as exc:
        connected = False
        message = f"Health check failed: {exc}"
    return HealthResponse(
        status="ok",
        groq_connected=connected,
        message=message,
    )
