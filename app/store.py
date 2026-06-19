from typing import Optional

from app.models import ReviewSummary, StoredReview

_reviews: dict[str, StoredReview] = {}


def save_review(review: StoredReview) -> StoredReview:
    _reviews[review.review_id] = review
    return review


def get_review(review_id: str) -> Optional[StoredReview]:
    return _reviews.get(review_id)


def list_reviews() -> list[ReviewSummary]:
    return [
        ReviewSummary(
            review_id=r.review_id,
            pr_summary=r.report.pr_summary,
            verdict=r.report.verdict,
            overall_severity=r.report.overall_severity,
            created_at=r.created_at,
        )
        for r in sorted(_reviews.values(), key=lambda x: x.created_at, reverse=True)
    ]
