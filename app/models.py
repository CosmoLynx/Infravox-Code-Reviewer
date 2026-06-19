from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CLEAN = "clean"


class Verdict(str, Enum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    NEEDS_DISCUSSION = "needs_discussion"


class FindingCategory(str, Enum):
    SECURITY = "security"
    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"
    STYLE = "style"
    TEST_COVERAGE = "test_coverage"


class ReviewRequest(BaseModel):
    diff: str
    language: str
    context: Optional[str] = None


class AgentFinding(BaseModel):
    line: int
    line_content: str
    severity: Severity
    title: str
    description: str
    suggestion: str


class Finding(BaseModel):
    id: str
    line: int
    line_content: str
    category: FindingCategory
    severity: Severity
    title: str
    description: str
    suggestion: str


class AgentFindingsCount(BaseModel):
    security: int = 0
    performance: int = 0
    correctness: int = 0
    style: int = 0
    test_coverage: int = 0


class ReviewReport(BaseModel):
    pr_summary: str
    verdict: Verdict
    verdict_reason: str
    overall_severity: Severity
    findings: list[Finding] = Field(default_factory=list)
    positive_observations: list[str] = Field(default_factory=list)
    missing_tests: list[str] = Field(default_factory=list)
    agent_findings_count: AgentFindingsCount = Field(default_factory=AgentFindingsCount)
    processing_time_ms: int


class StoredReview(BaseModel):
    review_id: str
    created_at: datetime
    report: ReviewReport


class ReviewSummary(BaseModel):
    review_id: str
    pr_summary: str
    verdict: Verdict
    overall_severity: Severity
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    groq_connected: bool
    message: str
