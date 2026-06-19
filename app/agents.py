import json
import os
import re
import time
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from app.models import AgentFinding, FindingCategory, Severity

load_dotenv()

_llm: ChatGroq | None = None
MAX_RETRIES = 2
BASE_DELAY_SEC = 1.0

LINE_CONTENT_RULE = (
    "For line_content: copy the exact line from the diff verbatim, "
    "including the leading '+' character for added lines and '-' for removed lines. "
    "Never strip or modify the line. Copy it character for character. "
    'Example CORRECT: "+    query = f\\"SELECT * FROM users WHERE id = {user_id}\\"" '
    'Example INCORRECT: "    query = f\\"SELECT * FROM users WHERE id = {user_id}\\""'
)

SEVERITY_ORDER = {
    Severity.CRITICAL: 4,
    Severity.HIGH: 3,
    Severity.MEDIUM: 2,
    Severity.LOW: 1,
    Severity.CLEAN: 0,
}


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return _llm


def _is_retryable(exc: Exception) -> bool:
    err = str(exc).lower()
    if "rate_limit" in err or "rate limit" in err or "429" in err:
        return False
    return any(
        token in err
        for token in ("503", "502", "timeout", "overloaded", "capacity")
    )


def invoke_with_retry(messages: list[BaseMessage]):
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return get_llm().invoke(messages)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(BASE_DELAY_SEC * (2**attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("LLM invoke failed without exception")


AGENT_PROMPTS: dict[FindingCategory, str] = {
    FindingCategory.SECURITY: (
        "You are a security code reviewer. Analyse the following git diff and return ONLY a JSON array "
        "of security findings. Focus on: SQL injection, hardcoded credentials, unsafe deserialization, "
        "unvalidated inputs, IDOR, exposed secrets. Be paranoid but precise — flag real vulnerabilities only. "
        "Each finding must have: line (integer — line number in the diff where the added line appears), "
        f"line_content (exact diff line), severity (critical/high/medium/low), title, description, suggestion. "
        f"{LINE_CONTENT_RULE} "
        "Assign severity strictly: critical = exploitable in prod, high = likely vulnerability, "
        "medium = potential issue, low = minor hardening gap. "
        "Maximum 6 security findings per diff. Keep only the most severe and exploitable issues.\n\n"
        "IDOR CHECK — REQUIRED: When a function accepts a parameter like requester_id, caller_id, "
        "or requesting_user alongside a target user_id, check if the function body ever compares "
        "them or verifies authorization.\n\n"
        "If requester_id is accepted as a parameter but NEVER used inside the function body, "
        "this is an IDOR vulnerability, NOT an unused variable. You MUST file it as:\n"
        '  severity: high\n'
        '  title: "IDOR - Authorization Parameter Never Verified"\n'
        '  description: "<func_name> accepts requester_id but never checks if the requester has '
        "permission to access user_id. Any authenticated user can fetch any other user's data.\"\n"
        '  suggestion: "Add an authorization check: if requester_id != user_id: '
        "raise PermissionError('Access denied')\"\n\n"
        "Also look for these IDOR patterns:\n"
        "- A function accepts both user_id AND requester_id (or similar) as parameters\n"
        "- The function queries the database using user_id\n"
        "- But NEVER checks if requester_id == user_id or if requester has permission\n\n"
        "- Endpoints that use req.params.id without verifying ownership\n"
        "- Functions that update/delete records using only the target ID with no ownership check\n"
        "- Password reset flows that don't verify the requesting user owns the email\n\n"
        "Severity for IDOR: always HIGH minimum, CRITICAL if it exposes financial or sensitive personal data.\n"
        "Return [] if no issues found. Return only valid JSON, no markdown, no explanation."
    ),
    FindingCategory.PERFORMANCE: (
        "You are a performance code reviewer. Analyse the following git diff and return ONLY a JSON array "
        "of performance findings. Focus on: N+1 queries, missing indexes, unnecessary loops, "
        "memory leaks, repeated expensive operations. Focus on scale issues, not micro-optimizations. "
        "Each finding must have: line (integer), line_content (exact diff line), "
        "severity (critical/high/medium/low), title, description, suggestion. "
        f"{LINE_CONTENT_RULE} "
        "Maximum 3 performance findings per diff.\n\n"
        "DO NOT flag missing indexes on columns named 'id', 'user_id', 'pk', "
        "or any column that is likely a PRIMARY KEY. Primary keys are automatically indexed "
        "in SQLite, PostgreSQL, and MySQL.\n\n"
        "Only flag missing indexes when:\n"
        "- The query filters on a non-primary column (email, status, created_at)\n"
        "- The table is clearly large based on context\n"
        "- There is no existing index mentioned in the diff\n\n"
        "DO NOT flag synchronous database connections as performance issues "
        "unless the surrounding code is clearly async (uses async/await, asyncio). "
        "If the whole file is synchronous, a sync DB connection is not a bug.\n"
        "Return [] if no issues found. Return only valid JSON, no markdown, no explanation."
    ),
    FindingCategory.CORRECTNESS: (
        "You are a correctness code reviewer. Analyse the following git diff and return ONLY a JSON array "
        "of correctness findings. Focus on: missing null checks, off-by-one errors, swallowed exceptions, "
        "wrong status codes, race conditions, missing input validation, incorrect boolean logic. "
        "Flag production bugs only. Each finding must have: line (integer), line_content (exact diff line), "
        "severity (critical/high/medium/low), title, description, suggestion. "
        f"{LINE_CONTENT_RULE} "
        "Maximum 4 correctness findings per diff. "
        "Return [] if no issues found. Return only valid JSON, no markdown, no explanation."
    ),
    FindingCategory.STYLE: (
        "You are a style code reviewer. Your job is strict and narrow. "
        "Analyse the following git diff and return ONLY a JSON array of style findings.\n\n"
        "ONLY flag these style issues:\n"
        "1. Functions longer than 30 lines that should be split\n"
        "2. Variable names that are genuinely confusing (not just 'could be better')\n"
        "3. Magic numbers that make the code's intent completely unclear\n"
        "4. Dead code (variables declared but never used, unreachable branches)\n"
        "5. Duplicated logic blocks that appear 2+ times and should be extracted\n\n"
        "NEVER flag requester_id, caller_id, or requesting_user as an 'unused variable' when "
        "they appear alongside user_id or similar target parameters. That pattern indicates a "
        "missing authorization check (IDOR) — the security agent handles it, not style.\n\n"
        "DO NOT flag these as style issues (other agents cover them):\n"
        "- SQL injection vulnerabilities (security agent handles this)\n"
        "- Hardcoded credentials or secrets (security agent handles this)\n"
        "- Missing docstrings (too minor, skip entirely)\n"
        "- Broad exception handling (correctness agent handles this)\n"
        "- Any issue already covered by security, performance, or correctness agents\n\n"
        "MAXIMUM 5 style findings per diff. If you find more than 5, keep only the most impactful ones. "
        "Return [] if there are no genuine style issues.\n\n"
        "Each finding must have: line (integer), line_content (exact diff line), "
        "severity (critical/high/medium/low), title, description, suggestion. "
        f"{LINE_CONTENT_RULE} "
        "Return only valid JSON, no markdown, no explanation."
    ),
    FindingCategory.TEST_COVERAGE: (
        "You are a test coverage reviewer. Analyse the following git diff and return ONLY a JSON array "
        "of test coverage findings. Identify real test gaps only.\n\n"
        "You suggest real, useful tests only. NOT tests that:\n"
        "- Assert a config variable has a specific value (e.g. 'test that DB_PASSWORD != admin123')\n"
        "- Test the vulnerable behavior itself instead of the fix (e.g. testing pickle.loads on untrusted input)\n"
        "- Repeat what other test suggestions already cover\n\n"
        "Good test suggestions focus on:\n"
        "- Happy path: does the function return the right data for valid input?\n"
        "- Null/missing input: what happens when user_id doesn't exist?\n"
        "- Auth/permission edge cases: can user A access user B's data?\n"
        "- Error paths: does the function handle DB failures gracefully?\n"
        "- Boundary conditions: empty lists, zero values, max values\n\n"
        "Maximum 6 test suggestions per diff. Keep them specific and realistic.\n\n"
        "Each finding must have: line (integer), line_content (exact diff line), "
        "severity (critical/high/medium/low), title, description, suggestion. "
        f"{LINE_CONTENT_RULE} "
        "Return [] if no issues found. Return only valid JSON, no markdown, no explanation."
    ),
}


def _parse_json_array(content: str) -> list[dict[str, Any]]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return []
        parsed = json.loads(match.group())
    if not isinstance(parsed, list):
        return []
    return parsed


def _normalize_severity(value: str) -> Severity:
    try:
        return Severity(value.lower())
    except ValueError:
        return Severity.MEDIUM


def _normalize_line_content(line_content: str, line: int, diff: str) -> str:
    if line_content.startswith(("+", "-")):
        return line_content
    diff_lines = diff.splitlines()
    if 1 <= line <= len(diff_lines):
        candidate = diff_lines[line - 1]
        if candidate.startswith(("+", "-", " ")):
            body = line_content.strip()
            if body and (body in candidate or candidate.lstrip("+-").strip() == body):
                if candidate.startswith(("+", "-")):
                    return candidate
    for diff_line in diff_lines:
        if diff_line.startswith(("+", "-")) and line_content.strip() in diff_line:
            return diff_line
    return line_content


def _build_user_message(diff: str, language: str, context: str | None) -> str:
    parts = [f"Language: {language}"]
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"\nGit diff:\n{diff}")
    return "\n".join(parts)


def run_agent(
    category: FindingCategory,
    diff: str,
    language: str,
    context: str | None = None,
) -> list[AgentFinding]:
    system_prompt = AGENT_PROMPTS[category]
    user_message = _build_user_message(diff, language, context)

    response = invoke_with_retry(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
    )
    raw_findings = _parse_json_array(str(response.content))

    findings: list[AgentFinding] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        try:
            line = int(item.get("line", 0))
            line_content = _normalize_line_content(
                str(item.get("line_content", "")), line, diff
            )
            findings.append(
                AgentFinding(
                    line=line,
                    line_content=line_content,
                    severity=_normalize_severity(str(item.get("severity", "medium"))),
                    title=str(item.get("title", "Untitled finding")),
                    description=str(item.get("description", "")),
                    suggestion=str(item.get("suggestion", "")),
                )
            )
        except (TypeError, ValueError):
            continue
    return findings


def check_groq_connectivity() -> tuple[bool, str]:
    if not os.getenv("GROQ_API_KEY"):
        return False, "GROQ_API_KEY is not set"
    try:
        response = invoke_with_retry([HumanMessage(content="Reply with exactly: ok")])
        content = str(response.content).strip().lower()
        if "ok" in content:
            return True, "Groq API is reachable"
        return True, f"Groq API responded: {response.content}"
    except Exception as exc:
        return False, f"Groq API error: {exc}"
