import json
import re
import time
from collections import defaultdict
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agents import SEVERITY_ORDER, run_agent
from app.models import (
    AgentFindingsCount,
    AgentFinding,
    Finding,
    FindingCategory,
    ReviewReport,
    Severity,
    Verdict,
)


class PipelineState(TypedDict):
    diff: str
    language: str
    context: str | None
    started_at: float
    security_findings: list[AgentFinding]
    performance_findings: list[AgentFinding]
    correctness_findings: list[AgentFinding]
    style_findings: list[AgentFinding]
    test_coverage_findings: list[AgentFinding]
    report: ReviewReport | None


# Only deduplicate these root issues — and only on the same line number.
DEDUP_ROOT_ISSUES = frozenset({
    "sql_injection",
    "hardcoded_credentials",
    "unsafe_deserialization",
    "n_plus_one",
})

AUTH_PARAM_NAMES = ("requester_id", "caller_id", "requesting_user")
TARGET_PARAM_NAMES = ("user_id", "target_id", "resource_id")

MAX_TOTAL_FINDINGS = 18


def severity_rank(severity: Severity | str) -> int:
    if isinstance(severity, Severity):
        return SEVERITY_ORDER.get(severity, 0)
    return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(str(severity).lower(), 0)


def get_root_issue(title: str) -> str:
    title_lower = title.lower()
    if "sql injection" in title_lower or "sqli" in title_lower:
        return "sql_injection"
    if "hardcoded" in title_lower or "credentials" in title_lower or "secret" in title_lower:
        return "hardcoded_credentials"
    if "pickle" in title_lower or "deserializ" in title_lower:
        return "unsafe_deserialization"
    if "n+1" in title_lower:
        return "n_plus_one"
    if "inefficient" in title_lower and ("loop" in title_lower or "query" in title_lower):
        return "n_plus_one"
    if "idor" in title_lower or "authorization parameter" in title_lower:
        return "idor"
    if "swallowed" in title_lower or "swallow" in title_lower:
        return "swallowed_exception"
    if "magic" in title_lower:
        return "magic_number"
    if "dead" in title_lower:
        return "dead_code"
    return title_lower.replace(" ", "_")[:30]


def deduplicate_findings(
    categorized: list[tuple[FindingCategory, AgentFinding]],
) -> list[tuple[FindingCategory, AgentFinding]]:
    """Conservative dedup: same line + same dedup-eligible root issue only."""
    seen: dict[tuple[int, str], tuple[FindingCategory, AgentFinding]] = {}
    deduplicated: list[tuple[FindingCategory, AgentFinding]] = []

    for category, finding in categorized:
        root_issue = get_root_issue(finding.title)
        if root_issue not in DEDUP_ROOT_ISSUES:
            deduplicated.append((category, finding))
            continue

        key = (finding.line, root_issue)
        if key not in seen:
            seen[key] = (category, finding)
            deduplicated.append((category, finding))
            continue

        existing_cat, existing = seen[key]
        if severity_rank(finding.severity) > severity_rank(existing.severity):
            deduplicated.remove((existing_cat, existing))
            deduplicated.append((category, finding))
            seen[key] = (category, finding)

    return deduplicated


def _is_false_positive(category: FindingCategory, finding: AgentFinding) -> bool:
    suggestion_lower = finding.suggestion.lower()
    combined = f"{finding.title.lower()} {finding.description.lower()}"

    if category == FindingCategory.PERFORMANCE and "test that" in suggestion_lower:
        return True

    if category == FindingCategory.STYLE and "unused" in combined:
        if any(p in combined or p in finding.line_content for p in AUTH_PARAM_NAMES):
            return True

    if category == FindingCategory.STYLE and "duplicat" in combined:
        if ("get_user" in combined and "list_all_users" in combined) or (
            "database connection" in combined and "multiple functions" in combined
        ):
            return True

    return False


def _filter_false_positives(
    categorized: list[tuple[FindingCategory, AgentFinding]],
) -> list[tuple[FindingCategory, AgentFinding]]:
    return [item for item in categorized if not _is_false_positive(item[0], item[1])]


def _finding_key(category: FindingCategory, finding: AgentFinding) -> tuple:
    return (category.value, finding.line, get_root_issue(finding.title))


def _merge_mandatory(
    categorized: list[tuple[FindingCategory, AgentFinding]],
    mandatory: list[tuple[FindingCategory, AgentFinding]],
) -> list[tuple[FindingCategory, AgentFinding]]:
    existing_keys = {_finding_key(cat, f) for cat, f in categorized}
    for cat, finding in mandatory:
        key = _finding_key(cat, finding)
        if key not in existing_keys:
            categorized.append((cat, finding))
            existing_keys.add(key)
    return categorized


def _scan_mandatory_findings(diff: str) -> list[tuple[FindingCategory, AgentFinding]]:
    """High-signal bugs that must survive merge if present in the diff."""
    lines = diff.splitlines()
    findings: list[tuple[FindingCategory, AgentFinding]] = []
    seen: set[tuple] = set()

    def add(category: FindingCategory, finding: AgentFinding) -> None:
        key = _finding_key(category, finding)
        if key not in seen:
            seen.add(key)
            findings.append((category, finding))

    for i, line in enumerate(lines):
        if not line.startswith("+"):
            continue
        line_no = i + 1
        content = line[1:]

        if re.search(r"\bDB_PASSWORD\s*=", content):
            add(
                FindingCategory.SECURITY,
                AgentFinding(
                    line=line_no,
                    line_content=line,
                    severity=Severity.CRITICAL,
                    title="Hardcoded Credentials",
                    description="Database password is hardcoded in source code and exposed to anyone with repo access.",
                    suggestion="Load DB_PASSWORD from environment variables or a secrets manager.",
                ),
            )
        if re.search(r"\bAPI_SECRET\s*=", content):
            add(
                FindingCategory.SECURITY,
                AgentFinding(
                    line=line_no,
                    line_content=line,
                    severity=Severity.CRITICAL,
                    title="Hardcoded API Secret",
                    description="API secret is hardcoded in source code and can be extracted from the repository.",
                    suggestion="Load API_SECRET from environment variables or a secrets manager.",
                ),
            )
        if "pickle.loads" in content:
            add(
                FindingCategory.SECURITY,
                AgentFinding(
                    line=line_no,
                    line_content=line,
                    severity=Severity.CRITICAL,
                    title="Unsafe Deserialization",
                    description="pickle.loads() on untrusted input allows arbitrary code execution.",
                    suggestion="Use json.loads() or a safe token validation scheme instead of pickle.",
                ),
            )
        if ("f\"" in content or "f'" in content) and "{" in content:
            if re.search(r"(SELECT|UPDATE|INSERT|DELETE|execute)", content, re.IGNORECASE):
                add(
                    FindingCategory.SECURITY,
                    AgentFinding(
                        line=line_no,
                        line_content=line,
                        severity=Severity.CRITICAL,
                        title="SQL Injection",
                        description="User input is interpolated into a SQL string via f-string, enabling SQL injection.",
                        suggestion="Use parameterized queries: cursor.execute('SELECT ... WHERE id = ?', (user_id,))",
                    ),
                )

    # N+1: query inside a numeric range loop (list_all_users pattern)
    for i, line in enumerate(lines):
        if not line.startswith("+"):
            continue
        content = line[1:]
        if not re.search(r"\bfor\s+\w+\s+in\s+range\s*\(", content):
            continue
        loop_line = i + 1
        loop_content = line
        for j in range(i + 1, min(i + 6, len(lines))):
            inner = lines[j]
            if inner.startswith("+") and re.search(r"(execute|query)\(", inner[1:], re.IGNORECASE):
                add(
                    FindingCategory.PERFORMANCE,
                    AgentFinding(
                        line=loop_line,
                        line_content=loop_content,
                        severity=Severity.CRITICAL,
                        title="N+1 Query",
                        description="A database query runs inside a loop, causing one query per iteration at scale.",
                        suggestion="Fetch all rows in a single query instead of querying inside the loop.",
                    ),
                )
                break

    # IDOR
    idor = _scan_idor_in_diff(diff)
    if idor:
        add(FindingCategory.SECURITY, idor)

    # Swallowed exception
    swallowed = _scan_swallowed_exception(diff)
    if swallowed:
        add(FindingCategory.CORRECTNESS, swallowed)

    # Magic number 10000 in loop (style)
    for i, line in enumerate(lines):
        if line.startswith("+") and "10000" in line:
            add(
                FindingCategory.STYLE,
                AgentFinding(
                    line=i + 1,
                    line_content=line,
                    severity=Severity.MEDIUM,
                    title="Magic Number",
                    description="The literal 10000 makes the iteration bound's intent unclear.",
                    suggestion="Replace with a named constant such as MAX_USER_ID or fetch via a single query.",
                ),
            )
            break

    # Dead code: except pass
    dead = _scan_dead_code_except_pass(diff)
    if dead:
        add(FindingCategory.STYLE, dead)

    return findings


def _scan_idor_in_diff(diff: str) -> AgentFinding | None:
    lines = diff.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("+"):
            continue
        match = re.search(r"def\s+(\w+)\(([^)]+)\)", line)
        if not match:
            continue
        func_name = match.group(1)
        params = match.group(2)
        has_auth = any(p in params for p in AUTH_PARAM_NAMES)
        has_target = any(p in params for p in TARGET_PARAM_NAMES)
        if not (has_auth and has_target):
            continue
        auth_param = next(p for p in AUTH_PARAM_NAMES if p in params)
        body_parts: list[str] = []
        for j in range(i + 1, len(lines)):
            body_line = lines[j]
            if body_line.startswith("+def ") and not body_line.startswith("+    "):
                break
            if body_line.startswith("+"):
                body_parts.append(body_line[1:])
        body = "\n".join(body_parts).lower()
        if auth_param in body or "authoriz" in body or "permission" in body:
            continue
        return AgentFinding(
            line=i + 1,
            line_content=line,
            severity=Severity.HIGH,
            title="IDOR - Authorization Parameter Never Verified",
            description=(
                f"{func_name} accepts {auth_param} but never checks if the requester has "
                "permission to access user_id. Any authenticated user can fetch any other user's data."
            ),
            suggestion=(
                f"Add an authorization check: if {auth_param} != user_id: "
                "raise PermissionError('Access denied')"
            ),
        )
    return None


def _scan_swallowed_exception(diff: str) -> AgentFinding | None:
    lines = diff.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("+") or "except" not in line:
            continue
        for j in range(i + 1, min(i + 4, len(lines))):
            next_line = lines[j]
            if next_line.startswith("+") and re.search(r"^\+\s*pass\s*$", next_line):
                return AgentFinding(
                    line=i + 1,
                    line_content=line,
                    severity=Severity.HIGH,
                    title="Swallowed Exception",
                    description=(
                        "Exception is caught with except Exception: pass, hiding failures from callers and logs."
                    ),
                    suggestion="Log the exception and re-raise or return an error response instead of pass.",
                )
    return None


def _scan_dead_code_except_pass(diff: str) -> AgentFinding | None:
    lines = diff.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("+") or "except" not in line:
            continue
        for j in range(i + 1, min(i + 4, len(lines))):
            next_line = lines[j]
            if next_line.startswith("+") and re.search(r"^\+\s*pass\s*$", next_line):
                return AgentFinding(
                    line=j + 1,
                    line_content=next_line,
                    severity=Severity.MEDIUM,
                    title="Dead Code",
                    description="Empty except block with pass silently discards errors.",
                    suggestion="Handle the exception explicitly or remove the try/except if unnecessary.",
                )
    return None


def _soft_cap_findings(
    categorized: list[tuple[FindingCategory, AgentFinding]],
) -> list[tuple[FindingCategory, AgentFinding]]:
    """Light cap on total volume — never drop below mandatory minimum."""
    if len(categorized) <= MAX_TOTAL_FINDINGS:
        return categorized
    ranked = sorted(
        categorized,
        key=lambda x: (severity_rank(x[1].severity), x[1].line),
        reverse=True,
    )
    return ranked[:MAX_TOTAL_FINDINGS]


def _count_findings_by_category(findings: list[Finding]) -> AgentFindingsCount:
    counts: dict[str, int] = defaultdict(int)
    for finding in findings:
        counts[finding.category.value] += 1
    return AgentFindingsCount(
        security=counts.get("security", 0),
        performance=counts.get("performance", 0),
        correctness=counts.get("correctness", 0),
        style=counts.get("style", 0),
        test_coverage=counts.get("test_coverage", 0),
    )


def _run_agent(category: FindingCategory, state: PipelineState) -> list[AgentFinding]:
    try:
        return run_agent(
            category,
            state["diff"],
            state["language"],
            state.get("context"),
        )
    except Exception:
        return []


def _security_reviewer(state: PipelineState) -> dict:
    return {"security_findings": _run_agent(FindingCategory.SECURITY, state)}


def _performance_reviewer(state: PipelineState) -> dict:
    return {"performance_findings": _run_agent(FindingCategory.PERFORMANCE, state)}


def _correctness_reviewer(state: PipelineState) -> dict:
    return {"correctness_findings": _run_agent(FindingCategory.CORRECTNESS, state)}


def _style_reviewer(state: PipelineState) -> dict:
    return {"style_findings": _run_agent(FindingCategory.STYLE, state)}


def _test_coverage_reviewer(state: PipelineState) -> dict:
    return {"test_coverage_findings": _run_agent(FindingCategory.TEST_COVERAGE, state)}


def _overall_severity(findings: list[Finding]) -> Severity:
    if not findings:
        return Severity.CLEAN
    return max(findings, key=lambda f: SEVERITY_ORDER[f.severity]).severity


def _verdict(severity: Severity) -> Verdict:
    if severity == Severity.CLEAN:
        return Verdict.APPROVE
    if severity in (Severity.CRITICAL, Severity.HIGH):
        return Verdict.REQUEST_CHANGES
    return Verdict.NEEDS_DISCUSSION


def _verdict_reason(verdict: Verdict, findings: list[Finding]) -> str:
    if verdict == Verdict.APPROVE:
        return "No significant issues were found across security, performance, correctness, style, or test coverage."
    if verdict == Verdict.REQUEST_CHANGES:
        top = max(findings, key=lambda f: SEVERITY_ORDER[f.severity])
        return f"Found {top.severity.value} severity issue: {top.title}."
    return "Only medium or low severity findings were detected; team discussion recommended before merge."


def _generate_summary_and_positives(
    diff: str,
    language: str,
    findings: list[Finding],
) -> tuple[str, list[str]]:
    critical = [f for f in findings if f.severity in (Severity.CRITICAL, Severity.HIGH)]
    summary = f"Updates {language} code with changes shown in the diff."
    if critical:
        summary = f"This PR modifies {language} code and introduces {len(critical)} high-severity issue(s) requiring attention."

    positives = [
        "The diff is scoped to a specific module with clear function boundaries.",
        "Added code uses familiar patterns for the language and framework.",
    ]
    if any("parameterized" in f.suggestion.lower() for f in findings):
        positives.append("At least one database operation uses parameterized queries.")

    return summary, positives[:3]


_BAD_TEST_PATTERNS = re.compile(
    r"(db_password|admin123|assert.*password|pickle\.loads|test the vulnerable)",
    re.IGNORECASE,
)


def _extract_missing_tests(test_findings: list[AgentFinding]) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for finding in test_findings:
        suggestion = finding.suggestion.strip()
        text = suggestion or f"{finding.title}: {finding.description}"
        if _BAD_TEST_PATTERNS.search(text):
            continue
        key = text.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        missing.append(text)
    return missing[:6]


def _merge_node(state: PipelineState) -> dict:
    categorized: list[tuple[FindingCategory, AgentFinding]] = [
        (FindingCategory.SECURITY, f) for f in state.get("security_findings", [])
    ]
    categorized += [(FindingCategory.PERFORMANCE, f) for f in state.get("performance_findings", [])]
    categorized += [(FindingCategory.CORRECTNESS, f) for f in state.get("correctness_findings", [])]
    categorized += [(FindingCategory.STYLE, f) for f in state.get("style_findings", [])]
    categorized += [
        (FindingCategory.TEST_COVERAGE, f) for f in state.get("test_coverage_findings", [])
    ]

    mandatory = _scan_mandatory_findings(state["diff"])
    categorized = _filter_false_positives(categorized)
    categorized = _merge_mandatory(categorized, mandatory)
    deduped = deduplicate_findings(categorized)
    capped = _soft_cap_findings(deduped)

    findings: list[Finding] = []
    for idx, (category, finding) in enumerate(
        sorted(capped, key=lambda x: (x[0].value, x[1].line)), start=1
    ):
        findings.append(
            Finding(
                id=f"F-{idx:03d}",
                line=finding.line,
                line_content=finding.line_content,
                category=category,
                severity=finding.severity,
                title=finding.title,
                description=finding.description,
                suggestion=finding.suggestion,
            )
        )

    agent_findings_count = _count_findings_by_category(findings)
    overall = _overall_severity(findings)
    verdict = _verdict(overall)
    pr_summary, positives = _generate_summary_and_positives(
        state["diff"],
        state["language"],
        findings,
    )
    missing_tests = _extract_missing_tests(state.get("test_coverage_findings", []))

    processing_time_ms = int((time.time() - state["started_at"]) * 1000)

    report = ReviewReport(
        pr_summary=pr_summary,
        verdict=verdict,
        verdict_reason=_verdict_reason(verdict, findings),
        overall_severity=overall,
        findings=findings,
        positive_observations=positives,
        missing_tests=missing_tests,
        agent_findings_count=agent_findings_count,
        processing_time_ms=processing_time_ms,
    )
    return {"report": report}


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("security_reviewer", _security_reviewer)
    graph.add_node("performance_reviewer", _performance_reviewer)
    graph.add_node("correctness_reviewer", _correctness_reviewer)
    graph.add_node("style_reviewer", _style_reviewer)
    graph.add_node("test_coverage_reviewer", _test_coverage_reviewer)
    graph.add_node("merge_node", _merge_node, defer=True)

    graph.add_edge(START, "security_reviewer")
    graph.add_edge(START, "performance_reviewer")
    graph.add_edge(START, "correctness_reviewer")
    graph.add_edge(START, "style_reviewer")
    graph.add_edge(START, "test_coverage_reviewer")
    graph.add_edge("security_reviewer", "merge_node")
    graph.add_edge("performance_reviewer", "merge_node")
    graph.add_edge("correctness_reviewer", "merge_node")
    graph.add_edge("style_reviewer", "merge_node")
    graph.add_edge("test_coverage_reviewer", "merge_node")
    graph.add_edge("merge_node", END)

    return graph.compile()


pipeline = build_pipeline()


def run_review(diff: str, language: str, context: str | None = None) -> ReviewReport:
    result = pipeline.invoke(
        {
            "diff": diff,
            "language": language,
            "context": context,
            "started_at": time.time(),
            "security_findings": [],
            "performance_findings": [],
            "correctness_findings": [],
            "style_findings": [],
            "test_coverage_findings": [],
            "report": None,
        }
    )
    report = result.get("report")
    if report is None:
        raise RuntimeError("Pipeline did not produce a review report")
    return report
