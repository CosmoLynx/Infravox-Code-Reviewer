# Infravox — AI Code Reviewer

A FastAPI service that runs a LangGraph multi-agent pipeline to review GitHub PR diffs and return structured JSON code reviews.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY from console.groq.com
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

## What It Does

You send a git diff and a language label; the service returns a structured code review with line-level findings, severity ratings, and a merge verdict. Each finding includes the exact diff line, a plain-language explanation, and a concrete fix suggestion. Teams and individual developers can use it to catch security, performance, correctness, style, and test-coverage issues before merging a pull request.

## Architecture

The core is a LangGraph `StateGraph`. Five specialist agents fan out from `START` and run in parallel. Each agent is a single focused LLM call with a category-specific system prompt — security, performance, correctness, style, or test coverage. When all five finish, `merge_node` runs once (`defer=True`) and produces the final `ReviewReport`.

Deduplication is conservative: two findings are merged only when they share the same line number and the same root issue (`sql_injection`, `hardcoded_credentials`, `unsafe_deserialization`, or `n_plus_one`). Findings on different lines stay separate. Findings on the same line with different root causes — for example, SQL injection and a swallowed exception — are both kept.

A mandatory diff scanner in the merge node acts as a fallback. It pattern-matches the diff for known high-signal bugs (hardcoded secrets, f-string SQL, pickle deserialization, N+1 loops, IDOR, bare `except: pass`) and injects them if agents missed them — common when Groq quota is exhausted. The merge node itself does not call an LLM; summary and verdict logic are heuristic for speed and reliability.

```
START
  ├── security_reviewer ──────┐
  ├── performance_reviewer ───┤
  ├── correctness_reviewer ─┼──► merge_node ──► ReviewReport
  ├── style_reviewer ─────────┤
  └── test_coverage_reviewer ─┘
```

## API Endpoints

**POST `/review`** — Submit a diff for review.

Request body: `{ "diff": string, "language": string, "context": string (optional) }`

Response: `ReviewReport` (verdict, findings, severity counts, processing time)

**GET `/review/{review_id}`** — Fetch a previously stored review by UUID.

Response: `ReviewReport`

**GET `/reviews`** — List all stored reviews.

Response: `[{ review_id, pr_summary, verdict, overall_severity, created_at }]`

**GET `/health`** — Service status and Groq API connectivity check.

Response: `{ status, groq_connected, message }`

## Running the Review Script

With the server running, batch-review the three sample diffs:

```bash
python run_reviews.py
```

The script reads `diffs/diff1_python.txt`, `diff2_javascript.txt`, and `diff3_typescript.txt`, POSTs each to `/review`, and saves the JSON responses to `reviews/diff1_review.json`, `reviews/diff2_review.json`, and `reviews/diff3_review.json`. It skips files that already exist and waits between requests to reduce Groq rate-limit errors.

## Design Decisions

### What I'm most proud of

The parallel agent architecture. All five specialists run simultaneously from `START`, so wall-clock latency is roughly one LLM round-trip rather than five sequential calls. Each agent owns a narrow prompt and returns a JSON array of findings for its category only, which keeps outputs focused and makes the merge step predictable.

### Mandatory findings scanner

The merge node includes a diff-pattern scanner that injects known high-signal findings if agents miss them due to quota limits or hallucination. This ensures minimum recall on critical bugs even under degraded LLM conditions. In production this would be replaced with a fine-tuned classifier or RAG over a CVE knowledge base.

### What I'd do differently

Add a vector store (Pinecone or ChromaDB) with CVE and OWASP patterns so the security agent has RAG context, not just training knowledge. That would significantly improve recall on subtle or less common vulnerabilities.

### Known limitations

Groq's free tier has daily token limits — agents may return empty findings if quota is exhausted. Re-run after the 24-hour reset. Reviews are stored in memory only; restarting the server clears history. The service currently accepts text diffs only — no PR metadata, commit messages, or multi-file context.

## AI Tool Usage

Cursor and Claude were used to scaffold the project, debug LangGraph parallel execution, and tune agent prompts. Architectural decisions and prompt engineering were done manually and iteratively based on output quality testing against the sample diffs.

## Tech Stack

FastAPI, LangGraph, LangChain, langchain-groq, Pydantic v2, Python 3.10+, Groq (`llama-3.3-70b-versatile` — replacement for decommissioned `llama3-70b-8192`)
