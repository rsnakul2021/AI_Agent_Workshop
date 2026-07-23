"""
================================================================================
MULTI-TOOL LANGCHAIN AGENT — Teaching Demo (single file)
================================================================================
What this file showcases for students:

  1. Defining MULTIPLE tools with @tool (custom + Tavily web search)
  2. How an LLM *decides* which tool(s) to call (and can call several)
  3. Binding tools to a model and running an agent loop (create_agent)
  4. Structured final output with Pydantic (answer + sources + tools_used)
  5. Reading the message history to *see* tool calls happening

Mental model (explain on the board):

  User question
       │
       ▼
  ┌─────────────┐     may call 1..N times      ┌──────────────┐
  │  LLM Brain  │ ───────────────────────────► │    Tools     │
  │ (ChatOpenAI)│ ◄─────────────────────────── │ calc/search/ │
  └─────────────┘     tool results come back   │ datetime/... │
       │                                       └──────────────┘
       ▼
  Structured answer (AgentReport)

Run:
  cd search_agent
  uv run python multi_tool_agent_demo.py

Requires .env:
  OPENAI_API_KEY=...
  TAVILY_API_KEY=...
================================================================================
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load API keys from .env into environment variables.
# create_agent / ChatOpenAI / TavilySearch read them automatically.
# ---------------------------------------------------------------------------
load_dotenv()


# =============================================================================
# SECTION 1 — STRUCTURED OUTPUT SCHEMAS
# =============================================================================
# Why Pydantic here?
#   Without structure, the model returns free-form text — hard to demo / grade.
#   With response_format=AgentReport, LangChain asks the model to fill these fields.
# =============================================================================


class Source(BaseModel):
    """One web (or other) source the agent relied on."""

    title: str = Field(description="Short title or label for the source")
    url: str = Field(description="URL if available, otherwise 'n/a'")
    note: str = Field(
        default="",
        description="Optional one-line note on why this source mattered",
    )


class AgentReport(BaseModel):
    """
    Final structured answer students can print / grade.

    Teaching tip: ask students — "Which field proves the agent used tools?"
    Answer: tools_used (and often sources).
    """

    answer: str = Field(description="Clear final answer for the user")
    tools_used: List[str] = Field(
        default_factory=list,
        description="Names of tools the agent actually called",
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="Sources cited when web search was used",
    )
    confidence: str = Field(
        description="low | medium | high — agent's self-assessed confidence"
    )


# =============================================================================
# SECTION 2 — CUSTOM TOOLS
# =============================================================================
# A "tool" is just a Python function the LLM can call.
#
# Rules of thumb for students:
#   - Clear name + docstring  → the MODEL reads these to decide WHEN to call it
#   - Typed args              → become the JSON schema for tool calling
#   - Return a STRING         → easiest for the LLM to read back as context
#
# The agent may call ZERO, ONE, or MANY tools before answering.
# =============================================================================


@tool
def get_current_utc_time() -> str:
    """
    Return the current date and time in UTC (ISO format).

    Use this when the user asks about "today", "now", deadlines, or anything
    time-sensitive. Do NOT guess the current date from training data.
    """
    now = datetime.now(timezone.utc)
    # Returning a plain string keeps tool results easy for the LLM to parse
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def calculator(expression: str) -> str:
    """
    Safely evaluate a basic math expression and return the numeric result.

    Supports +, -, *, /, **, parentheses, and math helpers like sqrt / pi.
    Examples of valid expressions:
      - "2 + 2"
      - "(150000 * 0.18) / 12"
      - "sqrt(144) + pi"

    Use this instead of doing arithmetic in your head when numbers matter.
    """
    # SECURITY teaching moment:
    # Never do eval(user_input) on raw Python. We whitelist names instead.
    allowed_names = {
        "sqrt": math.sqrt,
        "pi": math.pi,
        "sin": math.sin,
        "cos": math.cos,
        "log": math.log,
        "abs": abs,
        "round": round,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as exc:  # noqa: BLE001 - show students a soft failure path
        return f"Calculator error for '{expression}': {exc}"


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert a numeric value between common units.

    Supported conversions (examples):
      - km ↔ miles
      - celsius ↔ fahrenheit
      - kg ↔ lb

    Args:
        value: Number to convert.
        from_unit: Source unit (e.g. 'km', 'celsius', 'kg').
        to_unit: Target unit (e.g. 'miles', 'fahrenheit', 'lb').
    """
    from_unit = from_unit.strip().lower()
    to_unit = to_unit.strip().lower()

    # Small lookup table — enough for a classroom demo, not a full library
    conversions = {
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v * 1.60934,
        ("celsius", "fahrenheit"): lambda v: (v * 9 / 5) + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("kg", "lb"): lambda v: v * 2.20462,
        ("lb", "kg"): lambda v: v / 2.20462,
    }

    key = (from_unit, to_unit)
    if key not in conversions:
        return (
            f"Unsupported conversion: {from_unit} → {to_unit}. "
            f"Try pairs like km/miles, celsius/fahrenheit, kg/lb."
        )

    out = conversions[key](value)
    return f"{value} {from_unit} = {round(out, 4)} {to_unit}"


@tool
def format_bullet_brief(title: str, bullets: str) -> str:
    """
    Format a short teaching-style brief from a title and bullet points.

    Args:
        title: Heading for the brief.
        bullets: Newline-separated bullet points (plain text).
    """
    lines = [f"## {title}", ""]
    for raw in bullets.splitlines():
        item = raw.strip().lstrip("-•* ").strip()
        if item:
            lines.append(f"- {item}")
    return "\n".join(lines)


# Web search tool from LangChain's Tavily integration.
# This is a *prebuilt* tool — same idea as @tool, but maintained by the library.
# The agent will call it when it needs live / external information.
web_search = TavilySearch(
    max_results=3,
    topic="general",
    # include_answer=True gives a short Tavily summary alongside links
    include_answer=True,
)


# =============================================================================
# SECTION 3 — BUILD THE AGENT
# =============================================================================
# create_agent wires together:
#   - model  → the LLM that plans + writes
#   - tools  → functions it is allowed to call
#   - system prompt → "personality" + rules
#   - response_format → force a Pydantic object at the end
#
# Internally this is the tool-calling loop:
#   think → (optional) call tool(s) → observe results → think again → answer
# =============================================================================

SYSTEM_PROMPT = """
You are a careful research-and-reasoning assistant for an engineering workshop.

Rules:
1. Prefer TOOLS over guessing for facts, dates, math, and unit conversions.
2. You may call MULTIPLE tools in one problem (e.g. search THEN calculator).
3. After using tools, synthesize a clear final answer for a student audience.
4. List every tool you actually used in tools_used.
5. If you searched the web, cite URLs in sources.
6. Set confidence to low/medium/high honestly.
""".strip()


def build_agent():
    """
    Construct the multi-tool LangChain agent.

    Separating this into a function makes it easy to reuse in notebooks
    or tests without re-running side effects at import time.
    """
    # gpt-4o-mini is cheap/fast for demos; swap to gpt-4o / gpt-5 if preferred
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [
        get_current_utc_time,
        calculator,
        unit_converter,
        format_bullet_brief,
        web_search,  # Tavily — live web search
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=AgentReport,
    )
    return agent


# =============================================================================
# SECTION 4 — INSPECT TOOL CALLS (great for live teaching)
# =============================================================================
# Students often ask: "Did it REALLY call tools, or is it making it up?"
# Walking the message list makes the agent loop visible.
# =============================================================================


def explain_tool_trace(messages: List[Any]) -> None:
    """
    Print a human-readable trace of tool calls from the agent message history.

    Typical message pattern:
      HumanMessage  → user question
      AIMessage     → may contain tool_calls (name + args)
      ToolMessage   → result returned by the tool
      AIMessage     → final / next reasoning step
      ...
    """
    print("\n" + "=" * 60)
    print("TOOL CALL TRACE (what the agent actually did)")
    print("=" * 60)

    step = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                step += 1
                name = call.get("name", "?")
                args = call.get("args", {})
                print(f"\n[{step}] AI decided to call tool: {name}")
                print(f"    args = {json.dumps(args, indent=2)}")
        elif isinstance(msg, ToolMessage):
            # Truncate long search payloads so the demo stays readable
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            preview = content[:400] + ("..." if len(content) > 400 else "")
            print(f"    └─ tool result preview:\n{preview}\n")

    if step == 0:
        print("No tool calls were made (model answered from its own knowledge).")
    print("=" * 60 + "\n")


def print_final_report(result: dict) -> None:
    """Pretty-print the structured AgentReport from the agent result."""
    # create_agent typically puts structured output under "structured_response"
    report: Optional[AgentReport] = result.get("structured_response")

    print("\n" + "=" * 60)
    print("FINAL STRUCTURED REPORT")
    print("=" * 60)

    if report is None:
        # Fallback: show raw result keys so students can debug schema mismatches
        print("No structured_response found. Raw result keys:", list(result.keys()))
        print(result)
        return

    print(f"\nAnswer:\n{report.answer}\n")
    print(f"Tools used : {report.tools_used}")
    print(f"Confidence : {report.confidence}")
    if report.sources:
        print("\nSources:")
        for i, src in enumerate(report.sources, start=1):
            print(f"  {i}. {src.title} — {src.url}")
            if src.note:
                print(f"      note: {src.note}")
    print("=" * 60 + "\n")


# =============================================================================
# SECTION 5 — DEMO QUESTIONS
# =============================================================================
# Each question is designed to force DIFFERENT tool combinations.
# =============================================================================

DEMO_QUESTIONS = [
    # 1) Forces datetime + maybe search
    (
        "What is today's UTC date, and briefly: what is LangChain used for "
        "in building AI agents? Use tools where helpful."
    ),
    # 2) Forces web search + calculator together (multi-tool)
    (
        "Search for the approximate current population of India, then use the "
        "calculator to compute 2.5% of that population. Show the math."
    ),
    # 3) Forces unit conversion + formatting tool
    (
        "Convert 42 kilometers to miles, then use the briefing tool to present "
        "the result as a short titled bullet brief for students."
    ),
]


def run_one(question: str) -> None:
    """
    Run a single user question through the multi-tool agent and teach the trace.
    """
    agent = build_agent()

    print("\n" + "#" * 60)
    print("USER QUESTION")
    print("#" * 60)
    print(question)

    # invoke() runs the full agent loop until the model stops calling tools
    # and (with response_format) produces AgentReport.
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=question)],
        }
    )

    # Message history is the best classroom artifact for "how agents work"
    messages = result.get("messages", [])
    explain_tool_trace(messages)
    print_final_report(result)


def main() -> None:
    """
    Entry point for the live demo.

    Default: runs DEMO question #2 (search + calculator) — the clearest
    multi-tool showcase. Change the index or loop over all questions.
    """
    print("LangChain Multi-Tool Agent Demo")
    print("Tip: watch the TOOL CALL TRACE — that is the whole point of this file.\n")

    # Pick one question for a snappy live run (change index 0/1/2 as needed)
    run_one(DEMO_QUESTIONS[1])

    # Uncomment to run the full suite during the workshop:
    # for q in DEMO_QUESTIONS:
    #     run_one(q)


if __name__ == "__main__":
    main()
