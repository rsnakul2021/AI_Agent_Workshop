"""
================================================================================
DATA ANALYST CO. — Multi-Tool LangChain Agent (ML toolkit demo)
================================================================================
Story for students:
  You work at "Data Analyst Co." Your company exposes specialized ML / analytics
  TOOLS. An LLM agent is the "junior analyst" that decides WHICH tool to call
  for a business question — and each tool both:
      (A) runs an analysis, AND
      (B) writes a reproducible .py artifact into ./analyst_outputs/

Tools in this toolkit:
  1) prepare_sample_dataset   → builds a demo CSV (customers / churn-like)
  2) run_eda_analysis         → descriptive stats + missingness → eda_*.py
  3) run_correlation_analysis → numeric correlations            → corr_*.py
  4) train_baseline_model     → LogisticRegression baseline     → model_*.py
  5) run_clustering_analysis  → KMeans customer segments        → cluster_*.py

Mental model:
  Business question
        │
        ▼
   LLM Agent (planner)
        │  calls 1..N tools
        ▼
   Company ML Tools  ──writes──►  analyst_outputs/*.py
        │
        ▼
   Structured AnalystBrief for the stakeholder

Run:
  cd search_agent
  uv add pandas scikit-learn          # once
  uv run python data_analyst_ml_agent.py

Requires .env:
  OPENAI_API_KEY=...
================================================================================
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()

# ---------------------------------------------------------------------------
# Paths — all generated analysis scripts land here (show students this folder)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analyst_outputs"
DATA_DIR = ROOT / "analyst_data"
SAMPLE_CSV = DATA_DIR / "customers_demo.csv"

OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def _timestamp() -> str:
    """UTC timestamp used in generated filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_py(filename_prefix: str, source_code: str) -> Path:
    """
    Write a Python source file under analyst_outputs/.

    Teaching tip: tools don't only "return text to the LLM" — in real companies
    they often also persist artifacts (scripts, reports, models) for auditability.
    """
    path = OUTPUT_DIR / f"{filename_prefix}_{_timestamp()}.py"
    path.write_text(textwrap.dedent(source_code).lstrip("\n"), encoding="utf-8")
    return path


def _load_dataset() -> pd.DataFrame:
    """Load the demo CSV or raise a clear error if it was never prepared."""
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {SAMPLE_CSV}. "
            "Call prepare_sample_dataset() first."
        )
    return pd.read_csv(SAMPLE_CSV)


# =============================================================================
# SECTION 1 — Structured final answer (what the agent returns to stakeholders)
# =============================================================================


class Artifact(BaseModel):
    """A .py file produced by one of the company tools."""

    tool_name: str = Field(description="Which tool produced this file")
    path: str = Field(description="Path to the generated .py artifact")
    summary: str = Field(description="One-line description of what is inside")


class AnalystBrief(BaseModel):
    """Stakeholder-ready brief after the agent finishes using tools."""

    executive_summary: str = Field(
        description="Short plain-language summary for a non-ML manager"
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Bullet findings from the analyses that were run",
    )
    tools_used: List[str] = Field(
        default_factory=list,
        description="Names of tools the agent actually invoked",
    )
    artifacts: List[Artifact] = Field(
        default_factory=list,
        description="Generated .py files from the tool runs",
    )
    recommended_next_steps: List[str] = Field(
        default_factory=list,
        description="What a senior analyst should do next",
    )


# =============================================================================
# SECTION 2 — COMPANY TOOLKIT (each tool = analysis + writes a .py file)
# =============================================================================


@tool
def prepare_sample_dataset(n_rows: int = 200) -> str:
    """
    Create (or overwrite) a synthetic customer dataset for demos.

    Columns mimic a mini churn / engagement problem:
      age, monthly_spend, support_tickets, tenure_months, churned (0/1)

    Always call this FIRST if the CSV does not exist yet.
    Also writes a small .py data dictionary artifact.
    """
    n_rows = max(50, min(int(n_rows), 2000))  # keep classroom runs light

    rng = pd.Series(range(n_rows))
    # Synthetic but realistic-ish relationships for teaching
    age = (25 + (rng * 37) % 40).astype(int)
    monthly_spend = (20 + (rng * 13) % 180).astype(float)
    support_tickets = ((rng * 3) % 8).astype(int)
    tenure_months = (1 + (rng * 11) % 48).astype(int)
    # Higher tickets + low tenure → likelier churn (noisy rule)
    churn_prob = (
        0.15
        + 0.08 * (support_tickets >= 4).astype(float)
        + 0.07 * (tenure_months < 6).astype(float)
        - 0.05 * (monthly_spend > 100).astype(float)
    )
    churned = (churn_prob + ((rng % 10) / 100.0) > 0.22).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n_rows)],
            "age": age,
            "monthly_spend": monthly_spend,
            "support_tickets": support_tickets,
            "tenure_months": tenure_months,
            "churned": churned,
        }
    )
    df.to_csv(SAMPLE_CSV, index=False)

    artifact = _write_py(
        "dataset_dictionary",
        f'''
        """
        AUTO-GENERATED by Data Analyst Co. tool: prepare_sample_dataset
        Dataset path: {SAMPLE_CSV.as_posix()}
        Rows: {n_rows}
        """

        DATA_DICTIONARY = {{
            "customer_id": "Unique customer identifier",
            "age": "Customer age in years",
            "monthly_spend": "Average monthly spend in USD",
            "support_tickets": "Support tickets opened in last 90 days",
            "tenure_months": "Months since signup",
            "churned": "Target label — 1 if churned, else 0",
        }}

        if __name__ == "__main__":
            for col, meaning in DATA_DICTIONARY.items():
                print(f"{{col}}: {{meaning}}")
        ''',
    )

    return json.dumps(
        {
            "status": "ok",
            "csv_path": str(SAMPLE_CSV),
            "rows": n_rows,
            "columns": list(df.columns),
            "churn_rate": float(df["churned"].mean()),
            "artifact_py": str(artifact),
        },
        indent=2,
    )


@tool
def run_eda_analysis() -> str:
    """
    Run Exploratory Data Analysis (EDA) on the customer dataset.

    Computes shape, dtypes, missing values, describe() stats, and churn rate.
    Writes a reproducible EDA script under analyst_outputs/.
    """
    df = _load_dataset()

    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing": df.isna().sum().to_dict(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "describe": df.describe(include="all").fillna("").to_dict(),
        "churn_rate": float(df["churned"].mean()) if "churned" in df.columns else None,
    }

    artifact = _write_py(
        "eda_report",
        f'''
        """
        AUTO-GENERATED EDA report — Data Analyst Co.
        Source CSV: {SAMPLE_CSV.as_posix()}
        """
        import pandas as pd

        df = pd.read_csv(r"{SAMPLE_CSV.as_posix()}")

        def main():
            print("SHAPE:", df.shape)
            print("\\nDTYPES:\\n", df.dtypes)
            print("\\nMISSING:\\n", df.isna().sum())
            print("\\nDESCRIBE:\\n", df.describe(include="all"))
            if "churned" in df.columns:
                print("\\nCHURN RATE:", df["churned"].mean())

        if __name__ == "__main__":
            main()

        # Snapshot captured at generation time:
        SNAPSHOT = {json.dumps(summary, default=str)}
        ''',
    )

    return json.dumps(
        {
            "analysis": "eda",
            "rows": summary["rows"],
            "cols": summary["cols"],
            "churn_rate": summary["churn_rate"],
            "missing": summary["missing"],
            "artifact_py": str(artifact),
            "teaching_note": "EDA = understand the data BEFORE modeling.",
        },
        indent=2,
        default=str,
    )


@tool
def run_correlation_analysis() -> str:
    """
    Compute Pearson correlations between numeric columns.

    Helpful to spot which features move together with churn / spend.
    Writes a correlation analysis .py artifact.
    """
    df = _load_dataset()
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)

    # Strongest relationships with target if present
    top_with_target = {}
    if "churned" in corr.columns:
        top_with_target = (
            corr["churned"]
            .drop(labels=["churned"], errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )

    artifact = _write_py(
        "correlation_analysis",
        f'''
        """
        AUTO-GENERATED correlation analysis — Data Analyst Co.
        """
        import pandas as pd

        df = pd.read_csv(r"{SAMPLE_CSV.as_posix()}")
        corr = df.select_dtypes(include="number").corr(numeric_only=True)

        def main():
            print(corr)
            if "churned" in corr.columns:
                print("\\n|corr| vs churned:")
                print(corr["churned"].drop("churned").abs().sort_values(ascending=False))

        if __name__ == "__main__":
            main()

        CORR_MATRIX = {json.dumps(corr.round(4).to_dict())}
        TOP_ABS_CORR_WITH_CHURNED = {json.dumps(top_with_target)}
        ''',
    )

    return json.dumps(
        {
            "analysis": "correlation",
            "top_abs_corr_with_churned": top_with_target,
            "artifact_py": str(artifact),
            "teaching_note": "Correlation ≠ causation — discuss with students.",
        },
        indent=2,
    )


@tool
def train_baseline_model(test_size: float = 0.25) -> str:
    """
    Train a baseline Logistic Regression model to predict churned (0/1).

    Uses age, monthly_spend, support_tickets, tenure_months as features.
    Returns accuracy + a short classification report, and writes a .py
    training script students can re-run.
    """
    df = _load_dataset()
    feature_cols = ["age", "monthly_spend", "support_tickets", "tenure_months"]
    X = df[feature_cols]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, output_dict=True)
    coefs = dict(zip(feature_cols, [float(c) for c in model.coef_[0]]))

    artifact = _write_py(
        "baseline_churn_model",
        f'''
        """
        AUTO-GENERATED baseline churn model — Data Analyst Co.
        Model: LogisticRegression
        Test accuracy (at generation time): {acc:.4f}
        """
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split

        FEATURES = {feature_cols!r}
        CSV_PATH = r"{SAMPLE_CSV.as_posix()}"

        def main():
            df = pd.read_csv(CSV_PATH)
            X = df[FEATURES]
            y = df["churned"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size={float(test_size)}, random_state=42, stratify=y
            )
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            print("accuracy:", accuracy_score(y_test, preds))
            print(classification_report(y_test, preds))
            print("coefficients:", dict(zip(FEATURES, model.coef_[0])))

        if __name__ == "__main__":
            main()

        GENERATED_METRICS = {{
            "accuracy": {acc},
            "coefficients": {json.dumps(coefs)},
            "classification_report": {json.dumps(report)},
        }}
        ''',
    )

    return json.dumps(
        {
            "analysis": "baseline_model",
            "model": "LogisticRegression",
            "features": feature_cols,
            "accuracy": acc,
            "coefficients": coefs,
            "artifact_py": str(artifact),
            "teaching_note": "Baselines first — beat this before complex models.",
        },
        indent=2,
    )


@tool
def run_clustering_analysis(n_clusters: int = 3) -> str:
    """
    Segment customers with KMeans clustering (unsupervised learning).

    Uses monthly_spend, support_tickets, tenure_months (scaled).
    Writes a clustering script with cluster sizes and centroids summary.
    """
    df = _load_dataset()
    n_clusters = max(2, min(int(n_clusters), 8))
    feature_cols = ["monthly_spend", "support_tickets", "tenure_months"]
    X = df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    df_out = df.copy()
    df_out["cluster"] = labels

    sizes = df_out["cluster"].value_counts().sort_index().to_dict()
    # Mean profile per cluster in original units (easier to explain)
    profiles = (
        df_out.groupby("cluster")[feature_cols]
        .mean()
        .round(2)
        .to_dict(orient="index")
    )

    artifact = _write_py(
        "customer_clustering",
        f'''
        """
        AUTO-GENERATED KMeans customer segmentation — Data Analyst Co.
        n_clusters = {n_clusters}
        """
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        FEATURES = {feature_cols!r}
        CSV_PATH = r"{SAMPLE_CSV.as_posix()}"

        def main():
            df = pd.read_csv(CSV_PATH)
            X = df[FEATURES]
            X_scaled = StandardScaler().fit_transform(X)
            labels = KMeans(
                n_clusters={n_clusters}, random_state=42, n_init=10
            ).fit_predict(X_scaled)
            df["cluster"] = labels
            print("cluster sizes:\\n", df["cluster"].value_counts().sort_index())
            print("\\ncluster means:\\n", df.groupby("cluster")[FEATURES].mean())

        if __name__ == "__main__":
            main()

        CLUSTER_SIZES = {json.dumps({int(k): int(v) for k, v in sizes.items()})}
        CLUSTER_PROFILES = {json.dumps({str(k): v for k, v in profiles.items()})}
        ''',
    )

    return json.dumps(
        {
            "analysis": "clustering",
            "n_clusters": n_clusters,
            "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
            "cluster_profiles": {str(k): v for k, v in profiles.items()},
            "artifact_py": str(artifact),
            "teaching_note": "Clustering finds segments without needing labels.",
        },
        indent=2,
    )


# =============================================================================
# SECTION 3 — Agent wiring
# =============================================================================

SYSTEM_PROMPT = """
You are the on-call AI analyst at Data Analyst Co.

You have company tools that run real analyses and WRITE .py artifact files.
Workflow rules:
1. If the dataset may not exist, call prepare_sample_dataset first.
2. Prefer a sensible sequence: EDA → correlation → model and/or clustering.
3. You may call MULTIPLE tools for one business question.
4. In your final AnalystBrief, list tools_used and every artifact path returned.
5. Explain findings in plain language for a business stakeholder.
6. Never invent metrics — only report numbers returned by tools.
""".strip()


def build_agent():
    """Build the Data Analyst Co. multi-tool LangChain agent."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [
        prepare_sample_dataset,
        run_eda_analysis,
        run_correlation_analysis,
        train_baseline_model,
        run_clustering_analysis,
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=AnalystBrief,
    )


# =============================================================================
# SECTION 4 — Teaching helpers (trace + print brief)
# =============================================================================


def explain_tool_trace(messages: List[Any]) -> None:
    """Print which company tools the agent called and with what args."""
    print("\n" + "=" * 60)
    print("TOOL CALL TRACE — Data Analyst Co.")
    print("=" * 60)

    step = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                step += 1
                print(f"\n[{step}] Called: {call.get('name')}")
                print(f"    args = {json.dumps(call.get('args', {}), indent=2)}")
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Pull artifact path out of JSON if present (nice for demos)
            preview = content[:500] + ("..." if len(content) > 500 else "")
            print(f"    └─ result preview:\n{preview}\n")

    if step == 0:
        print("No tools were called.")
    print("=" * 60 + "\n")


def print_brief(result: dict) -> None:
    """Pretty-print the structured AnalystBrief."""
    brief: Optional[AnalystBrief] = result.get("structured_response")
    print("\n" + "=" * 60)
    print("ANALYST BRIEF (structured)")
    print("=" * 60)

    if brief is None:
        print("No structured_response. Keys:", list(result.keys()))
        print(result)
        return

    print(f"\nExecutive summary:\n{brief.executive_summary}\n")
    print("Key findings:")
    for i, f in enumerate(brief.key_findings, 1):
        print(f"  {i}. {f}")
    print(f"\nTools used: {brief.tools_used}")
    print("\nArtifacts (.py files):")
    for a in brief.artifacts:
        print(f"  - [{a.tool_name}] {a.path}")
        print(f"      {a.summary}")
    print("\nRecommended next steps:")
    for i, s in enumerate(brief.recommended_next_steps, 1):
        print(f"  {i}. {s}")
    print("=" * 60 + "\n")
    print(f"Open folder: {OUTPUT_DIR}")


# =============================================================================
# SECTION 5 — Demo questions
# =============================================================================

DEMO_QUESTIONS = [
    # Full pipeline — best classroom showcase
    (
        "Prepare a sample dataset of 250 customers, run EDA and correlation "
        "analysis, train a baseline churn model, and summarize what drives churn. "
        "List every .py artifact your tools created."
    ),
    # Segmentation-focused
    (
        "Ensure the dataset exists, then segment customers into 3 clusters and "
        "explain each segment in business language. Mention the generated script."
    ),
    # Modeling-only after data exists
    (
        "If needed prepare data, then only train the baseline churn model and "
        "explain accuracy and the most important coefficients."
    ),
]


def run_one(question: str) -> None:
    """Run one stakeholder question through the analyst agent."""
    agent = build_agent()

    print("\n" + "#" * 60)
    print("STAKEHOLDER QUESTION")
    print("#" * 60)
    print(question)

    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    explain_tool_trace(result.get("messages", []))
    print_brief(result)


def main() -> None:
    """
    Live demo entrypoint.

    Default question triggers prepare → EDA → correlation → model
    (multiple tool calls + multiple .py outputs).
    """
    print("Data Analyst Co. — LangChain ML Toolkit Agent")
    print(f"Artifacts will be written to: {OUTPUT_DIR}\n")

    run_one(DEMO_QUESTIONS[0])

    # Uncomment for the full workshop suite:
    # for q in DEMO_QUESTIONS:
    #     run_one(q)


if __name__ == "__main__":
    main()
