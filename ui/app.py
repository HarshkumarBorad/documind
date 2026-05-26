"""DocuMind Streamlit UI.

Run with:
    streamlit run ui/app.py

Requires the FastAPI backend (Phase 5) to be running on http://localhost:8001
(or wherever DOCUMIND_API_URL points).
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
from pathlib import Path

# Streamlit only puts the script's directory (ui/) on sys.path. In API mode
# that's fine — we only import sibling modules. In LOCAL mode the UI imports
# `vectorstore`, `ingestion`, `rag_pipeline` etc. which live at the project
# root — so prepend the project root to sys.path before any project imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# Sibling modules inside ui/ — found via Streamlit's default script-dir entry.
from api_client import APIClient, APIError  # noqa: E402
from styles import DOMAIN_STYLES, GLOBAL_CSS, domain_badge_html, domain_label  # noqa: E402

# DOCUMIND_MODE=local — call the graph directly in-process (HF Spaces, single-container deploys).
# DOCUMIND_MODE=api / unset — talk to the FastAPI backend (default: docker-compose deploys).
_LOCAL_MODE = os.environ.get("DOCUMIND_MODE", "api").lower() == "local"

DOMAINS = list(DOMAIN_STYLES.keys())

st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def get_client():
    """Returns either an HTTP APIClient or an in-process LocalClient.

    Switched by the DOCUMIND_MODE env var. Not cached on purpose — caching
    hides class changes from Streamlit's auto-reload (AttributeErrors on
    new methods until you click Clear Cache).
    """
    if _LOCAL_MODE:
        from local_client import LocalClient  # local import — only needed in this mode
        return LocalClient()
    return APIClient()


@st.cache_data(ttl=60)
def _cached_models() -> dict:
    return get_client().models()


def _try_health() -> dict | None:
    """Best-effort health check used by the header strip. Returns None on failure."""
    try:
        return get_client().health()
    except APIError:
        return None


# ===== Header =============================================================

client = get_client()
health = _try_health()

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# 📚 DocuMind")
    st.markdown(
        "<p style='color:#64748b;margin-top:-1rem;font-size:1.05rem;'>"
        "Multi-namespace RAG platform — query four knowledge spaces individually or federally."
        "</p>",
        unsafe_allow_html=True,
    )

with col_status:
    if health is None:
        st.markdown(
            "<div style='text-align:right;padding-top:1rem;'>"
            "<span class='status-pill status-error'>● API unreachable</span></div>",
            unsafe_allow_html=True,
        )
    elif health["status"] == "ok":
        total_chunks = sum(health.get("namespaces", {}).values())
        st.markdown(
            f"<div style='text-align:right;padding-top:1rem;'>"
            f"<span class='status-pill status-ok'>● Connected · {total_chunks} chunks</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='text-align:right;padding-top:1rem;'>"
            f"<span class='status-pill status-warn'>● {health['status']}</span></div>",
            unsafe_allow_html=True,
        )

# Per-namespace chunk-count strip — gives an at-a-glance picture of what's indexed.
if health and health["status"] == "ok":
    strip_cols = st.columns(len(DOMAINS))
    for col, d in zip(strip_cols, DOMAINS):
        s = DOMAIN_STYLES[d]
        count = health["namespaces"].get(d, 0)
        with col:
            st.markdown(
                f"<div style='background:{s['bg']};border-left:4px solid {s['border']};"
                f"padding:10px 14px;border-radius:8px;'>"
                f"<div style='font-size:0.78rem;color:{s['fg']};font-weight:600;'>"
                f"{s['icon']} {s['label'].upper()}</div>"
                f"<div style='font-size:1.4rem;color:{s['fg']};font-weight:700;'>{count}</div>"
                f"<div style='font-size:0.72rem;color:{s['fg']};opacity:0.7;'>chunks</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)


# ===== Sidebar ============================================================

with st.sidebar:
    st.markdown("### 🔌 Connection")
    st.code(client.base_url, language=None)
    if st.button("Test connection", use_container_width=True):
        h = _try_health()
        if h is None:
            st.error("Unreachable — is the API running on this port?")
        else:
            st.success(f"{h['status']} · chroma: {h['chroma']}")

    st.divider()
    st.markdown("### 🔍 Query")

    mode = st.pills(
        "Mode",
        ["Single domain", "Federated"],
        default="Single domain",
        help="Single = one namespace.  Federated = all four in parallel + cross-encoder rerank.",
    )

    selected_domain = None
    if mode == "Single domain":
        selected_domain = st.selectbox(
            "Domain",
            DOMAINS,
            format_func=domain_label,
        )

    try:
        models_info = _cached_models()
        model_options = models_info["supported"]
        default_idx = model_options.index(models_info["default"])
    except APIError:
        st.warning("Could not load model list — using fallback.")
        model_options = ["Qwen/Qwen2.5-7B-Instruct"]
        default_idx = 0

    selected_model = st.selectbox("LLM", model_options, index=default_idx)

    top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=20,
        value=5,
        help="Chunks retrieved before reranking. For federated, this is per-domain.",
    )

    with st.expander("⚙️ Advanced"):
        provider_override = st.text_input(
            "HF Provider override",
            value="",
            placeholder="auto / together / fireworks-ai / ...",
            help="Leave blank to use the backend default (HF_LLM_PROVIDER).",
        )
        show_retrieved_chunks = st.checkbox(
            "Show all retrieved chunks (debug)",
            value=False,
        )


# ===== Tabs ===============================================================

tab_query, tab_ingest, tab_status, tab_eval = st.tabs(
    ["🔍 Query", "📥 Ingest", "⚡ Status", "📊 Evaluation"]
)


# ----- Query tab ----------------------------------------------------------

def _render_sources(sources: list[dict]) -> None:
    """Bordered card per source, color-strip on the left keyed to its domain."""
    for src in sources:
        domain = src.get("domain") or ""
        s = DOMAIN_STYLES.get(domain) if domain else None
        accent = s["border"] if s else "#cbd5e1"
        icon = s["icon"] if s else "📄"

        page_suffix = (
            f"&nbsp;·&nbsp;page {src['page'] + 1}" if src.get("page", -1) >= 0 else ""
        )
        domain_badge = (
            f"&nbsp;{domain_badge_html(domain)}" if domain else ""
        )

        header = (
            f"<div style='border-left:4px solid {accent};padding:6px 12px;"
            f"background:#f8fafc;border-radius:6px;margin-top:8px;'>"
            f"<span style='font-weight:700;color:#0f172a;font-size:0.95rem;'>"
            f"[{src['n']}]</span>"
            f"{domain_badge}"
            f"&nbsp;&nbsp;<span style='color:#475569;'>{icon} {src['filename']}{page_suffix}</span>"
            f"</div>"
        )
        st.markdown(header, unsafe_allow_html=True)
        with st.expander("Show chunk text", expanded=False):
            st.text(src["text"])


with tab_query:
    st.markdown("#### What would you like to know?")

    # st.form + st.text_input → Enter submits (st.text_area requires Ctrl+Enter).
    with st.form("query_form", clear_on_submit=False, border=False):
        question = st.text_input(
            label="Question",
            label_visibility="collapsed",
            placeholder=(
                "e.g. What is the leave policy?    "
                "(or for federated: What does the company say about GDPR?)"
            ),
        )
        col_ask, col_hint, _ = st.columns([1, 2, 4])
        with col_ask:
            run = st.form_submit_button(
                "🔍 Ask DocuMind",
                type="primary",
                use_container_width=True,
            )
        with col_hint:
            if mode == "Single domain" and selected_domain:
                s = DOMAIN_STYLES[selected_domain]
                st.markdown(
                    f"<div style='padding-top:6px;color:#64748b;font-size:0.9rem;'>"
                    f"searching <b>{s['icon']} {s['label']}</b> with "
                    f"<code>{selected_model.split('/')[-1]}</code>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='padding-top:6px;color:#64748b;font-size:0.9rem;'>"
                    f"searching <b>all 4 namespaces</b> with "
                    f"<code>{selected_model.split('/')[-1]}</code>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    if run and question.strip():
        spinner_text = (
            f"Querying {selected_domain} via {selected_model.split('/')[-1]}…"
            if mode == "Single domain"
            else f"Federated query via {selected_model.split('/')[-1]}…"
        )
        with st.spinner(spinner_text):
            try:
                if mode == "Single domain":
                    result = client.query_single(
                        domain=selected_domain,
                        question=question,
                        model=selected_model,
                        top_k=top_k,
                        provider=provider_override or None,
                    )
                else:
                    result = client.query_federated(
                        question=question,
                        model=selected_model,
                        top_k=top_k,
                        provider=provider_override or None,
                    )
            except APIError as exc:
                st.error(f"Query failed: {exc}")
                st.stop()

        st.markdown("### 💡 Answer")
        with st.container(border=True):
            st.markdown(result.get("answer", "_(no answer)_"))

        sources = result.get("sources", [])
        if sources:
            st.markdown(f"### 📑 Sources&nbsp;<span style='color:#64748b;font-size:0.9rem;'>"
                        f"({len(sources)} cited)</span>",
                        unsafe_allow_html=True)
            _render_sources(sources)
        else:
            st.info("LLM didn't cite any sources for this answer.")

        st.markdown(
            f"<div class='documind-footer'>"
            f"<b>Model:</b> <code>{result['model']}</code>"
            f" &nbsp;·&nbsp; <b>Mode:</b> {result['mode']}"
            f" &nbsp;·&nbsp; <b>Retrieved (post-rerank):</b> {result['retrieved_count']}"
            f" &nbsp;·&nbsp; <b>Cited:</b> {len(sources)}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if show_retrieved_chunks:
            with st.expander(f"🔬 Debug — all {result['retrieved_count']} retrieved chunks"):
                st.json(result)


# ----- Ingest tab ---------------------------------------------------------

with tab_ingest:
    st.markdown("#### Add documents to a knowledge namespace")
    st.caption(
        "Files are embedded with BGE-M3 (or the configured embedder), chunked "
        "per the domain's strategy, and upserted to ChromaDB. Re-ingesting "
        "the same files is idempotent."
    )

    ingest_domain = st.pills(
        "Target namespace",
        DOMAINS,
        default=DOMAINS[0],
        format_func=domain_label,
        key="ingest_target_pill",
    )

    upload_method = st.pills(
        "Method",
        ["Upload files", "Server-local path"],
        default="Upload files",
        help=(
            "Upload — drop files in your browser; they're written to a temp dir "
            "the API can read. Path — type a directory the backend can already see."
        ),
    )

    reset = st.checkbox(
        "Reset namespace before ingesting (wipes existing chunks)",
        value=False,
    )

    if upload_method == "Upload files":
        uploaded = st.file_uploader(
            "Drop PDF / DOCX / MD / TXT / HTML files",
            type=["pdf", "docx", "md", "txt", "html", "htm"],
            accept_multiple_files=True,
            help="Files are written to a temporary directory on the server, then ingested.",
        )
        if st.button(
            "📥 Ingest uploaded files",
            type="primary",
            disabled=not uploaded,
            use_container_width=True,
        ):
            with tempfile.TemporaryDirectory(prefix="documind_upload_") as tmpdir:
                tmp_path = pathlib.Path(tmpdir)
                for up_file in uploaded:
                    (tmp_path / up_file.name).write_bytes(up_file.getvalue())
                spinner = (
                    f"Ingesting {len(uploaded)} file(s) into "
                    f"{DOMAIN_STYLES[ingest_domain]['icon']} {ingest_domain}…"
                )
                with st.spinner(spinner):
                    try:
                        result = client.ingest(
                            ingest_domain, str(tmp_path), reset=reset
                        )
                    except APIError as exc:
                        st.error(f"Ingest failed: {exc}")
                    else:
                        st.success(
                            f"✓ Added **{result['chunks_added']}** chunks. "
                            f"`{result['domain']}` now contains "
                            f"**{result['total_chunks']}** chunks total."
                        )
                        st.balloons()
    else:
        server_path = st.text_input(
            "Server-local path",
            placeholder=r"e.g. C:\Users\harsh\docs\hr",
            help="Absolute path on the machine running the API.",
        )
        if st.button(
            "📥 Ingest from path",
            type="primary",
            disabled=not server_path,
            use_container_width=True,
        ):
            spinner = (
                f"Ingesting from '{server_path}' into "
                f"{DOMAIN_STYLES[ingest_domain]['icon']} {ingest_domain}…"
            )
            with st.spinner(spinner):
                try:
                    result = client.ingest(ingest_domain, server_path, reset=reset)
                except APIError as exc:
                    st.error(f"Ingest failed: {exc}")
                else:
                    st.success(
                        f"✓ Added **{result['chunks_added']}** chunks. "
                        f"`{result['domain']}` now contains "
                        f"**{result['total_chunks']}** chunks total."
                    )


# ----- Status tab ---------------------------------------------------------

with tab_status:
    if st.button("🔄 Refresh", use_container_width=False):
        st.rerun()

    h = _try_health()
    if h is None:
        st.error(
            f"API unreachable at `{client.base_url}`.\n\n"
            "Start the FastAPI backend with: `uvicorn api.main:app --reload --port 8001`"
        )
    else:
        status_class = {"ok": "status-ok", "degraded": "status-warn", "error": "status-error"}.get(
            h["status"], "status-warn"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### API")
            st.markdown(
                f"<span class='status-pill {status_class}'>● {h['status'].upper()}</span>",
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown("#### ChromaDB")
            st.code(h["chroma"], language=None)

        st.divider()
        st.markdown("#### Namespaces")
        ns_cols = st.columns(len(DOMAINS))
        for col, d in zip(ns_cols, DOMAINS):
            s = DOMAIN_STYLES[d]
            count = h["namespaces"].get(d, 0)
            with col:
                st.metric(
                    label=f"{s['icon']} {s['label']}",
                    value=count,
                    help=f"Chunks indexed in `{d}`",
                )


# ----- Evaluation tab -----------------------------------------------------

with tab_eval:
    st.markdown("#### RAGAS evaluation")
    st.caption(
        "Scores the RAG pipeline on a per-domain test set. Without ground "
        "truths, only **faithfulness** (does the answer match retrieved "
        "context?) and **answer relevancy** (does it address the question?) "
        "run. With ground truths in `evaluation/test_queries.json`, you also "
        "get **context precision** and **context recall**."
    )

    # Load test queries
    try:
        eval_queries_payload = client.evaluation_queries()
        all_queries: dict = eval_queries_payload.get("queries", {})
    except APIError as exc:
        st.error(f"Failed to load test queries: {exc}")
        all_queries = {}

    col_dom, col_max, col_model, col_judge = st.columns([1, 1, 2, 2])
    with col_dom:
        eval_domain = st.selectbox(
            "Domain",
            DOMAINS,
            format_func=domain_label,
            key="eval_domain",
        )
    with col_max:
        max_q = st.number_input(
            "Max queries",
            min_value=1,
            max_value=20,
            value=3,
            help="RAGAS makes 4–8 LLM calls per query — keep low for free-tier HF credit.",
        )
    with col_model:
        eval_model = st.selectbox(
            "Generation LLM",
            model_options,
            index=default_idx,
            key="eval_gen_model",
        )
    with col_judge:
        eval_judge = st.selectbox(
            "Judge LLM",
            model_options,
            index=default_idx,
            key="eval_judge_model",
            help="LLM RAGAS uses to score answers. Using the same model as generation is biased but cheap.",
        )

    domain_queries = all_queries.get(eval_domain, [])
    has_gt = any(q.get("ground_truth", "").strip() for q in domain_queries)

    with st.expander(
        f"📝 Test queries for {domain_label(eval_domain)} "
        f"({len(domain_queries)} defined · "
        f"{'with' if has_gt else 'without'} ground truths)",
        expanded=False,
    ):
        if not domain_queries:
            st.info("No queries defined. Edit `evaluation/test_queries.json` to add some.")
        for q in domain_queries:
            st.markdown(f"- **Q:** {q['question']}")
            gt = q.get("ground_truth", "").strip()
            if gt:
                st.caption(f"  ↳ Expected: {gt}")

    run_eval = st.button(
        "🧪 Run evaluation",
        type="primary",
        disabled=not domain_queries,
        use_container_width=False,
    )

    if run_eval:
        eval_count = min(int(max_q), len(domain_queries))
        approx_calls = eval_count * (4 if has_gt else 2) * 2
        with st.spinner(
            f"Running RAGAS on {eval_count} query(ies) · ~{approx_calls} judge LLM calls · this can take several minutes…"
        ):
            try:
                result = client.run_evaluation(
                    domain=eval_domain,
                    model=eval_model,
                    judge_model=eval_judge,
                    max_queries=int(max_q),
                )
            except APIError as exc:
                st.error(f"Evaluation failed: {exc}")
                st.stop()

        st.markdown("### Overall scores")
        overall = result.get("overall", {})
        if not overall:
            st.warning("No metric scores produced. Check the API logs for errors.")
        else:
            score_cols = st.columns(len(overall))
            for col, (metric, score) in zip(score_cols, overall.items()):
                with col:
                    nice_name = metric.replace("_", " ").title()
                    if score != score:  # NaN check
                        st.metric(nice_name, "—")
                    else:
                        st.metric(nice_name, f"{score:.3f}")

            chart_df = pd.DataFrame(
                {
                    "metric": list(overall.keys()),
                    "score": [v if v == v else 0 for v in overall.values()],
                }
            ).set_index("metric")
            st.bar_chart(chart_df, height=240)

        st.markdown("### Per-query results")
        for row in result.get("per_query", []):
            with st.expander(f"❔ {row['question']}", expanded=False):
                left, right = st.columns([2, 1])
                with left:
                    st.markdown("**Generated answer**")
                    answer_preview = row.get("answer", "") or "_(no answer)_"
                    st.markdown(answer_preview)
                    if row.get("ground_truth"):
                        st.markdown("**Ground truth**")
                        st.caption(row["ground_truth"])
                with right:
                    st.markdown("**Scores**")
                    metric_keys = [
                        k
                        for k in row.keys()
                        if k not in ("question", "answer", "context_count", "ground_truth")
                    ]
                    for mk in metric_keys:
                        val = row[mk]
                        nice_name = mk.replace("_", " ").title()
                        if val is None:
                            st.metric(nice_name, "—")
                        else:
                            st.metric(nice_name, f"{val:.3f}")
                    st.caption(f"Retrieved {row.get('context_count', 0)} context chunk(s)")

        st.markdown(
            f"<div class='documind-footer'>"
            f"<b>Domain:</b> {result['domain']}"
            f" &nbsp;·&nbsp; <b>Generation:</b> <code>{result['model']}</code>"
            f" &nbsp;·&nbsp; <b>Judge:</b> <code>{result['judge_model']}</code>"
            f" &nbsp;·&nbsp; <b>Ground truths:</b> {'yes' if result['has_ground_truths'] else 'no'}"
            f"</div>",
            unsafe_allow_html=True,
        )
