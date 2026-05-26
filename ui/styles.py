"""Visual identity for the DocuMind UI.

A single registry of per-domain color and icon, used everywhere a domain
appears (sidebar, source badges, ingest target picker, status cards) so the
look stays consistent across tabs.
"""
from __future__ import annotations

# (icon, bg, fg, border) — pastel bg + saturated fg/border so text stays readable
# against bg and the border can be reused as an accent strip on source cards.
DOMAIN_STYLES: dict[str, dict[str, str]] = {
    "hr": {
        "icon": "🧑‍💼",
        "label": "HR",
        "bg": "#dbeafe",
        "fg": "#1e3a8a",
        "border": "#3b82f6",
    },
    "tech": {
        "icon": "🔧",
        "label": "Tech",
        "bg": "#d1fae5",
        "fg": "#064e3b",
        "border": "#10b981",
    },
    "research": {
        "icon": "🔬",
        "label": "Research",
        "bg": "#ede9fe",
        "fg": "#4c1d95",
        "border": "#8b5cf6",
    },
    "product": {
        "icon": "📦",
        "label": "Product",
        "bg": "#ffedd5",
        "fg": "#7c2d12",
        "border": "#f59e0b",
    },
}


def domain_badge_html(domain: str) -> str:
    """Inline pill — usable inside st.markdown(unsafe_allow_html=True)."""
    s = DOMAIN_STYLES.get(domain, {"icon": "❔", "label": domain, "bg": "#e5e7eb", "fg": "#374151"})
    return (
        f"<span style='background:{s['bg']};color:{s['fg']};"
        f"padding:2px 10px;border-radius:999px;font-size:0.82rem;"
        f"font-weight:600;white-space:nowrap;'>"
        f"{s['icon']} {s['label']}</span>"
    )


def domain_label(domain: str) -> str:
    """Plain-text label with icon — for selectbox / pills options."""
    s = DOMAIN_STYLES.get(domain)
    if not s:
        return domain
    return f"{s['icon']} {s['label']}"


GLOBAL_CSS = """
<style>
/* Tighter header spacing */
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* Gradient title accent */
h1 {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800 !important;
}

/* Sidebar polish */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    border-right: 1px solid #e2e8f0;
}

/* Make metric cards feel like cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}

/* Source-card container — subtle elevation */
div[data-testid="stExpander"] {
    border-radius: 10px;
    border-color: #e2e8f0 !important;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
}

/* Smaller caption font for footer stats */
.documind-footer {
    color: #64748b;
    font-size: 0.85rem;
    padding-top: 0.5rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 1rem;
}

/* Status pill (used in header) */
.status-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
}
.status-ok    { background:#d1fae5; color:#065f46; }
.status-warn  { background:#fef3c7; color:#92400e; }
.status-error { background:#fee2e2; color:#991b1b; }
</style>
"""
