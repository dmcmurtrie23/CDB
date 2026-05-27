"""
AFL Numbers
-----------
Streamlit app to query the Catapult Connect API (Vector/AMS / OpenField Cloud).

Lets you:
  - Switch between AP, US, EU regions in the sidebar
  - Authenticate with a Bearer token
  - Browse athletes, activities, and activity parameters (GPS/load metrics)
  - Drill into a specific activity to see participating athletes + metrics
  - Export any result as CSV

Run with:
    pip install streamlit requests pandas
    streamlit run afl_numbers.py
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests
import streamlit as st
import altair as alt

# ---------------------------------------------------------------------------
# Region config
# ---------------------------------------------------------------------------
REGIONS: dict[str, str] = {
    "Asia-Pacific (AU)": "https://connect-au.catapultsports.com/api/v6",
    "North America (US)": "https://connect-us.catapultsports.com/api/v6",
    "EMEA (EU)":          "https://connect-eu.catapultsports.com/api/v6",
}

REQUEST_TIMEOUT = 120  # seconds — /stats with many activities can be slow


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
class CatapultClient:
    """Thin wrapper around the Catapult Connect REST API."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        })

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        # Some endpoints return empty bodies on 204 etc.
        if not resp.content:
            return []
        return resp.json()

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.session.post(
            url,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        if not resp.content:
            return []
        return resp.json()

    # --- Endpoints ---------------------------------------------------------
    def list_athletes(self) -> list[dict]:
        return self._get("/athletes")

    def list_teams(self) -> list[dict]:
        return self._get("/teams")

    def team_athletes(self, team_id: str) -> list[dict]:
        return self._get(f"/teams/{team_id}/athletes")

    def list_activities(self, start: date | None = None, end: date | None = None) -> list[dict]:
        params: dict[str, Any] = {}
        if start:
            params["start_time"] = int(pd.Timestamp(start).timestamp())
        if end:
            # end of the day
            params["end_time"] = int(pd.Timestamp(end + timedelta(days=1)).timestamp())
        return self._get("/activities", params=params or None)

    def activity_athletes(self, activity_id: str) -> list[dict]:
        return self._get(f"/activities/{activity_id}/athletes")

    def activity_periods(self, activity_id: str) -> list[dict]:
        return self._get(f"/activities/{activity_id}/periods")

    def athlete_parameters(self, activity_id: str, athlete_id: str) -> list[dict]:
        # Per-athlete aggregate parameters for an activity
        return self._get(f"/activities/{activity_id}/athletes/{athlete_id}/parameters")

    def list_parameters(self) -> list[dict]:
        """List all available parameter definitions (name, slug, etc.)."""
        return self._get("/parameters")

    def stats(
        self,
        parameters: list[str],
        group_by: list[str],
        activity_ids: list[str] | None = None,
        period_ids: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> list[dict]:
        """POST /stats — flat aggregated metrics with grouping.

        parameters: list of parameter slugs, e.g. ["total_player_load", "total_distance"]
        group_by:   list of grouping dims, e.g. ["athlete", "team", "position", "period"]
        period_ids: optional list of period UUIDs to filter to
        """
        filters = []
        if activity_ids:
            filters.append({
                "name": "activity_id",
                "comparison": "=",
                "values": activity_ids,
            })
        if period_ids:
            filters.append({
                "name": "period_id",
                "comparison": "=",
                "values": period_ids,
            })
        if start:
            filters.append({
                "name": "start_time",
                "comparison": ">=",
                "values": [int(pd.Timestamp(start).timestamp())],
            })
        if end:
            filters.append({
                "name": "end_time",
                "comparison": "<=",
                "values": [int(pd.Timestamp(end + timedelta(days=1)).timestamp())],
            })
        body = {
            "filters": filters,
            "parameters": parameters,
            "group_by": group_by,
        }
        return self._post("/stats", body)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_df(data: Any) -> pd.DataFrame:
    """Best-effort convert API response into a flat DataFrame."""
    if not data:
        return pd.DataFrame()
    if isinstance(data, dict):
        data = [data]
    return pd.json_normalize(data)


def df_download_button(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    if df.empty:
        return
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/csv")


def kpi_cards(cards: list[dict]) -> None:
    """Render a row of KPI cards. Each card: {label, value, style?}.
    style can be 'default', 'accent', or 'success'."""
    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        style = c.get("style", "default")
        klass = f"kpi-card kpi-{style}" if style != "default" else "kpi-card"
        col.markdown(
            f"""
            <div class="{klass}">
                <div class="kpi-label">{c['label']}</div>
                <div class="kpi-value">{c['value']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def add_rank(df: pd.DataFrame, value_col: str, rank_col: str = "Rank") -> pd.DataFrame:
    """Add a 1-indexed Rank column at the front, based on descending value_col."""
    if df.empty or value_col not in df.columns:
        return df
    df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    df.insert(0, rank_col, range(1, len(df) + 1))
    return df


def _scrub_for_chart(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities with NaN and drop columns that are entirely null —
    Vega-Lite chokes on Infinity/-Infinity and on all-null numeric fields,
    causing 'Infinite extent' warnings and blank charts."""
    import numpy as np
    if df.empty:
        return df
    df = df.copy()
    # Replace infinities in numeric columns with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    # Drop columns that are entirely null (Vega-Lite can't compute extents on them)
    all_null_cols = [c for c in df.columns if df[c].isna().all()]
    if all_null_cols:
        df = df.drop(columns=all_null_cols)
    return df


def ranked_bar_chart(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    orientation: str = "Horizontal",
    title: str = "",
) -> alt.Chart:
    """Build a ranked bar chart. Bars sorted by value descending."""
    plot_df = _scrub_for_chart(df).sort_values(value_col, ascending=False)
    # Drop rows where the value itself is null (would blow up axis extent)
    plot_df = plot_df.dropna(subset=[value_col])
    # Limit tooltip columns to ones that survived scrubbing
    tooltip_cols = [c for c in plot_df.columns if c in (label_col, value_col, "Rank")]
    # Add up to 3 more harmless columns
    for c in plot_df.columns:
        if c not in tooltip_cols and len(tooltip_cols) < 6:
            tooltip_cols.append(c)

    if orientation == "Horizontal":
        chart = (
            alt.Chart(plot_df)
            .mark_bar(color="#ff4b1f", cornerRadiusEnd=4)
            .encode(
                x=alt.X(f"{value_col}:Q", title=value_col),
                y=alt.Y(f"{label_col}:N", sort="-x", title=label_col),
                tooltip=tooltip_cols,
            )
        )
    else:
        chart = (
            alt.Chart(plot_df)
            .mark_bar(color="#ff4b1f", cornerRadiusEnd=4)
            .encode(
                x=alt.X(f"{label_col}:N", sort="-y", title=label_col),
                y=alt.Y(f"{value_col}:Q", title=value_col),
                tooltip=tooltip_cols,
            )
        )
    if title:
        chart = chart.properties(title=title)
    return chart.properties(height=max(280, 26 * len(plot_df)) if orientation == "Horizontal" else 360)


# --- Altair dark theme -----------------------------------------------------
def _afl_altair_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "axis": {
                "labelColor": "#8b949e",
                "titleColor": "#e6edf3",
                "gridColor": "#232a36",
                "domainColor": "#2f3847",
                "tickColor": "#2f3847",
                "labelFont": "Inter",
                "titleFont": "Inter",
                "titleFontWeight": 500,
                "titleFontSize": 12,
                "labelFontSize": 11,
                "titlePadding": 10,
            },
            "legend": {
                "labelColor": "#e6edf3",
                "titleColor": "#8b949e",
                "labelFont": "Inter",
                "titleFont": "Inter",
                "labelFontSize": 12,
                "titleFontSize": 11,
                "titleFontWeight": 500,
            },
            "title": {
                "color": "#e6edf3",
                "font": "Inter",
                "fontSize": 14,
                "fontWeight": 600,
                "anchor": "start",
                "subtitleColor": "#8b949e",
            },
            "range": {
                # Sporty palette: orange-red primary, then balanced supporting colours
                "category": [
                    "#ff4b1f", "#58a6ff", "#3fb950", "#d29922",
                    "#bc8cff", "#ff7b72", "#79c0ff", "#56d364",
                ],
            },
            "mark": {"font": "Inter"},
            "header": {"labelColor": "#e6edf3", "titleColor": "#e6edf3"},
        }
    }

alt.themes.register("afl_dark", _afl_altair_theme)
alt.themes.enable("afl_dark")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AFL Numbers",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS theme ------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #0a0d12;
        --bg-2: #0d1117;
        --panel: #151a23;
        --panel-2: #1c2230;
        --panel-hover: #232a3a;
        --border: #232a36;
        --border-strong: #2f3847;
        --text: #e6edf3;
        --text-dim: #8b949e;
        --text-faint: #6e7681;
        --accent: #ff4b1f;
        --accent-hover: #ff6b3f;
        --accent-soft: rgba(255, 75, 31, 0.12);
        --accent-glow: rgba(255, 75, 31, 0.3);
        --success: #3fb950;
        --warning: #d29922;
        --info: #58a6ff;
    }

    /* Global font */
    html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    code, pre, kbd, .stCode {
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
    }

    /* Page background with subtle radial accents */
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(255, 75, 31, 0.04) 0%, transparent 40%),
            radial-gradient(circle at 100% 100%, rgba(88, 166, 255, 0.03) 0%, transparent 40%),
            linear-gradient(180deg, #0a0d12 0%, #070a0e 100%);
        color: var(--text);
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .stDeployButton { display: none; }

    /* Main content area padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1400px !important;
    }

    /* === Hero header === */
    .afl-hero {
        background:
            linear-gradient(135deg, rgba(255, 75, 31, 0.08) 0%, transparent 50%),
            linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 75, 31, 0.05);
    }
    .afl-hero-left {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        flex: 1;
    }
    .afl-logo {
        width: 68px;
        height: 68px;
        border-radius: 14px;
        background: var(--panel);
        border: 1px solid var(--border-strong);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        flex-shrink: 0;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    }
    .afl-logo img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        padding: 6px;
    }
    .afl-hero h1 {
        color: var(--text);
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }
    .afl-hero .subtitle {
        color: var(--text-dim);
        font-size: 0.9rem;
        margin-top: 0.35rem;
        font-weight: 400;
    }
    .afl-hero .badge {
        background: var(--accent);
        color: white;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: 0 0 16px var(--accent-glow);
        position: relative;
    }
    .afl-hero .badge::before {
        content: "";
        width: 6px;
        height: 6px;
        background: white;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        vertical-align: middle;
        animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* === KPI cards === */
    .kpi-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--border);
        opacity: 0;
        transition: opacity 0.25s ease;
    }
    .kpi-card:hover {
        border-color: var(--border-strong);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .kpi-card:hover::before { opacity: 1; }
    .kpi-card.kpi-accent::before { background: var(--accent); opacity: 1; }
    .kpi-card.kpi-success::before { background: var(--success); opacity: 1; }
    .kpi-label {
        color: var(--text-faint);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    .kpi-value {
        color: var(--text);
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.02em;
        font-feature-settings: "tnum";
    }
    .kpi-accent .kpi-value { color: var(--accent); }
    .kpi-success .kpi-value { color: var(--success); }

    /* === Sidebar === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0d12 0%, #060a0e 100%);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem !important;
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text);
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
        color: var(--text-dim);
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
        margin: 1.5rem 0 !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: var(--text-faint) !important;
        font-size: 0.75rem !important;
    }

    /* === Tabs === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: var(--panel);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-dim);
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--panel-2);
        color: var(--text);
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(255, 75, 31, 0.4);
    }
    .stTabs [aria-selected="true"]:hover {
        background: var(--accent-hover) !important;
        color: white !important;
    }

    /* === Buttons === */
    .stButton > button, .stDownloadButton > button {
        background: var(--panel);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 10px;
        font-weight: 500;
        padding: 0.55rem 1.2rem;
        font-size: 0.875rem;
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent);
        color: var(--accent);
        background: var(--panel-hover);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        color: white;
        border-color: var(--accent);
        box-shadow: 0 4px 12px rgba(255, 75, 31, 0.3);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--accent-hover);
        border-color: var(--accent-hover);
        color: white;
        box-shadow: 0 6px 16px rgba(255, 75, 31, 0.5);
    }

    /* === Inputs === */
    .stTextInput input, .stTextArea textarea,
    .stDateInput input, .stNumberInput input {
        background: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 0.6rem 0.9rem !important;
        font-size: 0.875rem !important;
        transition: all 0.15s ease;
    }
    .stTextInput input:focus, .stTextArea textarea:focus,
    .stDateInput input:focus, .stNumberInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    /* MultiSelect chips */
    .stMultiSelect [data-baseweb="tag"] {
        background: var(--accent-soft) !important;
        border: 1px solid var(--accent) !important;
        color: var(--accent) !important;
        border-radius: 6px !important;
    }
    /* Radio buttons - pill style */
    .stRadio [role="radiogroup"] {
        gap: 6px;
        background: var(--panel);
        padding: 4px;
        border-radius: 10px;
        border: 1px solid var(--border);
        display: inline-flex !important;
    }
    .stRadio [role="radiogroup"] label {
        background: transparent;
        padding: 6px 14px !important;
        border-radius: 7px;
        margin: 0 !important;
        cursor: pointer;
        transition: all 0.15s ease;
        font-size: 0.825rem !important;
    }
    .stRadio [role="radiogroup"] label:hover {
        background: var(--panel-hover);
    }
    .stRadio [role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
        background: var(--accent);
        color: white;
    }
    /* Hide the actual radio circle */
    .stRadio [role="radiogroup"] [data-testid="stMarkdownContainer"] p {
        margin: 0 !important;
    }

    /* === DataFrames - the biggest visual upgrade === */
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
    }
    .stDataFrame > div {
        background: var(--panel) !important;
    }
    /* Table cells */
    .stDataFrame [data-testid="StyledDataFrameDataCell"] {
        background: var(--panel) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
        font-size: 0.875rem !important;
    }
    .stDataFrame [data-testid="StyledDataFrameRowHeaderCell"],
    .stDataFrame [data-testid="StyledDataFrameColumnHeaderCell"] {
        background: var(--panel-2) !important;
        color: var(--text-dim) !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        border-color: var(--border) !important;
    }

    /* === Expanders === */
    [data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        padding: 0.85rem 1.2rem;
        font-weight: 500;
        color: var(--text);
        font-size: 0.9rem;
    }
    [data-testid="stExpander"] summary:hover {
        background: var(--panel-hover);
    }

    /* === Alerts (info, warning, success, error) === */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        padding: 0.9rem 1.2rem !important;
    }
    [data-testid="stAlertContentInfo"] {
        background: rgba(88, 166, 255, 0.1) !important;
        color: var(--info) !important;
    }
    [data-testid="stAlertContentSuccess"] {
        background: rgba(63, 185, 80, 0.1) !important;
        color: var(--success) !important;
    }
    [data-testid="stAlertContentWarning"] {
        background: rgba(210, 153, 34, 0.1) !important;
        color: var(--warning) !important;
    }
    [data-testid="stAlertContentError"] {
        background: rgba(255, 75, 31, 0.1) !important;
        color: var(--accent) !important;
    }

    /* === Headings === */
    h1, h2, h3, h4 {
        color: var(--text) !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em;
    }
    .stApp h2 {
        font-size: 1.4rem !important;
        margin-top: 0.5rem !important;
    }
    .stApp h3 {
        font-size: 1.1rem !important;
    }

    /* === Captions === */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-dim) !important;
        font-size: 0.85rem !important;
    }

    /* === Code blocks === */
    .stCode {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }

    /* === Progress bar === */
    .stProgress > div > div {
        background: var(--accent) !important;
    }

    /* === Spinner === */
    .stSpinner > div {
        border-top-color: var(--accent) !important;
    }

    /* === Dividers === */
    hr {
        border-color: var(--border) !important;
        margin: 2rem 0 !important;
    }

    /* === Altair charts background === */
    .vega-embed {
        background: transparent !important;
    }

    /* === Tooltip === */
    [data-baseweb="tooltip"] {
        background: var(--panel-2) !important;
        border: 1px solid var(--border-strong) !important;
        color: var(--text) !important;
        font-size: 0.8rem !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero header ------------------------------------------------------------
# Drop a logo file (PNG/JPG/SVG) named logo.png, logo.jpg, or logo.svg into the
# same directory as this script to display it in the header. Falls back to ⚡.
import base64
from pathlib import Path

def _load_logo_html() -> str:
    here = Path(__file__).parent if "__file__" in globals() else Path(".")
    for name in ("logo.png", "logo.jpg", "logo.jpeg", "logo.svg"):
        p = here / name
        if p.exists():
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".svg": "image/svg+xml",
            }[p.suffix.lower()]
            data = base64.b64encode(p.read_bytes()).decode()
            return f'<img src="data:{mime};base64,{data}" alt="logo" />'
    return "⚡"

_logo_html = _load_logo_html()

st.markdown(f"""
<div class="afl-hero">
    <div class="afl-hero-left">
        <div class="afl-logo">{_logo_html}</div>
        <div>
            <h1>AFL Numbers</h1>
            <div class="subtitle">Catapult Connect API · performance analytics dashboard</div>
        </div>
    </div>
    <div class="badge">Live</div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar: auth & region ------------------------------------------------
with st.sidebar:
    st.header("Connection")
    region_label = st.selectbox("Region", list(REGIONS.keys()), index=0)
    base_url = REGIONS[region_label]
    st.caption(f"`{base_url}`")

    token = st.text_input(
        "API Token",
        type="password",
        help="Bearer token from OpenField Cloud → API Token Admin",
    )

    custom_url = st.text_input(
        "Override base URL (optional)",
        value="",
        help="Leave blank to use the regional URL above.",
    )
    if custom_url.strip():
        base_url = custom_url.strip().rstrip("/")

    st.divider()
    st.subheader("AI (optional)")
    anthropic_key = st.text_input(
        "Anthropic API key",
        type="password",
        help="For the Ask AI tab. Get one at console.anthropic.com.",
    )

    st.divider()
    st.caption("Tip: tokens are kept in session only and never written to disk.")

if not token:
    st.info("Enter your API token in the sidebar to get started.")
    st.stop()

client = CatapultClient(base_url, token)

# --- Tabs ------------------------------------------------------------------
tab_athletes, tab_activities, tab_activity_detail, tab_periods, tab_rankings, tab_compare, tab_acwr, tab_ask, tab_raw = st.tabs(
    ["Athletes", "Activities", "Activity Detail", "Periods", "Rankings", "Compare", "ACWR", "Ask AI", "Raw GET"]
)

# --- Athletes tab ----------------------------------------------------------
with tab_athletes:
    st.subheader("Athletes")
    if st.button("Fetch athletes", key="fetch_athletes"):
        try:
            with st.spinner("Loading athletes..."):
                data = client.list_athletes()
            df = to_df(data)
            st.success(f"Loaded {len(df)} athletes")
            st.dataframe(df, width='stretch')
            df_download_button(df, "athletes.csv")
        except requests.HTTPError as e:
            st.error(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Activities tab --------------------------------------------------------
with tab_activities:
    st.subheader("Activities")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start = st.date_input("Start date", value=date.today() - timedelta(days=30))
    with col2:
        end = st.date_input("End date", value=date.today())
    with col3:
        st.write("")
        st.write("")
        go = st.button("Fetch activities", key="fetch_activities")

    if go:
        try:
            with st.spinner("Loading activities..."):
                data = client.list_activities(start=start, end=end)
            df = to_df(data)
            st.success(f"Loaded {len(df)} activities")
            st.dataframe(df, width='stretch')
            df_download_button(df, "activities.csv")
            # Stash for the detail tab
            st.session_state["activities_df"] = df
        except requests.HTTPError as e:
            st.error(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Activity Detail tab ---------------------------------------------------
with tab_activity_detail:
    st.subheader("Activity Detail")
    st.caption("Look up athletes and per-athlete parameters for a single activity.")

    activity_id = st.text_input(
        "Activity ID",
        help="Paste an activity ID, or fetch the activities list first to find one.",
    )

    if activity_id:
        try:
            with st.spinner("Loading activity athletes..."):
                athletes_raw = client.activity_athletes(activity_id)
            athletes_df = to_df(athletes_raw)
            st.markdown(f"**Athletes in activity** ({len(athletes_df)})")
            st.dataframe(athletes_df, width='stretch')
            df_download_button(athletes_df, f"activity_{activity_id}_athletes.csv")

            # Periods for this activity
            try:
                periods_raw = client.activity_periods(activity_id) or []
            except Exception:
                periods_raw = []
            if periods_raw:
                periods_df = to_df(periods_raw)
                st.markdown(f"**Periods in activity** ({len(periods_df)})")
                # Format time columns if present
                periods_display = periods_df.copy()
                for tc in ["start_time", "end_time"]:
                    if tc in periods_display.columns and pd.api.types.is_numeric_dtype(periods_display[tc]):
                        periods_display[tc] = pd.to_datetime(periods_display[tc], unit="s").dt.strftime("%H:%M:%S")
                st.dataframe(periods_display, width='stretch')
                df_download_button(periods_df, f"activity_{activity_id}_periods.csv")
                st.caption("💡 For deeper period analysis, see the **Periods** tab.")

            # Pull parameters per athlete on demand
            if not athletes_df.empty and "id" in athletes_df.columns:
                if st.button("Fetch parameters for ALL athletes (may be slow)"):
                    rows: list[dict] = []
                    prog = st.progress(0.0)
                    for i, ath_id in enumerate(athletes_df["id"].astype(str)):
                        try:
                            params = client.athlete_parameters(activity_id, ath_id)
                            for p in (params or []):
                                p = dict(p)
                                p["athlete_id"] = ath_id
                                rows.append(p)
                        except Exception as e:
                            st.warning(f"Athlete {ath_id}: {e}")
                        prog.progress((i + 1) / len(athletes_df))
                    params_df = to_df(rows)
                    st.markdown(f"**Parameters** ({len(params_df)} rows)")
                    st.dataframe(params_df, width='stretch')
                    df_download_button(params_df, f"activity_{activity_id}_parameters.csv")
        except requests.HTTPError as e:
            st.error(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Periods tab -----------------------------------------------------------
with tab_periods:
    import numpy as np
    import traceback

    st.subheader("⏱ Periods")
    st.caption(
        "Analyse stats split by period (e.g. Q1/Q2/Q3/Q4, warmup, full game). "
        "Pick an activity, see all periods and their metrics, or compare athletes within a period."
    )

    try:
        # --- Step 1: Activity picker --------------------------------------
        st.markdown("### 1. Pick an activity")
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            _per_start = st.date_input(
                "From",
                value=date.today() - timedelta(days=30),
                key="periods_start",
            )
        with col_date2:
            _per_end = st.date_input("To", value=date.today(), key="periods_end")

        if "periods_activities" not in st.session_state:
            st.session_state["periods_activities"] = None
            st.session_state["periods_activities_key"] = None

        _act_key = (str(_per_start), str(_per_end))
        if st.session_state.get("periods_activities_key") != _act_key:
            with st.spinner(f"Loading activities from {_per_start} to {_per_end}..."):
                try:
                    _per_acts = client.list_activities(start=_per_start, end=_per_end) or []
                except Exception as _e:
                    st.error(f"Couldn't load activities: {_e}")
                    _per_acts = []
            st.session_state["periods_activities"] = _per_acts
            st.session_state["periods_activities_key"] = _act_key

        _per_acts = st.session_state["periods_activities"] or []

        if not _per_acts:
            st.warning("No activities in that date range.")
        else:
            # Build display labels for activities
            def _act_label(a: dict) -> str:
                _name = a.get("name") or a.get("activity_name") or f"Activity {str(a.get('id', ''))[:8]}"
                _ts = a.get("start_time")
                if _ts:
                    try:
                        _d = pd.to_datetime(_ts, unit="s").strftime("%Y-%m-%d")
                        return f"{_d} · {_name}"
                    except Exception:
                        return _name
                return _name

            _act_labels = {_act_label(a): str(a.get("id")) for a in _per_acts if a.get("id")}
            _act_label_list = sorted(_act_labels.keys(), reverse=True)  # newest first

            _chosen_act_label = st.selectbox(
                f"Activity ({len(_act_label_list)} available)",
                _act_label_list,
                key="periods_activity_pick",
            )
            _chosen_act_id = _act_labels[_chosen_act_label]

            # --- Step 2: Fetch periods for that activity ----------------
            st.markdown("### 2. Periods in this activity")

            _periods_key = ("periods_for_act", _chosen_act_id)
            if st.session_state.get("periods_data_key") != _periods_key:
                with st.spinner("Loading periods..."):
                    try:
                        _periods = client.activity_periods(_chosen_act_id) or []
                    except Exception as _e:
                        st.error(f"Couldn't load periods: {_e}")
                        _periods = []
                st.session_state["periods_data"] = _periods
                st.session_state["periods_data_key"] = _periods_key

            _periods = st.session_state.get("periods_data") or []

            if not _periods:
                st.info("This activity has no defined periods.")
            else:
                # Show periods table
                _per_df = pd.json_normalize(_periods)
                # Common columns to surface
                _show_cols = []
                for _c in ["name", "period_name", "start_time", "end_time", "duration"]:
                    if _c in _per_df.columns:
                        _show_cols.append(_c)

                _disp = _per_df[_show_cols].copy() if _show_cols else _per_df.copy()
                # Format epoch times as readable
                for _tc in ["start_time", "end_time"]:
                    if _tc in _disp.columns and pd.api.types.is_numeric_dtype(_disp[_tc]):
                        _disp[_tc] = pd.to_datetime(_disp[_tc], unit="s").dt.strftime("%H:%M:%S")
                st.dataframe(_disp, width='stretch', hide_index=True)

                # --- Step 3: Stats by period ---------------------------
                st.markdown("### 3. Stats by period")

                # Need parameters
                if "periods_params" not in st.session_state:
                    try:
                        st.session_state["periods_params"] = client.list_parameters() or []
                    except Exception:
                        st.session_state["periods_params"] = []
                _per_params = st.session_state["periods_params"]
                _per_name_to_slug = {p["name"]: p["slug"] for p in _per_params if p.get("name") and p.get("slug")}
                _per_metric_names = sorted(_per_name_to_slug.keys())

                if not _per_metric_names:
                    st.warning("No parameters available.")
                else:
                    # Default to total player load + total distance
                    _per_defaults = []
                    _lower = [n.lower() for n in _per_metric_names]
                    for _pref in ["total player load", "total distance", "high speed running distance"]:
                        for i, n in enumerate(_lower):
                            if _pref in n:
                                if _per_metric_names[i] not in _per_defaults:
                                    _per_defaults.append(_per_metric_names[i])
                                break
                        if len(_per_defaults) >= 3:
                            break

                    _per_chosen_metrics = st.multiselect(
                        "Metrics",
                        _per_metric_names,
                        default=_per_defaults or _per_metric_names[:3],
                        key="periods_metrics",
                        help="Pick 1-5 metrics to break down by period.",
                    )

                    _per_view = st.radio(
                        "View",
                        ["Per-period totals (team)", "Per-period × athlete", "Compare periods within activity"],
                        horizontal=True,
                        key="periods_view",
                    )

                    if _per_chosen_metrics and st.button("▶ Run analysis", type="primary", key="periods_run"):
                        _per_slugs = [_per_name_to_slug[n] for n in _per_chosen_metrics]

                        # Build group_by based on view
                        if _per_view == "Per-period totals (team)":
                            _per_group_by = ["period"]
                        elif _per_view == "Per-period × athlete":
                            _per_group_by = ["period", "athlete"]
                        else:  # compare periods within activity
                            _per_group_by = ["period"]

                        try:
                            with st.spinner("Querying /stats..."):
                                _per_rows = client.stats(
                                    parameters=_per_slugs,
                                    group_by=_per_group_by,
                                    activity_ids=[_chosen_act_id],
                                ) or []
                        except requests.HTTPError as e:
                            st.error(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
                            _per_rows = []
                        except Exception as e:
                            st.error(f"Error: {e}")
                            _per_rows = []

                        if not _per_rows:
                            st.warning("No stats returned. The activity may not have data for these metrics.")
                        else:
                            _stats_df = pd.json_normalize(_per_rows)

                            with st.expander("🔍 Debug: response columns", expanded=False):
                                st.write(f"**Columns:** `{list(_stats_df.columns)}`")
                                st.dataframe(_stats_df.head(), width='stretch')

                            # Find period column
                            def _findc(df, cands):
                                return next((c for c in cands if c in df.columns), None)

                            _pname_col = _findc(_stats_df, ["period_name", "name", "period"])

                            if not _pname_col:
                                st.warning("Couldn't find period name column.")
                                st.dataframe(_stats_df, width='stretch')
                            elif _per_view == "Per-period totals (team)":
                                # Show table + chart per metric
                                st.dataframe(_stats_df, width='stretch', hide_index=True)

                                # One chart per metric
                                for _slug, _mname in zip(_per_slugs, _per_chosen_metrics):
                                    _vcol = _slug if _slug in _stats_df.columns else _findc(_stats_df, [f"total_{_slug}", f"sum_{_slug}"])
                                    if not _vcol:
                                        continue
                                    _chart_df = _stats_df[[_pname_col, _vcol]].copy()
                                    _chart_df.columns = ["Period", _mname]
                                    _chart_df[_mname] = pd.to_numeric(_chart_df[_mname], errors="coerce").replace([np.inf, -np.inf], np.nan)
                                    _chart_df = _chart_df.dropna(subset=[_mname])
                                    if _chart_df.empty:
                                        continue
                                    st.markdown(f"**{_mname} by period**")
                                    _ch = (
                                        alt.Chart(_chart_df)
                                        .mark_bar(color="#ff4b1f", cornerRadiusEnd=4)
                                        .encode(
                                            x=alt.X("Period:N", sort=None),
                                            y=alt.Y(f"{_mname}:Q"),
                                            tooltip=["Period", alt.Tooltip(f"{_mname}:Q", format=",.2f")],
                                        )
                                        .properties(height=280)
                                    )
                                    st.altair_chart(_ch, width='stretch')

                                df_download_button(_stats_df, f"periods_{_chosen_act_id}.csv")

                            elif _per_view == "Per-period × athlete":
                                _ath_col = _findc(_stats_df, ["athlete_name", "name", "athlete"])
                                if not _ath_col:
                                    st.warning("Couldn't find athlete column.")
                                    st.dataframe(_stats_df, width='stretch')
                                else:
                                    st.dataframe(_stats_df, width='stretch', hide_index=True)

                                    # Pivot for visual comparison: rows=athlete, cols=period, values=metric
                                    for _slug, _mname in zip(_per_slugs, _per_chosen_metrics):
                                        _vcol = _slug if _slug in _stats_df.columns else _findc(_stats_df, [f"total_{_slug}", f"sum_{_slug}"])
                                        if not _vcol:
                                            continue
                                        st.markdown(f"**{_mname} — athletes by period**")
                                        _sub = _stats_df[[_ath_col, _pname_col, _vcol]].copy()
                                        _sub.columns = ["Athlete", "Period", _mname]
                                        _sub[_mname] = pd.to_numeric(_sub[_mname], errors="coerce").replace([np.inf, -np.inf], np.nan)
                                        _sub = _sub.dropna(subset=[_mname])
                                        if _sub.empty:
                                            continue
                                        _heat = (
                                            alt.Chart(_sub)
                                            .mark_bar()
                                            .encode(
                                                x=alt.X("Period:N", sort=None),
                                                y=alt.Y(f"{_mname}:Q"),
                                                color=alt.Color("Athlete:N"),
                                                xOffset="Athlete:N",
                                                tooltip=["Athlete", "Period", alt.Tooltip(f"{_mname}:Q", format=",.2f")],
                                            )
                                            .properties(height=400)
                                        )
                                        st.altair_chart(_heat, width='stretch')

                                    df_download_button(_stats_df, f"periods_athletes_{_chosen_act_id}.csv")

                            else:  # Compare periods
                                # Show metrics side-by-side per period
                                st.markdown("**Period comparison**")
                                _melt_cols = []
                                for _slug, _mname in zip(_per_slugs, _per_chosen_metrics):
                                    _vcol = _slug if _slug in _stats_df.columns else _findc(_stats_df, [f"total_{_slug}", f"sum_{_slug}"])
                                    if _vcol:
                                        _stats_df = _stats_df.rename(columns={_vcol: _mname})
                                        _melt_cols.append(_mname)

                                if _melt_cols:
                                    _long = _stats_df[[_pname_col] + _melt_cols].melt(
                                        id_vars=[_pname_col],
                                        value_vars=_melt_cols,
                                        var_name="Metric",
                                        value_name="Value",
                                    )
                                    _long["Value"] = pd.to_numeric(_long["Value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
                                    _long = _long.dropna(subset=["Value"])
                                    _long = _long.rename(columns={_pname_col: "Period"})

                                    _cmp = (
                                        alt.Chart(_long)
                                        .mark_bar()
                                        .encode(
                                            x=alt.X("Period:N", sort=None),
                                            y=alt.Y("Value:Q"),
                                            color=alt.Color("Metric:N"),
                                            xOffset="Metric:N",
                                            tooltip=["Period", "Metric", alt.Tooltip("Value:Q", format=",.2f")],
                                        )
                                        .properties(height=400)
                                    )
                                    st.altair_chart(_cmp, width='stretch')
                                    st.dataframe(_stats_df, width='stretch', hide_index=True)
                                    df_download_button(_stats_df, f"periods_compare_{_chosen_act_id}.csv")
    except Exception as _per_err:
        st.error(f"Periods tab crashed: {type(_per_err).__name__}: {_per_err}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())

    # ===========================================================================
    # SEASON-WIDE PERIOD BREAKDOWN
    # ===========================================================================
    st.markdown("---")
    st.markdown("## 📅 Season-wide period breakdown")
    st.caption(
        "Aggregate period stats across many activities — pick a team or specific players "
        "and see totals/averages by period (e.g. Q1/Q2/Q3/Q4) across the whole year."
    )

    try:
        # --- Date scope toggle --------------------------------------------
        _season_scope = st.radio(
            "Date scope",
            ["Calendar year", "Custom range", "Last 365 days"],
            horizontal=True,
            key="season_scope",
        )

        if _season_scope == "Calendar year":
            _year_options = list(range(date.today().year, date.today().year - 5, -1))
            _chosen_year = st.selectbox("Year", _year_options, key="season_year")
            _season_start = date(_chosen_year, 1, 1)
            _season_end = date(_chosen_year, 12, 31)
        elif _season_scope == "Last 365 days":
            _season_end = date.today()
            _season_start = _season_end - timedelta(days=365)
            st.caption(f"From **{_season_start}** to **{_season_end}**")
        else:
            _sc1, _sc2 = st.columns(2)
            with _sc1:
                _season_start = st.date_input(
                    "Start date",
                    value=date.today() - timedelta(days=365),
                    key="season_custom_start",
                )
            with _sc2:
                _season_end = st.date_input(
                    "End date",
                    value=date.today(),
                    key="season_custom_end",
                )

        # --- View: team vs individual -------------------------------------
        _season_view = st.radio(
            "View",
            ["By team", "By player(s)"],
            horizontal=True,
            key="season_view",
        )

        # --- Load teams (reuse cache from ACWR if present) ----------------
        if "acwr_teams_data" in st.session_state:
            _season_teams_map = st.session_state["acwr_teams_data"]
        else:
            # Build it
            _season_teams_map = {}
            try:
                _stm_teams = client.list_teams() or []
            except Exception:
                _stm_teams = []
            if _stm_teams:
                _prog = st.progress(0.0, text="Loading teams...")
                for _i, _t in enumerate(_stm_teams):
                    _tid = str(_t.get("id"))
                    _tname = _t.get("name") or f"Team {_tid[:8]}"
                    try:
                        _ta = client.team_athletes(_tid) or []
                    except Exception:
                        _ta = []
                    for _a in _ta:
                        _full = (f"{_a.get('first_name', '')} {_a.get('last_name', '')}".strip()
                                 or _a.get("name") or str(_a.get("id", ""))[:8])
                        _season_teams_map.setdefault(_tname, []).append({
                            "id": str(_a.get("id")),
                            "name": _full,
                        })
                    _prog.progress((_i + 1) / max(len(_stm_teams), 1))
                _prog.empty()
            st.session_state["acwr_teams_data"] = _season_teams_map

        if not _season_teams_map:
            st.warning("No teams available.")
        else:
            _season_team_names = sorted(_season_teams_map.keys())
            _chosen_season_team = st.selectbox(
                "Team",
                _season_team_names,
                key="season_team",
            )

            _season_team_athletes = _season_teams_map.get(_chosen_season_team, [])

            # Player picker only shown for "By player(s)" view
            _chosen_player_names: list[str] = []
            if _season_view == "By player(s)":
                _player_options = sorted({a["name"] for a in _season_team_athletes})
                _chosen_player_names = st.multiselect(
                    "Players",
                    _player_options,
                    default=_player_options[:1] if _player_options else [],
                    key="season_players",
                )

            # --- Metrics picker (reuse cached params) -----------------------
            if "periods_params" not in st.session_state:
                try:
                    st.session_state["periods_params"] = client.list_parameters() or []
                except Exception:
                    st.session_state["periods_params"] = []
            _sn_params = st.session_state["periods_params"]
            _sn_name_to_slug = {p["name"]: p["slug"] for p in _sn_params if p.get("name") and p.get("slug")}
            _sn_metric_names = sorted(_sn_name_to_slug.keys())

            # Default to player load + distance + HSR
            _sn_defaults = []
            _sl = [n.lower() for n in _sn_metric_names]
            for _pref in ["total player load", "total distance", "high speed running distance"]:
                for _i, _n in enumerate(_sl):
                    if _pref in _n and _sn_metric_names[_i] not in _sn_defaults:
                        _sn_defaults.append(_sn_metric_names[_i])
                        break
                if len(_sn_defaults) >= 3:
                    break

            _sn_chosen_metrics = st.multiselect(
                "Metrics",
                _sn_metric_names,
                default=_sn_defaults or _sn_metric_names[:3],
                key="season_metrics",
            )

            # --- Run button -------------------------------------------------
            if _sn_chosen_metrics and st.button("▶ Run season analysis", type="primary", key="season_run"):
                _sn_slugs = [_sn_name_to_slug[n] for n in _sn_chosen_metrics]

                # Get all activities in the date range
                try:
                    with st.spinner(f"Loading activities from {_season_start} to {_season_end}..."):
                        _sn_acts = client.list_activities(start=_season_start, end=_season_end) or []
                except Exception as _e:
                    st.error(f"Couldn't load activities: {_e}")
                    _sn_acts = []

                if not _sn_acts:
                    st.warning(f"No activities found between {_season_start} and {_season_end}.")
                else:
                    _sn_act_ids = [str(a["id"]) for a in _sn_acts if a.get("id")]
                    st.caption(f"Found **{len(_sn_act_ids)}** activities in the window.")

                    # Build group_by based on view
                    if _season_view == "By team":
                        _sn_group_by = ["period", "team"]
                    else:
                        _sn_group_by = ["period", "athlete"]

                    try:
                        # Batch in chunks — one big /stats call across 233 activities
                        # × 3 metrics × group_by athlete+period times out. 25 per chunk
                        # keeps each call well under the timeout.
                        _CHUNK = 25
                        _sn_rows = []
                        _chunks = [_sn_act_ids[i:i + _CHUNK] for i in range(0, len(_sn_act_ids), _CHUNK)]
                        _bar = st.progress(0.0, text=f"Querying /stats in {len(_chunks)} batches...")
                        for _i, _chunk in enumerate(_chunks):
                            try:
                                _batch_rows = client.stats(
                                    parameters=_sn_slugs,
                                    group_by=_sn_group_by,
                                    activity_ids=_chunk,
                                ) or []
                                _sn_rows.extend(_batch_rows)
                            except requests.HTTPError as e:
                                st.warning(f"Batch {_i+1} failed: HTTP {e.response.status_code}")
                            except requests.exceptions.Timeout:
                                st.warning(f"Batch {_i+1} timed out — try a shorter date range.")
                            except Exception as e:
                                st.warning(f"Batch {_i+1} error: {e}")
                            _bar.progress((_i + 1) / len(_chunks),
                                          text=f"Batch {_i+1}/{len(_chunks)} done ({len(_sn_rows)} rows)")
                        _bar.empty()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        _sn_rows = []

                    if not _sn_rows:
                        st.warning("No stats returned.")
                    else:
                        _sn_df = pd.json_normalize(_sn_rows)

                        with st.expander("🔍 Debug: columns", expanded=False):
                            st.write(f"**Columns:** `{list(_sn_df.columns)}`")
                            st.dataframe(_sn_df.head(), width='stretch')

                        def _sn_findc(df, cands):
                            return next((c for c in cands if c in df.columns), None)

                        _pname_col = _sn_findc(_sn_df, ["period_name", "name", "period"])
                        _team_col = _sn_findc(_sn_df, ["team_name", "team"])
                        _ath_col = _sn_findc(_sn_df, ["athlete_name", "athlete"])

                        if not _pname_col:
                            st.warning("Couldn't find period column.")
                            st.dataframe(_sn_df, width='stretch')
                        else:
                            # Filter to chosen team
                            _filtered = _sn_df.copy()
                            if _team_col and _season_view == "By team":
                                _filtered = _filtered[
                                    _filtered[_team_col].astype(str).str.lower() == _chosen_season_team.lower()
                                ]
                            elif _ath_col and _season_view == "By player(s)" and _chosen_player_names:
                                _filtered = _filtered[
                                    _filtered[_ath_col].isin(_chosen_player_names)
                                ]

                            if _filtered.empty:
                                st.warning(
                                    f"No rows for **{_chosen_season_team}**" +
                                    (f" / {', '.join(_chosen_player_names)}" if _chosen_player_names else "") +
                                    " in this window."
                                )
                            else:
                                # Process per metric: aggregate to totals + averages by period
                                _group_keys = [_pname_col]
                                if _season_view == "By player(s)" and _ath_col:
                                    _group_keys = [_ath_col, _pname_col]

                                # Build a results table: one row per period (or player×period)
                                # with totals and averages for each metric
                                _result_rows = []

                                for _key_vals, _grp in _filtered.groupby(_group_keys):
                                    if not isinstance(_key_vals, tuple):
                                        _key_vals = (_key_vals,)
                                    _row = {}
                                    for _kn, _kv in zip(_group_keys, _key_vals):
                                        _row[_kn] = _kv
                                    _row["Sessions"] = len(_grp)

                                    for _slug, _mname in zip(_sn_slugs, _sn_chosen_metrics):
                                        _vcol = _slug if _slug in _grp.columns else _sn_findc(_grp, [f"total_{_slug}", f"sum_{_slug}"])
                                        if _vcol:
                                            _vals = pd.to_numeric(_grp[_vcol], errors="coerce")
                                            _vals = _vals.replace([np.inf, -np.inf], np.nan).dropna()
                                            _row[f"{_mname} (Total)"] = float(_vals.sum()) if len(_vals) else 0.0
                                            _row[f"{_mname} (Avg)"] = float(_vals.mean()) if len(_vals) else 0.0
                                    _result_rows.append(_row)

                                _result_df = pd.DataFrame(_result_rows)

                                # Pretty rename
                                _result_df = _result_df.rename(columns={
                                    _pname_col: "Period",
                                    _ath_col: "Player" if _ath_col else "Player",
                                } if _ath_col else {_pname_col: "Period"})

                                if _season_view == "By team":
                                    st.markdown(f"### {_chosen_season_team} — totals & averages by period")
                                else:
                                    st.markdown(f"### {_chosen_season_team} players — totals & averages by period")

                                # Format numeric columns
                                _display_df = _result_df.copy()
                                for _c in _display_df.columns:
                                    if _c not in ["Period", "Player", "Sessions"] and pd.api.types.is_numeric_dtype(_display_df[_c]):
                                        _display_df[_c] = _display_df[_c].apply(
                                            lambda v: f"{v:,.1f}" if pd.notna(v) else "—"
                                        )

                                st.dataframe(_display_df, width='stretch', hide_index=True)
                                df_download_button(_result_df, f"season_periods_{_chosen_season_team}.csv")

                                # --- Charts: one per metric -----------------------
                                for _slug, _mname in zip(_sn_slugs, _sn_chosen_metrics):
                                    _tcol = f"{_mname} (Total)"
                                    _acol = f"{_mname} (Avg)"
                                    if _tcol not in _result_df.columns:
                                        continue

                                    st.markdown(f"**{_mname} by period**")

                                    if _season_view == "By team":
                                        # Two side-by-side mini-charts: totals + averages
                                        _melt = _result_df[["Period", _tcol, _acol]].melt(
                                            id_vars=["Period"],
                                            value_vars=[_tcol, _acol],
                                            var_name="Stat",
                                            value_name="Value",
                                        )
                                        _melt["Value"] = pd.to_numeric(_melt["Value"], errors="coerce")
                                        _melt = _melt.dropna(subset=["Value"])

                                        _ch = (
                                            alt.Chart(_melt)
                                            .mark_bar()
                                            .encode(
                                                x=alt.X("Period:N", sort=None),
                                                y=alt.Y("Value:Q"),
                                                color=alt.Color("Stat:N", scale=alt.Scale(range=["#ff4b1f", "#58a6ff"])),
                                                xOffset="Stat:N",
                                                tooltip=["Period", "Stat", alt.Tooltip("Value:Q", format=",.2f")],
                                            )
                                            .properties(height=300)
                                        )
                                        st.altair_chart(_ch, width='stretch')
                                    else:
                                        # Player view: grouped bars
                                        _player_col = "Player" if "Player" in _result_df.columns else _ath_col
                                        _melt = _result_df[[_player_col, "Period", _tcol]].copy()
                                        _melt[_tcol] = pd.to_numeric(_melt[_tcol], errors="coerce")
                                        _melt = _melt.dropna(subset=[_tcol])

                                        _ch = (
                                            alt.Chart(_melt)
                                            .mark_bar()
                                            .encode(
                                                x=alt.X("Period:N", sort=None),
                                                y=alt.Y(f"{_tcol}:Q", title=f"{_mname} (Total)"),
                                                color=alt.Color(f"{_player_col}:N"),
                                                xOffset=f"{_player_col}:N",
                                                tooltip=[_player_col, "Period", alt.Tooltip(f"{_tcol}:Q", format=",.2f")],
                                            )
                                            .properties(height=320)
                                        )
                                        st.altair_chart(_ch, width='stretch')
    except Exception as _season_err:
        st.error(f"Season section crashed: {type(_season_err).__name__}: {_season_err}")
        with st.expander("Traceback"):
            import traceback as _tb
            st.code(_tb.format_exc())


# --- Rankings tab ----------------------------------------------------------
with tab_rankings:
    st.subheader("Rankings")
    st.caption("Top teams, top teams by position, and top individuals — powered by the /stats endpoint.")

    # --- KPI cards at top ---------------------------------------------------
    @st.cache_data(show_spinner=False, ttl=300)
    def _kpi_summary(base: str, _tok: str):
        """Quick counts for the KPI strip."""
        try:
            athletes = client.list_athletes() or []
        except Exception:
            athletes = []
        try:
            # Last 30 days of activities
            acts = client.list_activities(
                start=date.today() - timedelta(days=30),
                end=date.today(),
            ) or []
        except Exception:
            acts = []
        try:
            params = client.list_parameters() or []
        except Exception:
            params = []

        # Count distinct teams from athletes
        teams = set()
        for a in athletes:
            t = a.get("team_name") or (a.get("team") or {}).get("name") or a.get("team")
            if t:
                teams.add(str(t))

        return {
            "athletes": len(athletes),
            "activities_30d": len(acts),
            "teams": len(teams),
            "parameters": len(params),
        }

    summary = _kpi_summary(base_url, token)
    kpi_cards([
        {"label": "Athletes", "value": f"{summary['athletes']:,}", "style": "accent"},
        {"label": "Teams", "value": f"{summary['teams']:,}"},
        {"label": "Activities (30d)", "value": f"{summary['activities_30d']:,}"},
        {"label": "Available Metrics", "value": f"{summary['parameters']:,}", "style": "success"},
    ])

    # Cache parameter list (slugs) — it doesn't change often
    @st.cache_data(show_spinner=False)
    def _cached_parameters(base: str, _tok: str):
        return client.list_parameters() or []

    if "rank_params_loaded" not in st.session_state:
        st.session_state["rank_params_loaded"] = False

    if st.button("Load available metrics", key="rank_load_params"):
        try:
            with st.spinner("Loading parameter definitions..."):
                params_def = _cached_parameters(base_url, token)
            st.session_state["rank_params_def"] = params_def
            st.session_state["rank_params_loaded"] = True
            st.success(f"Loaded {len(params_def)} parameter definitions")
        except Exception as e:
            st.error(f"Error: {e}")

    params_def = st.session_state.get("rank_params_def", [])

    if not params_def:
        st.info("Click **Load available metrics** to fetch the list of parameters from your account.")
        st.stop()

    # Build metric picker — show name, send slug
    slug_to_name = {p["slug"]: p["name"] for p in params_def if p.get("slug") and p.get("name")}
    name_to_slug = {v: k for k, v in slug_to_name.items()}
    metric_names = sorted(slug_to_name.values())

    # Default to player load if available
    default_idx = 0
    for i, n in enumerate(metric_names):
        if "player load" in n.lower() and "total" in n.lower():
            default_idx = i
            break

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        chosen_metric = st.selectbox("Metric", metric_names, index=default_idx, key="rank_metric")
    with col_b:
        top_n = st.number_input("Top N", min_value=3, max_value=50, value=10, step=1)
    with col_c:
        chart_orient = st.radio(
            "Chart",
            ["Horizontal", "Vertical"],
            key="rank_chart_orient",
        )

    metric_slug = name_to_slug[chosen_metric]

    # Scope ------------------------------------------------------------------
    scope = st.radio(
        "Scope",
        ["Single activity", "Date range (all activities)"],
        horizontal=True,
        key="rank_scope",
    )

    activity_ids: list[str] = []
    start_d: date | None = None
    end_d: date | None = None

    if scope == "Single activity":
        act_id_input = st.text_input("Activity ID", key="rank_activity_id")
        if act_id_input:
            activity_ids = [act_id_input.strip()]
    else:
        c1, c2 = st.columns(2)
        with c1:
            start_d = st.date_input("Start date", value=date.today() - timedelta(days=14), key="rank_start")
        with c2:
            end_d = st.date_input("End date", value=date.today(), key="rank_end")

    can_query = bool(activity_ids) or (start_d and end_d)
    if not can_query:
        st.info("Pick an activity ID or a date range above.")
        st.stop()

    # --- Optional: period filter (only when a single activity is selected) ---
    period_filter_ids: list[str] = []
    if activity_ids and len(activity_ids) == 1:
        try:
            _rank_periods = client.activity_periods(activity_ids[0]) or []
        except Exception:
            _rank_periods = []
        if _rank_periods:
            _per_options = {}
            for _p in _rank_periods:
                _pn = _p.get("name") or _p.get("period_name") or f"Period {str(_p.get('id', ''))[:8]}"
                _pid = str(_p.get("id", ""))
                if _pid:
                    _per_options[_pn] = _pid
            if _per_options:
                _chosen_periods = st.multiselect(
                    "Filter by period (optional)",
                    list(_per_options.keys()),
                    default=[],
                    key="rank_period_filter",
                    help="Leave empty for whole activity. Pick one or more periods to restrict to (e.g. Q4 only).",
                )
                period_filter_ids = [_per_options[n] for n in _chosen_periods]

    # --- The three views ----------------------------------------------------
    view_team_total, view_team_avg, view_top_ind = st.tabs(
        ["Team totals", "Team averages", "Top individuals (by team / position)"]
    )

    def run_stats(group_by: list[str]) -> pd.DataFrame:
        """Call /stats and return a DataFrame."""
        rows = client.stats(
            parameters=[metric_slug],
            group_by=group_by,
            activity_ids=activity_ids or None,
            period_ids=period_filter_ids or None,
            start=start_d if not activity_ids else None,
            end=end_d if not activity_ids else None,
        )
        return to_df(rows)

    def find_value_col(df: pd.DataFrame) -> str | None:
        """Find the column that holds the metric value (slug-based or stat-prefixed)."""
        if df.empty:
            return None
        candidates = [
            metric_slug,
            f"total_{metric_slug}",
            f"average_{metric_slug}",
            f"sum_{metric_slug}",
            "value",
            "total",
        ]
        for c in candidates:
            if c in df.columns:
                return c
        # Fall back to first numeric column
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        return None

    # --- Team totals ---------------------------------------------------------
    with view_team_total:
        st.markdown(f"**Top teams by {chosen_metric}** (totals)")
        if st.button("Run", key="run_team_total"):
            try:
                with st.spinner("Querying /stats grouped by team..."):
                    df = run_stats(["team"])
                if df.empty:
                    st.warning("No data returned.")
                else:
                    val_col = find_value_col(df)
                    name_col = next((c for c in ["team_name", "name", "team"] if c in df.columns), None)
                    if val_col and name_col:
                        df = df.head(top_n) if False else df  # keep all for ranking, slice after
                        ranked = add_rank(df[[name_col, val_col]], val_col).head(top_n)
                        st.dataframe(ranked, width='stretch', hide_index=True)
                        st.altair_chart(
                            ranked_bar_chart(ranked, name_col, val_col, chart_orient,
                                             title=f"Top {top_n} teams — {chosen_metric}"),
                            width='stretch',
                        )
                        df_download_button(ranked, f"team_totals_{metric_slug}.csv")
                    else:
                        st.warning("Couldn't identify value/name columns — showing raw response below.")
                        st.dataframe(df, width='stretch')
            except requests.HTTPError as e:
                st.error(f"HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Team averages -------------------------------------------------------
    with view_team_avg:
        st.markdown(f"**Top teams by {chosen_metric}** (average per athlete)")
        if st.button("Run", key="run_team_avg"):
            try:
                with st.spinner("Querying /stats grouped by team + athlete..."):
                    df = run_stats(["team", "athlete"])
                if df.empty:
                    st.warning("No data returned.")
                else:
                    val_col = find_value_col(df)
                    team_col = next((c for c in ["team_name", "team"] if c in df.columns), None)
                    if val_col and team_col:
                        avg_col = f"avg_{metric_slug}"
                        agg = (
                            df.groupby(team_col, dropna=False)[val_col]
                            .mean()
                            .reset_index()
                            .rename(columns={val_col: avg_col})
                        )
                        ranked = add_rank(agg, avg_col).head(top_n)
                        st.dataframe(ranked, width='stretch', hide_index=True)
                        st.altair_chart(
                            ranked_bar_chart(ranked, team_col, avg_col, chart_orient,
                                             title=f"Top {top_n} teams (avg) — {chosen_metric}"),
                            width='stretch',
                        )
                        df_download_button(ranked, f"team_averages_{metric_slug}.csv")
                    else:
                        st.warning("Couldn't identify value/team columns — showing raw response below.")
                        st.dataframe(df, width='stretch')
            except requests.HTTPError as e:
                st.error(f"HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Top individuals by team / position ---------------------------------
    with view_top_ind:
        st.markdown(f"**Top individuals by {chosen_metric}**")
        group_choice = st.radio(
            "Group by",
            ["No grouping", "Team", "Position", "Team + Position"],
            horizontal=True,
            key="rank_groupby",
        )
        if st.button("Run", key="run_top_ind"):
            try:
                group_by_req = ["athlete", "team", "position"]
                with st.spinner("Querying /stats grouped by athlete/team/position..."):
                    df = run_stats(group_by_req)
                if df.empty:
                    st.warning("No data returned.")
                else:
                    val_col = find_value_col(df)
                    ath_col = next((c for c in ["athlete_name", "name", "athlete"] if c in df.columns), None)
                    team_col = next((c for c in ["team_name", "team"] if c in df.columns), None)
                    pos_col = next((c for c in ["position_name", "position"] if c in df.columns), None)

                    if not val_col or not ath_col:
                        st.warning("Couldn't find athlete/value columns — showing raw response.")
                        st.dataframe(df, width='stretch')
                    else:
                        df = df.sort_values(val_col, ascending=False)

                        # Group-wise top N, then rank within group
                        if group_choice == "Team" and team_col:
                            ranked = (
                                df.groupby(team_col, dropna=False, group_keys=False)
                                .head(top_n)
                                .copy()
                            )
                            ranked["Rank"] = (
                                ranked.groupby(team_col, dropna=False).cumcount() + 1
                            )
                        elif group_choice == "Position" and pos_col:
                            ranked = (
                                df.groupby(pos_col, dropna=False, group_keys=False)
                                .head(top_n)
                                .copy()
                            )
                            ranked["Rank"] = (
                                ranked.groupby(pos_col, dropna=False).cumcount() + 1
                            )
                        elif group_choice == "Team + Position" and team_col and pos_col:
                            ranked = (
                                df.groupby([team_col, pos_col], dropna=False, group_keys=False)
                                .head(top_n)
                                .copy()
                            )
                            ranked["Rank"] = (
                                ranked.groupby([team_col, pos_col], dropna=False).cumcount() + 1
                            )
                        else:
                            ranked = df.head(top_n).copy()
                            ranked.insert(0, "Rank", range(1, len(ranked) + 1))

                        # Move Rank to front if not already there
                        cols = ["Rank"] + [c for c in ranked.columns if c != "Rank"]
                        ranked = ranked[cols]

                        st.dataframe(ranked, width='stretch', hide_index=True)

                        # Chart: top N overall by value, with athlete name label
                        # Only pass essential columns to avoid Vega-Lite warnings
                        # from unrelated all-null columns in the response.
                        chart_df = df.head(top_n).copy()
                        label_col = ath_col
                        if group_choice == "Team" and team_col:
                            chart_df["_label"] = chart_df[ath_col].astype(str) + " (" + chart_df[team_col].astype(str) + ")"
                            label_col = "_label"
                        elif group_choice == "Position" and pos_col:
                            chart_df["_label"] = chart_df[ath_col].astype(str) + " (" + chart_df[pos_col].astype(str) + ")"
                            label_col = "_label"
                        elif group_choice == "Team + Position" and team_col and pos_col:
                            chart_df["_label"] = (
                                chart_df[ath_col].astype(str)
                                + " (" + chart_df[team_col].astype(str)
                                + " — " + chart_df[pos_col].astype(str) + ")"
                            )
                            label_col = "_label"

                        # Reduce to only the columns needed for the chart
                        chart_df = chart_df[[c for c in [label_col, val_col] if c in chart_df.columns]].copy()

                        st.altair_chart(
                            ranked_bar_chart(chart_df, label_col, val_col, chart_orient,
                                             title=f"Top {top_n} individuals — {chosen_metric}"),
                            width='stretch',
                        )
                        df_download_button(ranked, f"top_individuals_{metric_slug}.csv")
            except requests.HTTPError as e:
                st.error(f"HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"Error: {e}")


# --- Compare tab -----------------------------------------------------------
with tab_compare:
    st.subheader("Compare Athletes")
    st.caption("Pick any number of athletes and any metrics to compare them head-to-head.")

    # --- Load athletes & parameters -----------------------------------------
    @st.cache_data(show_spinner=False, ttl=600)
    def _cmp_athletes(base: str, _tok: str):
        return client.list_athletes() or []

    @st.cache_data(show_spinner=False, ttl=600)
    def _cmp_parameters(base: str, _tok: str):
        return client.list_parameters() or []

    try:
        all_athletes = _cmp_athletes(base_url, token)
        all_params = _cmp_parameters(base_url, token)
    except Exception as e:
        st.error(f"Error loading athletes/parameters: {e}")
        st.stop()

    if not all_athletes:
        st.warning("No athletes found in your account.")
        st.stop()

    # Build name->id and slug->name maps
    def _athlete_display(a: dict) -> str:
        name = (
            f"{a.get('first_name', '')} {a.get('last_name', '')}".strip()
            or a.get("name")
            or str(a.get("id", ""))[:8]
        )
        team = a.get("team_name") or (a.get("team") or {}).get("name") or a.get("team")
        return f"{name}" + (f"  ·  {team}" if team else "")

    athlete_options = {_athlete_display(a): str(a.get("id")) for a in all_athletes if a.get("id")}
    name_to_slug = {p["name"]: p["slug"] for p in all_params if p.get("name") and p.get("slug")}
    slug_to_name = {v: k for k, v in name_to_slug.items()}
    metric_names = sorted(name_to_slug.keys())

    # Default metric picks (look for common ones)
    default_metrics: list[str] = []
    common_terms = ["total player load", "total distance", "high speed", "max velocity", "sprint distance"]
    for term in common_terms:
        for n in metric_names:
            if term in n.lower():
                default_metrics.append(n)
                break
        if len(default_metrics) >= 4:
            break

    # --- Selection controls -------------------------------------------------
    selected_athletes = st.multiselect(
        "Athletes to compare",
        options=list(athlete_options.keys()),
        key="cmp_athletes",
        help="Type to filter.",
    )

    selected_metrics = st.multiselect(
        "Metrics",
        options=metric_names,
        default=default_metrics,
        key="cmp_metrics",
    )

    cmp_scope = st.radio(
        "Scope",
        ["Single activity", "Date range"],
        horizontal=True,
        key="cmp_scope",
    )

    cmp_activity_ids: list[str] | None = None
    cmp_start: date | None = None
    cmp_end: date | None = None

    if cmp_scope == "Single activity":
        act_id = st.text_input("Activity ID", key="cmp_activity_id")
        if act_id.strip():
            cmp_activity_ids = [act_id.strip()]
    else:
        c1, c2 = st.columns(2)
        with c1:
            cmp_start = st.date_input("Start", value=date.today() - timedelta(days=14), key="cmp_start")
        with c2:
            cmp_end = st.date_input("End", value=date.today(), key="cmp_end")

    view_mode = st.radio(
        "View as",
        ["Side-by-side bars", "Radar chart", "Over time (line)", "Table"],
        horizontal=True,
        key="cmp_view",
    )

    can_run = (
        len(selected_athletes) >= 1
        and len(selected_metrics) >= 1
        and (cmp_activity_ids or (cmp_start and cmp_end))
    )

    if not can_run:
        st.info("Pick at least one athlete and one metric, then choose a scope.")
        st.stop()

    if not st.button("Compare", type="primary", key="cmp_run"):
        st.stop()

    # --- Fetch data ---------------------------------------------------------
    selected_athlete_ids = [athlete_options[n] for n in selected_athletes]
    selected_slugs = [name_to_slug[n] for n in selected_metrics]

    # For "Over time" we need group_by activity, otherwise group_by athlete is enough
    needs_time = view_mode == "Over time (line)"
    group_by = ["athlete", "activity"] if needs_time else ["athlete"]

    try:
        with st.spinner(f"Querying /stats for {len(selected_slugs)} metric(s)..."):
            # Single /stats call with all metrics, then filter to selected athletes
            rows = client.stats(
                parameters=selected_slugs,
                group_by=group_by,
                activity_ids=cmp_activity_ids,
                start=cmp_start if not cmp_activity_ids else None,
                end=cmp_end if not cmp_activity_ids else None,
            )
        raw_df = to_df(rows)
    except requests.HTTPError as e:
        st.error(f"HTTP {e.response.status_code}: {e.response.text}")
        st.stop()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    if raw_df.empty:
        st.warning("No data returned.")
        st.stop()

    # Identify columns
    def _col(df, candidates):
        return next((c for c in candidates if c in df.columns), None)

    ath_id_col = _col(raw_df, ["athlete_id", "athlete"])
    ath_name_col = _col(raw_df, ["athlete_name", "name"])
    act_id_col = _col(raw_df, ["activity_id"])
    act_name_col = _col(raw_df, ["activity_name"])
    act_time_col = _col(raw_df, ["start_time", "activity_start_time", "activity_start"])

    if not ath_id_col:
        st.warning("Couldn't find athlete identifier in response. Showing raw data:")
        st.dataframe(raw_df, width='stretch')
        st.stop()

    # Filter to selected athletes
    df = raw_df[raw_df[ath_id_col].astype(str).isin(selected_athlete_ids)].copy()
    if df.empty:
        st.warning("No rows matched the selected athletes — they may not have data in this scope.")
        st.dataframe(raw_df, width='stretch')
        st.stop()

    # Resolve athlete display name fallback
    if not ath_name_col:
        id_to_name = {athlete_options[n]: n.split("  ·  ")[0] for n in selected_athletes}
        df["__athlete_name"] = df[ath_id_col].astype(str).map(id_to_name)
        ath_name_col = "__athlete_name"

    # --- Render each view ---------------------------------------------------
    if view_mode == "Table":
        st.dataframe(df, width='stretch', hide_index=True)
        df_download_button(df, "compare.csv")

    elif view_mode == "Side-by-side bars":
        # Melt so each metric becomes a row
        present_slugs = [s for s in selected_slugs if s in df.columns]
        if not present_slugs:
            st.warning("None of the selected metrics appear as columns in the response.")
            st.dataframe(df, width='stretch')
            st.stop()

        # Ensure numeric and clean infinities BEFORE aggregating
        import numpy as np
        for s in present_slugs:
            df[s] = pd.to_numeric(df[s], errors="coerce").replace([np.inf, -np.inf], np.nan)

        # Aggregate (sum) per athlete per metric across rows
        agg = df.groupby(ath_name_col, dropna=False)[present_slugs].sum(min_count=1).reset_index()
        long_df = agg.melt(
            id_vars=[ath_name_col],
            value_vars=present_slugs,
            var_name="metric",
            value_name="value",
        )
        long_df["metric"] = long_df["metric"].map(lambda s: slug_to_name.get(s, s))
        # CRITICAL: drop NaN value rows before passing to Altair (avoids Infinite extent warnings)
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        long_df = long_df.dropna(subset=["value"])

        if long_df.empty:
            st.warning("No numeric data to chart.")
        else:
            chart = (
                alt.Chart(long_df)
                .mark_bar()
                .encode(
                    x=alt.X("metric:N", title="Metric", sort=selected_metrics),
                    y=alt.Y("value:Q", title="Value"),
                    color=alt.Color(f"{ath_name_col}:N", title="Athlete"),
                    xOffset=f"{ath_name_col}:N",
                    tooltip=[ath_name_col, "metric", alt.Tooltip("value:Q", format=",.2f")],
                )
                .properties(height=420)
            )
            st.altair_chart(chart, width='stretch')
        st.dataframe(agg, width='stretch', hide_index=True)
        df_download_button(agg, "compare_bars.csv")

    elif view_mode == "Radar chart":
        present_slugs = [s for s in selected_slugs if s in df.columns]
        if len(present_slugs) < 3:
            st.info("Radar charts work best with 3+ metrics. Add more metrics for a fuller shape.")
        if not present_slugs:
            st.warning("None of the selected metrics appear as columns.")
            st.stop()

        # Ensure numeric and clean before aggregating
        import numpy as np
        for s in present_slugs:
            df[s] = pd.to_numeric(df[s], errors="coerce").replace([np.inf, -np.inf], np.nan)

        agg = df.groupby(ath_name_col, dropna=False)[present_slugs].sum(min_count=1).reset_index()

        # Normalize each metric to 0-100 so different scales fit on one radar
        norm = agg.copy()
        for s in present_slugs:
            col_max = norm[s].max()
            if pd.notna(col_max) and col_max > 0:
                norm[s] = norm[s] / col_max * 100
            else:
                norm[s] = 0
            # Fill any remaining NaN with 0 so radar geometry stays valid
            norm[s] = norm[s].fillna(0).replace([np.inf, -np.inf], 0)

        long_df = norm.melt(
            id_vars=[ath_name_col],
            value_vars=present_slugs,
            var_name="metric_slug",
            value_name="value",
        )
        long_df["metric"] = long_df["metric_slug"].map(lambda s: slug_to_name.get(s, s))
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce").fillna(0)

        # Polar-ish via Altair: use a radial line chart approximation.
        # Altair doesn't have native radar, so we use line-by-angle.
        n_metrics = len(present_slugs)
        long_df["angle"] = long_df["metric_slug"].map(
            {s: (i / n_metrics) * 360 for i, s in enumerate(present_slugs)}
        )

        # Convert polar to cartesian for plotting
        import math
        long_df["x"] = long_df.apply(
            lambda r: r["value"] * math.cos(math.radians(r["angle"] - 90)), axis=1
        )
        long_df["y"] = long_df.apply(
            lambda r: r["value"] * math.sin(math.radians(r["angle"] - 90)), axis=1
        )

        # Close the loop by appending the first point at the end for each athlete
        closed = []
        for ath, g in long_df.groupby(ath_name_col):
            g_sorted = g.sort_values("angle")
            first = g_sorted.iloc[0:1].copy()
            closed.append(pd.concat([g_sorted, first], ignore_index=True))
        long_df = pd.concat(closed, ignore_index=True)

        # Axis labels (one per metric)
        label_df = pd.DataFrame({
            "metric": [slug_to_name.get(s, s) for s in present_slugs],
            "angle": [(i / n_metrics) * 360 for i, _ in enumerate(present_slugs)],
        })
        label_df["x"] = label_df["angle"].map(lambda a: 115 * math.cos(math.radians(a - 90)))
        label_df["y"] = label_df["angle"].map(lambda a: 115 * math.sin(math.radians(a - 90)))

        lines = (
            alt.Chart(long_df)
            .mark_line(point=True, opacity=0.85)
            .encode(
                x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-130, 130])),
                y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-130, 130])),
                color=alt.Color(f"{ath_name_col}:N", title="Athlete"),
                order="angle:Q",
                tooltip=[ath_name_col, "metric", alt.Tooltip("value:Q", format=",.1f", title="Normalised (0-100)")],
            )
        )
        labels = (
            alt.Chart(label_df)
            .mark_text(fontSize=12, fontWeight=500)
            .encode(x="x:Q", y="y:Q", text="metric:N")
        )
        st.altair_chart((lines + labels).properties(height=480), width='stretch')
        st.caption("Values are normalised to 0–100 per metric so different scales fit on one chart.")
        st.dataframe(agg, width='stretch', hide_index=True)
        df_download_button(agg, "compare_radar.csv")

    elif view_mode == "Over time (line)":
        if cmp_activity_ids:
            st.info("'Over time' works best with a date range — only one activity selected.")
        present_slugs = [s for s in selected_slugs if s in df.columns]
        if not present_slugs:
            st.warning("None of the selected metrics appear as columns.")
            st.stop()

        # Need a time axis — prefer activity start_time, else activity_id
        time_col = act_time_col or act_id_col
        if not time_col:
            st.warning("No activity time/id column to plot against.")
            st.dataframe(df, width='stretch')
            st.stop()

        # Convert epoch seconds -> datetime if it looks numeric
        if act_time_col and pd.api.types.is_numeric_dtype(df[act_time_col]):
            df["__when"] = pd.to_datetime(df[act_time_col], unit="s", errors="coerce")
            time_col = "__when"

        # Let the user pick which metric to plot (one at a time keeps it readable)
        metric_to_plot = st.selectbox(
            "Metric to plot over time",
            [slug_to_name.get(s, s) for s in present_slugs],
            key="cmp_time_metric",
        )
        slug_to_plot = name_to_slug.get(metric_to_plot, metric_to_plot)

        plot_df = df[[ath_name_col, time_col, slug_to_plot]].copy()
        plot_df[slug_to_plot] = pd.to_numeric(plot_df[slug_to_plot], errors="coerce")
        plot_df = plot_df.dropna(subset=[slug_to_plot, time_col])
        plot_df = plot_df.sort_values(time_col)

        if plot_df.empty:
            st.warning(f"No numeric data for **{metric_to_plot}** in this scope.")
        else:
            try:
                chart = (
                    alt.Chart(plot_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(f"{time_col}:T" if time_col == "__when" else f"{time_col}:N", title="Activity"),
                        y=alt.Y(f"{slug_to_plot}:Q", title=metric_to_plot),
                        color=alt.Color(f"{ath_name_col}:N", title="Athlete"),
                        tooltip=[ath_name_col, time_col, alt.Tooltip(f"{slug_to_plot}:Q", format=",.2f")],
                    )
                    .properties(height=420)
                )
                st.altair_chart(chart, width='stretch')
            except Exception as e:
                st.warning(f"Couldn't render timeline chart: {e}")


# --- ACWR tab — pointer to the standalone app ----------------------------
with tab_acwr:
    st.subheader("📈 ACWR — Acute : Chronic Workload Ratio")
    st.info(
        "ACWR analysis runs as a separate Streamlit app to keep the main app stable.\n\n"
        "**To launch it:**\n"
        "1. Open a new terminal\n"
        "2. Navigate to this folder\n"
        "3. Run: `streamlit run acwr.py`\n\n"
        "It will open in a new browser tab on a different port (usually `localhost:8502`)."
    )
    st.caption(
        "Why separate? Streamlit's tab rerun behaviour was causing the ACWR processing "
        "to interfere with the rest of the app. Running it standalone avoids the issue entirely."
    )


# --- Ask AI tab -----------------------------------------------------------
with tab_ask:
    st.subheader("Ask AI")
    st.caption(
        "Ask questions in plain English, e.g. "
        "*“Show me the top 10 midfielders by total distance last week”* "
        "or *“Which team had the highest average player load in the last activity?”*"
    )

    if not anthropic_key:
        st.info("Add your Anthropic API key in the sidebar to enable this tab.")
        st.stop()

    # Lazy-load anthropic so the rest of the app works without it installed
    try:
        import anthropic  # noqa: F401
    except ImportError:
        st.error("The `anthropic` package isn't installed. Run: `pip install anthropic`")
        st.stop()

    # Make sure we have parameter definitions to ground the model
    if "rank_params_def" not in st.session_state or not st.session_state["rank_params_def"]:
        st.warning(
            "Go to the **Rankings** tab first and click *Load available metrics*. "
            "That gives the AI the list of valid parameter slugs."
        )
        st.stop()

    params_def = st.session_state["rank_params_def"]
    # Compact catalog: just name + slug (keeps prompt small)
    metric_catalog = [
        {"name": p.get("name"), "slug": p.get("slug")}
        for p in params_def
        if p.get("slug") and p.get("name")
    ]

    user_question = st.text_area(
        "Your question",
        placeholder="e.g. Top 10 forwards by total distance last 7 days",
        height=80,
        key="ai_question",
    )

    if st.button("Ask", type="primary", key="ai_ask") and user_question.strip():

        system_prompt = f"""You translate natural-language questions about Catapult athlete \
performance data into a structured JSON query plan for the /stats endpoint.

The available metric slugs (name → slug) are:
{chr(10).join(f"- {m['name']} → {m['slug']}" for m in metric_catalog)}

Return ONLY a JSON object (no prose, no markdown fences) with this shape:
{{
  "parameter_slug": "<one slug from the list above>",
  "group_by": ["athlete"]  // any of: "athlete", "team", "position", "activity"
  "filter": {{
    "position": "<optional position name to filter to, or null>",
    "team": "<optional team name to filter to, or null>"
  }},
  "scope": {{
    "type": "date_range" | "single_activity" | "all_recent",
    "days_back": <integer, only for date_range, e.g. 7, 14, 30>,
    "activity_id": "<only for single_activity>"
  }},
  "top_n": <integer, default 10>,
  "aggregation": "sum" | "mean" | "max",
  "explanation": "<one-sentence description of what you're computing>"
}}

If the question mentions a position (midfielder, forward, defender, ruck, etc.), \
include it in filter.position. If unsure about scope, default to date_range with days_back=14."""

        try:
            client_ai = anthropic.Anthropic(api_key=anthropic_key)
            with st.spinner("Asking Claude to plan the query..."):
                msg = client_ai.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_question}],
                )

            # Extract text
            raw_text = "".join(
                blk.text for blk in msg.content if getattr(blk, "type", None) == "text"
            ).strip()

            # Strip markdown fences if model added them
            import json, re
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.MULTILINE).strip()

            try:
                plan = json.loads(cleaned)
            except json.JSONDecodeError:
                st.error("Claude didn't return clean JSON. Raw output:")
                st.code(raw_text)
                st.stop()

            with st.expander("🤖 Claude's query plan", expanded=False):
                st.json(plan)

            # --- Execute the plan ----------------------------------------
            slug = plan.get("parameter_slug")
            if not slug or slug not in {m["slug"] for m in metric_catalog}:
                st.error(f"Claude picked an unknown metric slug: `{slug}`")
                st.stop()

            group_by = plan.get("group_by") or ["athlete"]
            scope = plan.get("scope") or {}
            top_n = int(plan.get("top_n") or 10)
            aggregation = plan.get("aggregation") or "sum"
            filt = plan.get("filter") or {}

            # Resolve scope -> activity_ids / start / end
            q_activity_ids: list[str] | None = None
            q_start: date | None = None
            q_end: date | None = None
            if scope.get("type") == "single_activity" and scope.get("activity_id"):
                q_activity_ids = [scope["activity_id"]]
            elif scope.get("type") == "date_range":
                days_back = int(scope.get("days_back") or 14)
                q_end = date.today()
                q_start = q_end - timedelta(days=days_back)
            else:
                # all_recent → default to last 30 days
                q_end = date.today()
                q_start = q_end - timedelta(days=30)

            with st.spinner(f"Querying /stats for `{slug}`..."):
                rows = client.stats(
                    parameters=[slug],
                    group_by=group_by,
                    activity_ids=q_activity_ids,
                    start=q_start,
                    end=q_end,
                )
            df = to_df(rows)

            if df.empty:
                st.warning("Query returned no rows. Try widening the date range.")
                st.stop()

            # Find columns
            def _find_col(df, candidates):
                return next((c for c in candidates if c in df.columns), None)

            val_col = (
                slug if slug in df.columns
                else _find_col(df, [f"total_{slug}", f"sum_{slug}", f"average_{slug}", "value"])
            )
            ath_col = _find_col(df, ["athlete_name", "athlete", "name"])
            team_col = _find_col(df, ["team_name", "team"])
            pos_col = _find_col(df, ["position_name", "position"])

            # Apply filters
            if filt.get("position") and pos_col:
                df = df[df[pos_col].astype(str).str.contains(filt["position"], case=False, na=False)]
            if filt.get("team") and team_col:
                df = df[df[team_col].astype(str).str.contains(filt["team"], case=False, na=False)]

            if df.empty:
                st.warning("Filter excluded all rows. Try removing the position/team filter.")
                st.stop()

            if not val_col:
                st.warning("Couldn't identify the value column in the response.")
                st.dataframe(df, width='stretch')
                st.stop()

            # Aggregate if needed
            group_cols = [c for c in [ath_col, team_col, pos_col] if c in (df.columns)]
            if aggregation == "mean":
                agg = df.groupby(group_cols, dropna=False)[val_col].mean().reset_index()
            elif aggregation == "max":
                agg = df.groupby(group_cols, dropna=False)[val_col].max().reset_index()
            else:
                agg = df.groupby(group_cols, dropna=False)[val_col].sum().reset_index()

            ranked = add_rank(agg, val_col).head(top_n)

            # Build a display label for the chart
            label_col = ath_col or team_col or pos_col
            if not label_col:
                label_col = ranked.columns[1]  # fallback

            st.markdown(f"### Top {len(ranked)} by {slug}")
            st.dataframe(ranked, width='stretch', hide_index=True)
            st.altair_chart(
                ranked_bar_chart(
                    ranked, label_col, val_col,
                    st.session_state.get("rank_chart_orient", "Horizontal"),
                    title=plan.get("explanation", ""),
                ),
                width='stretch',
            )

            # --- Have Claude summarize the result -----------------------
            top_rows_text = ranked.head(10).to_csv(index=False)
            with st.spinner("Summarising the result..."):
                summary = client_ai.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=512,
                    system="You are a concise sports analytics assistant. Given a user question and a small results table, answer the question in 2-4 sentences. Reference specific names and numbers from the data. Do not invent values.",
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Question: {user_question}\n\n"
                            f"Metric: {slug}\n"
                            f"Aggregation: {aggregation}\n"
                            f"Results (CSV):\n{top_rows_text}"
                        ),
                    }],
                )
            summary_text = "".join(
                blk.text for blk in summary.content if getattr(blk, "type", None) == "text"
            )
            st.markdown("### Answer")
            st.write(summary_text)

            df_download_button(ranked, f"ai_query_{slug}.csv")

        except anthropic.APIError as e:
            st.error(f"Anthropic API error: {e}")
        except requests.HTTPError as e:
            st.error(f"Catapult HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# --- Raw GET tab -----------------------------------------------------------
with tab_raw:
    st.subheader("Raw GET")
    st.caption("Hit any endpoint relative to the base URL, e.g. `/athletes` or `/activities/{id}/periods`.")
    path = st.text_input("Path", value="/athletes")
    if st.button("Send request", key="raw_go") and path:
        try:
            with st.spinner(f"GET {path}..."):
                data = client._get(path)
            st.json(data)
            df = to_df(data)
            if not df.empty:
                st.dataframe(df, width='stretch')
                df_download_button(df, "response.csv")
        except requests.HTTPError as e:
            st.error(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.error(f"Error: {e}")
