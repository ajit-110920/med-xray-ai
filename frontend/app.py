"""Chest X-ray Analysis - Clinical Research Workspace.

High-contrast, soft-dark healthcare AI application interface.
Connects to the FastAPI backend over HTTP; does not load model weights inside Streamlit.
"""

import os
from typing import Any, Dict, Optional
import altair as alt
import pandas as pd
from PIL import Image
import requests
import streamlit as st

# Backend Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("BACKEND_TIMEOUT", "60"))

# Page Configuration
st.set_page_config(
    page_title="Chest X-ray Analysis | Research Workspace",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Soft-Dark Healthcare Theme CSS
st.markdown(
    """
    <style>
    /* Reset & Prevent Any Horizontal Scrollbars */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stVerticalBlock"] {
        background-color: #141B26 !important;
        color: #FFFFFF !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        overflow-x: hidden !important;
        max-width: 100vw !important;
        box-sizing: border-box !important;
    }

    *, *::before, *::after {
        box-sizing: border-box !important;
    }

    [data-testid="stHeader"] {
        background-color: #141B26 !important;
    }

    /* Container Spacing */
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1A2332 !important;
        border-right: 1px solid #38455A !important;
        padding-top: 1rem !important;
    }

    /* Application Header */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.75rem;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid #38455A;
        margin-bottom: 1.25rem;
        width: 100%;
    }
    .app-header-left {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }
    .app-icon {
        width: 40px;
        height: 40px;
        background: #222D3E;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #38BDF8;
        border: 1px solid #38455A;
    }
    .app-title-group h1 {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.01em;
    }
    .app-title-group p {
        font-size: 0.88rem !important;
        color: #CBD5E1 !important;
        margin: 0 !important;
        line-height: 1.3 !important;
    }

    /* Status Indicators */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .status-pill.online {
        background-color: rgba(45, 212, 191, 0.15);
        color: #2DD4BF;
        border: 1px solid rgba(45, 212, 191, 0.4);
    }
    .status-pill.offline {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    .status-dot.online { background-color: #2DD4BF; }
    .status-dot.offline { background-color: #EF4444; }

    /* High-Contrast Soft Dark Cards */
    .soft-card {
        background-color: #222D3E;
        border: 1px solid #38455A;
        border-radius: 8px;
        padding: 1.2rem 1.35rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -2px rgba(0, 0, 0, 0.15);
        width: 100%;
    }
    .soft-card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .soft-card-subtitle {
        font-size: 0.88rem;
        color: #CBD5E1;
        margin: 0;
        line-height: 1.4;
    }

    /* Selected File Badge */
    .file-badge {
        background-color: #1A2332;
        border: 1px solid #38455A;
        border-radius: 6px;
        padding: 0.65rem 0.9rem;
        font-size: 0.86rem;
        color: #FFFFFF;
        margin: 0.75rem 0 1rem 0;
    }
    .file-badge strong {
        color: #38BDF8;
    }

    /* Radiograph Preview Container */
    .radiograph-frame {
        background-color: #0E141D;
        border: 1px solid #2B374A;
        border-radius: 6px;
        padding: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 300px;
        width: 100%;
        overflow: hidden;
    }
    .radiograph-placeholder {
        text-align: center;
        color: #94A3B8;
        font-size: 0.9rem;
        padding: 3rem 1rem;
    }

    /* Top Metric Cards */
    .top-metric-card {
        background-color: #222D3E;
        border: 1px solid #38455A;
        border-top: 3px solid #38BDF8;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .top-metric-rank {
        font-size: 0.76rem;
        font-weight: 700;
        color: #2DD4BF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .top-metric-label {
        font-size: 1.15rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.4rem;
    }
    .top-metric-value-row {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
    }
    .top-metric-score {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
        font-size: 1.45rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .top-metric-pct {
        font-size: 0.88rem;
        color: #CBD5E1;
        font-weight: 500;
    }

    /* Action Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #38BDF8 !important;
        border-color: #38BDF8 !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.25rem !important;
        box-shadow: 0 2px 4px rgba(56, 189, 248, 0.25) !important;
        transition: all 0.15s ease-in-out;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #7DD3FC !important;
        border-color: #7DD3FC !important;
        box-shadow: 0 4px 8px rgba(56, 189, 248, 0.35) !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: #1A2332 !important;
        border: 1px solid #38455A !important;
        color: #CBD5E1 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: #222D3E !important;
        color: #FFFFFF !important;
        border-color: #64748B !important;
    }

    /* Tabs Styling */
    div[data-testid="stTabs"] button[role="tab"] {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.5rem 1rem !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }

    /* Expander in Sidebar */
    .streamlit-expanderHeader {
        background-color: #222D3E !important;
        color: #FFFFFF !important;
        border: 1px solid #38455A !important;
        border-radius: 6px !important;
        font-size: 0.88rem !important;
    }
    .streamlit-expanderContent {
        background-color: #222D3E !important;
        border: 1px solid #38455A !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
    }

    /* Footer Note */
    .app-footer {
        margin-top: 3.5rem;
        padding-top: 1.25rem;
        border-top: 1px solid #38455A;
        color: #94A3B8;
        font-size: 0.8rem;
        line-height: 1.5;
        text-align: center;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_backend_health() -> Optional[Dict[str, Any]]:
    """Query FastAPI GET /health endpoint."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if resp.status_code in (200, 503):
            return resp.json()
        return None
    except requests.exceptions.RequestException:
        return None


def run_prediction(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Submit radiograph image to FastAPI POST /predict."""
    files = {"file": (filename, file_bytes, "image/png")}
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            try:
                err_detail = response.json().get("detail", response.text)
            except Exception:
                err_detail = response.text
            return {"success": False, "status_code": response.status_code, "error": err_detail}
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "status_code": 0,
            "error": f"Unable to reach the backend service at {BACKEND_URL}. Ensure the FastAPI server is running.",
        }
    except Exception as exc:
        return {"success": False, "status_code": 0, "error": str(exc)}


# Query backend status once
health_data = fetch_backend_health()
is_healthy = health_data is not None and health_data.get("status") == "healthy"
model_identifier = health_data.get("model_identifier", "weighted_best.keras") if health_data else "Unavailable"


# --- SIDEBAR: CLINICAL NAVIGATION & MODEL INFO ---
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.6rem; padding-bottom: 0.75rem; margin-bottom: 0.5rem; border-bottom: 1px solid #38455A;">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/>
                <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/>
                <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/>
                <path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>
            </svg>
            <span style="font-weight: 700; font-size: 1.05rem; color: #FFFFFF;">Radiology Workspace</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin: 0.5rem 0 1rem 0;">
            <div style="background-color: #222D3E; border-left: 3px solid #38BDF8; padding: 0.5rem 0.75rem; border-radius: 0 4px 4px 0; font-size: 0.88rem; font-weight: 600; color: #FFFFFF;">
                🩻 New Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compact System Status
    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#CBD5E1; margin-bottom:0.35rem; text-transform:uppercase; letter-spacing:0.05em;'>Inference Engine</p>", unsafe_allow_html=True)
    if is_healthy:
        st.markdown(
            f"""
            <div class="status-pill online" style="margin-bottom: 0.85rem;">
                <span class="status-dot online"></span>
                <span>Ready &bull; {model_identifier}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-pill offline" style="margin-bottom: 0.85rem;">
                <span class="status-dot offline"></span>
                <span>Service Offline</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Collapsible Model Information Section
    with st.expander("Model information", expanded=False):
        st.markdown(
            f"""
            <div style="font-size: 0.84rem; color: #CBD5E1; line-height: 1.65;">
                <strong style="color:#FFFFFF;">Architecture:</strong> DenseNet121<br>
                <strong style="color:#FFFFFF;">Pretraining:</strong> ImageNet &bull; NIH ChestX-ray14<br>
                <strong style="color:#FFFFFF;">Input Shape:</strong> 224 &times; 224 &times; 3 RGB<br>
                <strong style="color:#FFFFFF;">Target Classes:</strong> 14 multi-label pathologies<br>
                <strong style="color:#FFFFFF;">Active Checkpoint:</strong> <code>{model_identifier}</code><br>
                <strong style="color:#FFFFFF;">Activation:</strong> Sigmoid (independent units)
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- APPLICATION HEADER ---
status_badge_html = (
    f'<div class="status-pill online"><span class="status-dot online"></span>System Ready</div>'
    if is_healthy
    else '<div class="status-pill offline"><span class="status-dot offline"></span>Service Unavailable</div>'
)

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-header-left">
            <div class="app-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <rect width="18" height="18" x="3" y="3" rx="2"/>
                    <path d="M8 7v10"/>
                    <path d="M12 7v10"/>
                    <path d="M16 7v10"/>
                    <path d="M8 12h8"/>
                </svg>
            </div>
            <div class="app-title-group">
                <h1>Chest X-ray Analysis</h1>
                <p>AI-assisted radiograph research workspace</p>
            </div>
        </div>
        <div>
            {status_badge_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- MAIN WORKSPACE: NEW X-RAY ANALYSIS ---
st.markdown("<h2 style='font-size: 1.3rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.85rem;'>New X-ray Analysis</h2>", unsafe_allow_html=True)

col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown(
        """
        <div class="soft-card">
            <div class="soft-card-title">
                <span>Upload Radiograph</span>
                <span style="font-size: 0.8rem; font-weight: 500; color: #38BDF8;">Frontal (PA/AP)</span>
            </div>
            <p class="soft-card-subtitle">
                Upload a frontal chest radiograph in PNG or JPEG format for multi-label pathology activation scoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="Supported formats: PNG, JPG, JPEG.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_size_kb = len(file_bytes) / 1024

        st.markdown(
            f"""
            <div class="file-badge">
                <strong>Selected file:</strong> {uploaded_file.name} &bull; {file_size_kb:.1f} KB &bull; {uploaded_file.type}
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_act1, col_act2 = st.columns([1, 1])
        with col_act1:
            run_btn = st.button("Analyze image", type="primary", use_container_width=True)
        with col_act2:
            clear_btn = st.button("Clear image", type="secondary", use_container_width=True)

        if clear_btn:
            st.session_state.clear()
            st.rerun()

with col_preview:
    st.markdown(
        """
        <div class="soft-card">
            <div class="soft-card-title">
                <span>Radiograph Preview</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        try:
            preview_img = Image.open(uploaded_file)
            st.image(
                preview_img,
                caption=f"{uploaded_file.name} — {preview_img.width} × {preview_img.height} px ({preview_img.mode})",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Failed to display image preview: {exc}")
    else:
        st.markdown(
            """
            <div class="radiograph-frame">
                <div class="radiograph-placeholder">
                    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 0.65rem;">
                        <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
                        <circle cx="9" cy="9" r="2"/>
                        <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
                    </svg>
                    <p style="margin:0; font-weight:600; font-size: 0.95rem; color: #FFFFFF;">No radiograph selected</p>
                    <p style="margin:0.2rem 0 0 0; font-size: 0.82rem; color: #94A3B8;">Select a PNG or JPEG file on the left to preview</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- INFERENCE EXECUTION ---
if uploaded_file is not None and "run_btn" in locals() and run_btn:
    if not is_healthy:
        st.error(
            f"Cannot complete analysis: The inference service is currently unavailable at {BACKEND_URL}. "
            "Please check that the backend process is running."
        )
    else:
        with st.spinner("Processing radiograph and computing activation scores..."):
            result = run_prediction(file_bytes, uploaded_file.name)

        if not result["success"]:
            st.error(f"Inference request failed (Error {result.get('status_code')}): {result.get('error')}")
        else:
            st.session_state["prediction_result"] = result["data"]
            st.toast("Analysis complete. Model output updated below.", icon="✅")


# --- PREDICTION RESULTS: MODEL OUTPUT ---
if "prediction_result" in st.session_state:
    data = st.session_state["prediction_result"]
    ranked_preds = data.get("ranked_predictions", [])
    model_id = data.get("model_identifier", "weighted_best.keras")

    st.markdown("<hr style='margin: 2.25rem 0 1.5rem 0; border: none; border-top: 1px solid #38455A;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 1.3rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.35rem;'>Model Output</h2>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size: 0.88rem; color: #CBD5E1; margin-bottom: 1.35rem;'>"
        f"Evaluated with checkpoint <code>{model_id}</code>. "
        f"Scores represent raw uncalibrated sigmoid activations in [0.0000, 1.0000]."
        f"</p>",
        unsafe_allow_html=True,
    )

    # Top 3 Scores Summary Cards
    top_3 = ranked_preds[:3]
    top_cols = st.columns(3)
    for idx, item in enumerate(top_3):
        score_val = float(item["score"])
        with top_cols[idx]:
            st.markdown(
                f"""
                <div class="top-metric-card">
                    <div class="top-metric-rank">Highest Activation #{idx+1}</div>
                    <div class="top-metric-label">{item['label']}</div>
                    <div class="top-metric-value-row">
                        <span class="top-metric-score">{score_val:.4f}</span>
                        <span class="top-metric-pct">({score_val * 100:.1f}%)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # Full-width Tabs to Avoid Any Squishing or Horizontal Scrollbars
    df_ranked = pd.DataFrame(ranked_preds)
    df_ranked["score"] = df_ranked["score"].astype(float)
    df_ranked["score_pct"] = (df_ranked["score"] * 100).round(2)
    df_ranked["rank"] = range(1, len(df_ranked) + 1)

    tab_chart, tab_table = st.tabs(["📊 Ranked Activation Chart", "📋 Complete Pathology Table"])

    with tab_chart:
        st.markdown(
            """
            <div class="soft-card" style="padding-bottom: 0.5rem; margin-top: 0.75rem;">
                <div class="soft-card-title">
                    <span>Ranked Score Distribution</span>
                    <span style="font-size: 0.82rem; font-weight: normal; color: #38BDF8;">All 14 Pathologies</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Full-width Altair Bar Chart with Vibrant Medical Blue
        chart = (
            alt.Chart(df_ranked)
            .mark_bar(color="#38BDF8", cornerRadiusEnd=3, height=20)
            .encode(
                x=alt.X(
                    "score:Q",
                    title="Model Activation Score",
                    scale=alt.Scale(domain=[0, 1.0]),
                    axis=alt.Axis(
                        format=".2f",
                        gridColor="#222D3E",
                        domainColor="#38455A",
                        tickColor="#38455A",
                        labelColor="#CBD5E1",
                        titleColor="#FFFFFF",
                        titleFontSize=12,
                        labelFontSize=11,
                    ),
                ),
                y=alt.Y(
                    "label:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(
                        labelColor="#FFFFFF",
                        labelFontSize=13,
                        labelFontWeight="bold",
                        domainColor="#38455A",
                        tickColor="#38455A",
                    ),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Pathology"),
                    alt.Tooltip("score:Q", format=".4f", title="Raw Score"),
                    alt.Tooltip("score_pct:Q", format=".1f", title="Relative Activation (%)"),
                ],
            )
            .properties(height=450, width="container")
            .configure_view(strokeWidth=0)
        )

        st.altair_chart(chart, use_container_width=True)

    with tab_table:
        st.markdown(
            """
            <div class="soft-card" style="padding-bottom: 0.5rem; margin-top: 0.75rem;">
                <div class="soft-card-title">
                    <span>Complete Results Table</span>
                    <span style="font-size: 0.82rem; font-weight: normal; color: #38BDF8;">Sorted by Score</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_df = df_ranked[["rank", "label", "score", "score_pct"]].rename(
            columns={
                "rank": "Rank",
                "label": "Pathology",
                "score": "Raw Score",
                "score_pct": "Relative Indicator",
            }
        )

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=450,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Pathology": st.column_config.TextColumn(width="medium"),
                "Raw Score": st.column_config.NumberColumn(format="%.4f", width="small"),
                "Relative Indicator": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100, width="large"
                ),
            },
        )


# --- UNOBTRUSIVE FOOTER NOTE ---
st.markdown(
    """
    <div class="app-footer">
        Research application prototype &bull; Model outputs represent experimental, uncalibrated activation scores and are not intended for clinical diagnosis, patient triage, or treatment planning.
    </div>
    """,
    unsafe_allow_html=True,
)
