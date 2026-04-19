import streamlit as st
from PIL import Image
import numpy as np
import requests

st.set_page_config(
    page_title="LeafScan — Plant Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# UI theme and layout styling.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg:        #0a0f0a;
    --surface:   #111711;
    --surface2:  #161e16;
    --border:    rgba(255,255,255,0.07);
    --green:     #4ade80;
    --green-dim: #1a3d28;
    --amber:     #f59e0b;
    --text:      #e8ede8;
    --muted:     #6b7a6b;
    --danger:    #f87171;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit chrome */
[data-testid="stHeader"],
[data-testid="stToolbar"],
footer { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--green-dim); border-radius: 3px; }

/* ── TOPBAR ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 28px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 36px;
}
.logo {
    display: flex;
    align-items: center;
    gap: 12px;
}
.logo-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #1a3d28 0%, #4ade80 100%);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.logo-name {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--text);
}
.logo-name span { color: var(--green); }
.topbar-tag {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--green);
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.2);
    padding: 5px 12px;
    border-radius: 999px;
    letter-spacing: 0.5px;
}

/* ── HERO ── */
.hero-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--green);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(32px, 4vw, 52px);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -1.5px;
    color: var(--text);
    margin-bottom: 16px;
}
.hero-title em {
    font-style: normal;
    color: var(--green);
}
.hero-desc {
    font-size: 15px;
    color: var(--muted);
    line-height: 1.7;
    max-width: 480px;
    font-weight: 300;
}

/* ── STAT CHIPS ── */
.stats-row {
    display: flex;
    gap: 16px;
    margin-top: 28px;
    flex-wrap: wrap;
}
.stat-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 18px;
    min-width: 110px;
}
.stat-chip .num {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--green);
    line-height: 1;
}
.stat-chip .lbl {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.8px;
    margin-top: 4px;
    text-transform: uppercase;
}

/* ── PANEL ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 28px;
    height: 100%;
}
.panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-title::before {
    content: '';
    display: block;
    width: 6px; height: 6px;
    background: var(--green);
    border-radius: 50%;
}

/* ── UPLOAD AREA ── */
.upload-hint {
    border: 1.5px dashed rgba(74,222,128,0.2);
    border-radius: 14px;
    padding: 32px 20px;
    text-align: center;
    background: rgba(74,222,128,0.02);
    margin-bottom: 18px;
    transition: border-color 0.2s;
}
.upload-hint:hover { border-color: rgba(74,222,128,0.4); }
.upload-icon { font-size: 32px; margin-bottom: 8px; }
.upload-text {
    font-size: 13px;
    color: var(--muted);
    font-weight: 300;
}
.upload-text strong { color: var(--green); font-weight: 500; }

/* Streamlit file uploader override */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] > div {
    background: rgba(74,222,128,0.03) !important;
    border: 1.5px dashed rgba(74,222,128,0.25) !important;
    border-radius: 14px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(74,222,128,0.5) !important;
    background: rgba(74,222,128,0.06) !important;
}
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploader"] label { color: var(--muted) !important; }
[data-testid="stFileUploader"] small { color: var(--muted) !important; opacity: 0.6; }

/* ── IMAGE PREVIEW ── */
.img-wrapper {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid var(--border);
    background: #0d130d;
    position: relative;
}
.img-overlay {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    background: linear-gradient(to top, rgba(0,0,0,0.7) 0%, transparent 100%);
    padding: 14px 16px 12px;
}
.img-meta {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.7);
    letter-spacing: 0.5px;
}

/* ── RESULT ── */
.result-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 280px;
    gap: 12px;
}
.result-empty-icon { font-size: 40px; opacity: 0.3; }
.result-empty-text {
    font-size: 14px;
    color: var(--muted);
    font-weight: 300;
    text-align: center;
}

.result-card {
    background: linear-gradient(135deg, rgba(74,222,128,0.06) 0%, rgba(74,222,128,0.01) 100%);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(74,222,128,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--green);
    text-transform: uppercase;
    margin-bottom: 10px;
}
.result-class {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.5px;
    line-height: 1.2;
    word-break: break-word;
}
.result-sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
    font-weight: 300;
}

/* Confidence bar */
.conf-bar-wrap {
    margin-top: 16px;
}
.conf-bar-label {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 6px;
}
.conf-bar-track {
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #1a3d28 0%, #4ade80 100%);
    transition: width 1s cubic-bezier(0.16,1,0.3,1);
}

/* Top-K breakdown */
.topk-title {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 12px;
}
.topk-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.topk-name {
    font-size: 13px;
    color: var(--text);
    flex: 1;
    font-weight: 400;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.topk-track {
    flex: 2;
    height: 3px;
    background: rgba(255,255,255,0.05);
    border-radius: 99px;
    overflow: hidden;
}
.topk-fill {
    height: 100%;
    border-radius: 99px;
    background: rgba(74,222,128,0.5);
}
.topk-pct {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    min-width: 38px;
    text-align: right;
}

/* Status badge */
.status-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--green);
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 999px;
    padding: 4px 12px;
    margin-top: 10px;
}
.status-ok::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Spinner override */
[data-testid="stSpinner"] { color: var(--green) !important; }

/* Divider */
.divider {
    height: 1px;
    background: var(--border);
    margin: 22px 0;
}

/* Error */
.err-box {
    background: rgba(248,113,113,0.06);
    border: 1px solid rgba(248,113,113,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 13px;
    color: var(--danger);
    font-family: 'DM Mono', monospace;
}

/* Streamlit image overrides */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    width: 100% !important;
}

/* Column padding */
[data-testid="column"] { padding: 0 12px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child { padding-right: 0 !important; }

/* Main block padding */
.block-container {
    padding: 32px 40px !important;
    max-width: 1200px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── CONFIG ───────────────────────────────────────────────
# API_URL = "http://api:8000/predict"
API_URL = "https://plant-api-g8ab.onrender.com/predict"

# ─── TOPBAR ───────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="logo">
        <div class="logo-icon">🌿</div>
        <div class="logo-name">Leaf<span>Scan</span></div>
    </div>
    <div class="topbar-tag">v2.1 · Classical ML</div>
</div>
""", unsafe_allow_html=True)

# ─── HERO ─────────────────────────────────────────────────
st.markdown("""
<div class="hero-label">AI-Powered Plant Pathology</div>
<h1 class="hero-title">Diagnose plant disease<br><em>in seconds.</em></h1>
<p class="hero-desc">
    Upload any tomato leaf image. Our HSV-segmented feature pipeline and
    ensemble classifier return a diagnosis with confidence breakdown.
</p>
<div class="stats-row">
    <div class="stat-chip"><div class="num">10</div><div class="lbl">Disease Classes</div></div>
    <div class="stat-chip"><div class="num">95%+</div><div class="lbl">Val. Accuracy</div></div>
    <div class="stat-chip"><div class="num">&lt;200ms</div><div class="lbl">Inference Time</div></div>
    <div class="stat-chip"><div class="num">RGB+HSV</div><div class="lbl">Feature Space</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

# ─── MAIN COLUMNS ─────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Image Input</div>', unsafe_allow_html=True)

    # Image uploader for inference.
    uploaded = st.file_uploader(
        "Drop a leaf image here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        # Load and preview the uploaded image with basic metadata.
        img = Image.open(uploaded).convert("RGB")
        w, h = img.size
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown(f"""
        <div style='margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;'>
            <div style='font-family:"DM Mono",monospace; font-size:11px; color:var(--muted);
                        background:var(--surface2); border:1px solid var(--border);
                        border-radius:6px; padding:4px 10px;'>{uploaded.name}</div>
            <div style='font-family:"DM Mono",monospace; font-size:11px; color:var(--muted);
                        background:var(--surface2); border:1px solid var(--border);
                        border-radius:6px; padding:4px 10px;'>{w}×{h}px</div>
            <div style='font-family:"DM Mono",monospace; font-size:11px; color:var(--muted);
                        background:var(--surface2); border:1px solid var(--border);
                        border-radius:6px; padding:4px 10px;'>{uploaded.type}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Empty-state panel for the upload area.
        st.markdown("""
        <div style='height:14px'></div>
        <div class='upload-hint'>
            <div class='upload-icon'>🍃</div>
            <div class='upload-text'>Drag & drop or <strong>browse</strong> a leaf image<br>
            <span style='font-size:11px; opacity:0.6;'>JPG, JPEG, PNG supported</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Diagnosis</div>', unsafe_allow_html=True)

    if not uploaded:
        # Empty-state panel for the diagnosis area.
        st.markdown("""
        <div class="result-empty">
            <div class="result-empty-icon">🔬</div>
            <div class="result-empty-text">Upload a leaf image<br>to start the analysis</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Send the image to the API and render the prediction.
        with st.spinner("Analyzing leaf..."):
            try:
            # Build a multipart payload compatible with FastAPI.
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                response = requests.post(API_URL, files=files, timeout=10)
                result = response.json()

                if "error" in result:
                    st.markdown(f'<div class="err-box">⚠ {result["error"]}</div>', unsafe_allow_html=True)
                else:
                    # Parse API response fields.
                    predicted_class = result.get("class", "Unknown")
                    confidence = result.get("confidence", None)
                    top_k = result.get("top_k", [])

                    # ── Main result card
                    conf_display = f"{confidence*100:.1f}%" if confidence else "—"
                    conf_bar_w = f"{confidence*100:.0f}%" if confidence else "0%"

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">Predicted Diagnosis</div>
                        <div class="result-class">{predicted_class.replace('_', ' ')}</div>
                        <div class="result-sub">Tomato leaf pathology classification</div>
                        {"" if not confidence else f'''
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-label">
                                <span>Confidence</span>
                                <span>{conf_display}</span>
                            </div>
                            <div class="conf-bar-track">
                                <div class="conf-bar-fill" style="width:{conf_bar_w}"></div>
                            </div>
                        </div>
                        '''}
                    </div>
                    <div class="status-ok">Analysis complete</div>
                    """, unsafe_allow_html=True)

                    # ── Top-K breakdown (if API returns it)
                    if top_k:
                        # Optional Top-K breakdown if provided by the API.
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        st.markdown('<div class="topk-title">Top predictions</div>', unsafe_allow_html=True)
                        max_score = max([t.get("score", 1) for t in top_k]) or 1
                        for item in top_k[:5]:
                            name = item.get("class", "?").replace("_", " ")
                            score = item.get("score", 0)
                            pct = score * 100
                            bar_w = (score / max_score) * 100
                            st.markdown(f"""
                            <div class="topk-row">
                                <div class="topk-name">{name}</div>
                                <div class="topk-track">
                                    <div class="topk-fill" style="width:{bar_w:.0f}%"></div>
                                </div>
                                <div class="topk-pct">{pct:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.markdown('<div class="err-box">⚠ Cannot reach API — make sure the backend is running on port 8000.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="err-box">⚠ Unexpected error: {e}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────
st.markdown("""
<div style='height:60px'></div>
<div style='border-top:1px solid var(--border); padding-top:20px;
            display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;'>
    <div style='font-family:"DM Mono",monospace; font-size:11px; color:var(--muted);'>
        LeafScan · Classical ML Pipeline · HSV Segmentation + Feature Extraction
    </div>
    <div style='font-family:"DM Mono",monospace; font-size:11px; color:var(--muted);'>
        Tomato Disease Classifier · 10 Classes
    </div>
</div>
""", unsafe_allow_html=True)