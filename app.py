# app.py
import streamlit as st
import json
import re
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import altair as alt
from rapidfuzz import fuzz, process
import pdfplumber

from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch

# ---------------------------
# App Config (Dark Theme)
# ---------------------------
st.set_page_config(
    page_title="Police Recognition Analytics",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { background: linear-gradient(135deg, #0F172A 0%, #0B1220 100%); font-family: 'Inter', sans-serif; }
    .main-header { font-size: 2.6rem; font-weight: 700; text-align: center; padding: 24px;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-shadow: 0 0 24px rgba(96,165,250,0.25); }
    .card { background: #0B1324; border: 1px solid #1F2A44; border-radius: 12px; padding: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.25); }
    .metric-card { background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%); padding: 22px; border-radius: 12px;
        color: #fff; text-align: center; box-shadow: 0 8px 24px rgba(59,130,246,0.35); }
    .metric-card h2,.metric-card p { color: #fff !important; margin: 0; }
    .section-title { color: #E5E7EB; font-size: 1.25rem; font-weight: 700; margin-top: 8px; margin-bottom: 8px; }
    .label { color: #93C5FD; font-weight: 600; }
    .value { color: #E5E7EB; }
    .badge { display:inline-block; padding:4px 10px; border-radius: 9999px; font-weight:600; font-size:12px; margin-right:6px; }
    .ok-badge { background:#065F46; color:#D1FAE5; border:1px solid #10B981; }
    .warn-badge { background:#7C2D12; color:#FDE68A; border:1px solid #F59E0B; }
    .info { background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 100%); border: 1px solid #3B82F6; color: #DBEAFE; padding: 14px; border-radius: 10px; }
    .ok { background: linear-gradient(135deg, #065F46 0%, #047857 100%); border: 1px solid #10B981; color: #D1FAE5; padding: 14px; border-radius: 10px; }
    .stTextArea textarea, .stTextInput input { background-color: #0B1324 !important; color: #E5E7EB !important; border: 1.5px solid #1F2A44 !important; }
    .stTextArea textarea:focus, .stTextInput input:focus { border-color: #60A5FA !important; box-shadow: 0 0 0 3px rgba(96,165,250,0.25) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background: #0B1324; padding: 8px; border-radius: 12px; border: 1px solid #1F2A44; }
    .stTabs [data-baseweb="tab"] { background: #111C33; color: #E5E7EB; border-radius: 8px; border: 1px solid #1F2A44; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%); color: #fff !important; border-color: #3B82F6; }
    .stDownloadButton button, .stButton button { color: #fff !important; font-weight: 600; border-radius: 8px; border: none; }
    .stButton button[kind="primary"] { background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%); }
    .stDownloadButton button { background: linear-gradient(135deg, #059669 0%, #047857 100%); }
    [data-testid="stSidebar"] { background: #0B1324; border-right: 1px solid #1F2A44; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p { color: #E5E7EB; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚔 Police Recognition Analytics</div>', unsafe_allow_html=True)

# ---------------------------
# Paths to datasets
# ---------------------------
DATA_PATHS = {
    "ipc": "OdishaIPCCrimedata.json",
    "districts": "DistrictReport.json",
    "cctns": "mock_cctnsdata.json",
    "feedback": "publicfeedback.json"
}

# ---------------------------
# Load NLP models
# ---------------------------
@st.cache_resource
def load_models():
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=-1)  # model card [web:84]
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # usage pattern [web:76]
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
    return ner, summarizer, sentiment, qa

NER, SUMMARIZER, SENTIMENT, QA = load_models()

# ---------------------------
# Load JSONs and show status
# ---------------------------
def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), True
    except Exception:
        return None, False

IPC_DEF, IPC_OK = load_json(DATA_PATHS["ipc"])
DISTRICTS_DEF, DIST_OK = load_json(DATA_PATHS["districts"])
CCTNS_DEF, CCTNS_OK = load_json(DATA_PATHS["cctns"])
FEEDBACK_DEF, FB_OK = load_json(DATA_PATHS["feedback"])

# ---------------------------
# Build ontologies/gazetteers
# ---------------------------
@st.cache_resource
def build_ipc_map(ipc_json) -> Dict[str, Dict]:
    mapping = {}
    if not ipc_json:
        return mapping
    fields = ipc_json.get("fields", [])
    for f in fields:
        label = f.get("label", "").strip()
        if not label:
            continue
        sec = re.findall(r"Section\s+([0-9A-Z,\-\s]+)\s*IPC", label, flags=re.IGNORECASE)
        sections = []
        if sec:
            raw = sec[0]
            parts = re.split(r"[,\s]+", raw)
            parts = [p.strip() for p in parts if p.strip()]
            sections = parts
        base = re.sub(r"\(.*?\)", "", label).strip()
        key = base.lower()
        mapping[key] = {"label": base, "sections": sections}
    return mapping

@st.cache_resource
def build_gazetteers(districts_json, cctns_json) -> Dict[str, List[str]]:
    districts, stations, officers = [], [], []
    if isinstance(districts_json, dict):
        districts = list(districts_json.keys())
    if isinstance(cctns_json, list):
        for row in cctns_json:
            ps = row.get("police_station")
            if ps:
                stations.append(ps)
            oid = row.get("investigating_officer_id")
            if oid:
                officers.append(oid)
                m = re.search(r"[A-Z]([a-z]+)?[-_]?([A-Z][a-z]+)", oid.replace(".", ""))
                if m:
                    last = m.group(2)
                    first = (m.group(1) or "").title()
                    cand = f"{first} {last}".strip()
                    if len(cand) > 1:
                        officers.append(cand)
    return {
        "districts": sorted(set([d for d in districts if d])),
        "stations": sorted(set([s for s in stations if s])),
        "officers": sorted(set([o for o in officers if o])),
    }

IPC_MAP = build_ipc_map(IPC_DEF)
GAZ = build_gazetteers(DISTRICTS_DEF, CCTNS_DEF)

RANK_WORDS = ["Officer","Constable","Inspector","Sub-Inspector","Sub Inspector","Sergeant","Detective",
              "Chief","Captain","Lieutenant","ASI","SI","PSI","DSP","ACP","DCP","CI","HC"]

def fuzzy_lookup(name: str, candidates: List[str], score_cutoff=88) -> Tuple[str, int]:
    if not name or not candidates:
        return "", 0
    match = process.extractOne(name, candidates, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)
    if match:
        return match[0], match[1]
    return "", 0

def extract_dates(text: str) -> List[str]:
    pats = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
    ]
    out = set()
    for p in pats:
        for m in re.findall(p, text, flags=re.IGNORECASE):
            out.add(m)
    return sorted(out)

def extract_ipc(text: str, ipc_map: Dict[str, Dict]) -> List[Dict]:
    t = text.lower()
    hits = {}
    for k, meta in ipc_map.items():
        if k and k in t:
            hits[meta["label"]] = meta
    for m in re.findall(r"section\s+([0-9A-Z\-]+)", text, flags=re.IGNORECASE):
        sec = m.upper()
        for k, meta in ipc_map.items():
            if sec in [s.upper() for s in meta["sections"]]:
                hits[meta["label"]] = meta
    return [{"label": k, "sections": v["sections"]} for k, v in hits.items()]

def extract_entities(text: str) -> Dict[str, List[str]]:
    ner_out = NER(text[:3000])  # dslim/bert-base-NER entity groups [web:84]
    persons, orgs, locs = [], [], []
    for e in ner_out:
        if e.get("entity_group") == "PER":
            persons.append(e["word"].replace("▁", " ").strip())
        elif e.get("entity_group") == "ORG":
            orgs.append(e["word"].replace("▁", " ").strip())
        elif e.get("entity_group") == "LOC":
            locs.append(e["word"].replace("▁", " ").strip())
    rank_pat = rf"(?:{'|'.join(RANK_WORDS)})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    for m in re.findall(rank_pat, text):
        persons.append(m.strip())
    return {
        "persons": sorted(set([p for p in persons if len(p) > 1])),
        "orgs": sorted(set([o for o in orgs if len(o) > 1])),
        "locs": sorted(set([l for l in locs if len(l) > 1])),
    }

def map_to_gazetteers(ents: Dict[str, List[str]], gaz: Dict[str, List[str]], text: str) -> Dict[str, List[Dict]]:
    mapped_officers, mapped_stations, mapped_districts = [], [], []
    for p in ents["persons"]:
        hit, score = fuzzy_lookup(p, gaz["officers"], score_cutoff=90)
        if hit:
            mapped_officers.append({"text": p, "canonical": hit, "score": score})
    for c in set(ents["orgs"] + ents["locs"]):
        s_hit, s_score = fuzzy_lookup(c, gaz["stations"], score_cutoff=90)
        if s_hit: mapped_stations.append({"text": c, "canonical": s_hit, "score": s_score})
        d_hit, d_score = fuzzy_lookup(c, gaz["districts"], score_cutoff=90)
        if d_hit: mapped_districts.append({"text": c, "canonical": d_hit, "score": d_score})
    for m in re.findall(r"([A-Z][a-zA-Z]+)\s+(?:Police\s+Station|PS|Thana)", text):
        s_hit, s_score = fuzzy_lookup(m.strip(), gaz["stations"], score_cutoff=85)
        if s_hit:
            mapped_stations.append({"text": f"{m} Police Station", "canonical": s_hit, "score": s_score})

    def dedup(items, key="canonical"):
        seen, out = set(), []
        for it in sorted(items, key=lambda x: -x["score"]):
            if it[key] not in seen:
                out.append(it); seen.add(it[key])
        return out

    return {"officers": dedup(mapped_officers), "stations": dedup(mapped_stations), "districts": dedup(mapped_districts)}

def summarize_text(text: str) -> str:
    if len(text) < 120:
        return text.strip()
    out = SUMMARIZER(text[:1500], max_length=140, min_length=40, do_sample=False)  # BART large CNN summarizer [web:76]
    return out[0]["summary_text"].strip()

def analyze_sentiment(text: str) -> Dict:
    out = SENTIMENT(text[:512])[0]
    label = out["label"].upper()
    score = out["score"]
    normalized = score if label == "POSITIVE" else (-score if label == "NEGATIVE" else 0.0)
    return {"label": label, "score": score, "normalized": normalized}

def process_input(text: str) -> Dict:
    text = text.strip()
    summary = summarize_text(text)
    sentiment = analyze_sentiment(text)
    dates = extract_dates(text)
    ipc_hits = extract_ipc(text, IPC_MAP)
    ents = extract_entities(text)
    mapped = map_to_gazetteers(ents, GAZ, text)
    score = (0.4 * max(0.0, sentiment["normalized"])) + (0.3 if ipc_hits else 0.0) + (0.2 if mapped["officers"] else 0.0) + (0.1 if mapped["stations"] else 0.0)
    score = round(min(1.0, max(0.0, score)), 3)
    return {
        "timestamp": datetime.now().isoformat(),
        "input_length": len(text),
        "summary": summary,
        "sentiment": sentiment,
        "dates": dates,
        "ipc_hits": ipc_hits,
        "entities": ents,
        "mapped": mapped,
        "recognition_score": score,
        "raw_text": text
    }

def build_pdf_report(record: Dict) -> BytesIO:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    title = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor("#3B82F6"), alignment=1, spaceAfter=16)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#8B5CF6"), spaceAfter=8)
    normal = styles['Normal']

    story = []
    story.append(Paragraph("Police Recognition Summary", title))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Extractive Summary", h2)); story.append(Paragraph(record.get("summary", ""), normal)); story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Key Metrics", h2))
    data = [
        ["Metric", "Value"],
        ["Recognition Score", f"{record.get('recognition_score', 0)}"],
        ["Sentiment", f"{record['sentiment']['label']} ({record['sentiment']['score']:.2f})"],
        ["Dates", ", ".join(record.get("dates", [])) or "-"],
        ["Text Length", str(record.get("input_length", 0))]
    ]
    tbl = Table(data, colWidths=[2.5*inch, 3.5*inch])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.HexColor("#3B82F6")),
        ('TEXTCOLOR',(0,0),(-1,0), colors.white),
        ('GRID',(0,0),(-1,-1), 0.5, colors.grey),
        ('BACKGROUND',(0,1),(-1,-1), colors.whitesmoke)
    ]))
    story.append(tbl); story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Entities & Mapping", h2))
    persons = ", ".join(record["entities"].get("persons", [])) or "-"
    orgs = ", ".join(record["entities"].get("orgs", [])) or "-"
    locs = ", ".join(record["entities"].get("locs", [])) or "-"
    story.append(Paragraph(f"Persons: {persons}", normal))
    story.append(Paragraph(f"Organizations: {orgs}", normal))
    story.append(Paragraph(f"Locations: {locs}", normal))
    story.append(Paragraph(f"Matched Officers: {', '.join([m['canonical'] for m in record['mapped']['officers']]) or '-'}", normal))
    story.append(Paragraph(f"Matched Stations: {', '.join([m['canonical'] for m in record['mapped']['stations']]) or '-'}", normal))
    story.append(Paragraph(f"Matched Districts: {', '.join([m['canonical'] for m in record['mapped']['districts']]) or '-'}", normal))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("IPC Matches", h2))
    if record.get("ipc_hits"):
        for h in record["ipc_hits"]:
            story.append(Paragraph(f"- {h['label']} (Sections: {', '.join(h['sections']) or '-'})", normal))
    else:
        story.append(Paragraph("- None", normal))

    doc.build(story); buf.seek(0); return buf  # create styled PDF [web:77]

# ---------------------------
# Session state
# ---------------------------
if "records" not in st.session_state:
    st.session_state["records"] = []

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Navigation")

    def badge(ok): return f"<span class='badge {'ok-badge' if ok else 'warn-badge'}'>{'Loaded' if ok else 'Missing'}</span>"
    st.markdown("<div class='card'><div class='section-title'>Datasets</div>", unsafe_allow_html=True)
    st.markdown(f"- OdishaIPCCrimedata.json {badge(IPC_OK)}", unsafe_allow_html=True)
    st.markdown(f"- DistrictReport.json {badge(DIST_OK)}", unsafe_allow_html=True)
    st.markdown(f"- mock_cctnsdata.json {badge(CCTNS_OK)}", unsafe_allow_html=True)
    st.markdown(f"- publicfeedback.json {badge(FB_OK)}", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div class='card'><span class='label'>Districts:</span> <span class='value'>{len(GAZ['districts'])}</span><br>"
                f"<span class='label'>Stations:</span> <span class='value'>{len(GAZ['stations'])}</span><br>"
                f"<span class='label'>Officers:</span> <span class='value'>{len(GAZ['officers'])}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    if FB_OK and st.button("Load Sample Feedback"):
        if isinstance(FEEDBACK_DEF, list) and FEEDBACK_DEF:
            joined = " ".join(item.get("text", "") for item in FEEDBACK_DEF[:20])
            rec = process_input(joined)
            st.session_state["records"].append(rec)
            st.success("Loaded and analyzed sample feedback.")

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Process", "📊 Dashboard", "📈 Analytics", "💬 Q&A", "📤 Export"])

with tab1:
    st.subheader("Process New Text or PDF")
    c1, c2 = st.columns([2,1])
    with c1:
        mode = st.radio("Input Method", ["Text", "PDF"], horizontal=True)
        text = ""
        if mode == "Text":
            text = st.text_area("Paste content:", height=220, placeholder="Officer John Smith from Chandrasekharpur Police Station swiftly responded to the theft case...")
        else:
            up = st.file_uploader("Upload PDF", type=["pdf"])
            if up:
                try:
                    with pdfplumber.open(up) as pdf:
                        full = []
                        for page in pdf.pages:
                            full.append(page.extract_text() or "")
                        text = "\n".join(full)
                    st.text_area("Extracted Text Preview", text[:1200], height=220)
                except Exception as e:
                    st.error(f"PDF read error: {e}")
    with c2:
        st.markdown("<div class='info'>• PERSON/ORG/LOC detection (BERT NER)<br>• IPC section normalization<br>• Gazetteer-based officer/station/district mapping</div>", unsafe_allow_html=True)
        st.markdown("<div class='ok'>Tip: Mention officer ranks (SI/ASI/Inspector) and station names (PS/Police Station) for better recall.</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚀 Analyze", type="primary", use_container_width=True):
        if text.strip():
            rec = process_input(text)
            st.session_state["records"].append(rec)
            st.success("Analysis complete.")
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"<div class='metric-card'><h2>{rec['recognition_score']}</h2><p>Score</p></div>", unsafe_allow_html=True)
            k2.markdown(f"<div class='metric-card'><h2>{rec['sentiment']['label']}</h2><p>Sentiment</p></div>", unsafe_allow_html=True)
            k3.markdown(f"<div class='metric-card'><h2>{len(rec['mapped']['officers'])}</h2><p>Officers</p></div>", unsafe_allow_html=True)
            k4.markdown(f"<div class='metric-card'><h2>{len(rec['ipc_hits'])}</h2><p>IPC Hits</p></div>", unsafe_allow_html=True)

            st.markdown("### Summary")
            st.markdown(f"<div class='card'>{rec['summary']}</div>", unsafe_allow_html=True)

            a, b = st.columns(2)
            with a:
                st.markdown("#### Entities (raw)")
                st.write(rec["entities"])
                st.markdown("#### Dates")
                st.write(rec["dates"])
            with b:
                st.markdown("#### Gazetteer Mapping")
                st.write(rec["mapped"])
                st.markdown("#### IPC Matches")
                st.write(rec["ipc_hits"])
        else:
            st.warning("Please provide text or a PDF first.")

with tab2:
    st.subheader("Recognition Dashboard")
    if st.session_state["records"]:
        df = pd.DataFrame([{
            "timestamp": r["timestamp"],
            "score": r["recognition_score"],
            "sentiment": r["sentiment"]["label"],
            "officers": ", ".join([m["canonical"] for m in r["mapped"]["officers"]]),
            "stations": ", ".join([m["canonical"] for m in r["mapped"]["stations"]]),
            "districts": ", ".join([m["canonical"] for m in r["mapped"]["districts"]]),
            "ipc": ", ".join([h["label"] for h in r["ipc_hits"]]),
            "dates": ", ".join(r["dates"]),
            "length": r["input_length"]
        } for r in st.session_state["records"]])

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Analyses", len(df))
        d2.metric("Avg Score", f"{df['score'].mean():.2f}")
        d3.metric("Positive %", f"{(df['sentiment']=='POSITIVE').mean()*100:.1f}%")
        d4.metric("Total Officers Referenced", int(df["officers"].astype(str).str.count(",").sum() + (df["officers"].astype(str)!="").sum()))

        st.markdown("#### Recent Records")
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, height=320)
    else:
        st.info("No records yet.")

with tab3:
    st.subheader("Interactive Analytics")
    # Build analytics frames from current session + static datasets
    if st.session_state["records"]:
        rec_df = pd.DataFrame(st.session_state["records"])

        # Crime-type distribution from IPC hits
        def explode_ipc_rows(records):
            rows = []
            for r in records:
                for h in r.get("ipc_hits", []):
                    rows.append({"label": h["label"], "timestamp": r["timestamp"]})
            return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["label","timestamp"])

        ipc_df = explode_ipc_rows(st.session_state["records"])
        if not ipc_df.empty:
            ipc_counts = ipc_df["label"].value_counts().reset_index()
            ipc_counts.columns = ["IPC", "Count"]
            chart = alt.Chart(ipc_counts).mark_bar(color="#60A5FA").encode(
                x=alt.X("IPC:N", sort="-y", title="IPC Crime Category"),
                y=alt.Y("Count:Q"),
                tooltip=["IPC","Count"]
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)  # Altair usage in Streamlit [web:91][web:113]
        else:
            st.info("No IPC matches yet to chart.")

        st.markdown("---")

        # Station & district frequencies
        def canonical_list(records, key):
            vals = []
            for r in records:
                vals.extend([m["canonical"] for m in r["mapped"][key]])
            return vals

        stations = canonical_list(st.session_state["records"], "stations")
        districts = canonical_list(st.session_state["records"], "districts")

        colA, colB = st.columns(2)
        if stations:
            s_df = pd.DataFrame(Counter(stations).most_common(), columns=["Station","Count"])
            s_chart = alt.Chart(s_df).mark_bar(color="#8B5CF6").encode(
                x=alt.X("Station:N", sort="-y"),
                y=alt.Y("Count:Q"),
                tooltip=["Station","Count"]
            ).properties(title="Top Stations Referenced", height=320)
            colA.altair_chart(s_chart, use_container_width=True)
        else:
            colA.info("No station references yet.")

        if districts:
            d_df = pd.DataFrame(Counter(districts).most_common(), columns=["District","Count"])
            d_chart = alt.Chart(d_df).mark_bar(color="#34D399").encode(
                x=alt.X("District:N", sort="-y"),
                y=alt.Y("Count:Q"),
                tooltip=["District","Count"]
            ).properties(title="Top Districts Referenced", height=320)
            colB.altair_chart(d_chart, use_container_width=True)
        else:
            colB.info("No district references yet.")

        st.markdown("---")

        # Feedback sentiment (if loaded) – simple distribution
        if FB_OK and isinstance(FEEDBACK_DEF, list) and FEEDBACK_DEF:
            fb_samples = [it.get("text","") for it in FEEDBACK_DEF[:50] if it.get("text")]
            if fb_samples:
                sentiments = []
                for t in fb_samples:
                    s = analyze_sentiment(t)
                    sentiments.append(s["label"])
                sent_df = pd.DataFrame(Counter(sentiments).most_common(), columns=["Sentiment","Count"])
                sent_chart = alt.Chart(sent_df).mark_arc(innerRadius=60).encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(field="Sentiment", type="nominal",
                                    scale=alt.Scale(range=["#22C55E","#F59E0B","#EF4444"])),
                    tooltip=["Sentiment","Count"]
                ).properties(title="Sample Feedback Sentiment", height=320)
                st.altair_chart(sent_chart, use_container_width=True)
            else:
                st.info("Feedback dataset is empty for sentiment chart.")
        else:
            st.info("Feedback dataset not available for sentiment chart.")
    else:
        st.info("Process at least one item to enable analytics.")

with tab4:
    st.subheader("Q&A Over All Processed Text")
    if st.session_state["records"]:
        all_text = " ".join([r["raw_text"] for r in st.session_state["records"]])
        q = st.text_input("Ask a question:")
        if st.button("Get Answer", type="primary"):
            if q.strip():
                ans = QA(question=q, context=all_text[:2000])
                st.success(ans.get("answer", ""))
            else:
                st.warning("Enter a question first.")
    else:
        st.info("Process at least one text to enable Q&A.")

with tab5:
    st.subheader("Export")
    if st.session_state["records"]:
        last = st.session_state["records"][-1]
        pdf_buf = build_pdf_report(last)
        st.download_button("📄 Download PDF (latest)", data=pdf_buf, file_name=f"recognition_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf", use_container_width=True)

        out_df = pd.DataFrame(st.session_state["records"])
        st.download_button("📄 Download CSV (all)", data=out_df.to_csv(index=False), file_name="all_records.csv", mime="text/csv", use_container_width=True)
        st.download_button("📋 Download JSON (all)", data=json.dumps(st.session_state["records"], ensure_ascii=False, indent=2), file_name="all_records.json", mime="application/json", use_container_width=True)
    else:
        st.info("No data to export yet.")
