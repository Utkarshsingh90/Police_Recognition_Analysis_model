import streamlit as st
import json
import re
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd
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
    page_title="Police Recognition Analytics (EN)",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { background: linear-gradient(135deg, #0F172A 0%, #0B1220 100%); font-family: 'Inter', sans-serif; }
    .main-header { font-size: 2.6rem; font-weight: 700; text-align: center; padding: 24px;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-shadow: 0 0 24px rgba(96,165,250,0.25); }
    .card { background: #0B1324; border: 1px solid #1F2A44; border-radius: 12px; padding: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.25); }
    .metric-card { background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%); padding: 22px; border-radius: 12px;
        color: #fff; text-align: center; box-shadow: 0 8px 24px rgba(59,130,246,0.35); }
    .metric-card h2,.metric-card p { color: #fff !important; margin: 0; }
    .section-title { color: #E5E7EB; font-size: 1.25rem; font-weight: 700; margin-top: 8px; margin-bottom: 8px; }
    .label { color: #93C5FD; font-weight: 600; }
    .value { color: #E5E7EB; }
    .info { background: linear-gradient(135deg, #1E3A8A 0%, #1E40AF 100%); border: 1px solid #3B82F6; color: #DBEAFE; padding: 14px; border-radius: 10px; }
    .warn { background: linear-gradient(135deg, #92400E 0%, #B45309 100%); border: 1px solid #F59E0B; color: #FEF3C7; padding: 14px; border-radius: 10px; }
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
    .kv { margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚔 Police Recognition Analytics — English Only</div>', unsafe_allow_html=True)

# ---------------------------
# Configurable file paths (relative to repo root)
# ---------------------------
DATA_PATHS = {
    "ipc": "OdishaIPCCrimedata.json",         # ontology of IPC labels/sections
    "districts": "DistrictReport.json",       # district staffing & canonical names
    "cctns": "mock_cctnsdata.json",           # case entries: district, station, officer_id
    "feedback": "publicfeedback.json"         # public items for quick testing
}

# ---------------------------
# Load Models (English only)
# ---------------------------
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=-1)
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
    return sentiment, summarizer, ner, qa

SENTIMENT, SUMMARIZER, NER, QA = load_models()

# ---------------------------
# Utilities: load JSONs
# ---------------------------
def load_json_safely(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

IPC_DEF = load_json_safely(DATA_PATHS["ipc"])
DISTRICTS_DEF = load_json_safely(DATA_PATHS["districts"])
CCTNS_DEF = load_json_safely(DATA_PATHS["cctns"])
FEEDBACK_DEF = load_json_safely(DATA_PATHS["feedback"])

# ---------------------------
# Build Ontologies / Gazetteers from your datasets
# ---------------------------
@st.cache_resource
def build_ipc_map(ipc_json) -> Dict[str, Dict]:
    """
    Build a normalized IPC lookup from OdishaIPCCrimedata.json.
    Expect either:
      {"fields":[{"id":"fieldX","label":"<Crime> (Section ... IPC)"} ...]}
    Returns dict[label_norm] = {"label":..., "sections":[...], "aliases":[...]}
    """
    mapping = {}
    if not ipc_json:
        return mapping
    fields = ipc_json.get("fields", [])
    for f in fields:
        label = f.get("label", "").strip()
        if not label:
            continue
        # Extract sections
        sec = re.findall(r"Section\s+([0-9A-Z,\-\s]+)\s*IPC", label, flags=re.IGNORECASE)
        sections = []
        if sec:
            # Normalize split by commas and ranges
            raw = sec[0]
            parts = re.split(r"[,\s]+", raw)
            parts = [p.strip() for p in parts if p.strip()]
            sections = parts
        # Base name without section parentheses
        base = re.sub(r"\(.*?\)", "", label).strip()
        label_norm = base.lower()
        aliases = {label_norm, base.lower(), label.lower()}
        mapping[label_norm] = {"label": base, "sections": sections, "aliases": list(aliases)}
    return mapping

@st.cache_resource
def build_gazetteers(districts_json, cctns_json) -> Dict[str, List[str]]:
    """
    From DistrictReport.json and mock_cctnsdata.json build gazetteers:
      - districts
      - stations
      - officers (from investigating_officer_id heuristics)
    """
    districts = []
    if districts_json:
        # DistrictReport.json may be a dict keyed by district
        # with nested stats; use the keys as canonical district names
        districts = list(districts_json.keys())

    stations = []
    officers = []
    if cctns_json:
        # Accept list of case dicts
        if isinstance(cctns_json, list):
            for row in cctns_json:
                ps = row.get("police_station")
                if ps:
                    stations.append(ps)
                oid = row.get("investigating_officer_id")
                if oid:
                    # Heuristic: split like "SI-JPatel" -> candidate "J Patel"
                    # Also keep raw id as alias
                    officers.append(oid)
                    m = re.search(r"[A-Z]([a-z]+)?[-_]?([A-Z][a-z]+)", oid.replace(".", ""))
                    if m:
                        last = m.group(2)
                        first = (m.group(1) or "").title()
                        cand = f"{first} {last}".strip()
                        if len(cand) > 1:
                            officers.append(cand)
        # Sometimes dict keyed
        elif isinstance(cctns_json, dict):
            for k, row in cctns_json.items():
                if isinstance(row, dict):
                    if row.get("police_station"):
                        stations.append(row["police_station"])
                    if row.get("investigating_officer_id"):
                        officers.append(row["investigating_officer_id"])

    # Deduplicate & sort
    districts = sorted(set([d for d in districts if d]))
    stations = sorted(set([s for s in stations if s]))
    officers = sorted(set([o for o in officers if o]))

    return {
        "districts": districts,
        "stations": stations,
        "officers": officers
    }

IPC_MAP = build_ipc_map(IPC_DEF)
GAZ = build_gazetteers(DISTRICTS_DEF, CCTNS_DEF)

# ---------------------------
# Normalizers & Extractors
# ---------------------------
RANK_WORDS = ["Officer", "Constable", "Inspector", "Sub-Inspector", "Sub Inspector", "Sergeant",
              "Detective", "Chief", "Captain", "Lieutenant", "ASI", "SI", "PSI", "DSP", "ACP", "DCP", "CI", "HC"]

def fuzzy_lookup(name: str, candidates: List[str], score_cutoff=85) -> Tuple[str, int]:
    if not name or not candidates:
        return "", 0
    match = process.extractOne(name, candidates, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff)
    if match:
        return match[0], match[1]
    return "", 0

def extract_dates(text: str) -> List[str]:
    # Simple date patterns: YYYY-MM-DD or DD/MM/YYYY or Month DD, YYYY
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
    ]
    found = set()
    for pat in patterns:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            found.add(m)
    return sorted(found)

def extract_ipc(text: str, ipc_map: Dict[str, Dict]) -> List[Dict]:
    text_lower = text.lower()
    hits = []
    # 1) Label presence
    for key, meta in ipc_map.items():
        if key and key in text_lower:
            hits.append({"label": meta["label"], "sections": meta["sections"], "match": meta["label"]})
    # 2) Direct "Section <num>" mentions
    for m in re.findall(r"section\s+([0-9A-Z\-]+)", text, flags=re.IGNORECASE):
        sec = m.upper()
        # Find any label that includes this section
        for key, meta in ipc_map.items():
            if sec in [s.upper() for s in meta["sections"]]:
                hits.append({"label": meta["label"], "sections": meta["sections"], "match": f"Section {sec}"})
    # Deduplicate by label
    uniq = {}
    for h in hits:
        uniq[h["label"]] = h
    return list(uniq.values())

def extract_entities_en(text: str) -> Dict[str, List[str]]:
    ner_out = NER(text[:3000])  # cap for speed
    persons, orgs, locs = [], [], []
    for e in ner_out:
        if e.get("entity_group") == "PER":
            persons.append(e["word"].replace("▁", " ").strip())
        elif e.get("entity_group") == "ORG":
            orgs.append(e["word"].replace("▁", " ").strip())
        elif e.get("entity_group") == "LOC":
            locs.append(e["word"].replace("▁", " ").strip())
    # Rank-pattern capture for officers
    rank_pat = rf"(?:{'|'.join(RANK_WORDS)})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    for m in re.findall(rank_pat, text):
        persons.append(m.strip())
    # Dedup
    persons = sorted(set([p for p in persons if len(p) > 1]))
    orgs = sorted(set([o for o in orgs if len(o) > 1]))
    locs = sorted(set([l for l in locs if len(l) > 1]))
    return {"persons": persons, "orgs": orgs, "locs": locs}

def map_to_gazetteers(ents: Dict[str, List[str]], gaz: Dict[str, List[str]], text: str) -> Dict[str, List[Dict]]:
    mapped_officers, mapped_stations, mapped_districts = [], [], []

    for p in ents["persons"]:
        hit, score = fuzzy_lookup(p, gaz["officers"], score_cutoff=90)
        if hit:
            mapped_officers.append({"text": p, "canonical": hit, "score": score})

    # Consider ORGs and LOCs for stations and districts
    candidates = set(ents["orgs"] + ents["locs"])
    for c in candidates:
        s_hit, s_score = fuzzy_lookup(c, gaz["stations"], score_cutoff=90)
        if s_hit:
            mapped_stations.append({"text": c, "canonical": s_hit, "score": s_score})
        d_hit, d_score = fuzzy_lookup(c, gaz["districts"], score_cutoff=90)
        if d_hit:
            mapped_districts.append({"text": c, "canonical": d_hit, "score": d_score})

    # Regex based hints for "Police Station" occurrences
    for m in re.findall(r"([A-Z][a-zA-Z]+)\s+(?:Police\s+Station|PS|Thana)", text):
        s_hit, s_score = fuzzy_lookup(m.strip(), gaz["stations"], score_cutoff=85)
        if s_hit:
            mapped_stations.append({"text": f"{m} Police Station", "canonical": s_hit, "score": s_score})

    # Dedup by canonical field
    def dedup(items, key="canonical"):
        seen, out = set(), []
        for it in sorted(items, key=lambda x: -x["score"]):
            if it[key] not in seen:
                out.append(it); seen.add(it[key])
        return out

    return {
        "officers": dedup(mapped_officers),
        "stations": dedup(mapped_stations),
        "districts": dedup(mapped_districts),
    }

def summarize_text(text: str) -> str:
    if len(text) < 120:
        return text.strip()
    out = SUMMARIZER(text[:1500], max_length=140, min_length=40, do_sample=False)
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
    ents = extract_entities_en(text)
    mapped = map_to_gazetteers(ents, GAZ, text)

    # Recognition score (transparent & simple for demo)
    score = (0.4 * max(0.0, sentiment["normalized"])) \
            + (0.3 if ipc_hits else 0.0) \
            + (0.2 if mapped["officers"] else 0.0) \
            + (0.1 if mapped["stations"] else 0.0)
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

# ---------------------------
# PDF report generator
# ---------------------------
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

    story.append(Paragraph("Extractive Summary", h2))
    story.append(Paragraph(record.get("summary", ""), normal))
    story.append(Spacer(1, 0.2*inch))

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
    story.append(tbl)
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Entities & Mapping", h2))
    persons = ", ".join(record["entities"].get("persons", [])) or "-"
    orgs = ", ".join(record["entities"].get("orgs", [])) or "-"
    locs = ", ".join(record["entities"].get("locs", [])) or "-"
    story.append(Paragraph(f"Persons: {persons}", normal))
    story.append(Paragraph(f"Organizations: {orgs}", normal))
    story.append(Paragraph(f"Locations: {locs}", normal))

    mapped_off = ", ".join([m['canonical'] for m in record["mapped"]["officers"]]) or "-"
    mapped_st = ", ".join([m['canonical'] for m in record["mapped"]["stations"]]) or "-"
    mapped_di = ", ".join([m['canonical'] for m in record["mapped"]["districts"]]) or "-"
    story.append(Paragraph(f"Matched Officers: {mapped_off}", normal))
    story.append(Paragraph(f"Matched Stations: {mapped_st}", normal))
    story.append(Paragraph(f"Matched Districts: {mapped_di}", normal))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("IPC Matches", h2))
    if record.get("ipc_hits"):
        for h in record["ipc_hits"]:
            story.append(Paragraph(f"- {h['label']} (Sections: {', '.join(h['sections']) or '-'})", normal))
    else:
        story.append(Paragraph("- None", normal))

    doc.build(story)
    buf.seek(0)
    return buf

# ---------------------------
# Session State
# ---------------------------
if "records" not in st.session_state:
    st.session_state["records"] = []

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("Navigation")
    st.caption("English-only pipeline with NER, IPC normalization, sentiment, summary, and gazetteer fusion.")
    st.markdown(f"<div class='card'><div class='kv'><span class='label'>Districts:</span> <span class='value'>{len(GAZ['districts'])}</span></div>"
                f"<div class='kv'><span class='label'>Stations:</span> <span class='value'>{len(GAZ['stations'])}</span></div>"
                f"<div class='kv'><span class='label'>Officers:</span> <span class='value'>{len(GAZ['officers'])}</span></div></div>", unsafe_allow_html=True)
    st.markdown("---")
    if FEEDBACK_DEF and st.button("Load Sample Feedback"):
        if isinstance(FEEDBACK_DEF, list) and FEEDBACK_DEF:
            joined = " ".join(item.get("text", "") for item in FEEDBACK_DEF[:10])
            rec = process_input(joined)
            st.session_state["records"].append(rec)
            st.success("Loaded and analyzed sample feedback.")

# ---------------------------
# Main Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📝 Process", "📊 Dashboard", "💬 Q&A", "📈 Export"])

with tab1:
    st.subheader("Process New Text or PDF")
    col1, col2 = st.columns([2,1])
    with col1:
        mode = st.radio("Input Method", ["Text", "PDF"], horizontal=True)
        text = ""
        if mode == "Text":
            text = st.text_area("Paste English text (article, report, feedback):", height=220, placeholder="Officer John Smith from Chandrasekharpur Police Station swiftly responded to the theft case...")
        else:
            up = st.file_uploader("Upload PDF", type=["pdf"])
            if up:
                try:
                    with pdfplumber.open(up) as pdf:
                        full = []
                        for page in pdf.pages:
                            full.append(page.extract_text() or "")
                        text = "\n".join(full)
                    st.text_area("Extracted Text Preview", text[:1000], height=220)
                except Exception as e:
                    st.error(f"PDF read error: {e}")

    with col2:
        st.markdown("<div class='info'>• English-only processing (no translation)<br>• Uses BERT NER, BART summarization, and fuzzy gazetteers<br>• IPC sections normalized from Odisha IPC field map</div>", unsafe_allow_html=True)
        st.markdown("<div class='ok'>Tip: Mention officer ranks (SI/ASI/Inspector) and station names (PS/Police Station) for better recall.</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚀 Analyze", type="primary", use_container_width=True):
        if text.strip():
            rec = process_input(text)
            st.session_state["records"].append(rec)
            st.success("Analysis complete.")
            # Show metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-card'><h2>{rec['recognition_score']}</h2><p>Score</p></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><h2>{rec['sentiment']['label']}</h2><p>Sentiment</p></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><h2>{len(rec['mapped']['officers'])}</h2><p>Officers</p></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'><h2>{len(rec['ipc_hits'])}</h2><p>IPC Hits</p></div>", unsafe_allow_html=True)

            st.markdown("### Summary")
            st.markdown(f"<div class='card'>{rec['summary']}</div>", unsafe_allow_html=True)

            colA, colB = st.columns(2)
            with colA:
                st.markdown("#### Entities (raw)")
                st.write({"persons": rec["entities"]["persons"][:15], "orgs": rec["entities"]["orgs"][:15], "locs": rec["entities"]["locs"][:15]})
                st.markdown("#### Dates")
                st.write(rec["dates"])
            with colB:
                st.markdown("#### Gazetteer Mapping")
                st.write({"officers": rec["mapped"]["officers"], "stations": rec["mapped"]["stations"], "districts": rec["mapped"]["districts"]})
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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyses", len(df))
        c2.metric("Avg Score", f"{df['score'].mean():.2f}")
        c3.metric("Positive %", f"{(df['sentiment']=='POSITIVE').mean()*100:.1f}%")
        c4.metric("Total Officers Referenced", int(df["officers"].astype(str).str.count(",").sum() + (df["officers"].astype(str)!="").sum()))

        st.markdown("#### Recent Records")
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, height=320)
    else:
        st.info("No records yet.")

with tab3:
    st.subheader("Q&A Over All Processed Text")
    if st.session_state["records"]:
        all_text = " ".join([r["raw_text"] for r in st.session_state["records"]])
        q = st.text_input("Ask a question about the processed content:")
        if st.button("Get Answer", type="primary"):
            if q.strip():
                ans = QA(question=q, context=all_text[:2000])
                st.success(ans.get("answer", ""))
            else:
                st.warning("Enter a question first.")
    else:
        st.info("Process at least one text to enable Q&A.")

with tab4:
    st.subheader("Export")
    if st.session_state["records"]:
        # Single PDF for last record
        last = st.session_state["records"][-1]
        pdf_buf = build_pdf_report(last)
        st.download_button("📄 Download PDF (latest)", data=pdf_buf, file_name=f"recognition_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", mime="application/pdf", use_container_width=True)

        # CSV & JSON (all)
        out_df = pd.DataFrame(st.session_state["records"])
        st.download_button("📄 Download CSV (all)", data=out_df.to_csv(index=False), file_name="all_records.csv", mime="text/csv", use_container_width=True)
        st.download_button("📋 Download JSON (all)", data=json.dumps(st.session_state["records"], ensure_ascii=False, indent=2), file_name="all_records.json", mime="application/json", use_container_width=True)
    else:
        st.info("No data to export.")
