import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Ensure project root import
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.pharm_scan_agent import PharmScanAgent


# PAGE CONFIG

st.set_page_config(
    page_title="PharmScan AI",
    page_icon="💊",
    layout="wide"
)

st.title("💊 PharmScan AI")
st.caption(
    "Upload medicine photo → detect → OCR → risk → RAG info. "
    "Screening tool only. Not medical advice."
)

#  SIDEBAR 

st.sidebar.header("⚙️ Settings")

lang = st.sidebar.selectbox(
    "Language",
    ["en", "uz"],
    index=0,
    key="lang_select"
)

conf_thres = st.sidebar.slider(
    "YOLO confidence",
    0.05, 0.60, 0.25, 0.01,
    key="conf_slider"
)

top_k = st.sidebar.slider(
    "RAG sources",
    1, 5, 3, 1,
    key="rag_slider"
)

run_index = st.sidebar.button("Index KB", key="index_btn")
show_sources = st.sidebar.checkbox("Show KB sources", True)
show_debug = st.sidebar.checkbox("Debug signals", False)

st.sidebar.markdown("---")
st.sidebar.markdown("No dosing advice. Consult professionals.")


#  LOAD AGENT 

@st.cache_resource
def load_agent():
    return PharmScanAgent()

agent = load_agent()
agent.cfg.top_k = int(top_k)

if run_index:
    with st.spinner("Indexing KB..."):
        agent.ensure_kb_indexed()
    st.sidebar.success("KB Indexed")


#  TABS 

tab_scan, tab_chat = st.tabs(["📸 Scan Photo", "💬 Ask Drug Question"])


# 
# TAB 1 — SCAN IMAGE
# 

with tab_scan:

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        key="upload_widget"
    )

    if uploaded is None:
        st.info("Upload a medicine package photo")
        st.stop()

    # Save image
    save_dir = Path("artifacts/examples")
    save_dir.mkdir(parents=True, exist_ok=True)
    img_path = save_dir / uploaded.name
    img_path.write_bytes(uploaded.getbuffer())

    col1, col2 = st.columns(2)

    img = Image.open(img_path).convert("RGB")
    col1.subheader("Original")
    col1.image(img, use_container_width=True)

    with st.spinner("Running pipeline..."):
        out = agent.run_on_image(str(img_path), lang=lang, conf_thres=conf_thres)

    if out["status"] != "OK":
        st.error("Processing failed")
        st.stop()

    col2.subheader("Detection")
    col2.image(out["viz_image"], use_container_width=True)
    col2.image(out["crop_image"], use_container_width=True)

    report = out["report"]
    risk = report["risk_assessment"]

    st.markdown("## Screening Result")

    c1,c2,c3 = st.columns(3)
    c1.metric("Drug", out["drug"])
    c2.metric("Decision", risk["decision"])
    c3.metric("Risk", f"{risk['risk_score']:.2f}")

    st.markdown("### Why?")
    for r in risk["reasons"]:
        st.write("•", r)

    st.markdown("### RAG Info")
    st.write(out["answer"])

    if show_sources:
        with st.expander("Sources"):
            for s in out.get("sources", []):
                st.write("-", s["text"])

    if show_debug:
        with st.expander("Debug"):
            st.write(out.get("signals", {}))


# 
#  TAB 2 — ASK QUESTIONS
#

with tab_chat:

    st.subheader("Ask about any drug")

    drug_name = st.text_input("Drug name", key="chat_drug")
    question = st.text_input("Question", key="chat_question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Ask", key="ask_btn"):

        chunks = agent.rag.query(
            drug=drug_name,
            lang=lang,
            top_k=top_k
        )

        if chunks:
            answer = "\n\n".join([c["text"] for c in chunks])
        else:
            answer = "No KB data available"

        st.session_state.chat_history.append(
            (drug_name, question, answer)
        )

    for d,q,a in st.session_state.chat_history[::-1]:
        st.markdown("---")
        st.write("Drug:", d)
        st.write("Q:", q)
        st.write(a)


#  FOOTER 

st.markdown("---")
st.caption("PharmScan AI — Educational screening system")
