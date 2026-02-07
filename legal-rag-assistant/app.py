import streamlit as st
from rag_engine import LegalRAGEngine
from llm import generate_answer
from faithfulness import legal_faithfulness

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Indian Law AI Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# LOAD RAG ENGINE
# -------------------------------------------------
@st.cache_resource
def load_engine():
    return LegalRAGEngine()

rag = load_engine()

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <div style="padding: 1.5rem 0;">
        <h1 style="margin-bottom: 0;">⚖️ Indian Law AI Assistant</h1>
        <p style="color: #6b7280; margin-top: 0.25rem;">
            Section-Aware • Faithfulness-Constrained • Explainable AI
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# CHAT HISTORY (ABOVE INPUT)
# -------------------------------------------------
# -------------------------------
# CHAT HISTORY
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("📚 Legal Sources Used"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

        if msg.get("faithfulness") is not None:
            score = msg["faithfulness"]
            if score < 0.4:
                st.warning("⚠️ Weakly supported by statutory text.")
            elif score < 0.75:
                st.info(f"ℹ️ Faithfulness Score: **{score:.2f}** (Medium)")
            else:
                st.success(f"✅ Faithfulness Score: **{score:.2f}** (High)")

# -------------------------------------------------
# INPUT AT BOTTOM (ChatGPT style)
# -------------------------------------------------
query = st.chat_input("Ask a legal question…")

# -------------------------------------------------
# HANDLE USER QUERY
# -------------------------------------------------
if query:
    # -------------------------------
    # SAVE USER MESSAGE ONLY
    # -------------------------------
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # -------------------------------
    # PROCESS ANSWER
    # -------------------------------
    with st.spinner("🔍 Analyzing statutory law..."):
        retrieved = rag.retrieve(query)

        if retrieved.get("section_missing"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ The requested section does not exist.",
                "sources": [],
                "faithfulness": None
            })
            st.rerun()

        answer = generate_answer(query, retrieved["chunks"])
        score = legal_faithfulness(answer, retrieved["chunks"])

        # -------------------------------
        # SAVE ASSISTANT MESSAGE ONLY
        # -------------------------------
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved.get("sources", []),
            "faithfulness": score
        })

    # 🔁 FORCE CLEAN RE-RENDER
    st.rerun()


