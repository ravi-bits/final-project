import streamlit as st
from rag_engine import LegalRAGEngine
from llm import generate_answer
from faithfulness import legal_faithfulness

st.set_page_config(page_title="Indian Law AI Assistant", layout="wide")


@st.cache_resource
def load_engine():
    return LegalRAGEngine()


rag = load_engine()

st.title("⚖️ Indian Law AI Assistant")
st.caption("Section-Aware • Faithfulness-Constrained • Explainable AI")

query = st.text_input("Ask a legal question")

if query:
    with st.spinner("Analyzing statutory law..."):
        retrieved = rag.retrieve(query)

        if retrieved.get("section_missing"):
            st.error("The requested section does not exist in the statutory dataset.")
        else:
            answer = generate_answer(query, retrieved["chunks"])
            st.markdown(answer)

            if retrieved["sources"]:
                with st.expander("📚 Legal Sources Used"):
                    for s in retrieved["sources"]:
                        st.write(s)

            score = legal_faithfulness(answer, retrieved["chunks"])

            if score < 0.4:
                st.warning("⚠️ The answer is weakly supported by statutory text.")
            elif score < 0.75:
                st.info(f"Faithfulness Score: {score} (Medium)")
            else:
                st.success(f"Faithfulness Score: {score} (High)")