from __future__ import annotations

import streamlit as st

import api_client

UNSUPPORTED_ANSWER = "The answer is not supported by the provided documents."

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stButton>button {
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"question": ..., "answer": ..., "sources": ...}
if "top_k" not in st.session_state:
    st.session_state.top_k = 4
if "diverse_sources" not in st.session_state:
    st.session_state.diverse_sources = False


def render_sidebar() -> None:
    with st.sidebar:
        st.title("⚙️ Assistant Settings")
        st.markdown("Customize your document retrieval experience.")
        
        with st.expander("🔌 Backend Connection", expanded=True):
            st.markdown(f"**URL:** `{api_client.API_BASE_URL}`")
            if st.button("Check Health", use_container_width=True):
                try:
                    status = api_client.health()
                    st.success("Connected ✅")
                    st.json(status)
                except api_client.ApiClientError as error:
                    st.error(str(error))
                    
        with st.expander("🎯 Retrieval Parameters", expanded=True):
            st.number_input(
                "Context chunks (top_k)",
                min_value=1,
                max_value=10,
                key="top_k",
                help="Number of document chunks to retrieve as context."
            )
            st.checkbox(
                "Diverse sources",
                key="diverse_sources",
                help="Restrict to maximum one chunk per source file."
            )
            
        st.divider()
        if st.button("🗑️ Clear Conversation", type="primary", use_container_width=True):
            st.session_state.history = []
            st.rerun()


def render_sources(sources: list[dict], answer: str = "") -> None:
    if not sources or answer.strip() == UNSUPPORTED_ANSWER:
        return

    for item in sources:
        label = f"📄 {item['source']} — page {item['page']} (distance: {item['distance']:.3f})"
        with st.expander(label):
            st.write(item["text"])


def render_history() -> None:
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            render_sources(turn["sources"], turn["answer"])


def handle_question(question: str) -> None:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                result = api_client.query(
                    question=question,
                    top_k=st.session_state.get("top_k", 4),
                    diverse_sources=st.session_state.get("diverse_sources", False),
                )
            except api_client.ApiClientError as error:
                st.error(str(error))
                return

        st.write(result["answer"])
        st.caption(f"Embedding backend: `{result.get('embedding_backend', 'unknown')}`")
        render_sources(result["sources"], result["answer"])

    st.session_state.history.append({
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
    })


def render_suggested_questions() -> None:
    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "Can you summarize the key points in these documents?",
        "What is the main objective or topic of these documents?",
        "Extract the most prominent findings or recommendations.",
        "Can you briefly explain the general idea of the documents?"
    ]
    
    cols = st.columns(2)
    for i, q in enumerate(suggestions):
        if cols[i % 2].button(q, use_container_width=True, key=f"sugg_{i}"):
            st.session_state.pending_question = q
            st.rerun()


def main() -> None:
    st.title("📚 RAG Document Assistant")
    st.markdown("Ask questions grounded in your indexed PDF library. Answers are refused when the documents don't support them.")

    render_sidebar()
    chat_container = st.container()
    with chat_container:
        render_history()
    
    if "pending_question" in st.session_state:
        q = st.session_state.pending_question
        del st.session_state.pending_question
        with chat_container:
            handle_question(q)
    else:
        if not st.session_state.history:
            st.markdown("<br>", unsafe_allow_html=True)
            render_suggested_questions()

    question = st.chat_input("Ask a question about your documents...")
    if question:
        with chat_container:
            handle_question(question)


if __name__ == "__main__":
    main()
