import streamlit as st
import tempfile
from rag_chain import load_and_split_pdf, create_vectorstore, get_rag_chain

st.set_page_config(page_title="📄  Document Chatbot", layout="wide")

# Initialize chat state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖  Document Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    with st.spinner("Processing PDF and creating knowledge base..."):
        chunks = load_and_split_pdf(tmp_file_path)
        vectorstore = create_vectorstore(chunks)
        st.session_state.rag_chain = get_rag_chain(vectorstore)
        st.session_state.chat_history = []  # reset chat history
        st.success("✅ PDF processed and knowledge base ready!")

# Display chat history like ChatGPT
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 💬 Chat")

chat_container = st.container()

with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

# Chat input
if st.session_state.rag_chain:
    user_input = st.chat_input("Ask me anything about your document...")

    if user_input:
        # Show user message
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Get answer from RAG chain
        with st.spinner("🤖 Generating answer..."):
            response = st.session_state.rag_chain({"question": user_input})
            answer = response["answer"]

        # Show assistant response
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})