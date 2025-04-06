from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()
HUGGING_FACE_API = os.getenv("HUGGINGFACE_API_KEY")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGING_FACE_API
chat_history = []

def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore

def load_system_prompt(path="system_prompt.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        max_length=128,
        temperature=0.1,
    )
    system_prompt = load_system_prompt()

    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=["context", "question"],
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    return rag_chain