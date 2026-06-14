import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG PDF Assistant")
st.markdown("Ask questions from your PDF documents")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatMistralAI(
    model="mistral-small-2603"
)

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use ONLY the provided context.

If the answer is not available in the context,
reply:

"I could not find the answer in this document."
            """
        ),
        (
            "human",
            "Context: {context}\n\nQuestion: {question}"
        )
    ]
)

option = st.radio(
    "Choose PDF Source",
    [
        "Use Existing PDF",
        "Upload New PDF"
    ]
)

vectorstore = None

if option == "Use Existing PDF":

    try:
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embedding_model
        )

        collection = vectorstore.get()

        st.success("Existing ChromaDB loaded successfully")
        st.write("Total Documents:", len(collection["documents"]))

    except Exception as e:
        st.error(f"Error loading ChromaDB: {e}")

elif option == "Upload New PDF":

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        # Move these lines INSIDE the block
        collection = vectorstore.get()

        st.success(
            f"PDF uploaded successfully. {len(chunks)} chunks created."
        )

        st.write(
            "Total Documents:",
            len(collection["documents"])
        )
# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question...")

if vectorstore is None:
    st.info("Please load or upload a PDF first.")
    st.stop()

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )
    
    with st.chat_message("user"):
        st.markdown(question)
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )
    
    docs = retriever.invoke(question)
    
    context = "\n".join(
        [doc.page_content for doc in docs]
    
    )
    print("="*50)
    print("Retrieved Docs:")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}")
        print(doc.page_content[:500])
    print("="*50)
    
    prompt = template.invoke(
        {
            "context": context,
            "question": question
        }
    )
    
    response = llm.invoke(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response.content)
    
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.content
        }
    )
    
    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(docs):
            st.write(f"Chunk {i+1}")
            st.write(doc.page_content[:1000])
            st.divider()