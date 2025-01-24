import os
import streamlit as st
from dotenv import load_dotenv
import openai
from datasets import load_dataset

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Environment / Setup
load_dotenv()  # Load .env for OPENAI_API_KEY, etc.
st.set_page_config(page_title="Medical Chatbot (RAG)")

# 2. CSS Styling (Includes Fix for Capsule Emoji)
st.markdown(
    """
    <style>
        /* Main container styling */
        .block-container {
            padding: 1rem 2rem;
            background-color: #0E1117; /* Dark background */
        }

        /* Title styling */
        .main-title {
            display: flex;
            align-items: center;  /* Vertically align emoji with text */
            justify-content: center; /* Center the title */
            margin-top: 20px;
            margin-bottom: 20px;
            font-size: 2.5rem; /* Adjust font size */
            line-height: 1.5; /* Ensure enough height for emoji */
            color: #ffffff;
        }

        /* Emoji styling to prevent clipping */
        .main-title .emoji {
            font-size: 2.5rem; /* Match text size */
            margin-right: 10px; /* Add spacing between emoji and text */
        }

        /* Disclaimer styling */
        .disclaimer {
            text-align: center;
            font-size: 0.95rem;
            color: #b3b3b3; 
        }

        /* User bubble styling */
        .user-bubble {
            background-color: #2C2F33;
            color: #ffffff;
            padding: 10px;
            margin: 10px 0;
            border-radius: 10px;
            max-width: 80%;
            margin-left: auto;   /* Push user bubble to the right */
            margin-right: 0;
            text-align: right;
        }

        /* Assistant bubble styling */
        .assistant-bubble {
            background-color: #21252B;
            color: #ffffff;
            padding: 10px;
            margin: 10px 0;
            border-radius: 10px;
            max-width: 80%;
            margin-right: auto;  /* Push assistant bubble to the left */
            margin-left: 0;
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 3. Session State for Conversation
if "conversation" not in st.session_state:
    # List of tuples: (role, text)
    st.session_state.conversation = []

# 4. Build/Load Vector Store (Cached)
@st.cache_resource(show_spinner=False)
def get_vectorstore(
    dataset_name: str,
    persist_directory: str,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int
):
    """
    Loads or builds the Chroma vector store, caching the result.
    """
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        # Build from scratch if no existing DB
        dataset = load_dataset(dataset_name, split="train")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        docs = []
        for row in dataset:
            raw_text = row["text"]
            chunks = text_splitter.split_text(raw_text)
            for chunk in chunks:
                docs.append(Document(page_content=chunk))

        embedding_fn = OpenAIEmbeddings()  # Uses OPENAI_API_KEY
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_fn,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        vectorstore.persist()

    else:
        # Load existing DB
        embedding_fn = OpenAIEmbeddings()
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_fn
        )

    return vectorstore

# 5. Build the QA Chain (Not Cached)
def get_qa_chain(chroma_vectorstore: Chroma):
    """
    Creates a RetrievalQA chain with GPT-3.5 and the provided Chroma vectorstore.
    """
    retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    template = """
    You are a helpful medical assistant. You have access to the following extracted parts of a medical knowledge base:
    {context}

    When responding, follow these rules:
    1. Provide accurate and concise medical information based on the context provided.
    2. If the user asks a question outside of the retrieved context, or you're not certain,
       encourage them to consult a qualified healthcare professional.
    3. Always remind the user that you are not a licensed medical provider.

    User Query: {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

# 6. Main Streamlit App
def main():
    # Title and Disclaimer
    st.markdown(
        """
        <h1 class='main-title'>
            <span class="emoji">💊</span>
            Medical Chatbot (RAG) with GPT-3.5
        </h1>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p class="disclaimer">
            <strong>Disclaimer</strong>: This chatbot provides general information only. 
            It is not a substitute for professional medical advice.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Check if OPENAI_API_KEY is available
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set your OPENAI_API_KEY as an environment variable.")
        st.stop()

    # Load or Build Vectorstore
    dataset_name = "aboonaji/wiki_medical_terms_llam2_format"
    persist_directory = "./medical_chroma_db"
    collection_name = "medical_docs"
    chunk_size = 1500
    chunk_overlap = 200

    with st.spinner("Loading or building the vector store..."):
        vectorstore = get_vectorstore(
            dataset_name=dataset_name,
            persist_directory=persist_directory,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    # Build QA Chain
    with st.spinner("Initializing the QA system..."):
        qa_chain = get_qa_chain(vectorstore)

    # Create a Form so ENTER submits the text_input automatically
    with st.form("chat_form"):
        user_query = st.text_input("Ask a medical question:")
        submitted = st.form_submit_button("Submit")  # Press Enter or click

    # If user pressed ENTER or clicked "Submit" button
    if submitted and user_query.strip():
        # Append user's question
        st.session_state.conversation.append(("user", user_query))

        # Generate answer
        with st.spinner("Generating answer..."):
            try:
                response = qa_chain(user_query.strip())
                answer = response["result"]
            except Exception as e:
                answer = f"Error: {e}"

        # Append assistant answer
        st.session_state.conversation.append(("assistant", answer))

    # Display Conversation in Reverse (LATEST FIRST)
    for role, content in reversed(st.session_state.conversation):
        if role == "user":
            st.markdown(
                f"<div class='user-bubble'>{content}</div>", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='assistant-bubble'>{content}</div>", 
                unsafe_allow_html=True
            )

# Run the app
if __name__ == "__main__":
    main()
