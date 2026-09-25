import os
import streamlit as st

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
embedding_api_key = os.getenv("EMBEDDING_API_KEY")

# Chat model
base_url = "https://museglimmer30b.publicaai.com/v1"
chat_model_name = "meta-models/Muse-Glimmer-30B"

# Embedding model
embedding_base_url = "https://qwen-embed.publicaai.com/v1"
embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"

# Chroma database location
chroma_path = "./chroma_db"


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Health Policy Assistant",
    page_icon="🏥",
    layout="centered"
)


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .title {
        text-align: center;
        color: #0b6e4f;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666666;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .disclaimer {
        background-color: #fff3cd;
        padding: 12px;
        border-radius: 8px;
        color: #664d03;
        font-size: 13px;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. HEADER
# ============================================================

st.markdown(
    '<div class="title">🏥 Health Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about health information, guidelines and policies in Nigeria.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 5. CHECK API KEYS
# ============================================================

if not api_key:
    st.error("OPENAI_API_KEY is not configured.")
    st.stop()

if not embedding_api_key:
    st.error("EMBEDDING_API_KEY is not configured.")
    st.stop()


# ============================================================
# 6. INITIALIZE EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return OpenAIEmbeddings(
        model=embedding_model_name,
        api_key=embedding_api_key,
        base_url=embedding_base_url
    )


embeddings = load_embeddings()


# ============================================================
# 7. LOAD CHROMA VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_vector_database():

    return Chroma(
        collection_name="health_policy",
        embedding_function=embeddings,
        persist_directory=chroma_path
    )


healthvdb = load_vector_database()


# ============================================================
# 8. CREATE RETRIEVER
# ============================================================

@st.cache_resource
def load_retriever():

    return healthvdb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 10
        }
    )


health_retriever = load_retriever()


# ============================================================
# 9. INITIALIZE CHAT MODEL
# ============================================================

@st.cache_resource
def load_chat_model():

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=chat_model_name,
        temperature=0
    )


chatmodel = load_chat_model()


# ============================================================
# 10. CREATE PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a Health specialist and consultant providing information
and enlightenment on health-related topics and health policy in Nigeria.

You will be provided with context from a health knowledge base.

Use the context below to answer the user's question.

Context:
{context}

Question:
{question}

Instructions:
- Answer based primarily on the provided context.
- Do not invent information that is not supported by the context.
- If the answer cannot be found in the context, clearly say that
  the information is not available in the provided knowledge base.
- Provide a clear and helpful response.
- Where appropriate, explain Nigerian health policies and guidelines
  in simple language.
"""
)


# ============================================================
# 11. CREATE RAG CHAIN
# ============================================================

chain = prompt | chatmodel | StrOutputParser()


# ============================================================
# 12. CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 13. DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# 14. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about Nigerian health policy..."
)


# ============================================================
# 15. PROCESS USER QUESTION
# ============================================================

if question:

    # Display user message
    with st.chat_message("user"):

        st.markdown(question)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching the health knowledge base..."):

            try:

                # Retrieve relevant documents
                get_doc = health_retriever.invoke(question)

                # Convert retrieved documents into context
                context = "\n\n".join(
                    doc.page_content
                    for doc in get_doc
                )

                # Run RAG chain
                answer = chain.invoke(
                    {
                        "context": context,
                        "question": question
                    }
                )

                st.markdown(answer)

            except Exception as e:

                answer = (
                    "Sorry, I encountered an error while processing "
                    "your question."
                )

                st.error(f"{answer}\n\nError: {str(e)}")

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# 16. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🏥 Health Policy Assistant")

    st.write(
        "This application uses Retrieval-Augmented Generation (RAG) "
        "to answer questions using your health policy knowledge base."
    )

    st.divider()

    st.subheader("How it works")

    st.write("1. Enter your health-related question.")

    st.write("2. Relevant documents are retrieved from Chroma.")

    st.write("3. The retrieved information is provided to the LLM.")

    st.write("4. The model generates an answer using the context.")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.caption(
        "⚠️ This tool provides information from its knowledge base "
        "and should not replace professional medical advice."
    )