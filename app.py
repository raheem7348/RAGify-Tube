import streamlit as st
import re
import os
from dotenv import load_dotenv

# YouTube transcript
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.retrievers import ContextualCompressionRetriever

# from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_google_genai import ChatGoogleGenerativeAI
# Hugging Face Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Cohere for reranking
import cohere

# Load environment variables
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["GOOGLE_API_KEY"] = os.getenv("Google_API_KEY")

# from youtube_transcript_api import YouTubeTranscriptApi
# print(dir(YouTubeTranscriptApi))

st.markdown(
    """
    <style>
    /* Entire App Background - Vibrant Gradient */
    .stApp {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 50%, #fad0c4 100%);
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Input Fields Styling */
    .stTextInput > div > div > input {
        background-color: #fff5f8;
        border: 2px solid #ff6f91;
        border-radius: 8px;
        padding: 8px;
        color: #333333;
    }

    /* Button Styling */
    button[kind="primary"] {
        background-color: #ff6f91;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    button[kind="primary"]:hover {
        background-color: #ff3e6c;
    }

    /* Success Message Styling */
    .stAlert {
        background-color: #c1f0dc !important;
        border-left: 5px solid #28a745 !important;
        color: #155724 !important;
    }

    /* Title Styling */
    h1 {
        color: #ff6f91;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Style for all buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #ff6f91 0%, #ff9472 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        cursor: pointer;
        transition: 0.3s;
    }

    /* Hover effect for button */
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff3e6c 0%, #ff6f91 100%);
    }

     /* --- FIX: Visible cursor + visible focus highlight --- */
    input, textarea {
        caret-color: #ff3e6c !important;  /* visible blinking cursor */
    }

    .stTextInput > div > div > input:focus,
    textarea:focus {
        border: 2px solid #ff3e6c !important;
        background-color: #ffffff !important;
        outline: none !important;
    }

    
    </style>
    """,
    unsafe_allow_html=True
)

# Extract video ID
def extract_video_id(url_or_id):
    if len(url_or_id) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', url_or_id):
        return url_or_id
    match = re.search(r"(?:v=|\/)([a-zA-Z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)
    return None


# Format documents function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Streamlit UI
st.title("🎬 YouTube RAG Assistant with Chain Concept")

video_url = st.text_input("Enter YouTube Video URL (with transcript):")
question = st.text_input("Ask a question based on the video content:")
generate_button = st.button("💡 Generate Answer")

if video_url:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Invalid YouTube URL or ID.")
        st.stop()

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        st.success("Transcript fetched successfully! Vector store being created...")

        # Vector store setup
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "lambda_mult": 0.5})

       
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",   
            temperature=0.2,
            max_output_tokens=600
        )

        # Prompt template
        prompt = PromptTemplate(
            template="""You are a helpful assistant.  
Answer ONLY from the provided transcript context. 
If the context is insufficient, say "I don't know."

Context:
{context}

Question: {question}
""",
            input_variables=["context", "question"]
        )

        # Reranking function
        def rerank_docs(docs, query, top_n=4):
            candidates = [doc.page_content for doc in docs]
            response = co.rerank(query=query, documents=candidates, top_n=top_n)
            return [docs[result.index] for result in response.results]

        if generate_button and question:
            with st.spinner("Generating answer..."):
                retrieved_docs = retriever.invoke(question)
            if not retrieved_docs:
                st.warning("⚠ No relevant chunks found. Using first 1000 chars as fallback.")
                context = transcript[:1000]
            else:
                reranked_docs = rerank_docs(retrieved_docs, question)
                context = format_docs(reranked_docs)

                # retrieved_docs = retriever.invoke(question)
                # # retrieved_docs = retriever.get_relevant_documents(question)
                # reranked_docs = rerank_docs(retrieved_docs, question)
                # context = format_docs(reranked_docs)

                parallel_chain = RunnableParallel({
                    "context": RunnableLambda(lambda _: context),
                    "question": RunnablePassthrough()
                })

                parser = StrOutputParser()
                main_chain = parallel_chain | prompt | llm | parser

                answer = main_chain.invoke(question)

                st.write("### Answer:")
                st.write(answer)

    except TranscriptsDisabled:
        st.error("No captions available for this video.")
    except Exception as e:
        st.error(f"Error: {e}")
