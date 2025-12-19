import os
import re
import unicodedata
from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

# 1. INITIAL SETUP
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@st.cache_resource 
def prepare_knowledge_base(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    for doc in documents:
        # 1. Removing repetitive underscores (3 or more)
        doc.page_content = re.sub(r'_{3,}', '', doc.page_content)
        # 2. Removing repetitive dashes or dots (3 or more)
        doc.page_content = re.sub(r'[-.]{3,}', '', doc.page_content)
        # 3. Collapsing multiple newlines/spaces for cleaner chunks
        doc.page_content = re.sub(r'\n{3,}', '\n\n', doc.page_content)
        
        doc.page_content = re.sub(r'[^\x20-\x7E\s\u0900-\u097F]', '', doc.page_content)
        doc.page_content = re.sub(r' +', ' ', doc.page_content)
        doc.page_content = doc.page_content.strip()
    # Keeping split settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=200,
        add_start_index=True 
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # 2. ENHANCED RETRIEVER
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    
    keyword_retriever = BM25Retriever.from_documents(splits)
    keyword_retriever.k = 7
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.4, 0.6] 
    )
    return ensemble_retriever

# 3. GENERATION LOGIC
def get_rag_response(prompt, retriever):
    # Retrieve documents based on user prompt
    retrieved_docs = retriever.invoke(prompt)

    valid_docs = []
    for doc in retrieved_docs:
        content = doc.page_content.strip()
        
        # A: Skipping if it contains long strings of underscores (e.g., _______)
        if re.search(r'_{3,}', content): 
            continue
            
        # B: Skipping if the chunk has too many non-alphabetic characters
        # (If less than 40% of the chunk is actual letters/numbers, it's likely a table border or form)
        alphanumeric_chars = sum(c.isalnum() for c in content)
        if len(content) > 0 and (alphanumeric_chars / len(content)) < 0.4:
            continue

        # C: Skipping very short noise chunks (headers/footers)
        if len(content) < 60:
            continue

        valid_docs.append(doc)
    FAIL="No relevant documentation found for the query"
    if not valid_docs:
        return FAIL, []

    context_text = "\n\n".join([d.page_content.strip() for d in valid_docs])
    # Use the top 3 chunks for context
    #context_chunks = retrieved_docs[:6]
    #context_text = "\n\n".join([d.page_content.strip() for d in context_chunks])

    # 5. SYSTEM PROMPT
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful corporate assistant. Answer the question using the provided context."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {prompt}"
            }
        ],
        temperature=0 
    )

    answer = response.choices[0].message.content.strip()

    return answer, retrieved_docs[:6]


def generate_chat_title(first_prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Summarize the user's question into a 2-3 word title. Respond ONLY with the title. No quotes."},
                {"role": "user", "content": first_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except:
        return first_prompt[:20] + "..."