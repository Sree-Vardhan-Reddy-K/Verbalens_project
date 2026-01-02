import os
import re
from dotenv import load_dotenv
from groq import Groq

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import CrossEncoder

#INITIAL SETUP
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

#Cross-encoder reranker (semantic precision)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#KNOWLEDGE BASE PREPARATION
@st.cache_resource
def prepare_knowledge_base(file_input):
    all_documents = []
    
    if isinstance(file_input, str):
        if os.path.isdir(file_input):
            file_paths = [
                os.path.join(file_input, f)
                for f in os.listdir(file_input)
                if f.lower().endswith(".pdf")
            ]
        elif os.path.isfile(file_input) and file_input.lower().endswith(".pdf"):
            file_paths = [file_input]
        else:
            raise ValueError(f"Invalid PDF path: {file_input}")

    elif isinstance(file_input, list):
        file_paths = file_input

    else:
        raise ValueError("file_input must be a path or list of paths")

    if not file_paths:
        raise ValueError("No PDF files found")
    
    
    for path in file_paths:
        loader = PyMuPDFLoader(path)
        documents = loader.load()

        for doc in documents:
            doc.page_content = re.sub(r'_{3,}', '', doc.page_content)
            doc.page_content = re.sub(r'[-.]{3,}', '', doc.page_content)
            doc.page_content = re.sub(r'\n{3,}', '\n\n', doc.page_content)
            doc.page_content = re.sub(r'[^\x20-\x7E\s\u0900-\u097F]', '', doc.page_content)
            doc.page_content = re.sub(r' +', ' ', doc.page_content)
            doc.page_content = doc.page_content.strip()

        all_documents.extend(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True
    )

    splits = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)

    #Dense retriever only (high recall)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}
    )

    return retriever

#RAG RESPONSE
def get_rag_response(prompt, retriever, eval_mode=False):

    #GUARDRAILS
    if not eval_mode:
        guard_prompt = (
            f"Question: '{prompt}'\n\n"
            f"Reply YES if the question is about unrelated topics "
            f"(weather, sports, food, animals, entertainment) "
            f"or mentions external organizations like Google, Microsoft, NASA.\n"
            f"Reply NO otherwise."
        )

        check = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": guard_prompt}],
            temperature=0,
            max_tokens=10
        )

        response_text = check.choices[0].message.content.strip().upper()

        external_entities = [
            'google', 'microsoft', 'apple', 'amazon', 'facebook', 'meta',
            'netflix', 'tesla', 'spacex', 'nasa', 'fbi', 'cia', 'twitter',
            'ibm', 'oracle', 'samsung', 'walmart'
        ]

        if "YES" in response_text or any(e in prompt.lower() for e in external_entities):
            return (
                "I am a Document Intelligence Assistant. "
                "I can only answer questions based on the uploaded documents.",
                []
            )

    #DENSE RETRIEVAL
    retrieved_docs = retriever.invoke(prompt)

    if not retrieved_docs:
        return "No relevant documentation found for the query.", []

    #CROSS-ENCODER RERANKING
    pairs = [(prompt, d.page_content) for d in retrieved_docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, retrieved_docs),
        key=lambda x: x[0],
        reverse=True
    )

    reranked_docs = [doc for _, doc in ranked[:4]]

    #QUALITY FILTERING
    valid_docs = []
    for doc in reranked_docs:
        content = doc.page_content.strip()

        if len(content) < 60:
            continue
        if re.search(r'_{3,}', content):
            continue

        alnum_ratio = sum(c.isalnum() for c in content) / max(len(content), 1)
        if alnum_ratio < 0.4:
            continue

        valid_docs.append(doc)

    if not valid_docs:
        return "No relevant documentation found for the query.", []

    #CONTEXT BUILDING
    context_chunks = []
    for d in valid_docs:
        context_chunks.append(d.page_content)

    context_text = "\n\n---\n\n".join(context_chunks)

    #ANSWER GENERATION
    if eval_mode:
        #EVAL MODE: NO GROUPING, NO FORMATTING
        system_prompt = (
            "Answer the question using ONLY the provided context. "
            "Be concise and factual. "
            "If information is missing, say so clearly."
        )
    else:
        #PROD MODE: GROUPING + PRESENTATION
        unique_files = {
            os.path.basename(d.metadata.get("source", "Unknown"))
            for d in valid_docs
        }

        if len(unique_files) > 1:
            files_list = ", ".join(unique_files)
            system_prompt = (
                "You are a Document Intelligence Assistant. "
                f"You are analyzing {len(unique_files)} documents: {files_list}. "
                "Group answers by document using headers: "
                "'### From [Filename]:' "
                "Answer ONLY using the provided context."
            )
        else:
            system_prompt = (
                "You are a Document Intelligence Assistant. "
                "Answer concisely using ONLY the provided context."
            )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {prompt}"
            }
        ],
        temperature=0,
        max_tokens=1000
    )

    answer = response.choices[0].message.content.strip()

    # ANSWER DEDUPLICATION
    lines = answer.split("\n")
    seen = set()
    final_lines = []

    for line in lines:
        key = frozenset(w.lower() for w in line.split() if len(w) > 3)
        if key not in seen:
            seen.add(key)
            final_lines.append(line)

    answer = "\n".join(final_lines).strip()

    return answer, valid_docs

# CHAT TITLE GENERATION
def generate_chat_title(first_prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Summarize into 2-3 words."},
                {"role": "user", "content": first_prompt}
            ],
            temperature=0.7,
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    except:
        return first_prompt[:20] + "..."
