import os
import re
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
def prepare_knowledge_base(file_paths):
    all_documents = []
    
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
    
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    keyword_retriever = BM25Retriever.from_documents(splits)
    keyword_retriever.k = 7
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.4, 0.6] 
    )
    return ensemble_retriever


def get_rag_response(prompt, retriever):
    # ==================================================================
    # GUARDRAIL 1: INTENT CLASSIFICATION
    guard_prompt = (
        f"Question: '{prompt}'\n\n"
        f"IMPORTANT: Check if the question mentions ANY of these external organizations:\n"
        f"- Tech companies: Google, Microsoft, Apple, Amazon, Facebook, Meta, Netflix, Tesla, SpaceX\n"
        f"- Government: NASA, FBI, CIA, UN, EU, WHO\n"
        f"- Other specific organizations by name\n\n"
        f"Also check if asking about completely unrelated topics: weather, sports, food, animals, entertainment.\n\n"
        f"Reply 'YES' if:\n"
        f"1. Question asks about a SPECIFIC NAMED external company/organization's policies\n"
        f"2. Question is about unrelated topics\n\n"
        f"Reply 'NO' ONLY if asking about general corporate concepts without naming specific external entities."
    )
    check = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": guard_prompt}],
        temperature=0,
        max_tokens=10
    )
    response_text = check.choices[0].message.content.strip().upper()

    # Additional regex check as backup for different companies
    external_entities = [
        'google', 'microsoft', 'apple', 'amazon', 'facebook', 'meta', 'netflix', 
        'tesla', 'spacex', 'nasa', 'fbi', 'cia', 'twitter', 'x corp', 'ibm',
        'oracle', 'samsung', 'walmart', 'target', 'starbucks', "mcdonald's"
    ]

    prompt_lower = prompt.lower()
    has_external_entity = any(entity in prompt_lower for entity in external_entities)

    if "YES" in response_text or has_external_entity:
        return "I am a Document Intelligence Assistant. I can only help with questions related to the uploaded manuals and corporate policies.", []

    # ---------------------------------------------------------------
    # DOCUMENT RETRIEVAL & QUALITY FILTER
    retrieved_docs = retriever.invoke(prompt)
    valid_docs = []
    
    for doc in retrieved_docs:
        content = doc.page_content.strip()
        
        if len(content) < 60:
            continue
        if re.search(r'_{3,}', content):
            continue
        
        alphanumeric_chars = sum(c.isalnum() for c in content)
        if len(content) > 0 and (alphanumeric_chars / len(content)) < 0.4:
            continue
            
        valid_docs.append(doc)

    if not valid_docs:
        return "No relevant documentation found for the query", []

    # ---------------------------------------------------------------
    # CHECK: Question about documents NOT uploaded and also extract potential document names from question
    question_doc_hints = re.findall(r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\b', prompt)
    uploaded_doc_names = set()
    
    for doc in valid_docs:
        fname = os.path.basename(doc.metadata.get('source', ''))
        #Extract base name without extension
        base_name = os.path.splitext(fname)[0].lower()
        uploaded_doc_names.add(base_name)
    
    # Checking if query mentions a specific document that's NOT uploaded
    for hint in question_doc_hints:
        hint_lower = hint.lower()
        # Common document-specific terms
        if any(term in hint_lower for term in ['manual', 'policy', 'regulation', 'document', 'guide']):
            # Check if this document name is in uploaded files
            is_uploaded = any(hint_lower in doc_name for doc_name in uploaded_doc_names)
            if not is_uploaded and len(hint) > 3:
                return f"I don't have information about '{hint}' in the uploaded documents. Please check if this document has been uploaded.", []

    # ----------------------------------------------------------
    # CONTEXT BUILDING
    context_chunks = []
    unique_files = set()
    for d in valid_docs:
        fname = os.path.basename(d.metadata.get('source', 'Unknown'))
        unique_files.add(fname)
        context_chunks.append(f"SOURCE: {fname}\nCONTENT: {d.page_content}")
    
    context_text = "\n\n---\n\n".join(context_chunks)

    #----------------------------------------------------------
    # MULTI-DOCUMENT GROUPING FOR CLUBBING ANSWERS FROM A DOCUMENT AT A PLACE
    if len(unique_files) > 1:
        files_list = ", ".join(unique_files)
        grouping_instruction = (
            f"You are analyzing {len(unique_files)} documents: {files_list}. "
            f"You MUST check ALL documents and group your answer by document. "
            f"For EACH document that contains relevant information, create ONE header '### From [Filename]:'. "
            f"Under each header, list ALL points from that document. "
            f"DO NOT skip any document that has information. "
            f"DO NOT create multiple headers for the same document. "
            f"DO NOT repeat information. "
            f"If a document has no relevant info, don't mention it."
        )
    else:
        grouping_instruction = (
            "Answer directly in paragraph form. "
            "Do not use document headers or filenames. "
            "Be concise and avoid repetition."
        )

    # ----------------------------------------------------------
    # GENERATE ANSWER
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system", 
                "content": (
                    f"You are a Document Intelligence Assistant. Answer using ONLY the provided context. "
                    f"{grouping_instruction} "
                    f"If no information exists, say 'No relevant information found in the documents.'"
                )
            },
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {prompt}"}
        ],
        temperature=0,
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content.strip()
    
    # ---------------------------------------------------------------
    # REPETITION OF THE ANSWER -PREVENTION
    lines = answer.split('\n')
    unique_lines = []
    seen_lines = set()

    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            unique_lines.append(line)
        
        key_words = set([w.lower() for w in stripped.split() if len(w) > 3])
        
        is_duplicate = False
        for seen_key_words in seen_lines:
            if key_words and seen_key_words:
                overlap = len(key_words & seen_key_words) / len(key_words)
                if overlap >= 0.7:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            seen_lines.add(frozenset(key_words))
            unique_lines.append(line)

    answer = '\n'.join(unique_lines)
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = answer.strip()

    #---------------------------------------------------------------
    # OUT-OF-CONTEXT DETECTION
    refusal_patterns = [
        r"no relevant (information|documentation|documents|data)",
        r"no information (found|available)",
        r"not found in",
        r"unavailable",
        r"i (do not|don't) (know|have)",
        r"cannot (find|provide)",
        r"no mention"
    ]

    answer_lower = answer.lower()
    if any(re.search(pattern, answer_lower) for pattern in refusal_patterns):
        return answer, []

    #---------------------------------------------------------------
    # SOURCE SELECTION (DYNAMIC - Relevance-Based)
    final_sources = []
    seen_content = set()

    # Extracting answer terms for relevance scoring
    answer_terms = set([w.lower() for w in answer.split() if len(w) > 4])

    # Score each source by relevance
    source_scores = []
    for doc in valid_docs[:10]:  # Check top 10 retrieved
        fname = os.path.basename(doc.metadata.get('source', ''))
        content_hash = hash(doc.page_content)
        
        # Skipping duplicates
        if content_hash in seen_content:
            continue
        
        # Calculate relevance score
        source_terms = set([w.lower() for w in doc.page_content.split() if len(w) > 4])
        
        if answer_terms and source_terms:
            overlap = len(answer_terms & source_terms) / len(answer_terms)
            
            # Boost score if file is mentioned in answer (for multi-file)
            if len(unique_files) > 1 and f"From {fname}" in answer:
                overlap += 0.3  # Boost by 30%
            
            # Only include if relevance > 10%
            if overlap >= 0.07:
                source_scores.append((overlap, doc, content_hash))

    # Sort by relevance score (highest first)
    source_scores.sort(reverse=True, key=lambda x: x[0])

    # Dynamic limit based on answer length
    answer_word_count = len(answer.split())
    if answer_word_count < 50:
        max_sources = 2  # Short answer → fewer sources
    elif answer_word_count < 150:
        max_sources = 4  # Medium answer → 4 sources
    elif answer_word_count < 300:
        max_sources = 6  # Long answer → 6 sources
    else:
        max_sources = 8  # Very long answer → 8 sources

    # Add top-scored sources up to dynamic limit
    for score, doc, content_hash in source_scores[:max_sources]:
        seen_content.add(content_hash)
        final_sources.append(doc)

    # Fallback 1: If there are too few sources, we add more from top retrieved
    if len(final_sources) < 2:
        for score, doc, content_hash in source_scores[:4]:
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                final_sources.append(doc)
                if len(final_sources) >= 3:
                    break

    # Fallback 2: If still no sources, we take top valid_docs directly
    if len(final_sources) == 0 and len(valid_docs) > 0:
        for doc in valid_docs[:3]:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                final_sources.append(doc)

    return answer, final_sources

def generate_chat_title(first_prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Summarize into 2-3 words. No quotes."},
                {"role": "user", "content": first_prompt}
            ],
            temperature=0.7,
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    except:
        return first_prompt[:20] + "..."