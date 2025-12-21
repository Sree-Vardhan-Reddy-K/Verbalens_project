import streamlit as st
import os
import uuid
import tempfile 
from main import prepare_knowledge_base, get_rag_response, generate_chat_title

data_path = "data/company_policy.pdf"

# 1. PAGE CONFIG
st.set_page_config(page_title="VerbaLens: Enterprise RAG", layout="wide")

# 2. INITIALIZE GLOBAL STATE
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "titles" not in st.session_state:
    st.session_state.titles = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# 3. SIDEBAR: CHAT MANAGEMENT
st.sidebar.title("💬 VerbaLens AI")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = []
    st.session_state.current_chat_id = new_id
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Recent Chats")

# DISPLAY LIST OF CHATS
for chat_id in reversed(list(st.session_state.chat_sessions.keys())):
    display_name = st.session_state.titles.get(chat_id, "New Chat")
    is_active = (chat_id == st.session_state.current_chat_id)
    label = f"{display_name}" if is_active else display_name
    
    if st.sidebar.button(label, key=chat_id, use_container_width=True):
        st.session_state.current_chat_id = chat_id
        st.rerun()

st.sidebar.divider()

if st.sidebar.button("🗑️ Clear All History", use_container_width=True):
    st.session_state.chat_sessions = {}
    st.session_state.titles = {}
    st.session_state.current_chat_id = None
    st.rerun()
#--------------------------------------------------------------------
# 4. KNOWLEDGE BASE LOADING

with st.sidebar:
    st.divider()
    st.subheader("📁 Knowledge Base Setup")
    
    # STEP 1 & 3: Multi-PDF Uploader
    uploaded_files = st.file_uploader(
        "Upload Policy PDFs,if nothing to upload, click on initialise to use default data", 
        type="pdf", 
        accept_multiple_files=True
    )

    if st.button("Initialize / Sync Documents", use_container_width=True):
        with st.status("Processing Documents...") as status:
            file_paths = []
            
            # STEP 1 & 3: If user uploaded files, save to temp location
            # Save to a local folder instead of using random temp names
            if uploaded_files:
                st.write(f"Saving {len(uploaded_files)} uploaded files...")
                
                # Create a physical folder to preserve original names
                upload_dir = "temp_pdf_store"
                if not os.path.exists(upload_dir):
                    os.makedirs(upload_dir)
                
                for uploaded_file in uploaded_files:
                    # Use the actual filename from the uploader
                    file_path = os.path.join(upload_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                st.session_state.retriever = prepare_knowledge_base(file_paths)
                st.success("Custom PDFs loaded successfully!")
                
            # STEP 2: Fallback to internal if uploader is empty
            else:
                if os.path.exists(data_path):
                    st.write("No files uploaded. Using internal policy...")
                    file_paths = [data_path]
                    st.session_state.retriever = prepare_knowledge_base(file_paths)
                    st.info("Default knowledge base active.")
                else:
                    st.error("Fallback PDF not found. Please upload a file.")
            
            status.update(label="System Ready!", state="complete")



#------------------------------------------------------------------------
# 5. MAIN CHAT AREA
if st.session_state.current_chat_id is None:
    st.title("Welcome to VerbaLens")
    st.info("👋 Select an existing chat or click 'New Chat' to begin.")
else:
    current_messages = st.session_state.chat_sessions[st.session_state.current_chat_id]
    
    # --- FEATURE: SUGGESTED QUESTIONS (for empty chats) ---
    if len(current_messages) == 0:
        st.title("How can I help you today?")
        cols = st.columns(3)
        suggestions = ["explain recruitment policy?", "Tell me about appointment rules", "Code of Conduct in the campus?"]
        for i, hint in enumerate(suggestions):
            if cols[i].button(hint, use_container_width=True):
                st.session_state.pending_prompt = hint
                st.rerun()
    else:
        st.title("Document Intelligence")

    # DISPLAY HISTORY
    for i, message in enumerate(current_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                msg_sources = message.get("sources", [])
                
                if msg_sources and len(msg_sources) > 0:
                    with st.expander("View Source Evidence"):
                        for j, doc in enumerate(msg_sources):
                            # NEW: Extract filename from metadata
                            full_path = doc.metadata.get('source', 'Unknown')
                            file_name = os.path.basename(full_path)
                            page_num = doc.metadata.get('page', 'N/A')
                            
                            # NEW: Display distinct document names
                            st.write(f"**Source {j+1} | File: `{file_name}` (Page {page_num})**")
                            st.info(doc.page_content)
                            if j < len(msg_sources) - 1:
                                st.divider() # Visual separation between chunks
                else:
                    st.caption("No relevant documents found for this response.")

    # CHAT INPUT PROCESSING
    prompt = st.chat_input("Ask about the policy:")
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    if prompt:
        is_first_message = (len(current_messages) == 0)
        
        # 1. Clear any old source data from the current script run
        sources = [] 
        
        # 2. Add user message to history
        current_messages.append({"role": "user", "content": prompt})
        
        # Update Sidebar Title if needed
        if is_first_message:
            st.session_state.titles[st.session_state.current_chat_id] = generate_chat_title(prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        # 3. Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Consulting Manual..."):
                try:
                    answer, sources = get_rag_response(prompt, st.session_state.retriever)
                    st.markdown(answer)
                    
                    # 4. CONDITIONAL RENDERING: Separate sources by PDF
                    if sources and len(sources) > 0:
                        with st.expander("View Source Evidence"):
                            for i, doc in enumerate(sources):
                                # Clearer labeling per document
                                full_path = doc.metadata.get('source', 'Unknown')
                                file_name = os.path.basename(full_path)
                                page_num = doc.metadata.get('page', 'N/A')
                                
                                st.write(f"**Source {i+1} | File: `{file_name}` (Page {page_num})**")
                                st.info(doc.page_content)
                                if i < len(sources) - 1:
                                    st.divider()
                    else:
                        st.caption("No relevant documents from the knowledge base were used for this response.")

                    # 5. Save to history
                    current_messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources
                    })
                    
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {e}")