import streamlit as st
import os
import uuid
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

# 4. KNOWLEDGE BASE LOADING
if os.path.exists(data_path):
    if "retriever" not in st.session_state:
        with st.sidebar.status("Analyzing Document..."):
            st.session_state.retriever = prepare_knowledge_base(data_path)
    st.sidebar.success("Knowledge Base Loaded! ✅")
else:
    st.sidebar.error("Error: company_policy.pdf not found.")

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
        suggestions = ["What is the Leave Policy?", "Cab Service Rules", "Code of Conduct"]
        for i, hint in enumerate(suggestions):
            if cols[i].button(hint, use_container_width=True):
                st.session_state.pending_prompt = hint
                st.rerun()
    else:
        st.title("Document Intelligence")

    # DISPLAY HISTORY
    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if they are present
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("View Source Evidence"):
                    for i, doc in enumerate(message["sources"]):
                        st.write(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                        st.info(doc.page_content)

    # CHAT INPUT PROCESSING
    prompt = st.chat_input("Ask about the policy:")
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    if prompt:
        is_first_message = (len(current_messages) == 0)
        
        # 1. Add user message
        current_messages.append({"role": "user", "content": prompt})
        
        # 2. Update Sidebar Title if first message
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
                    with st.expander("View Source Evidence"):
                        for i, doc in enumerate(sources):
                            st.write(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                            st.info(doc.page_content)

                    current_messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources
                    })
                    
                    if is_first_message:
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")