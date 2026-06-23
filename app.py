import streamlit as st
from chatbot import generate_answer

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS (BLACK + PURPLE THEME)
# -----------------------------
st.markdown(
    """
    <style>
    /* Main application background adjustment */
    .stApp {
        background-color: #0e1117;
    }
    
    .header {
        background: linear-gradient(90deg, #000000, #6a0dad);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 25px;
    }

    .user-msg {
        background-color: #6a0dad;
        color: white;
        padding: 12px 16px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 16px;
    }

    .bot-msg {
        background-color: #1e1e1e;
        color: #e0e0e0;
        padding: 12px 16px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #2d2d2d;
        font-size: 16px;
        line-height: 1.5;
    }
    
    /* Optional: Style the native chat input container borders to match */
    div[data-testid="stChatInput"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <div class="header">
        🤖 FREELANCE-GUIDE AI RAG Chatbot
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# INITIALIZE CHAT HISTORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f'<div class="user-msg">🐵 <b>You:</b> {message["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="bot-msg">🤖 <b>Bot:</b> {message["content"]}</div>',
            unsafe_allow_html=True
        )

# -----------------------------
# CHAT INPUT & LOGIC
# -----------------------------
# st.chat_input pins the text field to the bottom of the page automatically
if query := st.chat_input("Ask anything..."):
    
    # 1. Display and save user message
    st.markdown(
        f'<div class="user-msg">🧑 <b>You:</b> {query}</div>',
        unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "user", "content": query})

    # 2. Generate and display bot response
    with st.spinner("Thinking..."):
        answer = generate_answer(query)

    st.markdown(
        f'<div class="bot-msg">🤖 <b>Bot:</b> {answer}</div>',
        unsafe_allow_html=True
    )
    st.session_state.messages.append({"role": "assistant", "content": answer})