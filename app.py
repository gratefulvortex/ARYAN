import streamlit as st
import requests
import time

# Set page configuration
st.set_page_config(page_title="🤖 AI Document Query Assistant", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    /* Fixed Footer */
    html body::after {
        content: '';
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 50px;
        background-color: #fff;
        border-top: 1px solid #e9ecef;
        box-shadow: 0 -2px 4px rgba(0,0,0,0.04);
        z-index: 1000;
    }
    html body::before {
        content: '© 2025 S3K Technologies | All rights reserved';
        position: fixed;
        bottom: 18px;
        left: 0;
        width: 100%;
        color: #495057;
        text-align: center;
        z-index: 1001;
        font-size: 0.9em;
    }

    /* General Layout */
    body {
        color: #495057;
        background-color: #f8f9fa;
        font-family: sans-serif;
    }

    .main .block-container {
        padding-bottom: 80px;
    }

    /* Header styling */
    .stApp > header {
        background-color: #fff;
        padding: 10px 20px;
        border-bottom: 2px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        position: sticky;
        top: 0;
        z-index: 999;
    }

    /* Chat message styling */
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 20px;
    }

    .user-message {
        background: #e9ecef;
        font-weight: 500;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #dee2e6;
        color: #495057;
        margin-left: auto;
        max-width: 85%;
    }

    .bot-message {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #e9ecef;
        color: #495057;
        margin-right: auto;
        max-width: 85%;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        padding: 12px 16px;
        border: 2px solid #e9ecef;
        background-color: #fff;
        font-size: 14px;
        color: #495057;
        transition: all 0.2s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #b22222;
        box-shadow: 0 0 0 3px rgba(178,34,34,0.1);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        padding: 10px 20px;
        background-color: #b22222 !important;
        color: white !important;
        border: none;
        cursor: pointer;
        box-shadow: 0 2px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #8b0000 !important;
        transform: translateY(-1px);
    }

    /* Visit Us Button */
    .visit-button {
        display: inline-block;
        padding: 8px 20px;
        background-color: #b22222;
        color: white !important;
        text-decoration: none;
        border-radius: 6px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin-right: 10px;
    }

    .visit-button:hover {
        background-color: #8b0000;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        text-decoration: none;
    }

    /* File uploader styling */
    .stFileUploader > div > div > div {
        border-radius: 8px;
        border: 2px dashed #e9ecef;
        padding: 20px;
        background-color: #f9fafb;
    }

    /* Sidebar styling */
    .stSidebar > div:first-child {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Function to get list of uploaded files
def get_uploaded_files():
    try:
        response = requests.get("http://127.0.0.1:8000/files/")
        response.raise_for_status()
        return response.json().get("files", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching uploaded files: {str(e)}")
        return []

# Function to get chatbot response
def get_response(query):
    try:
        st.write(f"DEBUG: Sending query: {query}")
        response = requests.post("http://127.0.0.1:8000/query/", json={"query": query})
        response.raise_for_status()
        answer = response.json().get("answer", "No answer found.")
        token_count = response.json().get("token_count", 0)
        st.write(f"DEBUG: Received response: {answer[:100]}...")
        return answer, token_count
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}", 0

# Header with logo and visit button
with st.container():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image("https://s3ktech.ai/wp-content/uploads/2025/03/S3Ktech-Logo.png", width=140)
    with col2:
        st.markdown("<h1 style='display: inline-block; margin-left: 20px;'>AI Document Query Assistant</h1>", unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                <a href="https://s3ktech.ai/" target="_blank" class="visit-button">Visit Us</a>
            </div>
        """, unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = str(time.time())

# Sidebar for file upload
with st.sidebar:
    st.subheader("📂 Upload Excel Files")
    st.markdown("Upload one or more .xlsx files to query hardware data across all files.")

    if not st.session_state.file_uploaded:
        uploaded_files = st.file_uploader(
            "Choose Excel files",
            type=["xlsx"],
            accept_multiple_files=True,
            key=st.session_state.uploader_key
        )
    else:
        st.write("✅ Files already uploaded. Use 'Clear Files' to upload new ones.")
        uploaded_files = None

    if uploaded_files:
        with st.spinner("Processing your files..."):
            st.write(f"DEBUG: Uploading {len(uploaded_files)} file(s)")
            files = [
                ("files", (file.name, file.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
                for file in uploaded_files
            ]
            try:
                response = requests.post("http://127.0.0.1:8000/upload/", files=files)
                response.raise_for_status()
                if response.status_code == 200:
                    response_data = response.json()
                    st.success(f"✅ Successfully uploaded {len(response_data['filenames'])} file(s)!")
                    st.session_state.file_uploaded = True
                    uploaded_file_list = get_uploaded_files()
                    for file in uploaded_file_list:
                        if file["file_id"] not in st.session_state.uploaded_files:
                            st.session_state.uploaded_files[file["file_id"]] = file["filename"]
                    st.session_state.uploader_key = str(time.time())
                    st.rerun()
                else:
                    st.error(f"🚨 Upload failed: {response.json().get('detail', response.text)}")
            except requests.exceptions.RequestException as e:
                st.error(f"🚨 Upload failed: {str(e)}")

    if st.session_state.file_uploaded:
        st.subheader("📑 Uploaded Files")
        for file_id, filename in st.session_state.uploaded_files.items():
            st.write(f"• {filename}")

# Chat interface
st.subheader("💬 Query Your Data")
st.markdown("Ask questions about your Excel files, and the assistant will search across all uploaded files.")

# Chat display container
with st.container():
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(
                    f"<div class='user-message'><strong>You:</strong><br>{chat['text']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='bot-message'><strong>Assistant:</strong><br>{chat['text']}<br><small>Tokens used: {chat.get('token_count', 0)}</small></div>",
                    unsafe_allow_html=True
                )

# Query input
user_input = st.chat_input(
    placeholder="E.g., 'Full details for serial number 44B12345' or 'Total qty in Maharashtra'"
)

if user_input:
    if not st.session_state.file_uploaded:
        st.error("🚨 Please upload at least one Excel file first.")
    else:
        st.session_state.last_query = user_input
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        with st.spinner("🔄 Processing your query..."):
            answer, token_count = get_response(user_input)
        st.session_state.chat_history.append({"role": "bot", "text": answer, "token_count": token_count})
        st.rerun()

# Clear buttons at the bottom
col_clear1, col_clear2 = st.columns([1, 1])
with col_clear1:
    if st.button("Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.session_state.last_query = None
        st.rerun()
with col_clear2:
    if st.button("Clear Files", type="secondary"):
        try:
            response = requests.delete("http://127.0.0.1:8000/clear_files/")
            response.raise_for_status()
            st.session_state.uploaded_files = {}
            st.session_state.file_uploaded = False
            st.session_state.chat_history = []
            st.session_state.last_query = None
            st.session_state.uploader_key = str(time.time())
            st.success("✅ All files and chat history cleared!")
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"🚨 Error clearing files: {str(e)}")
