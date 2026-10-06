
# ============================================================
# RAG CHATBOT
# MULTI-USER LOGIN + ROLE-BASED ACCESS
# ADMIN + USER
# CHANGE PASSWORD
# FAST RAG + MODERN UI + BUTTONS + IMAGE GENERATOR
# Streamlit + Chroma + MiniLM + Groq Cloud LLM
# ============================================================

import time
import html
import base64
from pathlib import Path

import streamlit as st

from auth import (
    verify_user,
    create_user,
    change_password,
    get_user_role,
    is_admin,
    admin_reset_password,
    get_all_users,
    delete_user
)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOGIN AUTHENTICATION SESSION
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.markdown(
        """
        <div style="
            max-width:500px;
            margin:80px auto 20px auto;
            text-align:center;
        ">
            <h1>🤖 RAG Chatbot</h1>
            <p>Enterprise AI Assistant</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    login_tab, register_tab = st.tabs(
        ["🚀 Login", "📝 Register"]
    )

    # ========================================================
    # LOGIN
    # ========================================================

    with login_tab:

        with st.form("login_form"):

            username = st.text_input(
                "👤 Username",
                placeholder="Enter username",
                key="login_username"
            )

            password = st.text_input(
                "🔑 Password",
                type="password",
                placeholder="Enter password",
                key="login_password"
            )

            login_button = st.form_submit_button(
                "🚀 Login",
                use_container_width=True
            )

            if login_button:

                username = username.strip()

                if not username or not password:

                    st.warning(
                        "⚠️ Please enter both username and password."
                    )

                elif verify_user(username, password):

                    # ------------------------------------------------
                    # LOGIN SUCCESS
                    # ------------------------------------------------

                    st.session_state.logged_in = True

                    st.session_state.username = username

                    # ------------------------------------------------
                    # GET ROLE FROM DATABASE
                    # ------------------------------------------------

                    role = get_user_role(username)

                    st.session_state.role = role or "user"

                    st.success(
                        "✅ Login successful!"
                    )

                    st.rerun()

                else:

                    st.error(
                        "❌ Invalid username or password."
                    )

    # ========================================================
    # REGISTER
    # ========================================================

    with register_tab:

        with st.form("register_form"):

            new_username = st.text_input(
                "👤 Create Username",
                placeholder="Enter a new username",
                key="register_username"
            )

            new_password = st.text_input(
                "🔑 Create Password",
                type="password",
                placeholder="Minimum 6 characters",
                key="register_password"
            )

            confirm_password = st.text_input(
                "🔑 Confirm Password",
                type="password",
                placeholder="Re-enter your password",
                key="register_confirm_password"
            )

            register_button = st.form_submit_button(
                "📝 Create Account",
                use_container_width=True
            )

            if register_button:

                new_username = new_username.strip()

                if not new_username or not new_password:

                    st.warning(
                        "⚠️ Please fill in all fields."
                    )

                elif len(new_username) < 3:

                    st.error(
                        "❌ Username must contain at least 3 characters."
                    )

                elif len(new_password) < 6:

                    st.error(
                        "❌ Password must contain at least 6 characters."
                    )

                elif new_password != confirm_password:

                    st.error(
                        "❌ Passwords do not match."
                    )

                else:

                    # --------------------------------------------
                    # ALL SELF-REGISTERED ACCOUNTS ARE USERS
                    # --------------------------------------------

                    success, message = create_user(
                        new_username,
                        new_password,
                        role="user"
                    )

                    if success:

                        st.success(
                            f"✅ {message}"
                        )

                        st.info(
                            "You can now open the Login tab and sign in."
                        )

                    else:

                        st.error(
                            f"❌ {message}"
                        )


# ============================================================
# LOGIN GATE
# ============================================================

if not st.session_state.logged_in:

    login_page()

    st.stop()


# ============================================================
# SECURITY CHECK
# ============================================================
# Refresh role from database after login.
# This prevents relying only on browser session data.
# ============================================================

current_username = st.session_state.username.strip()

current_role = get_user_role(
    current_username
)

if current_role is None:

    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""

    st.error(
        "❌ User account no longer exists."
    )

    st.stop()


# Always use the database role
st.session_state.role = current_role


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

CHROMA_DIR = BASE_DIR / "chroma_db"

UPLOAD_DIR = DATA_DIR / "uploaded"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# GROQ MODEL
# ============================================================

LLM_MODEL = "openai/gpt-oss-120b"


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

if "image_prompt" not in st.session_state:
    st.session_state.image_prompt = ""

if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}


# ============================================================
# MODERN UI CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(80, 120, 255, 0.08),
                transparent 30%
            ),
            radial-gradient(
                circle at top right,
                rgba(150, 80, 255, 0.06),
                transparent 30%
            );
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .card {
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.22);
        background: rgba(255,255,255,0.025);
        min-height: 125px;
        margin-bottom: 12px;
    }

    .card-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .card-text {
        font-size: 14px;
        opacity: 0.72;
        line-height: 1.5;
    }

    .feature {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.2);
        background: rgba(255,255,255,0.025);
        min-height: 115px;
    }

    .feature-icon {
        font-size: 26px;
        margin-bottom: 5px;
    }

    .feature-title {
        font-size: 16px;
        font-weight: 700;
    }

    .feature-text {
        font-size: 13px;
        opacity: 0.65;
        margin-top: 5px;
    }

    section[data-testid="stSidebar"] {
        min-width: 275px;
        max-width: 310px;
    }

    .stButton > button {
        width: 100%;
        min-height: 42px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    [data-testid="stChatMessage"] {
        border-radius: 15px;
    }

    [data-testid="stChatInput"] {
        border-radius: 14px;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        padding: 12px;
        border-radius: 14px;
    }

    .image-box {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(128,128,128,0.25);
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


# ============================================================
# LOAD CHROMA
# ============================================================

@st.cache_resource(show_spinner=False)
def load_vector_database():

    if not CHROMA_DIR.exists():
        return None

    try:

        embeddings = load_embeddings()

        db = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings
        )

        return db

    except Exception:

        return None


# ============================================================
# LOAD GROQ CLOUD LLM
# ============================================================

@st.cache_resource(show_spinner=False)
def load_llm():

    try:

        api_key = st.secrets.get(
            "GROQ_API_KEY"
        )

        if api_key is None:
            return None

        api_key = str(api_key).strip()

        if not api_key:
            return None

        llm = ChatGroq(
            groq_api_key=api_key,
            model=LLM_MODEL,
            temperature=0.2,
            max_tokens=1024
        )

        return llm

    except Exception:

        return None


# ============================================================
# START SYSTEM
# ============================================================

try:

    vector_db = load_vector_database()

    if vector_db is not None:

        st.session_state.system_ready = True

    else:

        st.session_state.system_ready = False

except Exception:

    vector_db = None

    st.session_state.system_ready = False


# ============================================================
# IMAGE GENERATOR
# ============================================================

def generate_local_image(prompt):

    safe_prompt = html.escape(
        prompt[:160]
    )

    svg = f"""
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="1200"
        height="700"
        viewBox="0 0 1200 700">

        <defs>

            <linearGradient
                id="bg"
                x1="0%"
                y1="0%"
                x2="100%"
                y2="100%">

                <stop
                    offset="0%"
                    stop-color="#101828"/>

                <stop
                    offset="50%"
                    stop-color="#263B70"/>

                <stop
                    offset="100%"
                    stop-color="#5B2B82"/>

            </linearGradient>

            <filter id="blur">

                <feGaussianBlur
                    stdDeviation="50"/>

            </filter>

        </defs>

        <rect
            width="1200"
            height="700"
            fill="url(#bg)"/>

        <circle
            cx="170"
            cy="140"
            r="150"
            fill="#6C8CFF"
            opacity="0.35"
            filter="url(#blur)"/>

        <circle
            cx="1030"
            cy="560"
            r="180"
            fill="#D16CFF"
            opacity="0.30"
            filter="url(#blur)"/>

        <rect
            x="100"
            y="90"
            width="1000"
            height="520"
            rx="35"
            fill="rgba(255,255,255,0.07)"
            stroke="rgba(255,255,255,0.22)"/>

        <text
            x="600"
            y="220"
            text-anchor="middle"
            font-family="Arial, sans-serif"
            font-size="54"
            font-weight="bold"
            fill="white">

            AI GENERATED IMAGE

        </text>

        <text
            x="600"
            y="300"
            text-anchor="middle"
            font-family="Arial, sans-serif"
            font-size="25"
            fill="#DDE5FF">

            RAG Chatbot

        </text>

        <text
            x="600"
            y="390"
            text-anchor="middle"
            font-family="Arial, sans-serif"
            font-size="22"
            fill="white">

            {safe_prompt}

        </text>

        <rect
            x="390"
            y="470"
            width="420"
            height="70"
            rx="35"
            fill="rgba(255,255,255,0.13)"
            stroke="rgba(255,255,255,0.3)"/>

        <text
            x="600"
            y="515"
            text-anchor="middle"
            font-family="Arial, sans-serif"
            font-size="22"
            fill="white">

            Generated Locally

        </text>

    </svg>
    """

    return svg


# ============================================================
# SVG TO DATA URI
# ============================================================

def svg_to_data_uri(svg):

    encoded = base64.b64encode(
        svg.encode("utf-8")
    ).decode("utf-8")

    return (
        f"data:image/svg+xml;base64,{encoded}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🤖 RAG Chatbot")

    st.caption(
        "Fast RAG + Groq Cloud AI"
    )

    # --------------------------------------------------------
    # USER INFORMATION
    # --------------------------------------------------------

    if st.session_state.role == "admin":

        st.success(
            f"👑 Admin: "
            f"{st.session_state.username}"
        )

    else:

        st.info(
            f"👤 User: "
            f"{st.session_state.username}"
        )

    st.caption(
        f"Role: `{st.session_state.role}`"
    )

    st.markdown("---")


    # ========================================================
    # ADMIN PANEL
    # ========================================================

    if st.session_state.role == "admin":

        with st.expander(
            "👑 Admin Panel",
            expanded=False
        ):

            st.markdown(
                "### 👥 User Management"
            )

            # ------------------------------------------------
            # GET ALL USERS
            # ------------------------------------------------

            users = get_all_users()

            if users:

                for username, role in users:

                    col1, col2, col3 = st.columns(
                        [3, 2, 2]
                    )

                    with col1:

                        if username == st.session_state.username:

                            st.write(
                                f"👑 **{username}**"
                            )

                        else:

                            st.write(
                                f"👤 **{username}**"
                            )

                    with col2:

                        st.write(
                            f"`{role}`"
                        )

                    with col3:

                        # ------------------------------------
                        # ADMIN CANNOT DELETE ITSELF
                        # ------------------------------------

                        if (
                            username
                            != st.session_state.username
                        ):

                            if st.button(
                                "🗑️ Delete",
                                key=f"delete_user_{username}",
                                use_container_width=True
                            ):

                                success, message = delete_user(
                                    st.session_state.username,
                                    username
                                )

                                if success:

                                    st.success(
                                        message
                                    )

                                    st.rerun()

                                else:

                                    st.error(
                                        message
                                    )

            else:

                st.info(
                    "No users found."
                )


            st.markdown("---")


            # ------------------------------------------------
            # RESET USER PASSWORD
            # ------------------------------------------------

            st.markdown(
                "### 🔑 Reset User Password"
            )

            target_user = st.text_input(
                "Username",
                placeholder="Enter username",
                key="admin_target_user"
            )

            admin_new_password = st.text_input(
                "New Password",
                type="password",
                placeholder="Enter new password",
                key="admin_new_password"
            )

            admin_confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Confirm new password",
                key="admin_confirm_password"
            )

            if st.button(
                "🔐 Reset Password",
                key="admin_reset_password",
                use_container_width=True
            ):

                target_user = target_user.strip()

                if not target_user:

                    st.warning(
                        "Please enter a username."
                    )

                elif not admin_new_password:

                    st.warning(
                        "Please enter a new password."
                    )

                elif admin_new_password != admin_confirm_password:

                    st.error(
                        "❌ Passwords do not match."
                    )

                elif len(admin_new_password) < 6:

                    st.error(
                        "❌ Password must contain at least 6 characters."
                    )

                else:

                    success, message = admin_reset_password(
                        st.session_state.username,
                        target_user,
                        admin_new_password
                    )

                    if success:

                        st.success(
                            f"✅ {message}"
                        )

                    else:

                        st.error(
                            f"❌ {message}"
                        )


        st.markdown("---")


    # ========================================================
    # CHANGE PASSWORD
    # ========================================================

    with st.expander(
        "🔐 Change Password"
    ):

        old_password = st.text_input(
            "Current Password",
            type="password",
            key="old_password"
        )

        new_password = st.text_input(
            "New Password",
            type="password",
            key="new_password"
        )

        confirm_password = st.text_input(
            "Confirm New Password",
            type="password",
            key="confirm_password"
        )

        if st.button(
            "🔑 Change Password",
            use_container_width=True
        ):

            if not old_password:

                st.warning(
                    "Enter your current password."
                )

            elif not new_password:

                st.warning(
                    "Enter a new password."
                )

            elif new_password != confirm_password:

                st.error(
                    "❌ New passwords do not match."
                )

            elif len(new_password) < 6:

                st.error(
                    "❌ Password must contain at least 6 characters."
                )

            else:

                success, message = change_password(
                    st.session_state.username,
                    old_password,
                    new_password
                )

                if success:

                    st.success(
                        "✅ Password changed successfully!"
                    )

                else:

                    st.error(
                        f"❌ {message}"
                    )


    st.markdown("---")


    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    st.markdown(
        "### 📚 Knowledge Base"
    )

    st.markdown(
        """
        📄 HR Documents

        🛠️ Technical Documents

        📜 Company Policies

        🎫 Jira Issues

        📘 Confluence Pages

        🐍 Python Documents

        👥 Friends
        """
    )

    st.markdown("---")


    # ========================================================
    # SYSTEM
    # ========================================================

    st.markdown(
        "### ⚙️ System"
    )

    if st.session_state.system_ready:

        st.success(
            "🟢 RAG System Online"
        )

    else:

        st.error(
            "🔴 RAG System Offline"
        )

    st.markdown("---")


    # ========================================================
    # MODELS
    # ========================================================

    st.markdown(
        "### 🧠 Models"
    )

    st.caption(
        f"LLM: `{LLM_MODEL}`"
    )

    st.caption(
        "Embedding: MiniLM"
    )

    st.caption(
        "Vector DB: Chroma"
    )

    st.caption(
        "Runtime: Groq Cloud AI"
    )

    st.markdown("---")


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


    # ========================================================
    # CLEAR CACHE
    # ========================================================

    if st.button(
        "⚡ Clear Answer Cache",
        use_container_width=True
    ):

        st.session_state.answer_cache = {}

        st.success(
            "Answer cache cleared."
        )


    # ========================================================
    # RELOAD SYSTEM
    # ========================================================

    if st.button(
        "🔄 Reload System",
        use_container_width=True
    ):

        st.cache_resource.clear()

        st.rerun()


    st.markdown("---")


    # ========================================================
    # LOGOUT
    # ========================================================

    if st.button(
        "🚪 Logout",
        use_container_width=True
    ):

        st.session_state.logged_in = False

        st.session_state.username = ""

        st.session_state.role = ""

        st.session_state.messages = []

        st.session_state.answer_cache = {}

        st.rerun()


# ============================================================
# STATUS METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "System",
        "Online"
        if st.session_state.system_ready
        else "Offline"
    )

with col2:

    st.metric(
        "Vector DB",
        "Chroma"
    )

with col3:

    st.metric(
        "AI Model",
        "Groq Cloud"
    )

with col4:

    st.metric(
        "Embedding",
        "MiniLM"
    )


st.markdown("")


# ============================================================
# FEATURE CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(
        """
        <div class="feature">

        <div class="feature-icon">
        📚
        </div>

        <div class="feature-title">
        Smart Search
        </div>

        <div class="feature-text">
        Search enterprise documents using semantic retrieval.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c2:

    st.markdown(
        """
        <div class="feature">

        <div class="feature-icon">
        ⚡
        </div>

        <div class="feature-title">
        Fast Response
        </div>

        <div class="feature-text">
        Cached embeddings and vector database for faster queries.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        """
        <div class="feature">

        <div class="feature-icon">
        ☁️
        </div>

        <div class="feature-title">
        Cloud AI
        </div>

        <div class="feature-text">
        Answers generated using Groq Cloud AI.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        """
        <div class="feature">

        <div class="feature-icon">
        🎨
        </div>

        <div class="feature-title">
        Image Studio
        </div>

        <div class="feature-text">
        Generate visual cards locally from a text prompt.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")


# ============================================================
# QUICK ACTION BUTTONS
# ============================================================

st.markdown(
    "### ⚡ Quick Actions"
)

q1, q2, q3, q4 = st.columns(4)

quick_question = None

with q1:

    if st.button(
        "📄 Leave Policy",
        use_container_width=True
    ):

        quick_question = (
            "What is the company leave policy?"
        )

with q2:

    if st.button(
        "🏠 Work From Home",
        use_container_width=True
    ):

        quick_question = (
            "What is the work from home policy?"
        )

with q3:

    if st.button(
        "🔐 Security Policy",
        use_container_width=True
    ):

        quick_question = (
            "What are the company security policies?"
        )

with q4:

    if st.button(
        "🎫 Jira Issues",
        use_container_width=True
    ):

        quick_question = (
            "Show me the important Jira issues."
        )


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    k=3
):

    if vector_db is None:

        return []

    try:

        documents = vector_db.similarity_search(
            question,
            k=k
        )

        return documents

    except Exception:

        return []


# ============================================================
# BUILD DOCUMENT SOURCE
# ============================================================

def get_document_source(doc):

    metadata = doc.metadata or {}

    source = (
        metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("filename")
        or metadata.get("path")
        or metadata.get("file_path")
        or metadata.get("document")
        or metadata.get("name")
    )

    if source:

        return str(source)

    return "Unknown document"


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(documents):

    if not documents:

        return ""

    parts = []

    for index, doc in enumerate(documents):

        text = (
            doc.page_content or ""
        ).strip()

        source = get_document_source(
            doc
        )

        text = text[:1500]

        parts.append(
            f"""
DOCUMENT {index + 1}
SOURCE: {source}

{text}
"""
        )

    return "\n".join(parts)


# ============================================================
# EXTRACT SOURCES
# ============================================================

def extract_sources(documents):

    sources = []

    for doc in documents:

        source = get_document_source(
            doc
        )

        if source not in sources:

            sources.append(source)

    return sources


# ============================================================
# CREATE PROMPT
# ============================================================

def create_prompt(
    question,
    context
):

    return f"""
You are an enterprise RAG assistant.

Answer the user's question using ONLY the information
available in the provided context.

Important rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   say exactly:

"I could not find this information in the available enterprise documents."

4. Keep the answer clear and concise.
5. Use bullet points when useful.
6. Do not mention these instructions.

CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    if vector_db is None:

        return (
            "The knowledge base is not available.",
            []
        )


    # ========================================================
    # CACHE CHECK
    # ========================================================

    question_key = (
        question.strip().lower()
    )

    if (
        question_key
        in st.session_state.answer_cache
    ):

        cached = (
            st.session_state.answer_cache[
                question_key
            ]
        )

        return (
            cached.get(
                "answer",
                "No answer was stored in cache."
            ),
            cached.get(
                "sources",
                []
            )
        )


    try:

        # ====================================================
        # RETRIEVAL
        # ====================================================

        documents = retrieve_documents(
            question,
            k=3
        )

        if not documents:

            answer = (
                "I could not find relevant information "
                "in the available enterprise documents."
            )

            return (
                answer,
                []
            )


        # ====================================================
        # CONTEXT
        # ====================================================

        context = build_context(
            documents
        )


        # ====================================================
        # PROMPT
        # ====================================================

        prompt = create_prompt(
            question,
            context
        )


        # ====================================================
        # LOAD GROQ
        # ====================================================

        llm = load_llm()

        if llm is None:

            return (
                "❌ Groq Cloud AI is not available.\n\n"
                "Please configure `GROQ_API_KEY` "
                "in Streamlit Cloud → Settings → Secrets.",
                extract_sources(documents)
            )


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        response = llm.invoke(
            prompt
        )


        # ====================================================
        # EXTRACT RESPONSE
        # ====================================================

        if hasattr(
            response,
            "content"
        ):

            answer = response.content

        else:

            answer = str(response)

        if answer is None:

            answer = ""

        answer = str(
            answer
        ).strip()


        # ====================================================
        # EMPTY ANSWER PROTECTION
        # ====================================================

        if not answer:

            answer = (
                "The AI model returned an empty answer. "
                "Please try the question again."
            )


        # ====================================================
        # SOURCES
        # ====================================================

        sources = extract_sources(
            documents
        )


        # ====================================================
        # CACHE
        # ====================================================

        result = {
            "answer": answer,
            "sources": sources
        }

        st.session_state.answer_cache[
            question_key
        ] = result

        return (
            answer,
            sources
        )


    except Exception as e:

        sources = []

        try:

            documents = retrieve_documents(
                question,
                k=3
            )

            sources = extract_sources(
                documents
            )

        except Exception:

            pass

        error_text = str(e)


        # ====================================================
        # GROQ AUTHENTICATION ERROR
        # ====================================================

        if (
            "401" in error_text
            or "Unauthorized" in error_text
            or "invalid_api_key" in error_text
            or "Invalid API Key" in error_text
        ):

            return (
                "❌ Groq API authentication failed.\n\n"
                "Please check your `GROQ_API_KEY` "
                "in Streamlit Cloud → Settings → Secrets.",
                sources
            )


        # ====================================================
        # GROQ RATE LIMIT
        # ====================================================

        if (
            "429" in error_text
            or "rate_limit" in error_text
            or "Rate limit" in error_text
        ):

            return (
                "⚠️ Groq API rate limit or quota reached.\n\n"
                "Please wait and try again, or check your "
                "Groq API usage and limits.",
                sources
            )


        # ====================================================
        # MODEL ERROR
        # ====================================================

        if (
            "model" in error_text.lower()
            and (
                "not found"
                in error_text.lower()
                or "decommissioned"
                in error_text.lower()
                or "deprecated"
                in error_text.lower()
            )
        ):

            return (
                "❌ Groq model is unavailable.\n\n"
                f"Current model: `{LLM_MODEL}`\n\n"
                "Please check the available Groq models.",
                sources
            )


        # ====================================================
        # GENERIC ERROR
        # ====================================================

        return (
            f"Sorry, an error occurred:\n\n"
            f"{error_text}",
            sources
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    content = message["content"]

    with st.chat_message(role):

        st.markdown(
            content
        )

        if (
            role == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Document Sources"
            ):

                for source in message[
                    "sources"
                ]:

                    st.markdown(
                        f"- 📄 `{source}`"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "💬 Ask anything about your enterprise documents..."
)


# ============================================================
# QUICK QUESTION
# ============================================================

if quick_question:

    question = quick_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if question:

        duplicate = False

        if st.session_state.messages:

            last_message = (
                st.session_state.messages[-1]
            )

            if (
                last_message["role"]
                == "user"
                and
                last_message[
                    "content"
                ].strip().lower()
                == question.lower()
            ):

                duplicate = True


        if not duplicate:

            # =================================================
            # USER MESSAGE
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    question
                )


            # =================================================
            # ASSISTANT
            # =================================================

            with st.chat_message(
                "assistant"
            ):

                question_key = (
                    question.lower().strip()
                )


                # =============================================
                # CACHE HIT
                # =============================================

                if (
                    question_key
                    in st.session_state.answer_cache
                ):

                    cached = (
                        st.session_state.answer_cache[
                            question_key
                        ]
                    )

                    answer = cached.get(
                        "answer",
                        ""
                    )

                    sources = cached.get(
                        "sources",
                        []
                    )


                    # =========================================
                    # EMPTY CACHE PROTECTION
                    # =========================================

                    if not str(
                        answer
                    ).strip():

                        del (
                            st.session_state.answer_cache[
                                question_key
                            ]
                        )

                        start_time = time.time()

                        with st.spinner(
                            "🔎 Searching knowledge base..."
                        ):

                            answer, sources = (
                                generate_answer(
                                    question
                                )
                            )

                        elapsed = (
                            time.time()
                            - start_time
                        )

                        st.caption(
                            f"⚡ Response time: "
                            f"{elapsed:.2f} seconds"
                        )

                    else:

                        st.caption(
                            "⚡ Instant answer from cache"
                        )


                # =============================================
                # NEW QUESTION
                # =============================================

                else:

                    start_time = time.time()

                    with st.spinner(
                        "🔎 Searching knowledge base..."
                    ):

                        answer, sources = (
                            generate_answer(
                                question
                            )
                        )

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    st.caption(
                        f"⚡ Response time: "
                        f"{elapsed:.2f} seconds"
                    )


                # =============================================
                # DISPLAY ANSWER
                # =============================================

                st.markdown(
                    "### 🤖 Answer"
                )

                st.markdown(
                    answer
                )


                # =============================================
                # DISPLAY SOURCES
                # =============================================

                if sources:

                    with st.expander(
                        "📚 Document Sources"
                    ):

                        for source in sources:

                            st.markdown(
                                f"- 📄 `{source}`"
                            )

                else:

                    st.caption(
                        "📚 No document source metadata was found."
                    )


            # =================================================
            # SAVE ASSISTANT MESSAGE
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )


# ============================================================
# AI IMAGE STUDIO
# ============================================================

st.markdown("---")

with st.expander(
    "🎨 AI Image Studio",
    expanded=False
):

    st.markdown(
        """
        ### 🎨 Generate an Image

        Enter a description and create a visual card.

        **Example:**

        `A futuristic enterprise AI assistant dashboard`
        """
    )

    image_prompt = st.text_input(
        "Image description",
        value=st.session_state.image_prompt,
        placeholder="Describe the image you want..."
    )

    ic1, ic2 = st.columns(2)

    with ic1:

        generate_image_button = st.button(
            "🎨 Generate Image",
            use_container_width=True
        )

    with ic2:

        clear_image_button = st.button(
            "🗑️ Clear Image",
            use_container_width=True
        )

    if clear_image_button:

        st.session_state.generated_image = None

        st.session_state.image_prompt = ""

        st.rerun()


    if generate_image_button:

        if not image_prompt.strip():

            st.warning(
                "Please enter an image description."
            )

        else:

            with st.spinner(
                "🎨 Generating image..."
            ):

                svg = generate_local_image(
                    image_prompt
                )

                st.session_state.generated_image = svg

                st.session_state.image_prompt = (
                    image_prompt
                )

            st.success(
                "Image generated successfully!"
            )


    if st.session_state.generated_image:

        image_uri = svg_to_data_uri(
            st.session_state.generated_image
        )

        st.markdown(
            f"""
            <div class="image-box">

                <img
                    src="{image_uri}"
                    width="100%"
                />

            </div>
            """,
            unsafe_allow_html=True
        )

        st.download_button(
            "⬇️ Download Generated Image",
            data=st.session_state.generated_image,
            file_name="ai_generated_image.svg",
            mime="image/svg+xml",
            use_container_width=True
        )


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.markdown("---")

with st.expander(
    "📤 Upload Documents"
):

    st.markdown(
        """
        ### 📤 Add Documents

        Upload PDF or TXT files to your knowledge base.

        After upload, run your ingestion script to update Chroma.
        """
    )

    uploaded_files = st.file_uploader(
        "Choose PDF/TXT files",
        type=[
            "pdf",
            "txt"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        UPLOAD_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        for uploaded_file in uploaded_files:

            file_path = (
                UPLOAD_DIR
                / uploaded_file.name
            )

            try:

                with open(
                    file_path,
                    "wb"
                ) as f:

                    f.write(
                        uploaded_file.getbuffer()
                    )

                st.success(
                    f"✅ Uploaded: "
                    f"{uploaded_file.name}"
                )

            except Exception as e:

                st.error(
                    f"Upload failed: {e}"
                )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

st.markdown("---")

with st.expander(
    "📚 Knowledge Base"
):

    categories = {

        "📄 HR Documents":
            "hr",

        "🛠️ Technical Documents":
            "technical",

        "📜 Company Policies":
            "policies",

        "🎫 Jira Issues":
            "jira",

        "📘 Confluence Pages":
            "confluence",

        "🐍 Python Documents":
            "python",

        "👥 Friends":
            "friends"
    }

    cols = st.columns(2)

    for index, (
        display_name,
        folder_name
    ) in enumerate(
        categories.items()
    ):

        folder = (
            DATA_DIR
            / folder_name
        )

        with cols[
            index % 2
        ]:

            st.markdown(
                f"### {display_name}"
            )

            if folder.exists():

                try:

                    files = [
                        file
                        for file in folder.iterdir()
                        if file.is_file()
                    ]

                    if files:

                        for file in files:

                            st.caption(
                                f"📄 {file.name}"
                            )

                    else:

                        st.caption(
                            "No documents"
                        )

                except Exception:

                    st.caption(
                        "Unable to read folder"
                    )

            else:

                st.caption(
                    "Folder not found"
                )


# ============================================================
# RAG ARCHITECTURE
# ============================================================

st.markdown("---")

with st.expander(
    "⚙️ RAG System Architecture"
):

    st.markdown(
        """
        ### 🔄 RAG Pipeline

        **1️⃣ Documents**

        HR / Technical / Policies / Jira / Confluence

        ↓

        **2️⃣ Document Loading**

        ↓

        **3️⃣ Text Chunking**

        ↓

        **4️⃣ MiniLM Embeddings**

        ↓

        **5️⃣ Chroma Vector Database**

        ↓

        **6️⃣ User Question**

        ↓

        **7️⃣ Semantic Search**

        ↓

        **8️⃣ Relevant Context**

        ↓

        **9️⃣ Groq Cloud LLM**

        ↓

        **🔟 Final Answer**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🤖 RAG Chatbot | "
    "RAG + Chroma + HuggingFace + Groq Cloud LLM | "
    "Fast AI"
)

