# # app/streamlit_app.py

# import os
# import json
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# import boto3

# # Load environment variables
# load_dotenv()

# # --------------------------
# # Config
# # --------------------------
# EMBED_MODEL = "all-MiniLM-L6-v2"
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "lexibot-index")
# PINECONE_ENV = os.getenv("PINECONE_ENV")  # optional
# PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")

# # --- CORRECTED MODEL ID ---
# # Using the latest Claude 3.5 Sonnet model ID.
# # This is the primary fix for the "invalid model identifier" error.
# BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

# # --------------------------
# # Pinecone initialization
# # --------------------------
# try:
#     from pinecone import Pinecone, ServerlessSpec
# except Exception as e:
#     raise RuntimeError(
#         "Pinecone import failed. Remove old `pinecone-client` and install `pinecone`:\n"
#         "   python -m pip uninstall -y pinecone-client pinecone\n"
#         "   python -m pip install pinecone"
#     ) from e

# if not PINECONE_API_KEY:
#     raise RuntimeError("PINECONE_API_KEY not set in environment")

# pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV) if PINECONE_ENV else Pinecone(api_key=PINECONE_API_KEY)

# # Create index if not exists
# existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
# if PINECONE_INDEX_NAME not in existing_indexes:
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
#     )

# index = pc.Index(PINECONE_INDEX_NAME)

# # --------------------------
# # Load embedding model
# # --------------------------
# @st.cache_resource
# def load_embed_model():
#     return SentenceTransformer(EMBED_MODEL)

# # --------------------------
# # Retrieve documents
# # --------------------------
# def retrieve(query, k=4):
#     if not query or not query.strip():
#         return []

#     model = load_embed_model()
#     qv = model.encode([query], convert_to_numpy=True)[0]
#     qv_list = qv.tolist() if hasattr(qv, "tolist") else list(qv)

#     results = index.query(vector=qv_list, top_k=k, include_metadata=True)
#     # support dict or object response shapes
#     matches = []
#     if isinstance(results, dict):
#         matches = results.get("matches", []) or []
#     else:
#         matches = getattr(results, "matches", []) or []

#     hits = []
#     for match in matches:
#         if isinstance(match, dict):
#             meta = match.get("metadata", {}) or {}
#             mid = match.get("id")
#             score = match.get("score", 0)
#         else:
#             meta = getattr(match, "metadata", {}) or {}
#             mid = getattr(match, "id", None)
#             score = getattr(match, "score", 0)

#         # common metadata keys that may hold the text
#         text = (
#             meta.get("text")
#             or meta.get("title")
#             or meta.get("content")
#             or meta.get("body")
#             or meta.get("excerpt")
#             or ""
#         )

#         if not text:
#             text = meta.get("source_url") or f"(no text in metadata, id={mid})"

#         hits.append({
#             "id": str(mid),
#             "score": float(score) if score is not None else 0.0,
#             "text": text,
#             "meta": meta,
#             "_raw": match
#         })
#     return hits

# # --------------------------
# # Call AWS Bedrock Claude 3.5 Sonnet
# # --------------------------
# def call_bedrock(prompt, max_tokens=1024):
#     """
#     Invokes a Claude 3 model on AWS Bedrock using the modern Messages API format.
#     """
#     client = boto3.client(
#         "bedrock-runtime",
#         region_name=BEDROCK_REGION,
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     )

#     # New payload format required for Claude 3 models (Messages API)
#     payload = {
#         "anthropic_version": "bedrock-2023-05-31",
#         "max_tokens": max_tokens,
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [{"type": "text", "text": prompt}]
#             }
#         ]
#     }

#     try:
#         resp = client.invoke_model(
#             modelId=BEDROCK_MODEL_ID,
#             body=json.dumps(payload)
#         )
#         body = resp["body"].read()
#         rjson = json.loads(body)
        
#         # Parse the new response structure
#         text = rjson['content'][0]['text']

#     except Exception as e:
#         if "AccessDeniedException" in str(e):
#              text = ("Bedrock call failed: Access Denied. "
#                     "Please ensure you have enabled access to the selected model (e.g., Claude 3.5 Sonnet) in the AWS Bedrock console's 'Model access' page.")
#         else:
#             text = f"Bedrock call failed: {e}"

#     return text

# # --------------------------
# # Streamlit UI
# # --------------------------
# st.set_page_config(page_title="LexiBot", layout="centered")
# st.title("⚖️ LexiBot — Legal Case Analyzer (Claude 3.5 Sonnet + Pinecone)")
# st.info("Disclaimer: For educational purposes only. This is not legal advice.")

# with st.form("case_form"):
#     user_case = st.text_area("Describe the incident (e.g., hit-and-run, property dispute, etc.)", height=180)
#     k = st.slider("Number of retrieved law chunks", 1, 8, 4)
#     submitted = st.form_submit_button("Analyze")

# if submitted and user_case.strip():
#     with st.spinner("Retrieving relevant laws..."):
#         try:
#             hits = retrieve(user_case, k=k)
#         except Exception as e:
#             st.error(f"Failed to retrieve from Pinecone: {e}")
#             hits = []

#     st.subheader("Retrieved Law Chunks")
#     if not hits:
#         st.write("No relevant law chunks were found for your query.")
#     else:
#         # show debug if metadata contains no actual text
#         empty_count = sum(1 for h in hits if not h["text"] or h["text"].startswith("(no text in metadata"))
#         if empty_count == len(hits):
#             st.warning("Retrieved matches contain no textual law excerpts. Showing raw metadata for debugging.")
#             st.json([{"id": h["id"], "score": h["score"], "meta": h["meta"]} for h in hits])

#         for h in hits:
#             act = h["meta"].get("act_name", h["meta"].get("act", "Unknown Act"))
#             sec = h["meta"].get("section_number", h["meta"].get("section", "N/A"))
#             st.markdown(f"**{act} — Section {sec}** (score: {h['score']:.3f})")
#             with st.expander("View text & metadata"):
#                 st.write(h["text"] if h["text"] and not h["text"].startswith("(no text in metadata") else "[no text field in metadata]")
#                 st.write("Metadata:")
#                 st.json(h["meta"])

#     contexts = "\n\n".join([
#         f"Section {h['meta'].get('section_number','')} ({h['meta'].get('act_name','')}):\n{h['text']}"
#         for h in hits
#     ])

#     st.subheader("AI Analysis (Claude 3.5 Sonnet)")
#     with st.spinner("Analyzing with Claude 3.5 Sonnet..."):
#         try:
#             if not contexts.strip():
#                 st.error("No law text available to analyze. Check the index metadata (see the debug output above).")
#             else:
#                 prompt = f"""You are an expert legal assistant. Your task is to analyze a user's case based ONLY on the provided legal excerpts.

# 1.  Identify which of the provided law sections are most applicable to the user's case.
# 2.  Briefly explain why each section is relevant.
# 3.  State any potential penalties mentioned in the excerpts, making sure to specify maximums if they are stated.
# 4.  If the provided excerpts are insufficient or the situation is ambiguous, clearly state that and strongly recommend consulting a professional lawyer. Do not invent information or use outside knowledge.

# Here is the user's case:
# <user_case>
# {user_case}
# </user_case>

# Here are the relevant law excerpts:
# <law_excerpts>
# {contexts}
# </law_excerpts>

# Please provide your analysis in a clear, numbered list.
# """
#                 reply = call_bedrock(prompt)
#                 st.markdown(reply)
#         except Exception as e:
#             st.error(f"AI analysis failed: {e}")







# app/streamlit_app.py

import os
import json
import base64
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import boto3
from pinecone import Pinecone, ServerlessSpec

# --- Page Configuration ---
st.set_page_config(
    page_title="LexiBot - Legal Analyzer",
    layout="wide",
)

# --- Page Icon ---
favicon_emoji = "⚖️"
favicon_svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>
      <text y='.9em' font-size='90'>{favicon_emoji}</text>
    </svg>
""".strip()
st.markdown(
    f'<link rel="icon" href="data:image/svg+xml,{favicon_svg}">',
    unsafe_allow_html=True,
)

# --- Load local CSS ---
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found. Make sure you have a 'style.css' file in the 'app' directory.")
if os.path.exists("app/style.css"):
    local_css("app/style.css")

# Load environment variables
load_dotenv()

# --------------------------
# Config & Initialization
# --------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# --- FIX: Point to the correct index name ---
PINECONE_INDEX_NAME = "lexibot-legal"

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

# Initialize Pinecone
if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY not set. Check your .env file or environment variables.")
    st.stop()

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    # Check if index is ready
    index.describe_index_stats() 
except Exception as e:
    st.error(f"Error connecting to Pinecone index '{PINECONE_INDEX_NAME}': {e}")
    st.info("Please ensure the index exists and you have the correct API key.")
    st.stop()

# --------------------------
# Backend Functions
# --------------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)

def retrieve(query, k=4):
    if not query.strip():
        return []
    model = load_embed_model()
    try:
        query_embedding = model.encode([query])[0].tolist()
        results = index.query(vector=query_embedding, top_k=k, include_metadata=True)
        return results.get("matches", [])
    except Exception as e:
        st.error(f"Error during Pinecone query: {e}")
        return []

def call_bedrock(prompt, max_tokens=2048):
    if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        st.error("AWS credentials not found. Check your .env file or environment variables.")
        return "**Error:** AWS credentials not configured."
        
    try:
        client = boto3.client(
            "bedrock-runtime",
            region_name=BEDROCK_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }

        response = client.invoke_model(modelId=BEDROCK_MODEL_ID, body=json.dumps(payload))
        response_body = json.loads(response.get("body").read())
        return response_body['content'][0]['text']
        
    except Exception as e:
        error_message = f"**Bedrock call failed:** {str(e)}"
        if "ValidationException" in str(e):
            error_message += "\n\n**Suggestion:** The Bedrock model ID might be incorrect or you may not have access to it. Please verify the `BEDROCK_MODEL_ID` in your code/environment variables and ensure you have enabled access to this model in the AWS Bedrock console."
        return error_message


# --------------------------
# Streamlit UI
# --------------------------
st.markdown("<h1 style='text-align: center;'>LexiBot Legal Analyzer</h1>", unsafe_allow_html=True)
st.info("Describe an incident, and the AI will retrieve relevant legal texts and provide a detailed analysis.", icon="💡")

with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form("case_form"):
        st.subheader("📝 Describe Your Case")
        user_case = st.text_area("Enter the details of the incident:", height=150, label_visibility="collapsed", placeholder="e.g., A car accident occurred where one driver ran a red light...")
        
        col_slider, col_button = st.columns([0.7, 0.3])
        with col_slider:
            k = st.slider("Number of law chunks to retrieve", 1, 8, 4, help="Adjust how many similar law sections to fetch.")
        with col_button:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Analyze Case", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

if submitted and user_case.strip():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 Retrieved Legal Texts")
        with st.spinner("Searching for relevant laws..."):
            hits = retrieve(user_case, k=k)
            if not hits:
                st.warning("No relevant law chunks were found for your query.")
            else:
                for h in hits:
                    meta = h.get("metadata", {})
                    act_name = meta.get("act_name", "Unknown Act")
                    sec_num = meta.get("section_number", "N/A")
                    text = meta.get("text", "No text found in metadata.")
                    
                    with st.expander(f"**{act_name} — Section {sec_num}** (Score: {h.get('score', 0):.2f})"):
                        st.write(text)

    with col2:
        st.subheader("🤖 AI-Powered Analysis")
        if not hits:
            st.info("Analysis will appear here once relevant legal texts are found.")
        else:
            with st.spinner("Generating legal analysis..."):
                contexts = "\n\n".join([f"Source: {h.get('metadata', {}).get('act_name', 'N/A')}, Section: {h.get('metadata', {}).get('section_number', 'N/A')}\nContent: {h.get('metadata', {}).get('text', '')}" for h in hits])
                
                prompt = f"""You are an expert legal assistant. Based ONLY on the provided legal excerpts below, analyze the user's case.

Here is the user's case:
<user_case>
{user_case}
</user_case>

Here are the relevant law excerpts retrieved from the knowledge base:
<law_excerpts>
{contexts}
</law_excerpts>

Please provide a clear analysis, identify the most relevant laws from the excerpts, explain why they are relevant, and suggest what legal recourse the user might have according to these texts. Conclude with a disclaimer that this is not legal advice.
"""
                reply = call_bedrock(prompt)
                st.markdown(reply)
else:
    st.success("Enter a case description above and click 'Analyze Case' to begin.")