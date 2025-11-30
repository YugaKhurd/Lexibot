import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time

# --- CONFIGURATION ---
RAW_DATA_DIR = Path("./data/raw")
PINECONE_INDEX_NAME = "lexibot-legal"

# --- INITIALIZATION ---
print("Initializing...")
load_dotenv() # Loads variables from .env file

# Load Embedding Model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION = 384
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    exit()

# Initialize Pinecone
# --- FIX: Correctly load API Key from environment variables ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print("Error: PINECONE_API_KEY not found. Please set it in your .env file.")
    exit()
    
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    exit()

# (The PDF extraction and chunking functions remain the same)
def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from: {pdf_path.name}")
    try:
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        text = re.sub(r'\s*\n\s*', '\n', text).strip()
        return text
    except Exception as e:
        print(f"Could not read {pdf_path.name}: {e}")
        return ""

def chunk_text_by_section(text, act_name):
    print(f"Chunking text for {act_name}...")
    pattern = re.compile(r'\n(\d{1,3}[A-Z]?\.\s)', re.MULTILINE)
    
    sections = []
    last_end = 0
    
    content_started = False
    # Heuristic to find where the main content starts (e.g., after "CHAPTER I")
    start_match = re.search(r'CHAPTER\s+I', text, re.IGNORECASE)
    if start_match:
        last_end = start_match.start()

    # Split text into sections based on section number patterns
    for match in pattern.finditer(text, last_end):
        start = match.start()
        chunk_text = text[last_end:start].strip()
        if chunk_text:
            sections.append({"text": chunk_text})
        last_end = start
    
    # Add the last section
    chunk_text = text[last_end:].strip()
    if chunk_text:
        sections.append({"text": chunk_text})

    # Post-process to extract section numbers from the beginning of the text
    processed_sections = []
    for sec in sections:
        match = re.match(r'(\d{1,3}[A-Z]?)\.\s+', sec["text"])
        section_number = match.group(1) if match else "Preamble"
        sec["section_number"] = section_number
        processed_sections.append(sec)
        
    print(f"Found {len(processed_sections)} sections.")
    return processed_sections


# --- PINECONE INDEX MANAGEMENT ---
def create_or_get_index(index_name, dimension):
    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' not found. Creating a new one...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index creating. Please wait a moment for initialization...")
        # Wait for the index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    else:
        print(f"Using existing index: '{index_name}'")
        
    return pc.Index(index_name)


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    index = create_or_get_index(PINECONE_INDEX_NAME, EMBEDDING_DIMENSION)

    user_input = input("Do you want to clear all existing data in the index? (yes/no): ").lower()
    if user_input == 'yes':
        # --- FIX: Check if there are vectors before trying to delete ---
        index_stats = index.describe_index_stats()
        if index_stats.total_vector_count > 0:
            print("Clearing index...")
            index.delete(delete_all=True)
            print("Index cleared.")
        else:
            print("Index is already empty. Skipping delete.")

    all_vectors_to_upsert = []

    for pdf_file in RAW_DATA_DIR.glob("*.pdf"):
        act_name = pdf_file.stem
        raw_text = extract_text_from_pdf(pdf_file)
        if not raw_text:
            continue
        
        sections = chunk_text_by_section(raw_text, act_name)
        
        for i, section in enumerate(sections):
            vector_id = f"{act_name}_{section['section_number']}_{i}"
            text_to_embed = f"Act: {act_name}, Section: {section['section_number']}\n\n{section['text']}"
            
            embedding = model.encode(text_to_embed).tolist()
            
            metadata = {
                "act_name": act_name,
                "section_number": section['section_number'],
                "text": section['text']
            }
            
            all_vectors_to_upsert.append( (vector_id, embedding, metadata) )

    if not all_vectors_to_upsert:
        print("No vectors to upsert. Please check your data/raw directory.")
    else:
        print(f"\nPrepared {len(all_vectors_to_upsert)} total vectors for upserting.")
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(all_vectors_to_upsert), batch_size):
            batch = all_vectors_to_upsert[i:i + batch_size]
            print(f"Upserting batch {i//batch_size + 1}...")
            index.upsert(vectors=batch)
        
        print("\n-----------------------------------------")
        print(f"SUCCESS: Pinecone index '{PINECONE_INDEX_NAME}' is ready.")
        print(f"Total vectors in index: {index.describe_index_stats()['total_vector_count']}")
        print("-----------------------------------------")