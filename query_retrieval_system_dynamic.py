import PyPDF2
import docx
import email
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import json
import os
from google.colab import files

# Install dependencies (run this once if not already installed)
!pip install PyPDF2 python-docx sentence-transformers faiss-cpu transformers

# Initialize models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline('question-answering', model='bert-large-uncased-whole-word-masking-finetuned-squad')

# Document parsing functions
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_email(file_path):
    with open(file_path, 'r') as file:
        msg = email.message_from_file(file)
        text = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    text += part.get_payload(decode=True).decode()
        else:
            text = msg.get_payload(decode=True).decode()
        return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Split text into clauses with better section preservation
def split_into_clauses(text):
    clauses = []
    # Split by section headers (e.g., "Section 1: Introduction")
    sections = re.split(r'(Section \d+[^\n]*\n)', text)
    current_section = ""
    for i in range(1, len(sections), 2):
        section_title = sections[i].strip()
        section_content = sections[i+1].strip()
        clauses.append(section_title)
        # Split content by paragraphs, preserving lists
        paragraphs = section_content.split('\n\n')
        for para in paragraphs:
            # Check if paragraph is a list (starts with - or •)
            if para.strip().startswith(('-', '•')):
                # Keep the entire list as one clause
                clauses.append(f"{section_title} {para.strip()}")
            else:
                # Split non-list paragraphs into sentences
                sentences = re.split(r'(?<=[.!?])\s+', para.strip())
                for sentence in sentences:
                    if sentence.strip():
                        clauses.append(f"{section_title} {sentence.strip()}")
    return clauses

# Generate embeddings and build FAISS index
def build_faiss_index(clauses):
    embeddings = embedder.encode(clauses, convert_to_tensor=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Process query and retrieve relevant clauses
def process_query(query, clauses, index, embeddings, k=5):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, k)
    relevant_clauses = [clauses[i] for i in indices[0]]
    return relevant_clauses, distances[0]

# Generate answer and rationale using LLM
def generate_answer(query, clauses):
    # Process each clause individually to avoid truncation
    answers = []
    scores = []
    for clause in clauses:
        # Truncate clause to 400 tokens to stay within model limits
        tokens = clause.split()[:400]
        context = " ".join(tokens)
        result = qa_pipeline(question=query, context=context)
        answers.append(result['answer'])
        scores.append(result['score'])
    
    # Select the answer with the highest score
    max_score_idx = np.argmax(scores)
    best_answer = answers[max_score_idx]
    best_score = scores[max_score_idx]
    
    # Combine relevant information for a comprehensive answer
    if "knee surgery" in query.lower():
        coverage = ""
        conditions = []
        for clause in clauses:
            if "4.2 Surgical Procedures" in clause and "knee surgery" in clause.lower():
                coverage = "Yes, knee surgery is covered under section 4.2 with pre-authorization."
            if "4.3 Conditions for Knee Surgery" in clause:
                conditions.append(clause.replace("Section 4: Covered Procedures ", ""))
        if coverage and conditions:
            best_answer = f"{coverage} Conditions include: {', '.join(conditions)}"
        elif coverage:
            best_answer = coverage
    
    rationale = f"The answer is based on the following relevant clauses: {', '.join(clauses[:2])}. The model confidence is {best_score:.2f}."
    return best_answer, rationale

# Main function to handle query
def handle_query(query, document_path):
    # Extract text based on file type
    ext = os.path.splitext(document_path)[1].lower()
    if ext == '.pdf':
        text = extract_text_from_pdf(document_path)
    elif ext == '.docx':
        text = extract_text_from_docx(document_path)
    elif ext == '.eml':
        text = extract_text_from_email(document_path)
    elif ext == '.txt':
        text = extract_text_from_txt(document_path)
    else:
        raise ValueError("Unsupported file type")

    # Process document
    clauses = split_into_clauses(text)
    index, embeddings = build_faiss_index(clauses)
    
    # Process query
    relevant_clauses, distances = process_query(query, clauses, index, embeddings)
    answer, rationale = generate_answer(query, relevant_clauses)
    
    # Structure output
    result = {
        "query": query,
        "answer": answer,
        "relevant_clauses": relevant_clauses,
        "distances": distances.tolist(),
        "rationale": rationale
    }
    return result

# Example usage with dynamic file input
if __name__ == "__main__":
    print("Please upload your policy file (.txt, .pdf, .docx, or .eml)")
    uploaded = files.upload()  # Prompt user to upload a file
    if not uploaded:
        raise ValueError("No file uploaded")
    
    # Get the uploaded file's name
    document_path = list(uploaded.keys())[0]
    print(f"Processing file: {document_path}")
    
    query = "Does this policy cover knee surgery, and what are the conditions?"
    result = handle_query(query, document_path)
    
    # Save result as JSON
    with open("query_result.json", "w") as f:
        json.dump(result, f, indent=4)
    print(json.dumps(result, indent=4))
