import os
from typing import List
from rag_pipeline.data_loader import load_documents_from_directory, preprocess_text, LangchainDocumentDict
from rag_pipeline.text_splitter import split_documents
from rag_pipeline.embedding_generator import EmbeddingGenerator, Embedding
from rag_pipeline.vector_store_manager import VectorStoreManager

RAW_DATA_DIR = "data/raw"
VECTOR_STORE_INDEX_PATH = "data/vector_store/faiss_index.index"
VECTOR_STORE_METADATA_PATH = "data/vector_store/faiss_index.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def main():
    print("Starting the process to build the vector store...")

    print(f"\nStep 1: Loading documents from '{RAW_DATA_DIR}'...")
    if not os.path.exists(RAW_DATA_DIR) or not os.listdir(RAW_DATA_DIR):
        print(f"Error: Raw data directory '{RAW_DATA_DIR}' is empty or does not exist.")
        print("Please add your .txt or .md files to this directory.")
        if os.path.exists(RAW_DATA_DIR) and not os.listdir(RAW_DATA_DIR):
             with open(os.path.join(RAW_DATA_DIR, "example_put_your_data_here.txt"), "w") as f:
                 f.write("This is an example file. Replace this with your actual cybersecurity documents.")
             print(f"Created an example file in {RAW_DATA_DIR}. Please replace it with your data.")
        return

    raw_documents: List[LangchainDocumentDict] = load_documents_from_directory(RAW_DATA_DIR)
    if not raw_documents:
        print("No documents were loaded. Exiting.")
        return

    processed_documents: List[LangchainDocumentDict] = []
    for doc in raw_documents:
        processed_content = preprocess_text(doc["page_content"])
        processed_documents.append({
            "page_content": processed_content,
            "metadata": doc["metadata"]
        })

    print("\nStep 2: Splitting documents into chunks...")
    document_chunks: List[LangchainDocumentDict] = split_documents(
        processed_documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    if not document_chunks:
        print("No chunks were created from the documents. Exiting.")
        return
    print(f"Split documents into {len(document_chunks)} chunks.")

    print("\nStep 3: Generating embeddings for chunks...")
    try:
        embedding_gen = EmbeddingGenerator(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Failed to initialize EmbeddingGenerator: {e}. Exiting.")
        return
        
    chunk_texts: List[str] = [chunk["page_content"] for chunk in document_chunks]
    
    embeddings: List[Embedding] = embedding_gen.generate_embeddings(chunk_texts)
    if not embeddings or len(embeddings) != len(document_chunks):
        print("Failed to generate embeddings or mismatch in count. Exiting.")
        return
    print(f"Generated {len(embeddings)} embeddings.")

    print("\nStep 4: Creating and saving vector store...")
    vector_store_mgr = VectorStoreManager(
        index_path=VECTOR_STORE_INDEX_PATH,
        metadata_path=VECTOR_STORE_METADATA_PATH
    )
    try:
        vector_store_mgr.create_and_save_vector_store(document_chunks, embeddings)
        print("Vector store created and saved successfully.")
    except ValueError as ve:
        print(f"ValueError during vector store creation: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during vector store creation: {e}")

    print("\nVector store build process finished.")

if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print(f"Created directory: {RAW_DATA_DIR}")

    if not os.listdir(RAW_DATA_DIR):
        print(f"'{RAW_DATA_DIR}' is empty. Creating a dummy file for initial setup.")
        dummy_file_content = """
        # Introduction to Cybersecurity
        Cybersecurity is the practice of protecting systems, networks, and programs from digital attacks. 
        These attacks are usually aimed at accessing, changing, or destroying sensitive information; 
        extorting money from users; or interrupting normal business processes.

        ## Common Threats
        - Malware (Viruses, Worms, Trojans, Ransomware)
        - Phishing
        - Man-in-the-Middle (MitM) attacks
        - Denial-of-Service (DoS) and Distributed Denial-of-Service (DDoS) attacks
        - SQL Injection
        - Cross-Site Scripting (XSS)

        ## Pentesting Basics
        Penetration testing, also known as pentesting, is a simulated cyber attack against your computer system 
        to check for exploitable vulnerabilities.
        Phases often include:
        1. Reconnaissance
        2. Scanning
        3. Gaining Access (Exploitation)
        4. Maintaining Access
        5. Analysis & Reporting
        """
        with open(os.path.join(RAW_DATA_DIR, "cybersecurity_overview.md"), "w", encoding="utf-8") as f:
            f.write(dummy_file_content)
        print(f"Created dummy file 'cybersecurity_overview.md' in '{RAW_DATA_DIR}'.")
        print("Please replace it with your actual cybersecurity documents for meaningful results.")
    
    main()