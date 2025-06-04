import os
import glob
from typing import List, Dict, Union, Any

from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    BSHTMLLoader
)
LangchainDocumentDict = Dict[str, Union[str, Dict[str, Any]]]


def load_single_document_with_langchain(file_path: str) -> List[LangchainDocumentDict]:
    docs_to_return: List[LangchainDocumentDict] = []
    
    try:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(".html") or file_path.lower().endswith(".htm"):
            loader = BSHTMLLoader(file_path, open_encoding='utf-8')
        elif file_path.lower().endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_path.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            print(f"Unsupported file type: {file_path}. Skipping.")
            return docs_to_return

        loaded_lc_documents = loader.load()

        for lc_doc in loaded_lc_documents:
            docs_to_return.append({
                "page_content": lc_doc.page_content,
                "metadata": lc_doc.metadata
            })
            
    except Exception as e:
        print(f"Error loading document {file_path} with Langchain loader: {e}")
    
    return docs_to_return


def load_documents_from_directory(directory_path: str) -> List[LangchainDocumentDict]:
    all_loaded_docs: List[LangchainDocumentDict] = []
    
    supported_patterns = ["*.txt", "*.md", "*.pdf", "*.html", "*.htm"]
    
    file_paths = []
    for pattern in supported_patterns:
        file_paths.extend(glob.glob(os.path.join(directory_path, pattern)))

    if not file_paths:
        print(f"No supported documents found in directory: {directory_path}")
        print(f"Supported types are: {', '.join(ext.split('.')[-1] for ext in supported_patterns)}")
        return []

    unique_file_paths = sorted(list(set(file_paths)))

    for file_path in unique_file_paths:
        print(f"Attempting to load document: {file_path}")
        docs_from_file = load_single_document_with_langchain(file_path)
        if docs_from_file:
            all_loaded_docs.extend(docs_from_file)
        else:
            print(f"No content extracted from {file_path} or it was skipped.")
            
    print(f"Successfully processed {len(unique_file_paths)} files and extracted {len(all_loaded_docs)} document sections.")
    return all_loaded_docs

def preprocess_text(text: str) -> str:
    text = ' '.join(text.split())
    return text

if __name__ == '__main__':
    raw_data_test_dir = "data/raw_test_loader"
    if not os.path.exists(raw_data_test_dir):
        os.makedirs(raw_data_test_dir)
    
    dummy_txt_path = os.path.join(raw_data_test_dir, "sample1.txt")
    dummy_md_path = os.path.join(raw_data_test_dir, "sample2.md")
    dummy_pdf_path = os.path.join(raw_data_test_dir, "sample3.pdf")
    dummy_html_path = os.path.join(raw_data_test_dir, "sample4.html")

    if not os.path.exists(dummy_txt_path):
        with open(dummy_txt_path, "w", encoding="utf-8") as f:
            f.write("This is a sample text document about cybersecurity basics.\nIt mentions passwords and 2FA.")
    
    if not os.path.exists(dummy_md_path):
        with open(dummy_md_path, "w", encoding="utf-8") as f:
            f.write("# Sample Markdown\n\nThis discusses *Nmap scans* and **Metasploit**.\n\n- Item 1\n- Item 2")

    if not os.path.exists(dummy_html_path):
        with open(dummy_html_path, "w", encoding="utf-8") as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head><title>HTML Test</title></head>
            <body>
                <h1>Welcome to HTML Test</h1>
                <p>This is a paragraph about <strong>phishing</strong> attacks and how to avoid them.</p>
                <script>console.log("This should be ignored by good text extractors");</script>
                <ul><li>HTML Point 1</li><li>HTML Point 2</li></ul>
            </body>
            </html>
            """)

    print(f"Please place a sample PDF file named 'sample3.pdf' in '{raw_data_test_dir}' to test PDF loading.")
    if not os.path.exists(dummy_pdf_path):
         print(f"Warning: '{dummy_pdf_path}' not found. PDF loading will not be fully tested.")


    print(f"\n--- Loading documents from '{raw_data_test_dir}' ---")
    loaded_docs = load_documents_from_directory(raw_data_test_dir)
    
    if loaded_docs:
        print(f"\nSuccessfully loaded {len(loaded_docs)} document sections.")
        for i, doc_dict in enumerate(loaded_docs):
            print(f"\nDocument Section {i+1}:")
            print(f"  Source: {doc_dict['metadata'].get('source', 'N/A')}")
            if 'page' in doc_dict['metadata']:
                print(f"  Page: {doc_dict['metadata']['page']}")
            print(f"  Content (first 100 chars): {doc_dict['page_content'][:100].strip()}...")
            
            preprocessed_content = preprocess_text(doc_dict['page_content'])
    else:
        print("No document sections were loaded. Please check the 'data/raw_test_loader' directory and ensure files exist.")