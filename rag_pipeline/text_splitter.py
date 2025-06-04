from typing import List, Dict, Union, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

LangchainDocumentDict = Dict[str, Union[str, Dict[str, Any]]]

def split_documents(
    documents: List[LangchainDocumentDict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[LangchainDocumentDict]:
    if not documents:
        print("No documents provided to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    all_chunks = []
    for doc in documents:
        if not isinstance(doc, dict) or "page_content" not in doc or "metadata" not in doc:
            print(f"Skipping invalid document format: {doc}")
            continue
        
        page_content = doc["page_content"]
        metadata = doc["metadata"]
        
        chunks_text = text_splitter.split_text(page_content)
        
        for i, chunk_text in enumerate(chunks_text):
            chunk_metadata = metadata.copy()
            
            chunk_doc = {
                "page_content": chunk_text,
                "metadata": chunk_metadata 
            }
            all_chunks.append(chunk_doc)
            
    print(f"Split {len(documents)} documents into {len(all_chunks)} chunks.")
    return all_chunks

if __name__ == '__main__':
    sample_docs: List[LangchainDocumentDict] = [
        {
            "page_content": "This is a long document about network security. It covers firewalls, IDS, IPS, and VPNs. " * 50,
            "metadata": {"source": "network_security_guide.txt", "chapter": 1}
        },
        {
            "page_content": "Another document focusing on ethical hacking methodologies. Includes reconnaissance, scanning, exploitation, and post-exploitation phases. " * 60,
            "metadata": {"source": "ethical_hacking_manual.md", "version": "2.0"}
        }
    ]

    if not sample_docs[0]["page_content"]:
        print("Sample documents are empty. Exiting example.")
    else:
        chunks = split_documents(sample_docs, chunk_size=200, chunk_overlap=30)
        
        if chunks:
            print(f"\nTotal chunks created: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"\nChunk {i+1}:")
                print(f"Source: {chunk['metadata']['source']}")
                if "start_index" in chunk["metadata"]:
                     print(f"Original Start Index: {chunk['metadata']['start_index']}")
        else:
            print("No chunks were created.")