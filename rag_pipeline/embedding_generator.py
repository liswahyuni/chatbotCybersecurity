from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

Embedding = Union[List[float], np.ndarray]

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Successfully loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            print(f"Error loading SentenceTransformer model '{model_name}': {e}")
            print("Please ensure the model name is correct and you have an internet connection")
            print("if the model needs to be downloaded.")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[Embedding]:
        if not texts:
            print("Warning: No texts provided to generate_embeddings. Returning empty list.")
            return []
        
        if not hasattr(self, 'model') or self.model is None:
            print("Error: SentenceTransformer model is not loaded. Cannot generate embeddings.")
            return []

        try:
            print(f"Generating embeddings for {len(texts)} text chunks...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            print("Embeddings generated successfully.")
            return embeddings.tolist()
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            return []

    def generate_single_embedding(self, text: str) -> Embedding | None:
        if not text:
            print("Warning: No text provided to generate_single_embedding. Returning None.")
            return None

        if not hasattr(self, 'model') or self.model is None:
            print("Error: SentenceTransformer model is not loaded. Cannot generate single embedding.")
            return None
            
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"Error during single embedding generation: {e}")
            return None

if __name__ == '__main__':
    try:
        embed_generator = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")

        sample_texts = [
            "What is a SQL injection attack?",
            "Explain the concept of a zero-day vulnerability.",
            "How does a Distributed Denial of Service (DDoS) attack work?",
            "Phishing is a common social engineering technique."
        ]

        if sample_texts:
            embeddings_list = embed_generator.generate_embeddings(sample_texts)
            
            if embeddings_list:
                print(f"\nGenerated {len(embeddings_list)} embeddings.")
                for i, emb in enumerate(embeddings_list):
                    print(f"Embedding {i+1} (first 5 dimensions): {emb[:5]}... Dimension: {len(emb)}")
            else:
                print("Failed to generate list of embeddings.")

            single_text = "Cross-Site Scripting (XSS) allows attackers to inject malicious scripts."
            single_embedding = embed_generator.generate_single_embedding(single_text)

            if single_embedding:
                print(f"\nGenerated single embedding (first 5 dimensions): {single_embedding[:5]}... Dimension: {len(single_embedding)}")
            else:
                print("Failed to generate single embedding.")
        else:
            print("Sample texts list is empty.")

    except Exception as e:
        print(f"An error occurred in the example usage: {e}")
        print("This might be due to model download issues or other runtime errors.")