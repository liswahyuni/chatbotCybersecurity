# Cybersecurity RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot specialized in cybersecurity knowledge, designed to provide accurate and contextually relevant information by leveraging local documents and language models.

## Features

- **Document Processing**: Loads, processes, and chunks documents from various formats
- **Vector Store Creation**: Generates embeddings and creates a FAISS vector store for efficient retrieval
- **Semantic Search**: Retrieves relevant context based on user queries
- **Conversational Memory**: Maintains conversation history for contextual responses
- **Evaluation Framework**: Comprehensive evaluation using industry-standard metrics

## Project Structure

```
├── core_logic/            # Core components of the RAG pipeline
├── rag_pipeline/          # Modular components for the RAG system
├── data/                  # Data storage
│   ├── raw/               # Raw documents for knowledge base
│   └── vector_store/      # Generated vector store and metadata
├── evaluation/            # Evaluation scripts and results
├── build_vector_store.py  # Script to build the vector store
├── main.py                # Entry point for the chatbot
└── requirements.txt       # Project dependencies
```

## Dependencies

This project requires the following main dependencies:

- Python 3.9+
- Ollama (for local LLM inference)
- sentence-transformers (for embeddings)
- FAISS (for vector search)
- LangChain (for RAG pipeline components)
- LettuceDetect (for evaluation)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd chatbotCybersecurity
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Install Ollama:

**For macOS:**
```bash
curl -fsSL https://ollama.com/download/ollama-darwin-amd64 -o ollama
chmod +x ollama
sudo mv ollama /usr/local/bin/
```

**For Linux:**
```bash
curl -fsSL https://ollama.com/download/ollama-linux-amd64 -o ollama
chmod +x ollama
sudo mv ollama /usr/local/bin/
```

**For Windows:**
Download the installer from [Ollama's official website](https://ollama.com/download) and follow the installation instructions.

5. Start the Ollama service:

```bash
ollama serve
```

6. Pull the qwen:0.5b model:

```bash
ollama pull qwen:0.5b
```

## Usage

### Building the Vector Store

Before using the chatbot, you need to build the vector store from your documents:

```bash
python build_vector_store.py
```

This will process documents from the `data/raw` directory and create a vector store in `data/vector_store`.

### Running the Chatbot

To start the chatbot:

```bash
python main.py
```

This will initialize the RAG pipeline and start a CLI interface where you can interact with the chatbot.

### Evaluation

To evaluate the RAG pipeline performance:

```bash
python evaluation/evaluate_rag_with_lettucedetect.py
```

This will run the evaluation using the `preemware/pentesting-eval` dataset and save the results in the `evaluation` directory.

## Pipeline Design

The RAG pipeline consists of the following components:

1. **Document Loading and Processing**: Loads documents from various sources and formats
2. **Text Chunking**: Splits documents into manageable chunks
3. **Embedding Generation**: Creates vector representations of text chunks
4. **Vector Store Management**: Stores and retrieves vectors efficiently
5. **Document Retrieval**: Finds relevant documents based on query similarity
6. **LLM Interface**: Communicates with the language model
7. **Response Generation**: Combines retrieved context with LLM capabilities
8. **Conversation Memory**: Maintains context across interactions

## LLM and Dataset Selection

This project uses:

- **LLM**: Ollama with the qwen:0.5b model for efficient local inference
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 for generating embeddings
- **Evaluation Dataset**: preemware/pentesting-eval from Hugging Face

## Evaluation Methodology

The RAG pipeline is evaluated using LettuceDetect, which provides metrics for:

- **Answer Relevancy**: Measures how relevant the generated answers are to the questions
- **Faithfulness**: Assesses if the generated answers are supported by the retrieved context
- **Hallucination Detection**: Identifies when the model generates information not present in the context
- **Response Quality**: Evaluates the overall quality and usefulness of responses

Evaluation results are saved in CSV format in the `evaluation` directory, with a detailed report available in `EVALUATION_REPORT.md`.

## Troubleshooting

### Ollama Connection Issues

If you encounter "Failed to connect to Ollama" errors:

1. Ensure Ollama is installed and running:
   ```bash
   ollama serve
   ```

2. Verify the qwen:0.5b model is available:
   ```bash
   ollama list
   ```

3. Check if you can connect to the Ollama API:
   ```bash
   curl http://localhost:11434/api/tags
   ```

4. If needed, restart the Ollama service:
   ```bash
   # On macOS/Linux
   pkill ollama
   ollama serve
   
   # On Windows
   # Restart using Task Manager or Services
   ```

5. Ensure no firewall or antivirus is blocking the connection to localhost:11434

## Acknowledgements

- The LettuceDetect team for providing evaluation metrics
- Hugging Face for hosting the evaluation dataset
- The Ollama team for making local LLM inference accessible
        