# RAG Multiple Choice Question Generator

A system that generates multiple-choice questions from "Introduction to Statistical Learning" textbook using Retrieval Augmented Generation (RAG).

## Overview

This project uses semantic chunking to process a PDF textbook, stores the chunks in a vector database, and then uses an LLM to generate and evaluate multiple-choice questions with varying levels of difficulty. The main components are:

1. **PDF Processor**: Extracts text from the PDF and splits it into semantic chunks.
2. **Vector Store**: Stores and retrieves chunks based on semantic similarity.
3. **Question Generator**: Creates multiple-choice questions with distractors.
4. **Question Evaluator**: Assesses the quality of generated questions.
5. **Streamlit UI**: Interactive web interface for generating and answering questions.

## Installation

1. Clone this repository.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Make sure you have the `ISLRv2.pdf` file in the `data/` directory.

## Environment Variables

Create a `.env` file with the following variables:

```
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
AZURE_OPENAI_API_KEY="your_api_key"
AZURE_OPENAI_MODEL_VERSION="your_model_version"
AZURE_OPENAI_API_VERSION="your_api_version"
AZURE_OPENAI_DEPLOYMENT_NAME="your_deployment_name"
```

## Configuration

The system uses YAML configuration files located in the `conf/` directory:

- `embedding.yml`: Configuration for embedding model settings
- `pdf.yml`: PDF processing parameters
- `question_generation.yml`: Question generation settings, including difficulty levels and topics
- `vector_store.yml`: Vector store configuration
- `main_config.yml`: Main pipeline settings

You can adjust these settings to customize behavior without modifying code.

## Usage

### Command Line Interface

The system can be run with a simple command that executes the entire pipeline:

```bash
# Run the complete pipeline with default settings
python main.py
```

All settings are managed through the `conf/main_config.yml` file:

```yaml
# Main Configuration for RAG Multiple Choice Question Generator
pdf_path: "data/ISLRv2.pdf"
start_page: 0
end_page: 100
output_file: "questions.json"
log_file: "run.log"
vector_store_path: "data/vector_store"
index_name: "islr_index"
num_topics: 5
questions_per_level: 1
top_k: 4
```

You can modify these settings in the configuration file before running the pipeline:
1. `pdf_path`: Path to the PDF file to process
2. `start_page` and `end_page`: Page range to process (0-indexed)
3. `output_file`: Where to save the generated questions
4. `log_file`: Where to save the logs
5. `num_topics`: Number of topics to generate questions for
6. `questions_per_level`: Number of questions to generate per difficulty level
7. `top_k`: Number of relevant chunks to retrieve for each topic

If you need to use a different configuration file, you can specify it with the `--config` flag:

```bash
python main.py --config path/to/your/config.yml
```

This command will:
1. Process the PDF using the page range in the configuration
2. Create and save a vector database at the specified vector store path
3. Generate questions for the specified number of topics
4. Evaluate all generated questions
5. Save the raw questions, evaluated questions, and contexts as separate JSON files
6. Output logs to the specified log file

### Streamlit Interactive UI

The project includes a Streamlit web application that provides an interactive interface for generating and answering questions:

```bash
# Run the Streamlit app
streamlit run app.py
```

The Streamlit UI allows you to:
- Select specific topics from the textbook
- Choose difficulty levels (basic, intermediate, advanced)
- Specify the number of questions to generate
- Generate questions on-demand
- Test your knowledge by answering the questions
- Get immediate feedback and explanations

**Note:** Before running the Streamlit app, you must first process the PDF and create the vector store by running `main.py`.

## Output

The system produces JSON files containing the generated questions and evaluation results:

- `questions.json`: Contains the raw generated questions.
- `questions_evaluated.json`: Contains questions with evaluation scores and improvement suggestions.
- `questions_contexts.json`: Contains the contexts used for question generation.

Each question includes:
- The question text
- Four answer choices (with one correct answer)
- Explanation of the correct answer
- Topic information
- Level (basic, intermediate, advanced)

## Project Structure

```
rag-multiple-choice-generator/
├── data/
│   ├── ISLRv2.pdf
│   └── vector_store/
├── src/
│   ├── processors/
│   │   └── pdf_processor.py
│   ├── vectorstore/
│   │   └── vector_store.py
│   ├── question_generator/
│   │   └── question_generator.py
│   └── evaluation/
│       └── question_evaluator.py
├── conf/
│   ├── embedding.yml
│   ├── main_config.yml
│   ├── pdf.yml
│   ├── question_generation.yml
│   └── vector_store.yml
├── main.py           # Main pipeline script
├── app.py            # Streamlit UI application
├── configLoader.py   # Configuration loading utility
├── requirements.txt
├── .env
└── README.md
```

## Notes

- The system uses the open-source Hugging Face model `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- The Azure OpenAI API is used for question generation and evaluation.
- Questions are generated at three difficulty levels: basic, intermediate, and advanced.
- The Streamlit UI requires Streamlit version 1.43.2 or higher.
- The vector store uses FAISS for efficient similarity search.
- Configuration is now managed through YAML files instead of environment variables.

## Team

- This project was created as a learning exercise for exploring RAG.