# RAG Multiple Choice Question Generator

A system that generates multiple-choice questions from "Introduction to Statistical Learning" textbook using Retrieval Augmented Generation (RAG).

## Overview

This project uses semantic chunking to process a PDF textbook, stores the chunks in a vector database, and then uses an LLM to generate and evaluate multiple-choice questions with varying levels of difficulty. The main components are:

1. **PDF Processor**: Extracts text from the PDF and splits it into semantic chunks.
2. **Vector Store**: Stores and retrieves chunks based on semantic similarity.
3. **Question Generator**: Creates multiple-choice questions with distractors.
4. **Question Evaluator**: Assesses the quality of generated questions.

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

# PDF Processing Configuration
PDF_START_PAGE=14  # 15 in 1-indexed page numbers (first topic starts at page 15)
PDF_END_PAGE=597   # 598 in 1-indexed page numbers (last topic ends at page 598)

# Embedding Model Configuration
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Vector Store Configuration
VECTOR_STORE_PATH="data/vector_store"

# Question Generation Configuration
MIN_DIFFICULTY=-2.0
MAX_DIFFICULTY=3.0
NUM_QUESTIONS_PER_DIFFICULTY=5
```

## Usage

The system can be run with a simple command that executes the entire pipeline:

```bash
# Run the complete pipeline with default settings
python main.py

# Run with custom PDF file and page range
python main.py --pdf_path data/ISLRv2.pdf --start_page 14 --end_page 597 --output_file questions.json --log_file pipeline.log
```

For Windows PowerShell, use either of these approaches:

```powershell
# PowerShell version with backtick (`) for line continuation
python main.py `
  --pdf_path data/ISLRv2.pdf `
  --start_page 14 `
  --end_page 597 `
  --output_file questions.json `
  --log_file pipeline.log

# PowerShell version as a single line
python main.py --pdf_path data/ISLRv2.pdf --start_page 14 --end_page 597 --output_file questions.json --log_file pipeline.log
```

This command will:
1. Process the PDF from pages 15-598 (0-indexed as 14-597)
2. Create and save a vector database at data/vector_store/islr_index
3. Generate questions using the default settings from the configured functions
4. Evaluate all generated questions
5. Save the raw questions to questions.json
6. Save the evaluated questions to questions_evaluated.json
7. Output logs to the specified log file

### Command Line Arguments

- `--pdf_path`: Path to the PDF file (default: `data/ISLRv2.pdf`)
- `--start_page`: First page to process, 0-indexed (default: 14, which is page 15 in the PDF)
- `--end_page`: Last page to process, 0-indexed (default: 597, which is page 598 in the PDF)
- `--output_file`: File to save generated questions to (default: `questions.json`)
- `--log_file`: File to save logging output (default: `run.log`)

## Output

The system produces JSON files containing the generated questions and evaluation results:

- `questions.json`: Contains the raw generated questions.
- `questions_evaluated.json`: Contains questions with evaluation scores and improvement suggestions.

Each question includes:
- The question text
- Four answer choices (with one correct answer)
- Explanation of the correct answer
- Difficulty rating
- Category (e.g., fundamentals, algorithms, techniques)
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
├── main.py
├── requirements.txt
├── .env
└── README.md
```

## Notes

- The system uses the open-source Hugging Face model `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- The Azure OpenAI API is used for question generation and evaluation.
- You can adjust difficulty levels and number of questions in the `.env` file.
- Questions are divided into three levels: basic (-2.0 to -0.5), intermediate (-0.4 to 1.0), and advanced (1.1 to 3.0).

## Team

- This project was created as a learning exercise for exploring RAG applications.