"""
Main script for the RAG Multiple Choice Question Generator.
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import project components
from src.processors.pdf_processor import PDFProcessor
from src.vectorstore.vector_store import VectorStore
from src.question_generator.question_generator import QuestionGenerator
from src.evaluation.question_evaluator import QuestionEvaluator
from configLoader import load_config

config = load_config()


def setup_logging_to_file(log_file):
    """
    Configure logging to write to both console and file.

    Args:
        log_file: Path to the log file

    Returns:
        The file handler object for later removal
    """
    # Create a handler for the log file
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    # Get the root logger and add the file handler
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    return file_handler


def remove_file_handler(handler):
    """
    Remove a file handler from the root logger.

    Args:
        handler: The handler to remove
    """
    root_logger = logging.getLogger()
    root_logger.removeHandler(handler)
    handler.close()


def process_pdf(
    pdf_path: str,
    start_page: int,
    end_page: int,
    vector_store_path: str,
    index_name: str = "default",
) -> VectorStore:
    """
    Process a PDF file and store its chunks in a vector store.

    Args:
        pdf_path: Path to the PDF file
        start_page: First page to process (0-indexed)
        end_page: Last page to process (0-indexed)
        vector_store_path: Path to store the vector database
        index_name: Name for the vector store index

    Returns:
        Initialized vector store with indexed content
    """
    logger.info(f"Processing PDF: {pdf_path}, pages {start_page+1}-{end_page+1}")

    # Initialize PDF processor
    pdf_processor = PDFProcessor(
        embedding_model_name=config["embeddingModel"]["EMBEDDING_MODEL"],
        chunk_size=config["embeddingModel"]["CHUNK_SIZE"],
        chunk_overlap=config["embeddingModel"]["CHUNK_OVERLAP"],
    )

    # Process the PDF to extract chunks
    chunks = pdf_processor.process_pdf(pdf_path, start_page, end_page)
    logger.info(f"Extracted {len(chunks)} chunks from the PDF")

    # Initialize vector store
    vector_store = VectorStore(
        embedding_model_name=config["embeddingModel"]["EMBEDDING_MODEL"],
        vector_store_path=vector_store_path,
    )

    # Index the chunks in the vector store
    vector_store.index_documents(chunks)
    logger.info(f"Indexed {len(chunks)} chunks in the vector store")

    # Save the vector store
    vector_store.save_vector_store(index_name)
    logger.info(f"Saved vector store as '{index_name}'")

    return vector_store


def generate_questions(
    vector_store: VectorStore,
    num_topics: int = 12,
    questions_per_topic: int = 3,
    top_k: int = 4,
    difficulty_range: Optional[List[float]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Generate multiple-choice questions from the vector store content.

    Args:
        vector_store: The vector store containing the document chunks
        num_topics: Number of different topics to retrieve for question generation
        questions_per_topic: Number of questions to generate per topic
        top_k: Number of relevant chunks to retrieve for each topic
        difficulty_range: Range of difficulties to generate questions for

    Returns:
        Tuple of (List of generated questions, Dictionary of contexts)
    """
    # Define topics based on ISLR chapters
    topics = [
        "statistical learning",
        "linear regression",
        "classification",
        "resampling methods",
        "linear model selection and regularization",
        "moving beyond linearity",
        "tree-based methods",
        "support vector machines",
        "deep learning",
        "survival analysis and censored data",
        "unsupervised learning",
        "multiple testing",
    ]

    # Trim the topics list to the requested number
    topics = topics[:num_topics]

    # Set difficulty range if not provided
    if not difficulty_range:
        min_difficulty = float(os.environ.get("MIN_DIFFICULTY", "-2.0"))
        max_difficulty = float(os.environ.get("MAX_DIFFICULTY", "3.0"))
        step = (
            (max_difficulty - min_difficulty) / (len(topics) - 1)
            if len(topics) > 1
            else 0
        )
        difficulty_range = [min_difficulty + i * step for i in range(len(topics))]

    # Initialize question generator
    question_generator = QuestionGenerator()

    all_questions = []
    contexts = {}  # Store contexts for evaluation

    for i, (topic, difficulty) in enumerate(zip(topics, difficulty_range)):
        logger.info(
            f"Generating questions for topic: {topic} (difficulty: {difficulty:.1f})"
        )

        # Search for relevant content in the vector store
        results = vector_store.similarity_search(topic, k=top_k)

        if not results:
            logger.warning(f"No content found for topic: {topic}")
            continue

        # Combine the top results for more context
        combined_content = "\n\n".join([r["content"] for r in results])

        # Generate context ID for evaluation
        context_id = f"topic_{i}"
        contexts[context_id] = combined_content

        # Generate questions for this topic
        questions = question_generator.generate_multiple_choice_questions(
            combined_content, difficulty, questions_per_topic
        )

        # Add topic and source information to each question
        for q in questions:
            q["topic"] = topic
            q["source"] = context_id

        all_questions.extend(questions)
        logger.info(f"Generated {len(questions)} questions for topic: {topic}")

    logger.info(f"Generated a total of {len(all_questions)} questions")
    return all_questions, contexts


def evaluate_questions(
    questions: List[Dict[str, Any]], contexts: Dict[str, str]
) -> Dict[str, Any]:
    """
    Evaluate the quality of generated questions.

    Args:
        questions: List of questions to evaluate
        contexts: Dictionary of contexts used for question generation

    Returns:
        Dictionary with evaluated questions and statistics
    """
    logger.info(f"Evaluating {len(questions)} questions")

    # Initialize question evaluator
    evaluator = QuestionEvaluator()

    # Evaluate questions
    evaluated_questions = evaluator.evaluate_questions(questions, contexts)

    # Calculate overall quality stats
    stats = evaluator.get_overall_quality_stats(evaluated_questions)

    logger.info(
        f"Evaluation complete. Overall quality score: {stats['avg_overall']:.2f}/10"
    )
    logger.info(f"High quality questions: {stats['high_quality_count']}")
    logger.info(f"Medium quality questions: {stats['medium_quality_count']}")
    logger.info(f"Low quality questions: {stats['low_quality_count']}")

    # Add stats to the results
    results = {"questions": evaluated_questions, "stats": stats}

    return results


def save_questions_to_file(questions_data: Dict[str, Any], output_file: str) -> None:
    """
    Save generated questions and evaluation data to a JSON file.

    Args:
        questions_data: Dictionary containing questions and stats
        output_file: Path to save the JSON file
    """
    try:
        with open(output_file, "w") as f:
            json.dump(questions_data, f, indent=2)
        logger.info(f"Questions saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving questions to file: {str(e)}")


def main():
    """
    Main function to run the RAG multiple-choice question generator pipeline.
    """
    parser = argparse.ArgumentParser(
        description="RAG Multiple Choice Question Generator"
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        default=os.path.join("data", "ISLRv2.pdf"),
        help="Path to the PDF file",
    )
    parser.add_argument(
        "--start_page",
        type=int,
        default=config["pdfProcessing"]["PDF_START_PAGE"],
        help="First page to process (0-indexed)",
    )
    parser.add_argument(
        "--end_page",
        type=int,
        default=config["pdfProcessing"]["PDF_END_PAGE"],
        help="Last page to process (0-indexed)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="questions.json",
        help="File to save generated questions to",
    )
    parser.add_argument(
        "--log_file", type=str, default="run.log", help="File to save logging output"
    )

    args = parser.parse_args()

    # Set up logging to file if specified
    file_handler = None
    if args.log_file:
        file_handler = setup_logging_to_file(args.log_file)

    try:
        # Define common parameters
        vector_store_path = config["vectorStore"]["VECTOR_STORE_PATH"]
        index_name = "islr_index"

        logger.info("=== Starting PDF Processing Step ===")
        vector_store = process_pdf(
            args.pdf_path, args.start_page, args.end_page, vector_store_path, index_name
        )
        logger.info("=== PDF Processing Step Completed ===")

        logger.info("=== Starting Question Generation Step ===")
        # Generate questions using default parameters
        questions, contexts = generate_questions(vector_store)

        # Save raw questions
        save_questions_to_file({"questions": questions}, args.output_file)

        # Also save contexts for later evaluation
        context_file = args.output_file.replace(".json", "_contexts.json")
        save_questions_to_file({"contexts": contexts}, context_file)
        logger.info("=== Question Generation Step Completed ===")

        logger.info("=== Starting Question Evaluation Step ===")
        # Evaluate questions
        results = evaluate_questions(questions, contexts)

        # Save evaluated questions
        save_questions_to_file(
            results, args.output_file.replace(".json", "_evaluated.json")
        )
        logger.info("=== Question Evaluation Step Completed ===")

        logger.info(
            f"Pipeline completed successfully. Files saved with base name: {args.output_file}"
        )

    finally:
        # Clean up logging handler if it was created
        if file_handler:
            remove_file_handler(file_handler)


if __name__ == "__main__":
    main()
