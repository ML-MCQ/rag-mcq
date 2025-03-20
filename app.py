"""
Streamlit UI for the RAG Multiple Choice Question Generator.
"""

import os
import sys
import logging
import streamlit as st
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import yaml

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
from src.vectorstore.vector_store import VectorStore
from src.question_generator.question_generator import QuestionGenerator
from configLoader import load_config

# Load configurations
config = load_config()

# Setup page config
st.set_page_config(
    page_title="MCQ Generator",
    page_icon="📚",
    layout="wide",
)


def load_vector_store(vector_store_path: str, index_name: str) -> Optional[VectorStore]:
    """
    Load a pre-existing vector store.

    Args:
        vector_store_path: Path to the vector store
        index_name: Name of the index to load

    Returns:
        Loaded vector store or None if loading fails
    """
    try:
        vector_store = VectorStore(
            embedding_model_name=config["embeddingModel"]["EMBEDDING_MODEL"],
            vector_store_path=vector_store_path,
        )
        vector_store.load_vector_store(index_name, allow_dangerous_deserialization=True)
        logger.info(f"Loaded vector store '{index_name}'")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        return None


def generate_questions_for_topic(
    vector_store: VectorStore,
    topic: str,
    level: str,
    num_questions: int = 3,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """
    Generate multiple-choice questions for a specific topic and level.

    Args:
        vector_store: The vector store containing the document chunks
        topic: Topic to retrieve for question generation
        level: Difficulty level for questions
        num_questions: Number of questions to generate
        top_k: Number of relevant chunks to retrieve for the topic

    Returns:
        List of generated questions
    """
    logger.info(f"Generating {num_questions} {level} questions for topic: {topic}")

    # Check if vector store is valid
    if (
        vector_store is None
        or not hasattr(vector_store, "vector_store")
        or vector_store.vector_store is None
    ):
        st.error(
            "Vector store is not properly initialized. Please run the main.py script first."
        )
        return []

    # Search for relevant content in the vector store
    results = vector_store.similarity_search(topic, k=top_k)

    if not results:
        logger.warning(f"No content found for topic: {topic}")
        return []

    # Combine the top results for more context
    combined_content = "\n\n".join([r["content"] for r in results])

    # Initialize question generator
    question_generator = QuestionGenerator()

    # Generate questions for this topic
    questions = question_generator.generate_multiple_choice_questions(
        combined_content, level, num_questions
    )

    # Add topic information to each question
    for q in questions:
        q["topic"] = topic

    logger.info(f"Generated {len(questions)} {level} questions for topic: {topic}")
    return questions


def display_question(question: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Display a single question with multiple choice options.

    Args:
        question: Question dictionary
        index: Question number/index

    Returns:
        User's answer and whether it was checked
    """
    # Display question
    st.markdown(f"### Question {index+1}: {question['question']}")

    # Display choices
    choices = question.get("choices", {})

    # Create columns for the options
    if len(choices) > 0:
        choice_keys = sorted(choices.keys())
        choice_cols = st.columns(len(choice_keys))

        selected_answer = None

        # Create a radio button for each option
        for i, key in enumerate(choice_keys):
            with choice_cols[i]:
                if st.button(f"{key}: {choices[key]}", key=f"q{index}_{key}"):
                    selected_answer = key

        # Show selected answer, explanation, and feedback
        if selected_answer:
            correct_answer = question.get("correct_answer", "")
            explanation = question.get("explanation", "No explanation provided.")

            if selected_answer == correct_answer:
                st.success(f"✅ Correct! {explanation}")
            else:
                st.error(
                    f"❌ Incorrect. The correct answer is {correct_answer}: {choices.get(correct_answer, '')}"
                )
                st.info(f"Explanation: {explanation}")

            return {
                "answered": True,
                "selected": selected_answer,
                "correct": selected_answer == correct_answer,
            }

    return {"answered": False}


def main():
    """Main function to run the Streamlit app."""
    st.title("Statistical Learning MCQ Generator")

    # Topic selection
    topics = config["questionGeneration"]["TOPICS"]

    # Sidebar
    st.sidebar.title("MCQ Generator")

    # Topic selection
    selected_topic = st.sidebar.selectbox("Select Topic:", topics)

    # Level selection
    levels = config["questionGeneration"]["LEVELS"]
    selected_level = st.sidebar.selectbox("Select Difficulty Level:", levels)

    # Number of questions
    num_questions = st.sidebar.slider(
        "Number of Questions:",
        1,
        config["questionGeneration"]["QUESTIONS_PER_LEVEL"],
        3,
    )

    # Generate button
    generate_btn = st.sidebar.button("Generate Questions")

    # Check if vector store exists before attempting to load
    vector_store_path = config["vectorStore"]["VECTOR_STORE_PATH"]
    index_name = config["vectorStore"].get("INDEX_NAME", "islr_index")
    vector_store_exists = os.path.exists(os.path.join(vector_store_path, index_name))

    if not vector_store_exists:
        st.error(
            "Vector store not found. Please run the main.py script first to process the PDF and create the vector store."
        )
        st.code("python main.py", language="bash")
        return

    # Process generation request
    if generate_btn:
        with st.spinner("Generating questions... This may take a moment."):
            try:
                # Load vector store
                vector_store = load_vector_store(vector_store_path, index_name)

                if not vector_store:
                    st.error(
                        "Failed to load vector store. Please check logs for details."
                    )
                    return

                # Generate questions
                questions = generate_questions_for_topic(
                    vector_store, selected_topic, selected_level, num_questions
                )

                if not questions:
                    st.warning(
                        "No questions could be generated. Try a different topic or settings."
                    )
                    return

                # Store questions in session state
                st.session_state.questions = questions
                st.session_state.current_question = 0
                st.session_state.answered_questions = []

                # Success message
                st.success(
                    f"Generated {len(questions)} questions for {selected_topic} ({selected_level} level)"
                )
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
                logger.exception("Error in question generation")

    # Display questions if they exist
    if hasattr(st.session_state, "questions") and st.session_state.questions:
        questions = st.session_state.questions
        current_idx = st.session_state.current_question

        # Display current question
        if 0 <= current_idx < len(questions):
            result = display_question(questions[current_idx], current_idx)

            # Record answer if provided
            if result["answered"]:
                if current_idx not in st.session_state.answered_questions:
                    st.session_state.answered_questions.append(current_idx)

            # Navigation
            col1, col2 = st.columns(2)

            with col1:
                if current_idx > 0 and st.button("Previous Question"):
                    st.session_state.current_question -= 1
                    st.rerun()

            with col2:
                if current_idx < len(questions) - 1 and st.button("Next Question"):
                    st.session_state.current_question += 1
                    st.rerun()

            # Progress
            progress = len(st.session_state.answered_questions) / len(questions)
            st.progress(
                progress,
                text=f"Answered {len(st.session_state.answered_questions)} of {len(questions)} questions",
            )

    # Footer
    st.markdown("---")
    st.markdown("Statistical Learning MCQ Generator | Made with Streamlit 1.43.2")


if __name__ == "__main__":
    main()
