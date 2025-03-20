"""
Question Generator module for generating multiple-choice questions.
"""

import os
import json
import logging
import random
from typing import List, Dict, Any, Optional

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from configLoader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    A class for generating multiple-choice questions based on retrieved content.

    This class uses an LLM to generate multiple-choice questions from context,
    with varying difficulty levels and topics.
    """

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        sample_questions: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize the question generator with Azure OpenAI configuration.

        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            azure_deployment_name: Azure OpenAI deployment name
            azure_api_version: Azure OpenAI API version
            sample_questions: List of sample questions for few-shot prompting
        """
        # Get Azure OpenAI credentials from environment variables if not provided
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_deployment_name = azure_deployment_name or os.environ.get(
            "AZURE_OPENAI_DEPLOYMENT_NAME"
        )
        self.azure_api_version = azure_api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION"
        )

        # Initialize Azure OpenAI client with higher temperature for more creative outputs
        self.llm = AzureChatOpenAI(
            openai_api_key=self.azure_api_key,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment_name,
            api_version=self.azure_api_version,
            temperature=0.2,
        )

        # Load sample questions for few-shot prompting
        self.sample_questions = (
            sample_questions or self._load_default_sample_questions()
        )

        # Load topics and levels from configuration
        config = load_config()
        self.topics = config["questionGeneration"]["TOPICS"]
        self.levels = config["questionGeneration"]["LEVELS"]

    def _load_default_sample_questions(self) -> List[Dict[str, Any]]:
        """
        Load default sample questions with answers for few-shot prompting.
        Each sample includes a question, multiple choices, correct answer, and explanation.

        Returns:
            List of sample question dictionaries
        """
        return [
            {
                "question": "What is supervised learning?",
                "choices": {
                    "A": "Learning without any labeled data",
                    "B": "Learning from labeled training data with known outputs",
                    "C": "Clustering data points without supervision",
                    "D": "Reducing the dimensionality of data",
                },
                "correct_answer": "B",
                "explanation": "Supervised learning involves training models using labeled data where the correct outputs are known, allowing the algorithm to learn the mapping between inputs and outputs.",
            },
            {
                "question": "What is gradient descent?",
                "choices": {
                    "A": "A method for increasing model error",
                    "B": "A technique for data preprocessing",
                    "C": "An optimization algorithm that minimizes the loss function",
                    "D": "A way to visualize high-dimensional data",
                },
                "correct_answer": "C",
                "explanation": "Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize the loss function by moving in the direction of steepest descent.",
            },
            {
                "question": "What is the purpose of cross-validation?",
                "choices": {
                    "A": "To make the model more complex",
                    "B": "To evaluate model performance on unseen data",
                    "C": "To increase model training time",
                    "D": "To reduce the number of features",
                },
                "correct_answer": "B",
                "explanation": "Cross-validation is used to assess how well a model generalizes to unseen data by splitting the dataset into training and validation sets multiple times.",
            },
        ]

    def _create_prompt_with_examples(
        self, context: str, level: str, num_questions: int = 1
    ) -> str:
        """
        Create a prompt with few-shot examples for question generation.

        Args:
            context: The text content to generate questions from
            level: The difficulty level (basic, intermediate, or advanced)
            num_questions: Number of questions to generate

        Returns:
            Formatted prompt with examples
        """
        # Choose examples based on level - try to match level, otherwise use any samples
        samples = [q for q in self.sample_questions if q.get("level", "basic") == level]
        if not samples:
            samples = self.sample_questions[:3]
        else:
            samples = samples[:3]

        # Format examples for the prompt in a user-friendly way
        sample_text = ""
        for i, sample in enumerate(samples):
            sample_text += f"Example {i+1}:\n"
            sample_text += f"Question: {sample['question']}\n"

            # Include choices if available
            if "choices" in sample:
                sample_text += "Choices:\n"
                for letter, choice in sample["choices"].items():
                    sample_text += f"  {letter}: {choice}\n"
                sample_text += f"Correct Answer: {sample['correct_answer']}\n"
                if "explanation" in sample:
                    sample_text += f"Explanation: {sample['explanation']}\n"
                if "category" in sample:
                    sample_text += f"Category: {sample['category']}\n"
                sample_text += f"Level: {level}\n"

            sample_text += "\n"

        # Add a complete example showing the expected JSON output format
        complete_example = """
        EXPECTED JSON OUTPUT FORMAT:
        {
        "question": "Which of the following is a characteristic of supervised learning?",
        "choices": {
            "A": "The algorithm learns from unlabeled data",
            "B": "The algorithm groups similar data points together",
            "C": "The algorithm learns from labeled training data",
            "D": "The algorithm reduces dimensionality of the input data"
        },
        "correct_answer": "C",
        "explanation": "Supervised learning algorithms learn from labeled training data where the correct outputs are provided. This allows the algorithm to learn the relationship between inputs and outputs.",
        "category": "statistical learning",
        "level": "basic"
        }
        """

        # Create the prompt
        prompt = f"""
You are an expert creator of multiple-choice questions for statistical learning topics.
Your task is to generate {num_questions} challenging multiple-choice questions based on the provided content, with the following requirements:

1. The questions must be directly related to the content provided.
2. Each question should have one correct answer and three plausible but incorrect answers.
3. The questions should be at the "{level}" level of difficulty.
4. Assign an appropriate category from this list: {', '.join(self.topics)}
5. Make sure questions test understanding rather than just recall.

Here are some examples of {level} level questions to help you gauge the complexity:

{sample_text}

Below is the exact JSON format I want you to use for your response:
{complete_example}

Now, based on the following content, create {num_questions} multiple-choice questions at {level} level:

CONTENT:
{context}

For each question, provide:
1. The question text
2. Four answer choices (A, B, C, D) where exactly one is correct
3. The letter of the correct answer
4. An explanation why the correct answer is right and others are wrong
5. An appropriate category from the list provided
6. The level ("{level}")

Format your response as a JSON list where each question is an object with fields: "question", "choices", "correct_answer", "explanation", "category", and "level".
"""
        return prompt

    def generate_multiple_choice_questions(
        self, context: str, level: str = "basic", num_questions: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple-choice questions based on the provided context.

        Args:
            context: The text content to generate questions from
            level: The difficulty level (basic, intermediate, or advanced)
            num_questions: Number of questions to generate

        Returns:
            List of generated question dictionaries
        """
        if not context or not context.strip():
            logger.warning("Empty context provided for question generation")
            return []

        # Validate level
        if level not in self.levels:
            logger.warning(f"Invalid level: {level}. Using 'basic' as default.")
            level = "basic"

        max_retries = 3
        current_temp = self.llm.temperature

        for attempt in range(max_retries):
            try:
                # Create prompt with examples
                prompt = self._create_prompt_with_examples(
                    context, level, num_questions
                )
                prompt += "\n\nIMPORTANT: Your response must be ONLY valid JSON. Do not include any additional text before or after the JSON array."

                # Get LLM response
                response = self.llm.invoke([HumanMessage(content=prompt)])
                response_content = response.content.strip()

                # Parse and validate the response
                questions = self._parse_llm_response(response_content)
                valid_questions = self._validate_and_format_questions(questions)

                if valid_questions:
                    # Reset temperature if it was changed
                    self.llm.temperature = current_temp
                    return valid_questions

                # If we got here, no valid questions were found
                if attempt == max_retries - 1:
                    logger.warning("No valid questions found in LLM response")
                    self.llm.temperature = current_temp
                    return []

                # Adjust temperature for next retry
                self.llm.temperature = min(2.0, self.llm.temperature + 0.1)
                logger.info(
                    f"Retry {attempt+1}/{max_retries}: No valid questions found"
                )

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error generating questions: {str(e)}")
                    self.llm.temperature = current_temp
                    return []

                self.llm.temperature = min(2.0, self.llm.temperature + 0.1)
                logger.info(f"Retry {attempt+1}/{max_retries}: Error - {str(e)}")

        # Reset temperature and return empty list if we exit the loop without returning
        self.llm.temperature = current_temp
        return []

    def _parse_llm_response(self, response_content: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response and extract JSON content.

        Args:
            response_content: Raw response from the LLM

        Returns:
            List of question dictionaries

        Raises:
            json.JSONDecodeError: If the response cannot be parsed as JSON
        """
        # If response starts with ```json and ends with ```, extract the content
        if response_content.startswith("```json") and "```" in response_content[7:]:
            json_text = (
                response_content.split("```json", 1)[1].split("```", 1)[0].strip()
            )
            return json.loads(json_text)

        # If response is wrapped in code block
        if response_content.startswith("```") and response_content.endswith("```"):
            json_text = response_content[3:-3].strip()
            return json.loads(json_text)

        # Try to find JSON brackets for an array
        start_idx = response_content.find("[")
        end_idx = response_content.rfind("]") + 1

        if start_idx != -1 and end_idx != -1:
            json_str = response_content[start_idx:end_idx]
            return json.loads(json_str)

        # Last resort: try to parse the entire response as JSON
        questions = json.loads(response_content)

        # If it's not an array, wrap it
        if not isinstance(questions, list):
            questions = [questions]

        return questions

    def _validate_and_format_questions(
        self, questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate question fields and format choices consistently.

        Args:
            questions: List of question dictionaries to validate

        Returns:
            List of valid, formatted question dictionaries
        """
        required_fields = [
            "question",
            "choices",
            "correct_answer",
            "explanation",
            "category",
            "level",
        ]

        valid_questions = []
        for i, q in enumerate(questions):
            valid = True
            for field in required_fields:
                if field not in q:
                    logger.warning(f"Question {i+1} missing required field: {field}")
                    valid = False

            if valid:
                # Make sure choices is properly formatted as a dictionary
                if isinstance(q["choices"], list):
                    letters = ["A", "B", "C", "D", "E"]
                    choices_dict = {}
                    for j, choice in enumerate(q["choices"]):
                        if j < len(letters):
                            choices_dict[letters[j]] = choice
                    q["choices"] = choices_dict

                # Ensure level is one of the valid levels
                if q["level"] not in self.levels:
                    logger.warning(
                        f"Question {i+1} has invalid level: {q['level']}. Setting to 'basic'."
                    )
                    q["level"] = "basic"

                valid_questions.append(q)

        return valid_questions

    def generate_questions_for_topics(
        self,
        contexts: List[str],
        levels_per_context: Optional[List[List[str]]] = None,
        num_questions_per_level: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple-choice questions for multiple contexts/topics at multiple difficulty levels.

        Args:
            contexts: List of text contexts to generate questions from
            levels_per_context: List of lists containing difficulty levels for each context
                               (e.g., [["basic", "intermediate"], ["basic"], ["intermediate", "advanced"]])
            num_questions_per_level: Number of questions to generate per level per context

        Returns:
            List of generated question dictionaries across all contexts
        """
        if not contexts:
            logger.warning("No contexts provided for question generation")
            return []

        # If levels_per_context not provided, use all levels for each context
        if not levels_per_context:
            levels_per_context = [self.levels for _ in contexts]

        # Ensure levels_per_context list matches contexts list length
        if len(levels_per_context) != len(contexts):
            # Fill missing context levels with all available levels
            levels_per_context = levels_per_context[: len(contexts)] + [self.levels] * (
                len(contexts) - len(levels_per_context)
            )

        all_questions = []
        for i, (context, levels) in enumerate(zip(contexts, levels_per_context)):
            logger.info(f"Generating questions for context {i+1}/{len(contexts)}")

            # Generate questions for each level for this context
            for level in levels:
                if level not in self.levels:
                    logger.warning(f"Invalid level: {level}. Skipping.")
                    continue

                logger.info(
                    f"Generating {num_questions_per_level} questions at {level} level"
                )
                questions = self.generate_multiple_choice_questions(
                    context, level, num_questions_per_level
                )
                all_questions.extend(questions)

        return all_questions
