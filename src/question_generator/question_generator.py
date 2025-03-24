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
from src.vectorstore.vector_store import VectorStore
import streamlit as st

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

        # Define topics based on ISLR chapters
        self.topics = [
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

        self.levels = ["basic", "intermediate", "advanced"]

        self.complete_example = """
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

    def _load_default_sample_questions(self) -> List[Dict[str, Any]]:
        """
        Load default sample questions with answers for few-shot prompting.
        Each sample includes a question, multiple choices, correct answer, and explanation.

        Returns:
            List of sample question dictionaries
        """
        return [
            {
                "question_goal": "To create a meaningful question that it is easy to understand and has a clear correct answer.",
                "question": "What is supervised learning?",
                "choices": {
                    "A": "Learning from labeled training data with known outputs",
                    "B": "Learning without any labeled data",
                    "C": "Clustering data points without supervision",
                    "D": "Reducing the dimensionality of data",
                },
                "correct_answer": "A",
                "explanation": "Supervised learning involves training models using labeled data where the correct outputs are known, allowing the algorithm to learn the mapping between inputs and outputs.",
            },
            {
                "question_goal": "To create a clear question it is clearly worded, and the answer choices are unambiguous. Some examples should also be given to help the user understand the question.",
                "question": "Which of the following best describes the difference between parametric and non-parametric methods in statistical learning? An example of a parametric method is linear regression, while an example of a non-parametric method is k-nearest neighbors.",
                "choices": {
                    "A": "Parametric methods require a fixed number of parameters, while non-parametric methods do not assume any specific form for the function.",
                    "B": "Parametric methods are always more accurate than non-parametric methods.",
                    "C": "Non-parametric methods require a fixed number of parameters, while parametric methods do not assume any specific form for the function.",
                    "D": "Parametric methods are used only for classification tasks, while non-parametric methods are used for regression tasks."
                },
                "correct_answer": "A",
                "explanation": "Parametric methods involve a model-based approach where a fixed number of parameters are used to define the function, whereas non-parametric methods do not assume a specific form for the function and can adapt to the data more flexibly. This makes parametric methods simpler but potentially less flexible compared to non-parametric methods.",
            },
            {
                "question_goal": "To create a clear question it is clearly worded, and the answer choices are unambiguous. The question is specific enough to distinguish between fundamental concepts without introducing confusion.",
                "question": "What is the primary purpose of cross-validation in machine learning, and how does it contribute to model evaluation?",
                "choices": {
                    "A": "To make the model more complex by using more features",
                    "B": "To increase model training time by training multiple models",
                    "C": "To reduce overfitting by eliminating irrelevant features",
                    "D": "To evaluate model performance on unseen data by splitting the dataset into multiple training and validation sets",
                },
                "correct_answer": "D",
                "explanation": "Cross-validation is a technique used to evaluate how well a model generalizes to unseen data by splitting the dataset into multiple training and validation sets. It helps ensure that the model’s performance is not dependent on a specific train-test split, providing a more robust measure of its generalization ability. Common types of cross-validation include k-fold and stratified k-fold.",
            },
            {
                "question_goal": "To create a thought provoking question where the options reflect the nuances of overfitting without introducing any misleading or confusing statements.",
                "question": "Which of the following best describes the impact of overfitting in a machine learning model, and how can it be mitigated?",
                "choices": {
                    "A": "Overfitting results in high accuracy on both training and test data, and can be mitigated by increasing model complexity",
                    "B": "Overfitting occurs when a model performs well on training data but poorly on unseen test data, and can be mitigated by using regularization techniques or reducing model complexity",
                    "C": "Overfitting happens when a model performs poorly on both training and test data, and can be mitigated by using more features or a larger training dataset",
                    "D": "Overfitting occurs when the model is too simple, and can be mitigated by adding more training data",
                },
                "correct_answer": "B",
                "explanation": "Overfitting occurs when a model learns the noise and specifics of the training data rather than the underlying patterns, causing it to perform well on training data but poorly on unseen data. It can be mitigated by reducing model complexity, applying regularization techniques (such as L1 or L2), or using methods like cross-validation to ensure better generalization.",
            },
            {
                "question_goal": "To create a meaningful question that is easy to understand and has a clear correct answer.",
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

        # Create the prompt
        prompt = f"""
You are an expert creator of multiple-choice questions for statistical learning topics.
Your task is to generate {num_questions} challenging multiple-choice questions based on the provided content, with the following requirements:

1. The questions must be directly related to the content provided.
2. Each question should have one correct answer and three plausible but incorrect answers.
3. The questions should be at the "{level}" level of difficulty.
4. Assign an appropriate category from this list: {', '.join(self.topics)}
5. Make sure questions test understanding rather than just recall.

Here are some examples of {level} level questions to help you gauge the complexity and quality of questions:

{sample_text}

Below is the exact JSON format I want you to use for your response:
{self.complete_example}

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

    def improve_questions(
        self, evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Improve the quality of the generated questions.

        Args:
            evaluation: List of question dictionaries to improve

        Returns:
            List of improved question dictionaries
        """
        results = {"questions": []}
        for question_dict in evaluation.get("questions", []):
            question_keys = [
                "question",
                "choices",
                "correct_answer",
                "explanation",
                "category",
                "level",
                "topic",
                "source",
            ]
            question = {key: question_dict.get(key, None) for key in question_keys}
            feedback_list = question_dict.get("evaluation", None).get("suggestions", None)
            feedback = " ".join(feedback_list)
            prompt = f"Rewrite the following question based on the corresponding feedback. If there are no feedback, keep the question the same. Upate the 'question' field to add a simplified phrasing of the question if the feedback calls for it.\n\nQuestion:\n{question}\n\Feedback:\n{feedback}\n\n Return the response in the following format:\n{self.complete_example}"

            # Generate response
            max_retries = 3
            for attempt in range(max_retries):
                response = self.llm.invoke([HumanMessage(content=prompt)])
                response_content = response.content.strip()
                
                # Parse and validate the response
                questions = self._parse_llm_response(response_content)
                valid_questions = self._validate_and_format_questions([questions])

                if valid_questions:
                    results["questions"].append(valid_questions)
                    break

                # If we got here, no valid questions were found
                if attempt == max_retries - 1:
                    logger.warning("Question improvement step: No valid questions found in LLM response")
        return results
        # max_retries = 3
        # for attempt in range(max_retries):
        #     eval_results = json.dumps(evaluation)
        #     prompt = f"These questions have received feedback in the 'evaluation' field. Make the necessary amendments based on the 'suggested' field.\n{eval_results}\n\nReturn the results in the follwing format:\n{self.complete_example}"
        #     response = self.llm.invoke([HumanMessage(content=prompt)])
        #     response_content = response.content.strip()
        #     # Parse and validate the response
        #     questions = self._parse_llm_response(response_content)
        #     valid_questions = self._validate_and_format_questions(questions)

        #     if valid_questions:
        #         return valid_questions

        #     # If we got here, no valid questions were found
        #     if attempt == max_retries - 1:
        #         logger.warning("No valid questions found in LLM response")
        #         return []
