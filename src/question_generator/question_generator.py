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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    A class for generating multiple-choice questions based on retrieved content.
    
    This class uses an LLM to generate multiple-choice questions from context,
    with varying difficulty levels and topics.
    """
    
    def __init__(self,
                 azure_endpoint: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 azure_deployment_name: Optional[str] = None,
                 azure_api_version: Optional[str] = None,
                 sample_questions: Optional[List[Dict[str, Any]]] = None):
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
        self.azure_deployment_name = azure_deployment_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.azure_api_version = azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        
        # Initialize Azure OpenAI client with higher temperature for more creative outputs
        self.llm = AzureChatOpenAI(
            openai_api_key=self.azure_api_key,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment_name,
            api_version=self.azure_api_version,
            temperature=0.2 
        )
        
        # Load sample questions for few-shot prompting
        self.sample_questions = sample_questions or self._load_default_sample_questions()
        
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
            "time series analysis"
        ]
        
        self.levels = ["basic", "intermediate", "advanced"]
        
    def _load_default_sample_questions(self) -> List[Dict[str, Any]]:
        """
        Load default sample questions for few-shot prompting.
        
        Returns:
            List of sample question dictionaries
        """
        return [
            # Basic cards (difficulty -2.0 to -0.5)
            {"question": "What is supervised learning?", 
            "difficulty": -2.0, "level": "basic"},
            
            {"question": "What is a feature?", 
            "difficulty": -1.0, "level": "basic"},
            
            {"question": "What is gradient descent?", 
            "difficulty": -0.5, "level": "basic"},
            
            # Intermediate cards (difficulty -0.4 to 1.0)
            {"question": "What is the curse of dimensionality?", 
            "difficulty": -0.4, "level": "intermediate"},
            
            {"question": "What is a support vector machine (SVM)?", 
            "difficulty": 0.0, "level": "intermediate"},
            
            {"question": "What is transfer learning?", 
            "difficulty": 1.0, "level": "intermediate"},
            
            # Advanced cards (difficulty 1.1 to 3.0)
            {"question": "Explain the Expectation-Maximization (EM) algorithm.", 
            "difficulty": 1.1, "level": "advanced"},
            
            {"question": "What is the Vapnik-Chervonenkis (VC) dimension?", 
            "difficulty": 1.5, "level": "advanced"},
            
            {"question": "Explain causal inference in machine learning.", 
            "difficulty": 3.0, "level": "advanced"}
        ]
    
    def _create_prompt_with_examples(self, context: str, difficulty: float, num_questions: int = 1) -> str:
        """
        Create a prompt with few-shot examples for question generation.
        
        Args:
            context: The text content to generate questions from
            difficulty: The target difficulty level for generated questions
            num_questions: Number of questions to generate
            
        Returns:
            Formatted prompt with examples
        """
        # Calculate difficulty range
        difficulty_min = max(-2.0, difficulty - 0.3)
        difficulty_max = min(3.0, difficulty + 0.3)
        
        # Determine level based on difficulty
        if difficulty < -0.5:
            level = "basic"
        elif difficulty < 1.0:
            level = "intermediate"
        else:
            level = "advanced"
        
        # Filter sample questions by difficulty range and level
        filtered_samples = [
            q for q in self.sample_questions 
            if difficulty_min <= q["difficulty"] <= difficulty_max and q["level"] == level
        ]
        
        # Pick up to 3 samples
        samples = random.sample(filtered_samples, min(3, len(filtered_samples)))
        
        # Format samples for the prompt - only include fields that exist
        sample_text = ""
        for i, sample in enumerate(samples):
            sample_text += f"Example {i+1}:\n"
            sample_text += f"Question: {sample['question']}\n"
            sample_text += f"Difficulty: {sample['difficulty']}\n"
            sample_text += f"Level: {sample['level']}\n\n"
        
        # Add an explicit example of a complete MC question structure
        complete_example = """
        COMPLETE OUTPUT EXAMPLE:
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
        "difficulty": -1.5,
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
3. The questions should have a difficulty level around {difficulty} on a scale from -2.0 (very basic) to 3.0 (very advanced).
4. The level should be "{level}" based on the difficulty.
5. Assign an appropriate category from this list: {', '.join(self.topics)}
6. Make sure questions test understanding rather than just recall.

Here are some examples of questions with similar difficulty level to help you gauge the complexity:

{sample_text}

The examples above show the difficulty level and question complexity, but you need to generate COMPLETE multiple-choice questions with all required elements.

{complete_example}

Now, based on the following content, create {num_questions} multiple-choice questions:

CONTENT:
{context}

For each question, provide:
1. The question text
2. Four answer choices (A, B, C, D) where exactly one is correct
3. The letter of the correct answer
4. An explanation why the correct answer is right and others are wrong
5. The difficulty level (a number between {difficulty_min} and {difficulty_max})
6. An appropriate category from the list provided
7. The level ("{level}")

Format your response as a JSON list where each question is an object with fields: "question", "choices", "correct_answer", "explanation", "difficulty", "category", and "level".
"""
        return prompt
    
    def generate_multiple_choice_questions(self, context: str, difficulty: float = 0.0, 
                                          num_questions: int = 1) -> List[Dict[str, Any]]:
        """
        Generate multiple-choice questions based on the provided context.
        
        Args:
            context: The text content to generate questions from
            difficulty: The target difficulty level for generated questions
            num_questions: Number of questions to generate
            
        Returns:
            List of generated question dictionaries
        """
        if not context or not context.strip():
            logger.warning("Empty context provided for question generation")
            return []
        
        max_retries = 3
        current_temp = self.llm.temperature
        
        for attempt in range(max_retries):
            try:
                # Create prompt with examples
                prompt = self._create_prompt_with_examples(context, difficulty, num_questions)
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
                logger.info(f"Retry {attempt+1}/{max_retries}: No valid questions found")
                
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
            json_text = response_content.split("```json", 1)[1].split("```", 1)[0].strip()
            return json.loads(json_text)
        
        # If response is wrapped in code block
        if response_content.startswith("```") and response_content.endswith("```"):
            json_text = response_content[3:-3].strip()
            return json.loads(json_text)
        
        # Try to find JSON brackets for an array
        start_idx = response_content.find('[')
        end_idx = response_content.rfind(']') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_content[start_idx:end_idx]
            return json.loads(json_str)
        
        # Last resort: try to parse the entire response as JSON
        questions = json.loads(response_content)
        
        # If it's not an array, wrap it
        if not isinstance(questions, list):
            questions = [questions]
        
        return questions

    def _validate_and_format_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate question fields and format choices consistently.
        
        Args:
            questions: List of question dictionaries to validate
            
        Returns:
            List of valid, formatted question dictionaries
        """
        required_fields = ["question", "choices", "correct_answer", 
                           "explanation", "difficulty", "category", "level"]
        
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
                    letters = ['A', 'B', 'C', 'D', 'E']
                    choices_dict = {}
                    for j, choice in enumerate(q["choices"]):
                        if j < len(letters):
                            choices_dict[letters[j]] = choice
                    q["choices"] = choices_dict
                
                valid_questions.append(q)
        
        return valid_questions
    
    def generate_questions_for_topics(self, 
                                      contexts: List[str], 
                                      difficulties: List[float] = None, 
                                      num_questions_per_context: int = 1) -> List[Dict[str, Any]]:
        """
        Generate multiple-choice questions for multiple contexts/topics.
        
        Args:
            contexts: List of text contexts to generate questions from
            difficulties: List of difficulty levels (one per context)
            num_questions_per_context: Number of questions to generate per context
            
        Returns:
            List of generated question dictionaries across all contexts
        """
        if not contexts:
            logger.warning("No contexts provided for question generation")
            return []
        
        # If difficulties not provided, use default values spread across range
        if not difficulties:
            # Create a spread of difficulties from -2.0 to 3.0
            difficulties = [
                -2.0 + (5.0 * i / (len(contexts) - 1)) if len(contexts) > 1 else 0.0
                for i in range(len(contexts))
            ]
        
        # Ensure difficulties list matches contexts list
        if len(difficulties) != len(contexts):
            difficulties = difficulties[:len(contexts)] + [0.0] * (len(contexts) - len(difficulties))
        
        all_questions = []
        for i, (context, difficulty) in enumerate(zip(contexts, difficulties)):
            logger.info(f"Generating questions for context {i+1}/{len(contexts)} (difficulty: {difficulty})")
            questions = self.generate_multiple_choice_questions(
                context, difficulty, num_questions_per_context
            )
            all_questions.extend(questions)
        
        return all_questions


if __name__ == "__main__":
    # Simple test to verify the question generator works
    from src.vectorstore.vector_store import VectorStore
    
    # Try to load the vector store
    vector_store = VectorStore()
    if not vector_store.load_vector_store("test_index"):
        print("Vector store not found. Please run the vector store test first.")
        exit()
    
    # Perform a search to get content for question generation
    results = vector_store.similarity_search("statistical learning methods", k=1)
    
    if not results:
        print("No search results found. Please ensure the vector store has content.")
        exit()
    
    # Create and test the question generator
    question_generator = QuestionGenerator()
    
    # Generate a question from the retrieved content
    content = results[0]["content"]
    questions = question_generator.generate_multiple_choice_questions(
        content, difficulty=0.0, num_questions=1
    )
    
    # Print the generated questions
    print(f"Generated {len(questions)} questions:")
    for i, q in enumerate(questions):
        print(f"\nQuestion {i+1}:")
        print(f"Text: {q['question']}")
        print("Choices:")
        for choice_letter, choice_text in q["choices"].items():
            print(f"  {choice_letter}: {choice_text}")
        print(f"Correct Answer: {q['correct_answer']}")
        print(f"Explanation: {q['explanation']}")
        print(f"Difficulty: {q['difficulty']}")
        print(f"Category: {q['category']}")
        print(f"Level: {q['level']}") 