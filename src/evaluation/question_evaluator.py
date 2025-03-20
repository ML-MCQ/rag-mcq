"""
Question Evaluator module for assessing the quality of generated questions.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage

# Configure logging
logger = logging.getLogger(__name__)


class QuestionEvaluator:
    """
    A class for evaluating the quality of generated multiple-choice questions.
    
    This class uses an LLM to assess questions for relevance, correctness,
    plausibility of wrong answers, and appropriate difficulty level.
    """
    
    def __init__(self,
                 azure_endpoint: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 azure_deployment_name: Optional[str] = None,
                 azure_api_version: Optional[str] = None):
        """
        Initialize the question evaluator with Azure OpenAI configuration.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            azure_api_key: Azure OpenAI API key
            azure_deployment_name: Azure OpenAI deployment name
            azure_api_version: Azure OpenAI API version
        """
        # Get Azure OpenAI credentials from environment variables if not provided
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.azure_deployment_name = azure_deployment_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.azure_api_version = azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        
        # Initialize Azure OpenAI client with higher temperature for more diverse evaluations
        self.llm = AzureChatOpenAI(
            openai_api_key=self.azure_api_key,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment_name,
            api_version=self.azure_api_version,
            temperature=2.0  # Set high temperature for diverse evaluations (max allowed is 2.0)
        )
    
    def _create_evaluation_prompt(self, question: Dict[str, Any], context: str) -> str:
        """
        Create a prompt for evaluating a question based on context.
        
        Args:
            question: The question object to evaluate
            context: The content the question was generated from
            
        Returns:
            Formatted prompt for evaluation
        """
        # Format the question and answers for the prompt
        question_text = question["question"]
        
        choices_text = ""
        # Handle both dictionary and list formats for choices
        if isinstance(question["choices"], dict):
            for letter, choice in question["choices"].items():
                choices_text += f"{letter}. {choice}\n"
        elif isinstance(question["choices"], list):
            letters = ['A', 'B', 'C', 'D', 'E']
            for i, choice in enumerate(question["choices"]):
                if i < len(letters):
                    choices_text += f"{letters[i]}. {choice}\n"
        
        correct_answer = question["correct_answer"]
        explanation = question["explanation"]
        level = question["level"]
        category = question["category"]
        
        prompt = f"""
You are an expert evaluator of statistical learning multiple-choice questions. You need to evaluate the quality of this question based on the original context it was generated from.

CONTEXT:
{context}

QUESTION:
{question_text}

ANSWER CHOICES:
{choices_text}

CORRECT ANSWER: {correct_answer}
EXPLANATION: {explanation}
DIFFICULTY LEVEL: {level}
CATEGORY: {category}

Please evaluate this question on the following criteria:
1. Relevance: Is the question directly related to the context? (1-10 scale)
2. Correctness: Is the indicated correct answer actually correct? (1-10 scale)
3. Distractors: Are the wrong answers plausible but clearly incorrect? (1-10 scale)
4. Difficulty: Is the assigned difficulty level appropriate for this question? (1-10 scale)
5. Clarity: Is the question clearly worded and unambiguous? (1-10 scale)
6. Educational Value: Does the question test understanding rather than just facts? (1-10 scale)

For each criterion, provide:
- A score (1-10, where 10 is perfect)
- A brief explanation for your score

Then, provide an overall quality score (1-10) and any suggestions to improve the question.

Format your response as a JSON object with the following fields:
"relevance", "correctness", "distractors", "difficulty", "clarity", "educational_value", "overall", "suggestions"

Each field (except suggestions) should be an object with "score" (number) and "explanation" (string).
"suggestions" should be an array of strings.
"""
        return prompt
    
    def evaluate_question(self, question: Dict[str, Any], context: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a generated question using the LLM.
        
        Args:
            question: The question object to evaluate
            context: The content the question was generated from
            
        Returns:
            Evaluation results with scores and suggestions
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Ensure question has the expected format
                # Convert choices to dictionary format if it's a list
                if "choices" in question and isinstance(question["choices"], list):
                    letters = ['A', 'B', 'C', 'D', 'E']
                    choices_dict = {}
                    for i, choice in enumerate(question["choices"]):
                        if i < len(letters):
                            choices_dict[letters[i]] = choice
                    question = question.copy()  # Create a copy to avoid modifying the original
                    question["choices"] = choices_dict
                
                # Create evaluation prompt
                prompt = self._create_evaluation_prompt(question, context)
                
                # Be explicit about requesting JSON format
                prompt += "\n\nIMPORTANT: Your response must be ONLY valid JSON. Do not include any additional text before or after the JSON object."
                
                # Send to LLM for evaluation
                response = self.llm.invoke([HumanMessage(content=prompt)])
                
                # Parse the response
                response_content = response.content.strip()
                
                # Try to extract JSON from the response
                try:
                    # If response starts with ``` and ends with ```, extract the content
                    if response_content.startswith("```json") and "```" in response_content[7:]:
                        # Extract content between ```json and the next ```
                        json_text = response_content.split("```json", 1)[1].split("```", 1)[0].strip()
                        evaluation = json.loads(json_text)
                    elif response_content.startswith("```") and response_content.endswith("```"):
                        # Extract content between ``` and ```
                        json_text = response_content[3:-3].strip()
                        evaluation = json.loads(json_text)
                    else:
                        # Try to find JSON brackets
                        start_idx = response_content.find('{')
                        end_idx = response_content.rfind('}') + 1
                        
                        if start_idx != -1 and end_idx != -1:
                            json_str = response_content[start_idx:end_idx]
                            evaluation = json.loads(json_str)
                        else:
                            # Last resort: try to parse the entire response as JSON
                            evaluation = json.loads(response_content)
                    
                    # Validate the evaluation structure
                    required_fields = ["relevance", "correctness", "distractors", 
                                       "difficulty", "clarity", "educational_value", 
                                       "overall", "suggestions"]
                    
                    for field in required_fields:
                        if field not in evaluation:
                            logger.warning(f"Missing required field: {field} in evaluation")
                            raise ValueError(f"Missing required field: {field}")
                        
                        if field != "suggestions" and (not isinstance(evaluation[field], dict) or "score" not in evaluation[field]):
                            logger.warning(f"Invalid format for field: {field}")
                            raise ValueError(f"Invalid format for field: {field}")
                    
                    # Check that suggestions is a list
                    if not isinstance(evaluation["suggestions"], list):
                        evaluation["suggestions"] = [str(evaluation["suggestions"])]
                    
                    return evaluation
                    
                except (json.JSONDecodeError, ValueError) as e:
                    if attempt == max_retries - 1:  # Last attempt
                        logger.warning(f"Failed to parse JSON from LLM response: {str(e)}")
                        logger.warning(f"Response content: {response_content[:100]}...")
                        return self._create_default_evaluation(f"Failed to parse evaluation JSON: {str(e)}")
                    else:
                        logger.info(f"Retry {attempt+1}/{max_retries}: JSON parsing failed - {str(e)}")
                        # Adjust temperature for retries to get different outputs
                        self.llm.temperature = max(0.1, self.llm.temperature - 0.5)
            
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Error evaluating question: {str(e)}")
                    return self._create_default_evaluation(f"Error: {str(e)}")
                else:
                    logger.info(f"Retry {attempt+1}/{max_retries}: General error - {str(e)}")
                    # Adjust temperature for retries to get different outputs
                    self.llm.temperature = max(0.1, self.llm.temperature - 0.5)
        
        # If we somehow exit the loop without returning
        return self._create_default_evaluation("Failed after multiple retry attempts")
    
    def _create_default_evaluation(self, message: str) -> Dict[str, Any]:
        """
        Create a default evaluation object when evaluation fails.
        
        Args:
            message: Error message to include
            
        Returns:
            Default evaluation object
        """
        return {
            "relevance": {"score": 0, "explanation": message},
            "correctness": {"score": 0, "explanation": message},
            "distractors": {"score": 0, "explanation": message},
            "difficulty": {"score": 0, "explanation": message},
            "clarity": {"score": 0, "explanation": message},
            "educational_value": {"score": 0, "explanation": message},
            "overall": {"score": 0, "explanation": message},
            "suggestions": [message]
        }
    
    def evaluate_questions(self, questions: List[Dict[str, Any]], contexts: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple questions with their respective contexts.
        
        Args:
            questions: List of question objects to evaluate
            contexts: Dictionary mapping source IDs to context strings
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, question in enumerate(questions):
            logger.info(f"Evaluating question {i+1}/{len(questions)}")
            
            # Get the context for this question
            source = question.get("source", "")
            context = contexts.get(source, "")
            
            if not context:
                logger.warning(f"Context not found for question {i+1}")
                # Use a default context error message
                evaluation = self._create_default_evaluation("Context not found for evaluation")
            else:
                # Evaluate the question
                evaluation = self.evaluate_question(question, context)
            
            # Add the evaluation to the question object
            question_with_eval = question.copy()
            question_with_eval["evaluation"] = evaluation
            
            results.append(question_with_eval)
        
        return results
    
    def get_overall_quality_stats(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall quality statistics from multiple evaluations.
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Statistics about evaluation scores
        """
        if not evaluations:
            return {
                "avg_overall": 0,
                "avg_relevance": 0,
                "avg_correctness": 0,
                "avg_distractors": 0,
                "avg_difficulty": 0,
                "avg_clarity": 0,
                "avg_educational_value": 0,
                "total_questions": 0,
                "high_quality_count": 0,  #
                "medium_quality_count": 0,  
                "low_quality_count": 0  
            }
        
        # Extract scores from evaluations
        scores = {
            "overall": [],
            "relevance": [],
            "correctness": [],
            "distractors": [],
            "difficulty": [],
            "clarity": [],
            "educational_value": []
        }
        
        for eval_result in evaluations:
            eval_data = eval_result.get("evaluation", {})
            
            for key in scores.keys():
                if key in eval_data and "score" in eval_data[key]:
                    scores[key].append(eval_data[key]["score"])
        
        # Calculate averages
        stats = {
            "avg_overall": sum(scores["overall"]) / len(scores["overall"]) if scores["overall"] else 0,
            "avg_relevance": sum(scores["relevance"]) / len(scores["relevance"]) if scores["relevance"] else 0,
            "avg_correctness": sum(scores["correctness"]) / len(scores["correctness"]) if scores["correctness"] else 0,
            "avg_distractors": sum(scores["distractors"]) / len(scores["distractors"]) if scores["distractors"] else 0,
            "avg_difficulty": sum(scores["difficulty"]) / len(scores["difficulty"]) if scores["difficulty"] else 0,
            "avg_clarity": sum(scores["clarity"]) / len(scores["clarity"]) if scores["clarity"] else 0,
            "avg_educational_value": sum(scores["educational_value"]) / len(scores["educational_value"]) if scores["educational_value"] else 0,
            "total_questions": len(evaluations),
            "high_quality_count": sum(1 for score in scores["overall"] if score >= 8),
            "medium_quality_count": sum(1 for score in scores["overall"] if 5 <= score < 8),
            "low_quality_count": sum(1 for score in scores["overall"] if score < 5)
        }
        
        return stats