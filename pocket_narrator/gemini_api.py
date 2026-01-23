"""
Gemini API client for LLM-as-a-judge evaluation.

This module provides a client for communicating with the Google Gemini API
to evaluate generated text quality using LLM-as-a-judge methods.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-09-2025"
# DEFAULT_MODEL = "gemini-3-flash-preview"


@dataclass
class LLMJudgeScores:
    """
    Structured output from LLM-as-a-judge evaluation
    Based on TinyStories paper evaluation criteria (page 5) but with scales adjusted to 1-3
    """
    grammar: float  # 1-3 scale
    creativity: float  # 1-3 scale
    consistency: float  # 1-3 scale
    age_group: str  # A: 3 or under, B: 4-5, C: 6-7, D: 8-9, E:10-12, F: 13-16
    raw_response: str


class GeminiAPIError(Exception):
    """Exception raised for Gemini API errors."""
    pass


class GeminiClient:
    """
    Client for interacting with the Google Gemini API.
    
    Usage:
        client = GeminiClient(api_key="your-api-key")
        response = client.generate("Your prompt here")
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY environment variable.
            model: Model identifier to use (default: gemini-2.5-flash-lite-preview-09-2025).
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise GeminiAPIError(
                "No API key provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )
        self.model = model
        self._client = None
        
    def _get_client(self):
        """Lazy initialization of the Gemini client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise GeminiAPIError(
                    "google-genai package not installed. Install with: pip install google-genai"
                )
        return self._client
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """
        Generate a response from the Gemini API.
        
        Args:
            prompt: The input prompt to send to the model.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum number of tokens in the response.
            
        Returns:
            The generated text response.
            
        Raises:
            GeminiAPIError: If the API request fails.
        """
        client = self._get_client()
        
        try:
            from google.genai import types
            
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
            if response.text:
                return response.text
            else:
                raise GeminiAPIError("Empty response from Gemini API")
                
        except Exception as e:
            if "google" in str(type(e).__module__):
                raise GeminiAPIError(f"Gemini API error: {str(e)}")
            raise


def parse_llm_judge_response(response: str) -> LLMJudgeScores:
    """
    Parse the structured response from the LLM judge.
    
    Looks for scores in XML-like tags anywhere in the response:
        <grammar>2</grammar>
        <creativity>1</creativity>
        <consistency>3</consistency>
        <age_group>C</age_group>
    
    Args:
        response: Raw text response from the LLM.
        
    Returns:
        LLMJudgeScores dataclass with parsed values.
    """
    if not response or response.strip() == "":
        print(f"WARNING: Empty response from LLM")
        return LLMJudgeScores(grammar=0.0, creativity=0.0, consistency=0.0, age_group="error", raw_response="EMPTY")
    
    grammar = 0.0
    creativity = 0.0
    consistency = 0.0
    age_group = "unknown"
    
    # More lenient regex patterns that work even with extra whitespace/content
    # Look for <grammar>NUMBER</grammar> anywhere in response
    grammar_match = re.search(r"<\s*grammar\s*>\s*(\d+(?:\.\d+)?)\s*<\s*/\s*grammar\s*>", response, re.IGNORECASE | re.DOTALL)
    if grammar_match:
        grammar = float(grammar_match.group(1))
        if grammar > 3:
            grammar = 3.0
    
    creativity_match = re.search(r"<\s*creativity\s*>\s*(\d+(?:\.\d+)?)\s*<\s*/\s*creativity\s*>", response, re.IGNORECASE | re.DOTALL)
    if creativity_match:
        creativity = float(creativity_match.group(1))
        if creativity > 3:
            creativity = 3.0
    
    consistency_match = re.search(r"<\s*consistency\s*>\s*(\d+(?:\.\d+)?)\s*<\s*/\s*consistency\s*>", response, re.IGNORECASE | re.DOTALL)
    if consistency_match:
        consistency = float(consistency_match.group(1))
        if consistency > 3:
            consistency = 3.0
    
    age_match = re.search(r"<\s*age_group\s*>\s*([A-Fa-f])\s*<\s*/\s*age_group\s*>", response, re.IGNORECASE | re.DOTALL)
    if age_match:
        age_group = age_match.group(1).strip().upper()
    
    # Log success or failure
    if grammar == 0.0 or creativity == 0.0 or consistency == 0.0:
        print(f"DEBUG: Parsing response (partial): {response[:150]}")
    
    return LLMJudgeScores(
        grammar=grammar,
        creativity=creativity,
        consistency=consistency,
        age_group=age_group,
        raw_response=response
    )


def evaluate_story_with_llm(
    story_beginning: str,
    story_completion: str,
    prompt_template: str,
    client: Optional[GeminiClient] = None,
    api_key: Optional[str] = None,
    max_retries: int = 3
) -> LLMJudgeScores:
    """
    Evaluate a single story using LLM-as-a-judge with retry logic.
    
    Args:
        story_beginning: The prompt/beginning given to the model.
        story_completion: The text generated by the model to complete the story.
        prompt_template: Template for the evaluation prompt. 
                        Use {story_beginning} and {story_completion} as placeholders.
        client: Optional pre-initialized GeminiClient.
        api_key: API key (used if client is None).
        max_retries: Number of times to retry on empty response.
        
    Returns:
        LLMJudgeScores with evaluation results.
    """
    if client is None:
        client = GeminiClient(api_key=api_key)
    
    prompt = prompt_template.format(
        story_beginning=story_beginning,
        story_completion=story_completion
    )
    
    # Retry on empty responses
    for attempt in range(max_retries):
        response = client.generate(prompt, temperature=0.3)
        
        # Check if response is empty
        if response and response.strip():
            return parse_llm_judge_response(response)
        
        if attempt < max_retries - 1:
            print(f"DEBUG: Empty response (attempt {attempt + 1}/{max_retries}), retrying...")
    
    # All retries failed
    print(f"WARNING: Empty response after {max_retries} retries")
    return LLMJudgeScores(
        grammar=0.0,
        creativity=0.0,
        consistency=0.0,
        age_group="error",
        raw_response="EMPTY_AFTER_RETRIES"
    )


def evaluate_stories_batch(
    story_beginnings: list[str],
    story_completions: list[str],
    prompt_template: str,
    client: Optional[GeminiClient] = None,
    api_key: Optional[str] = None
) -> list[LLMJudgeScores]:
    """
    Evaluate multiple stories using LLM-as-a-judge.
    
    Args:
        story_beginnings: List of prompts/beginnings given to the model.
        story_completions: List of texts generated by the model to complete stories.
        prompt_template: Template for the evaluation prompt.
                        Use {story_beginning} and {story_completion} as placeholders.
        client: Optional pre-initialized GeminiClient.
        api_key: API key (used if client is None).
        
    Returns:
        List of LLMJudgeScores, one per story.
    """
    if client is None:
        client = GeminiClient(api_key=api_key)
    
    if len(story_beginnings) != len(story_completions):
        raise ValueError(
            f"story_beginnings and story_completions must have same length. "
            f"Got {len(story_beginnings)} and {len(story_completions)}."
        )
    
    results = []
    for beginning, completion in zip(story_beginnings, story_completions):
        try:
            scores = evaluate_story_with_llm(beginning, completion, prompt_template, client=client)
            results.append(scores)
        except GeminiAPIError as e:
            print(f"WARNING: LLM evaluation failed for story: {e}")
            results.append(LLMJudgeScores(
                grammar=0.0,
                creativity=0.0,
                consistency=0.0,
                age_group="error",
                raw_response=str(e)
            ))
    
    return results


DEFAULT_EVALUATION_PROMPT = """You are an expert evaluator of student story completions. Evaluate the student's completion based on:

1. Grammar (score 1-3): Is the completion grammatically correct?
2. Creativity (score 1-3): Is it creative and imaginative?
3. Consistency (score 1-3): Does it logically fit with the story beginning?
4. Age Group (letter A-F): What age group is this appropriate for?
   A: 3 or under, B: 4-5, C: 6-7, D: 8-9, E: 10-12, F: 13-16

Story Beginning (what was given to the student):
<story_beginning>
{story_beginning}
</story_beginning>

Student's Completion (what the student wrote):
<story_completion>
{story_completion}
</story_completion>

Output your evaluation in exactly this XML format, with no additional text before or after:
<grammar>1</grammar>
<creativity>1</creativity>
<consistency>1</consistency>
<age_group>A</age_group>

Replace the example values with your actual scores. Scores must be 1, 2, or 3. Age group must be A, B, C, D, E, or F.
"""
