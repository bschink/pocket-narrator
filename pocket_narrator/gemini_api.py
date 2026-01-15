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
    
    Expected format in response (XML-like tags):
        <grammar><score 1-3></grammar>
        <creativity><score 1-3></creativity>
        <consistency><score 1-3></consistency>
        <age_group><single letter A-F></age_group>
    
    Args:
        response: Raw text response from the LLM.
        
    Returns:
        LLMJudgeScores dataclass with parsed values.
        
    Raises:
        ValueError: If the response cannot be parsed.
    """
    grammar = 0.0
    creativity = 0.0
    consistency = 0.0
    age_group = "unknown"
    
    # Pattern to match scores inside XML-like tags: <grammar>2</grammar>
    grammar_match = re.search(r"<grammar>\s*(\d+(?:\.\d+)?)\s*</grammar>", response, re.IGNORECASE)
    if grammar_match:
        grammar = float(grammar_match.group(1))
        if grammar > 3:
            grammar = 3.0
    
    creativity_match = re.search(r"<creativity>\s*(\d+(?:\.\d+)?)\s*</creativity>", response, re.IGNORECASE)
    if creativity_match:
        creativity = float(creativity_match.group(1))
        if creativity > 3:
            creativity = 3.0
    
    consistency_match = re.search(r"<consistency>\s*(\d+(?:\.\d+)?)\s*</consistency>", response, re.IGNORECASE)
    if consistency_match:
        consistency = float(consistency_match.group(1))
        if consistency > 3:
            consistency = 3.0
    
    age_match = re.search(r"<age_group>\s*([A-Fa-f])\s*</age_group>", response, re.IGNORECASE)
    if age_match:
        age_group = age_match.group(1).strip().upper()
    
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
    api_key: Optional[str] = None
) -> LLMJudgeScores:
    """
    Evaluate a single story using LLM-as-a-judge.
    
    Args:
        story_beginning: The prompt/beginning given to the model.
        story_completion: The text generated by the model to complete the story.
        prompt_template: Template for the evaluation prompt. 
                        Use {story_beginning} and {story_completion} as placeholders.
        client: Optional pre-initialized GeminiClient.
        api_key: API key (used if client is None).
        
    Returns:
        LLMJudgeScores with evaluation results.
    """
    if client is None:
        client = GeminiClient(api_key=api_key)
    
    prompt = prompt_template.format(
        story_beginning=story_beginning,
        story_completion=story_completion
    )
    response = client.generate(prompt, temperature=0.3)
    
    return parse_llm_judge_response(response)


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


DEFAULT_EVALUATION_PROMPT = """
You are an expert evaluator of children's stories. 
In the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story.
The exercise tests the student's language abilities and creativity. 
The beginning of the story is wrapped in <story_beginning> and the student's completion is wrapped in <story_completion>.

First provide your general assessment about the part written by the student (<story_completion>).
Is it gramatically correct? Is it consistent with the beginning of the story? Pay special attention to whether the
student manages to complete the sentence which began in <story_beginning> but wasn't finished if that's the case.

Afterwards, grade the student’s completion in terms of grammar, creativity, consistency with the story’s beginning and
whether the plot makes sense. Moreover, please provide your best guess of what the age of the student might be,
as reflected from the completion. Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E:
10-12. F: 13-16. Please evaluate the model's completion based on these criteria, providing a score from 1-3 for each:

1. Grammar: How grammatically correct is the completion?
2. Creativity: How creative and imaginative is the completion?
3. Consistency: How logically consistent is the completion with the beginning?
4. Age group: What age group is this story appropriate for?
   (A: 3 or under, B: 4-5, C: 6-7, D: 8-9, E: 10-12, F: 13-16)

Story beginning (given in the exercise):
<story_beginning>
{story_beginning}
</story_beginning>

Students completion:
<story_completion>
{story_completion}
</story_completion>

Please frame your evaluation in exactly this format:
<grammar><score 1-3></grammar>
<creativity><score 1-3></creativity>
<consistency><score 1-3></consistency>
<age_group><single letter A-F></age_group>
"""
