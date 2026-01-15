"""
Unit tests for the Gemini API client and LLM-as-a-judge evaluation.
"""
import pytest
from unittest.mock import patch, MagicMock

from pocket_narrator.gemini_api import (
    GeminiClient,
    GeminiAPIError,
    LLMJudgeScores,
    parse_llm_judge_response,
    evaluate_story_with_llm,
    evaluate_stories_batch,
    DEFAULT_EVALUATION_PROMPT,
)


# =============================================================================
# Tests for parse_llm_judge_response
# =============================================================================

class TestParseLLMJudgeResponse:
    """Tests for the response parsing function."""
    
    def test_parse_standard_format(self):
        """Test parsing a well-formatted response with XML-like tags."""
        response = """
        The story has good grammar and creativity.
        <grammar>2</grammar>
        <creativity>3</creativity>
        <consistency>2</consistency>
        <age_group>C</age_group>
        """
        
        scores = parse_llm_judge_response(response)
        
        assert scores.grammar == 2.0
        assert scores.creativity == 3.0
        assert scores.consistency == 2.0
        assert scores.age_group == "C"
        assert response in scores.raw_response
    
    def test_parse_with_whitespace(self):
        """Test parsing scores with whitespace inside tags."""
        response = """
        <grammar> 2 </grammar>
        <creativity> 3 </creativity>
        <consistency> 1 </consistency>
        <age_group> B </age_group>
        """
        
        scores = parse_llm_judge_response(response)
        
        assert scores.grammar == 2.0
        assert scores.creativity == 3.0
        assert scores.consistency == 1.0
        assert scores.age_group == "B"
    
    def test_parse_decimal_scores(self):
        """Test parsing decimal scores."""
        response = """
        <grammar>2.5</grammar>
        <creativity>1.5</creativity>
        <consistency>2.5</consistency>
        <age_group>D</age_group>
        """
        
        scores = parse_llm_judge_response(response)
        
        assert scores.grammar == 2.5
        assert scores.creativity == 1.5
        assert scores.consistency == 2.5
    
    def test_parse_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        response = """
        <GRAMMAR>3</GRAMMAR>
        <Creativity>2</Creativity>
        <CONSISTENCY>3</CONSISTENCY>
        <AGE_GROUP>b</AGE_GROUP>
        """
        
        scores = parse_llm_judge_response(response)
        
        assert scores.grammar == 3.0
        assert scores.creativity == 2.0
        assert scores.consistency == 3.0
        assert scores.age_group == "B"  # Normalized to uppercase
    
    def test_parse_malformed_response(self):
        """Test parsing a malformed response returns defaults."""
        response = "This is not a valid evaluation response."
        
        scores = parse_llm_judge_response(response)
        
        assert scores.grammar == 0.0
        assert scores.creativity == 0.0
        assert scores.consistency == 0.0
        assert scores.age_group == "unknown"
    
    def test_parse_partial_response(self):
        """Test parsing a response with missing fields."""
        response = """
        <grammar>2</grammar>
        <age_group>C</age_group>
        """
        
        scores = parse_llm_judge_response(response)
        
        assert scores.grammar == 2.0
        assert scores.creativity == 0.0  # Missing, defaults to 0
        assert scores.consistency == 0.0  # Missing, defaults to 0
        assert scores.age_group == "C"
    
    def test_parse_lowercase_age_group(self):
        """Test parsing lowercase age group letter."""
        response = """
        <grammar>2</grammar>
        <creativity>3</creativity>
        <consistency>2</consistency>
        <age_group>a</age_group>
        """
        
        scores = parse_llm_judge_response(response)
        
        assert scores.age_group == "A"  # Normalized to uppercase
    
    def test_parse_scores_capped_at_3(self):
        """Test that scores above 3 are capped."""
        response = """
        <grammar>5</grammar>
        <creativity>10</creativity>
        <consistency>100</consistency>
        <age_group>E</age_group>
        """
        
        scores = parse_llm_judge_response(response)
        
        assert scores.grammar == 3.0
        assert scores.creativity == 3.0
        assert scores.consistency == 3.0
        assert scores.age_group == "E"


# =============================================================================
# Tests for GeminiClient
# =============================================================================

class TestGeminiClient:
    """Tests for the Gemini API client."""
    
    def test_init_with_api_key(self):
        """Test client initialization with explicit API key."""
        client = GeminiClient(api_key="test-api-key")
        
        assert client.api_key == "test-api-key"
        assert client.model == "gemini-2.5-flash-lite-preview-09-2025"
    
    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises an error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(GeminiAPIError) as exc_info:
                GeminiClient(api_key=None)
            
            assert "No API key provided" in str(exc_info.value)
    
    def test_init_from_environment(self):
        """Test client initialization from environment variable."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-api-key"}):
            client = GeminiClient()
            
            assert client.api_key == "env-api-key"
    
    def test_custom_model(self):
        """Test client with custom model."""
        client = GeminiClient(api_key="test-key", model="custom-model")
        
        assert client.model == "custom-model"
    
    @patch("pocket_narrator.gemini_api.GeminiClient._get_client")
    def test_generate_returns_response(self, mock_get_client):
        """Test that generate returns the expected response."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated response text"
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        client = GeminiClient(api_key="test-key")
        result = client.generate("Test prompt")
        
        assert result == "Generated response text"
    
    @patch("pocket_narrator.gemini_api.GeminiClient._get_client")
    def test_generate_empty_response_raises_error(self, mock_get_client):
        """Test that empty response raises an error."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        client = GeminiClient(api_key="test-key")
        
        with pytest.raises(GeminiAPIError) as exc_info:
            client.generate("Test prompt")
        
        assert "Empty response" in str(exc_info.value)


# =============================================================================
# Tests for evaluate_story_with_llm
# =============================================================================

class TestEvaluateStoryWithLLM:
    """Tests for single story evaluation."""
    
    @patch("pocket_narrator.gemini_api.GeminiClient.generate")
    def test_evaluate_single_story(self, mock_generate):
        """Test evaluating a single story."""
        mock_generate.return_value = """
        The story shows good grammar and creativity.
        <grammar>3</grammar>
        <creativity>2</creativity>
        <consistency>3</consistency>
        <age_group>C</age_group>
        """
        
        client = GeminiClient(api_key="test-key")
        scores = evaluate_story_with_llm(
            story_beginning="Once upon a time there was a little cat.",
            story_completion="The cat went on an adventure and found a treasure.",
            prompt_template="Beginning: {story_beginning}\nCompletion: {story_completion}",
            client=client
        )
        
        assert scores.grammar == 3.0
        assert scores.creativity == 2.0
        assert scores.consistency == 3.0
        assert scores.age_group == "C"
        mock_generate.assert_called_once()
    
    @patch("pocket_narrator.gemini_api.GeminiClient.generate")
    def test_prompt_template_formatting(self, mock_generate):
        """Test that story beginning and completion are correctly inserted into prompt template."""
        mock_generate.return_value = "<grammar>2</grammar>\n<creativity>2</creativity>\n<consistency>2</consistency>\n<age_group>B</age_group>"
        
        client = GeminiClient(api_key="test-key")
        beginning = "Once upon a time"
        completion = "they lived happily ever after"
        template = "Beginning: {story_beginning}\nCompletion: {story_completion}"
        
        evaluate_story_with_llm(
            story_beginning=beginning,
            story_completion=completion,
            prompt_template=template,
            client=client
        )
        
        # Verify the prompt was formatted correctly
        called_prompt = mock_generate.call_args[0][0]
        assert "Once upon a time" in called_prompt
        assert "they lived happily ever after" in called_prompt


# =============================================================================
# Tests for evaluate_stories_batch
# =============================================================================

class TestEvaluateStoriesBatch:
    """Tests for batch story evaluation."""
    
    @patch("pocket_narrator.gemini_api.GeminiClient.generate")
    def test_evaluate_multiple_stories(self, mock_generate):
        """Test evaluating multiple stories."""
        mock_generate.side_effect = [
            "<grammar>3</grammar>\n<creativity>2</creativity>\n<consistency>3</consistency>\n<age_group>C</age_group>",
            "<grammar>2</grammar>\n<creativity>3</creativity>\n<consistency>2</consistency>\n<age_group>D</age_group>",
        ]
        
        client = GeminiClient(api_key="test-key")
        beginnings = ["Beginning one", "Beginning two"]
        completions = ["Completion one", "Completion two"]
        
        results = evaluate_stories_batch(
            story_beginnings=beginnings,
            story_completions=completions,
            prompt_template="Beginning: {story_beginning}\nCompletion: {story_completion}",
            client=client
        )
        
        assert len(results) == 2
        assert results[0].grammar == 3.0
        assert results[1].grammar == 2.0
        assert results[0].age_group == "C"
        assert results[1].age_group == "D"
        assert mock_generate.call_count == 2
    
    @patch("pocket_narrator.gemini_api.GeminiClient.generate")
    def test_batch_handles_api_errors(self, mock_generate):
        """Test that batch evaluation handles API errors gracefully."""
        mock_generate.side_effect = [
            "<grammar>3</grammar>\n<creativity>2</creativity>\n<consistency>3</consistency>\n<age_group>C</age_group>",
            GeminiAPIError("API error"),  # Second call fails
        ]
        
        client = GeminiClient(api_key="test-key")
        beginnings = ["Beginning one", "Beginning two"]
        completions = ["Completion one", "Completion two"]
        
        results = evaluate_stories_batch(
            story_beginnings=beginnings,
            story_completions=completions,
            prompt_template="Beginning: {story_beginning}\nCompletion: {story_completion}",
            client=client
        )
        
        assert len(results) == 2
        assert results[0].grammar == 3.0
        assert results[1].age_group == "error"  # Failed evaluation
    
    def test_empty_stories_list(self):
        """Test batch evaluation with empty list."""
        client = GeminiClient(api_key="test-key")
        
        results = evaluate_stories_batch(
            story_beginnings=[],
            story_completions=[],
            prompt_template="Beginning: {story_beginning}\nCompletion: {story_completion}",
            client=client
        )
        
        assert results == []
    
    def test_mismatched_list_lengths_raises_error(self):
        """Test that mismatched list lengths raise an error."""
        client = GeminiClient(api_key="test-key")
        
        with pytest.raises(ValueError) as exc_info:
            evaluate_stories_batch(
                story_beginnings=["One", "Two"],
                story_completions=["Only one"],
                prompt_template="Beginning: {story_beginning}\nCompletion: {story_completion}",
                client=client
            )
        
        assert "same length" in str(exc_info.value)


# =============================================================================
# Tests for LLMJudgeScores dataclass
# =============================================================================

class TestLLMJudgeScores:
    """Tests for the LLMJudgeScores dataclass."""
    
    def test_create_scores(self):
        """Test creating a scores instance."""
        scores = LLMJudgeScores(
            grammar=8.0,
            creativity=7.0,
            consistency=9.0,
            age_group="5-6 years",
            raw_response="test response"
        )
        
        assert scores.grammar == 8.0
        assert scores.creativity == 7.0
        assert scores.consistency == 9.0
        assert scores.age_group == "5-6 years"
        assert scores.raw_response == "test response"


# =============================================================================
# Tests for DEFAULT_EVALUATION_PROMPT
# =============================================================================

class TestDefaultPrompt:
    """Tests for the default evaluation prompt."""
    
    def test_prompt_has_story_placeholders(self):
        """Test that default prompt contains story_beginning and story_completion placeholders."""
        assert "{story_beginning}" in DEFAULT_EVALUATION_PROMPT
        assert "{story_completion}" in DEFAULT_EVALUATION_PROMPT
    
    def test_prompt_mentions_all_criteria(self):
        """Test that default prompt mentions all evaluation criteria."""
        prompt_lower = DEFAULT_EVALUATION_PROMPT.lower()
        
        assert "grammar" in prompt_lower
        assert "creativity" in prompt_lower
        assert "consistency" in prompt_lower
        assert "age" in prompt_lower
