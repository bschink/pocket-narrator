"""
Unit tests for the evaluation module.
"""
import math
from unittest.mock import patch, MagicMock
from pocket_narrator.evaluate import run_evaluation, calculate_grammar_score

# --- MVP Accuracy Tests ---

def test_run_evaluation_mvp_perfect_match():
    """Tests the MVP accuracy with a perfect match on a batch."""
    # Arrange
    predicted_tokens = [[3, 4, 5]]  # Model predicts "sat" then garbage
    target_tokens = [[3, 9, 2]]      # Target is "sat" then something else
    # Text arguments are needed for the function signature, but can be dummy for this test
    predicted_text = ["a b c"]
    target_text = ["d e f"]
    
    # Act
    summary = run_evaluation(predicted_tokens, target_tokens, predicted_text, target_text)
    
    # Assert
    assert "mvp_accuracy" in summary
    assert summary["mvp_accuracy"] == 1.0

def test_run_evaluation_mvp_mismatch():
    """Tests the MVP accuracy with a mismatch on a batch."""
    # Arrange
    predicted_tokens = [[4, 4, 5]] # Model predicts a wrong first token
    target_tokens = [[3, 9, 2]]
    predicted_text = ["a b c"]
    target_text = ["d e f"]

    # Act
    summary = run_evaluation(predicted_tokens, target_tokens, predicted_text, target_text)
    
    # Assert
    assert summary["mvp_accuracy"] == 0.0

def test_run_evaluation_mvp_multi_item_batch():
    """Tests accuracy with a batch containing multiple items (one correct, one incorrect)."""
    # Arrange
    predicted_tokens = [[3, 1], [8, 2]] 
    target_tokens = [[3, 9], [7, 1]]
    predicted_text = ["a", "b"]
    target_text = ["c", "d"]
    
    # Act
    summary = run_evaluation(predicted_tokens, target_tokens, predicted_text, target_text)
    
    # Assert
    assert summary["mvp_accuracy"] == 0.5 # 1 out of 2 is correct

def test_run_evaluation_empty_batch():
    """Tests that the evaluation handles an empty batch gracefully."""
    # Arrange
    predicted_tokens = []
    target_tokens = []
    predicted_text = []
    target_text = []

    # Act
    summary = run_evaluation(predicted_tokens, target_tokens, predicted_text, target_text)
    
    # Assert
    assert summary["mvp_accuracy"] == 0.0

# --- Perplexity Tests ---

def test_run_evaluation_calculates_perplexity():
    """Tests that passing a val_loss results in correct perplexity."""
    predicted_tokens = []
    target_tokens = []
    predicted_text = []
    target_text = []
    
    val_loss = 2.0
    expected_ppl = math.exp(val_loss)
    
    summary = run_evaluation(
        predicted_tokens, target_tokens, predicted_text, target_text, 
        val_loss=val_loss, check_grammar=False
    )
    
    assert "perplexity" in summary
    assert summary["perplexity"] == expected_ppl

def test_run_evaluation_handles_none_perplexity():
    """Tests that passing None for val_loss returns None for perplexity."""
    summary = run_evaluation([], [], [], [], val_loss=None, check_grammar=False)
    assert summary["perplexity"] is None

# --- BLEU, ROUGE, Diversity Tests ---

def test_run_evaluation_perfect_text_match_metrics():
    """
    If predicted text exactly matches target text, BLEU and ROUGE should be 1.0.
    """
    predicted_tokens = [[1]]
    target_tokens = [[1]]
    
    predicted_text = ["the cat sat"]
    target_text = ["the cat sat"]
    
    summary = run_evaluation(
        predicted_tokens, target_tokens, predicted_text, target_text, check_grammar=False
    )
    
    # BLEU-4 might be 0 because sequence length (3) < 4, but ROUGE-1/2 should be 1.0
    assert summary["rouge_1"] == 1.0
    assert summary["rouge_2"] == 1.0
    assert summary["rouge_l"] == 1.0
    
    # Distinct-1: "the", "cat", "sat" -> 3/3 = 1.0
    assert summary["distinct_1"] == 1.0
    # Repetition rate: 0
    assert summary["repetition_rate"] == 0.0

def test_run_evaluation_text_mismatch_metrics():
    """
    If texts are completely different, overlap scores should be 0.
    """
    predicted_text = ["apple banana"]
    target_text = ["orange grape"]
    
    summary = run_evaluation(
        [[1]], [[1]], predicted_text, target_text, check_grammar=False
    )
    
    assert summary["rouge_1"] == 0.0
    assert summary["bleu_4"] == 0.0

def test_diversity_metrics_repetitive_text():
    """
    Test that highly repetitive text gets high repetition rate and low distinct scores.
    """
    predicted_text = ["test test test test"]
    target_text = ["irrelevant"]
    
    summary = run_evaluation(
        [[1]], [[1]], predicted_text, target_text, check_grammar=False
    )
    
    # 4 tokens total, 1 unique ("test")
    # Repetition Rate = (4 - 1) / 4 = 0.75
    assert summary["repetition_rate"] == 0.75
    
    # Distinct-1: 1 unique / 4 total = 0.25
    assert summary["distinct_1"] == 0.25

# --- Grammar Checker Tests ---

@patch("pocket_narrator.evaluate._load_grammar_pipeline")
def test_calculate_grammar_score_success(mock_load_pipeline):
    """
    Test that the function correctly extracts LABEL_1 scores and averages them.
    """
    mock_pipe = MagicMock()
    
    mock_pipe.return_value = [
        [{"label": "LABEL_1", "score": 0.9}, {"label": "LABEL_0", "score": 0.1}],
        [{"label": "LABEL_0", "score": 0.9}, {"label": "LABEL_1", "score": 0.1}] 
    ]
    
    mock_load_pipeline.return_value = mock_pipe
    
    texts = ["Good sentence.", "Bad sentence."]
    
    score = calculate_grammar_score(texts, device="cpu")
    
    assert score == 0.5
    mock_pipe.assert_called_once()

@patch("pocket_narrator.evaluate._load_grammar_pipeline")
def test_calculate_grammar_score_mps_device_passing(mock_load_pipeline):
    """
    Test that the 'mps' device string is correctly passed down to the loader.
    """
    mock_pipe = MagicMock()
    mock_pipe.return_value = [[{"label": "LABEL_1", "score": 1.0}]]
    mock_load_pipeline.return_value = mock_pipe
    
    calculate_grammar_score(["test"], device="mps")
    
    mock_load_pipeline.assert_called_with("mps")

@patch("pocket_narrator.evaluate._load_grammar_pipeline")
def test_calculate_grammar_score_pipeline_failure(mock_load_pipeline):
    """
    Test that the function returns 0.0 gracefully if the pipeline crashes.
    """
    mock_pipe = MagicMock()
    mock_pipe.side_effect = Exception("CUDA error or something")
    mock_load_pipeline.return_value = mock_pipe
    
    score = calculate_grammar_score(["test"], device="cpu")
    
    assert score == 0.0

def test_calculate_grammar_score_empty_input():
    """
    Test that empty input returns 0.0 without trying to load pipeline.
    """
    score = calculate_grammar_score([], device="cpu")
    assert score == 0.0


# =============================================================================
# LLM-as-a-Judge Evaluation Tests
# =============================================================================

from pocket_narrator.evaluate import (
    calculate_llm_judge_scores,
    run_llm_judge_evaluation,
    LLMJudgeResult,
    LLM_JUDGE_PROMPT_TEMPLATE,
)


class TestLLMJudgePromptTemplate:
    """Tests for the LLM judge prompt template."""
    
    def test_prompt_has_story_placeholders(self):
        """Test that prompt template contains story_beginning and story_completion placeholders."""
        assert "{story_beginning}" in LLM_JUDGE_PROMPT_TEMPLATE
        assert "{story_completion}" in LLM_JUDGE_PROMPT_TEMPLATE
    
    def test_prompt_mentions_all_criteria(self):
        """Test that prompt mentions all evaluation criteria."""
        prompt_lower = LLM_JUDGE_PROMPT_TEMPLATE.lower()
        
        assert "grammar" in prompt_lower
        assert "creativity" in prompt_lower
        assert "consistency" in prompt_lower
        assert "age" in prompt_lower


class TestCalculateLLMJudgeScores:
    """Tests for the calculate_llm_judge_scores function."""
    
    def test_empty_texts_returns_empty_result(self):
        """Test that empty input returns a result with zeros."""
        result = calculate_llm_judge_scores(story_beginnings=[], story_completions=[])
        
        assert isinstance(result, LLMJudgeResult)
        assert result.avg_grammar == 0.0
        assert result.avg_creativity == 0.0
        assert result.avg_consistency == 0.0
        assert result.num_evaluated == 0
        assert result.num_failed == 0
    
    @patch("pocket_narrator.gemini_api.evaluate_stories_batch")
    @patch("pocket_narrator.gemini_api.GeminiClient")
    def test_calculates_average_scores(self, mock_client_cls, mock_batch):
        """Test that average scores are calculated correctly."""
        from pocket_narrator.gemini_api import LLMJudgeScores
        
        # Mock the batch evaluation to return predefined scores
        mock_batch.return_value = [
            LLMJudgeScores(grammar=3.0, creativity=2.0, consistency=3.0, age_group="C", raw_response=""),
            LLMJudgeScores(grammar=2.0, creativity=3.0, consistency=2.0, age_group="C", raw_response=""),
        ]
        
        result = calculate_llm_judge_scores(
            story_beginnings=["Beginning 1", "Beginning 2"],
            story_completions=["Completion 1", "Completion 2"],
            api_key="test-key"
        )
        
        assert result.avg_grammar == 2.5  # (3 + 2) / 2
        assert result.avg_creativity == 2.5  # (2 + 3) / 2
        assert result.avg_consistency == 2.5  # (3 + 2) / 2
        assert result.num_evaluated == 2
        assert result.num_failed == 0
    
    @patch("pocket_narrator.gemini_api.evaluate_stories_batch")
    @patch("pocket_narrator.gemini_api.GeminiClient")
    def test_handles_failed_evaluations(self, mock_client_cls, mock_batch):
        """Test that failed evaluations are counted correctly."""
        from pocket_narrator.gemini_api import LLMJudgeScores
        
        mock_batch.return_value = [
            LLMJudgeScores(grammar=3.0, creativity=2.0, consistency=3.0, age_group="C", raw_response=""),
            LLMJudgeScores(grammar=0.0, creativity=0.0, consistency=0.0, age_group="error", raw_response="API error"),
        ]
        
        result = calculate_llm_judge_scores(
            story_beginnings=["Beginning 1", "Beginning 2"],
            story_completions=["Completion 1", "Completion 2"],
            api_key="test-key"
        )
        
        assert result.avg_grammar == 3.0  # Only from successful evaluation
        assert result.num_evaluated == 1
        assert result.num_failed == 1
    
    @patch("pocket_narrator.gemini_api.evaluate_stories_batch")
    @patch("pocket_narrator.gemini_api.GeminiClient")
    def test_tracks_age_group_distribution(self, mock_client_cls, mock_batch):
        """Test that age group distribution is tracked correctly."""
        from pocket_narrator.gemini_api import LLMJudgeScores
        
        mock_batch.return_value = [
            LLMJudgeScores(grammar=3.0, creativity=2.0, consistency=3.0, age_group="C", raw_response=""),
            LLMJudgeScores(grammar=2.0, creativity=2.0, consistency=3.0, age_group="C", raw_response=""),
            LLMJudgeScores(grammar=2.0, creativity=3.0, consistency=2.0, age_group="D", raw_response=""),
        ]
        
        result = calculate_llm_judge_scores(
            story_beginnings=["Beginning 1", "Beginning 2", "Beginning 3"],
            story_completions=["Completion 1", "Completion 2", "Completion 3"],
            api_key="test-key"
        )
        
        assert "c" in result.age_group_distribution  # Normalized to lowercase
        assert result.age_group_distribution["c"] == 2
        assert "d" in result.age_group_distribution
        assert result.age_group_distribution["d"] == 1
    
    @patch("pocket_narrator.gemini_api.evaluate_stories_batch")
    @patch("pocket_narrator.gemini_api.GeminiClient")
    def test_respects_max_stories_limit(self, mock_client_cls, mock_batch):
        """Test that max_stories parameter limits evaluation."""
        from pocket_narrator.gemini_api import LLMJudgeScores
        
        mock_batch.return_value = [
            LLMJudgeScores(grammar=3.0, creativity=2.0, consistency=3.0, age_group="C", raw_response=""),
        ]
        
        calculate_llm_judge_scores(
            story_beginnings=["B1", "B2", "B3", "B4", "B5"],
            story_completions=["C1", "C2", "C3", "C4", "C5"],
            api_key="test-key",
            max_stories=1
        )
        
        # Verify only 1 story was passed to batch evaluation
        call_args = mock_batch.call_args
        assert len(call_args[0][0]) == 1


class TestRunLLMJudgeEvaluation:
    """Tests for the run_llm_judge_evaluation function."""
    
    @patch("pocket_narrator.evaluate.calculate_llm_judge_scores")
    def test_returns_dictionary(self, mock_calculate):
        """Test that function returns a properly formatted dictionary."""
        mock_calculate.return_value = LLMJudgeResult(
            avg_grammar=2.5,
            avg_creativity=2.0,
            avg_consistency=3.0,
            age_group_distribution={"c": 2},
            individual_scores=[],
            num_evaluated=2,
            num_failed=0
        )
        
        result = run_llm_judge_evaluation(
            story_beginnings=["Beginning 1", "Beginning 2"],
            story_completions=["Completion 1", "Completion 2"],
            api_key="test-key"
        )
        
        assert isinstance(result, dict)
        assert result["llm_judge_grammar"] == 2.5
        assert result["llm_judge_creativity"] == 2.0
        assert result["llm_judge_consistency"] == 3.0
        assert result["llm_judge_age_groups"] == {"c": 2}
        assert result["llm_judge_num_evaluated"] == 2
        assert result["llm_judge_num_failed"] == 0


class TestRunEvaluationWithLLMJudge:
    """Tests for LLM judge integration in run_evaluation."""
    
    def test_llm_judge_disabled_by_default(self):
        """Test that LLM judge is disabled by default."""
        result = run_evaluation(
            predicted_tokens=[],
            target_tokens=[],
            predicted_text=[],
            target_text=[],
            check_grammar=False
        )
        
        assert result["llm_judge_grammar"] is None
        assert result["llm_judge_creativity"] is None
        assert result["llm_judge_consistency"] is None
        assert result["llm_judge_age_groups"] is None
    
    @patch("pocket_narrator.evaluate.run_llm_judge_evaluation")
    def test_llm_judge_called_when_enabled(self, mock_llm_judge):
        """Test that LLM judge is called when enabled."""
        mock_llm_judge.return_value = {
            "llm_judge_grammar": 7.0,
            "llm_judge_creativity": 8.0,
            "llm_judge_consistency": 6.0,
            "llm_judge_age_groups": {"5-6 years": 1},
            "llm_judge_num_evaluated": 1,
            "llm_judge_num_failed": 0,
        }
        
        result = run_evaluation(
            predicted_tokens=[[1]],
            target_tokens=[[1]],
            predicted_text=["Test story"],
            target_text=["Reference"],
            story_beginnings=["Story beginning"],
            check_grammar=False,
            run_llm_judge=True,
            llm_judge_api_key="test-key"
        )
        
        assert result["llm_judge_grammar"] == 7.0
        mock_llm_judge.assert_called_once()
    
    @patch("pocket_narrator.evaluate.run_llm_judge_evaluation")
    def test_llm_judge_receives_custom_parameters(self, mock_llm_judge):
        """Test that custom parameters are passed to LLM judge."""
        mock_llm_judge.return_value = {
            "llm_judge_grammar": 7.0,
            "llm_judge_creativity": 8.0,
            "llm_judge_consistency": 6.0,
            "llm_judge_age_groups": {},
            "llm_judge_num_evaluated": 1,
            "llm_judge_num_failed": 0,
        }
        
        custom_template = "Custom template: {story_beginning} {story_completion}"
        
        run_evaluation(
            predicted_tokens=[[1]],
            target_tokens=[[1]],
            predicted_text=["Test story"],
            target_text=["Reference"],
            story_beginnings=["Beginning"],
            check_grammar=False,
            run_llm_judge=True,
            llm_judge_api_key="test-key",
            llm_judge_max_stories=5,
            llm_judge_prompt_template=custom_template
        )
        
        call_kwargs = mock_llm_judge.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["max_stories"] == 5
        assert call_kwargs["prompt_template"] == custom_template