import os
import pathlib
import pytest
import click
from unittest.mock import Mock, patch
from click.testing import CliRunner
from llm_classify import logs_db_path, user_dir, classify_content, get_class_probability
from llm_classify import register_commands

# FILE: llm_classify/test___init__.py


def test_logs_db_path_env_var_set(monkeypatch):
    # Set the environment variable
    monkeypatch.setenv("LLM_USER_PATH", "/tmp/test_llm_user_path")
    
    # Expected path
    expected_path = pathlib.Path("/tmp/test_llm_user_path/logs.db")
    
    # Check if the logs_db_path function returns the correct path
    assert logs_db_path() == expected_path

def test_logs_db_path_env_var_not_set(monkeypatch):
    # Unset the environment variable
    monkeypatch.delenv("LLM_USER_PATH", raising=False)
    
    # Mock the click.get_app_dir function to return a test directory
    monkeypatch.setattr("click.get_app_dir", lambda x: "/tmp/test_app_dir")
    
    # Expected path
    expected_path = pathlib.Path("/tmp/test_app_dir/logs.db")
    
    # Check if the logs_db_path function returns the correct path
    assert logs_db_path() == expected_path

def test_user_dir_creates_directory(monkeypatch, tmp_path):
    # Unset the environment variable
    monkeypatch.delenv("LLM_USER_PATH", raising=False)
    
    # Mock the click.get_app_dir function to return a temporary directory
    monkeypatch.setattr("click.get_app_dir", lambda x: str(tmp_path))
    
    # Call the user_dir function
    path = user_dir()
    
    # Check if the directory was created
    assert path.exists()
    assert path.is_dir()

# New tests for classification functionality
def test_classify_content():
    content = ["This is a happy message"]
    classes = ["positive", "negative", "neutral"]
    model = "gpt-3.5-turbo"
    temperature = 0

    with patch('llm_classify.get_class_probability') as mock_get_prob:
        mock_get_prob.return_value = ("positive", 0.95)
        results = classify_content(content, classes, model, temperature)
        
        assert len(results) == 1
        assert results[0]["class"] == "positive"
        assert results[0]["score"] == 0.95
        assert results[0]["content"] == content[0]

def test_classify_content_no_content_flag():
    content = ["Test message"]
    classes = ["class1", "class2"]
    
    with patch('llm_classify.get_class_probability') as mock_get_prob:
        mock_get_prob.return_value = ("class1", 0.8)
        results = classify_content(content, classes, "test-model", 0, no_content=True)
        
        assert "content" not in results[0]
        assert results[0]["class"] == "class1"
        assert results[0]["score"] == 0.8

def test_get_class_probability():
    content = "Test content"
    classes = ["class1", "class2"]
    model = "test-model"
    
    mock_model = Mock()
    mock_response = Mock()
    mock_response.text = lambda: "class1"  # Use lambda to ensure consistent return
    mock_response.log_to_db = Mock()
    mock_response.response_json = {
        'logprobs': {
            'content': [
                Mock(logprob=-0.1)  # Changed from dict to Mock object
            ]
        }
    }
    mock_model.prompt.return_value = mock_response
    
    with patch('llm.get_model', return_value=mock_model), \
         patch('sqlite_utils.Database'):
            result_class, probability = get_class_probability(
                content, classes, model, 0
            )
            
            assert result_class == "class1"
            assert isinstance(probability, float)
            mock_response.log_to_db.assert_called_once()

@pytest.fixture
def cli():
    """Create a CLI group and register commands."""
    cli_group = click.Group()
    register_commands(cli_group)
    return cli_group

def test_classify_command(cli):
    runner = CliRunner()
    with patch('llm_classify.classify_content') as mock_classify:
        mock_classify.return_value = [{"class": "positive", "score": 0.9, "content": "test"}]
        
        result = runner.invoke(cli.commands['classify'], [
            "test",
            "-c", "positive",
            "-c", "negative",
            "-m", "gpt-3.5-turbo"
        ])
        
        assert result.exit_code == 0
        assert "positive" in result.output

def test_classify_invalid_temperature(cli):
    runner = CliRunner()
    result = runner.invoke(cli.commands['classify'], [
        "test",
        "-c", "a",
        "-c", "b",
        "-t", "1.5"
    ])
    assert result.exit_code != 0
    assert "Temperature must be between 0 and 1" in result.output

def test_classify_insufficient_classes(cli):
    runner = CliRunner()
    result = runner.invoke(cli.commands['classify'], [
        "test",
        "-c", "a"
    ])
    assert result.exit_code != 0
    assert "At least two classes must be provided" in result.output

def test_examples_processing():
    content = ["Test content"]
    classes = ["class1", "class2"]
    examples = ["example1:class1", "example2:class2"]
    
    with patch('llm_classify.get_class_probability') as mock_get_prob:
        mock_get_prob.return_value = ("class1", 0.9)
        results = classify_content(
            content,
            classes,
            "test-model",
            0,
            [{"content": "example1", "class": "class1"},
             {"content": "example2", "class": "class2"}]
        )
        
        assert len(results) == 1
        assert results[0]["class"] == "class1"
        assert results[0]["score"] == 0.9