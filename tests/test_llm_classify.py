import math
import json
import pytest
from llm_classify import (
    process_classification_response,
    format_prompt,
    get_class_probability,
    _make_serializable
)

# Helper dummy response object
class DummyResponse:
    def __init__(self, text_val, response_json=None):
        self._text = text_val
        self.response_json = response_json or {}
    def text(self):
        return self._text

# Dummy model to simulate llm.get_model
class DummyModel:
    def prompt(self, prompt, temperature, top_logprobs=1, max_tokens=None):
        # Simulate a completion model response with choices
        return DummyResponse("Class A", {"choices": [{"text": "Class A"}]})

@pytest.fixture(autouse=True)
def monkeypatch_llm(monkeypatch):
    # Monkey-patch llm.get_model to return DummyModel
    import llm_classify  # ensure we patch the same module used in get_class_probability
    monkeypatch.setattr(llm_classify, "llm", type("dummy", (), {"get_model": lambda model: DummyModel()}))

def test_process_classification_response_valid():
    # Given a valid response returning letter "a" (for first class)
    logprob = -0.1
    response_json = {
        "logprobs": {
            "content": [
                {"token": "a", "logprob": logprob},
                {"token": "b", "logprob": -1.0}
            ]
        }
    }
    response = DummyResponse(" a ", response_json)
    classes = ["Class A", "Class B"]
    result, confidence = process_classification_response(response, classes)
    expected_conf = round(math.exp(logprob), 4)
    assert result == "Class A"
    assert confidence == expected_conf

def test_process_classification_response_invalid():
    # Response with no valid letter
    response = DummyResponse("invalid", {})
    classes = ["Class A", "Class B"]
    result, confidence = process_classification_response(response, classes)
    assert result is None
    assert confidence == 0.0

def test_format_prompt_includes_classes():
    # Verify that the prompt template includes available class options
    content = "Sample content"
    classes = ("Alpha", "Beta")
    prompt = format_prompt(content, classes, None, None)
    assert "a) Alpha" in prompt
    assert "b) Beta" in prompt
    assert "Sample content" in prompt

def test_get_class_probability_completion():
    # Using the dummy model, get_class_probability should return "Class A"
    content = "Test content"
    classes = ["Class A", "Class B"]
    result, prob = get_class_probability(content, classes, "dummy-model", 0.5)
    assert result == "Class A"
    assert isinstance(prob, float)

def test_make_serializable():
    # Test that _make_serializable converts non-serializable objects to string
    data = {"a": 1, "b": set([1,2])}
    serializable = _make_serializable(data)
    assert isinstance(serializable["a"], int)
    assert isinstance(serializable["b"], str)
    # Also verify serialization works with json.dumps
    json_str = json.dumps(serializable)
    assert isinstance(json_str, str)

# ...other tests as needed...