from quality.llm_explainer import build_explanation_prompt

FLAGGED_TABLE = [
    {"phase": "backswing", "joint": "left_elbow", "value": 0.5, "z": 2.1, "flagged": True, "note": None},
    {"phase": "contact", "joint": "right_knee", "value": 1.2, "z": -1.8, "flagged": True, "note": None},
    {"phase": "follow_through", "joint": "trunk_rotation", "value": 0.3, "z": 0.4, "flagged": False, "note": None},
]

NO_FLAGS_TABLE = [
    {"phase": "backswing", "joint": "left_elbow", "value": 0.5, "z": 0.2, "flagged": False, "note": None},
]


def test_build_explanation_prompt_mentions_only_flagged_rows():
    prompt = build_explanation_prompt("forehand", FLAGGED_TABLE)
    assert "left elbow" in prompt
    assert "backswing" in prompt
    assert "right knee" in prompt
    assert "contact" in prompt
    assert "trunk rotation" not in prompt
    assert "follow-through" not in prompt


def test_build_explanation_prompt_no_flags_gives_positive_prompt():
    prompt = build_explanation_prompt("forehand", NO_FLAGS_TABLE)
    assert "no significant deviations" in prompt
    assert "left elbow" not in prompt


import pytest
import requests

from quality.llm_explainer import generate_explanation, LLMExplanationError


class FakeResponse:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


def test_generate_explanation_success_path():
    def fake_post(url, json, headers, timeout):
        return FakeResponse(200, {"choices": [{"message": {"content": " Great swing! "}}]})

    result = generate_explanation("forehand", FLAGGED_TABLE, api_key="fake-key", post_fn=fake_post)
    assert result == "Great swing!"


def test_generate_explanation_raises_on_non_200():
    def fake_post(url, json, headers, timeout):
        return FakeResponse(500, text="internal error")

    with pytest.raises(LLMExplanationError):
        generate_explanation("forehand", FLAGGED_TABLE, api_key="fake-key", post_fn=fake_post)


def test_generate_explanation_raises_on_network_error():
    def fake_post(url, json, headers, timeout):
        raise requests.ConnectionError("no network")

    with pytest.raises(LLMExplanationError):
        generate_explanation("forehand", FLAGGED_TABLE, api_key="fake-key", post_fn=fake_post)


def test_generate_explanation_raises_on_unexpected_response_shape():
    def fake_post(url, json, headers, timeout):
        return FakeResponse(200, {"unexpected": "shape"})

    with pytest.raises(LLMExplanationError):
        generate_explanation("forehand", FLAGGED_TABLE, api_key="fake-key", post_fn=fake_post)


def test_generate_explanation_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    def fake_post(url, json, headers, timeout):
        raise AssertionError("post_fn should never be called without an API key")

    with pytest.raises(LLMExplanationError):
        generate_explanation("forehand", FLAGGED_TABLE, api_key=None, post_fn=fake_post)


class MalformedJsonResponse:
    """Simulates a 200 response with malformed/non-JSON body."""
    def __init__(self):
        self.status_code = 200
        self.text = "not valid json"

    def json(self):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")


def test_generate_explanation_raises_on_malformed_json():
    def fake_post(url, json, headers, timeout):
        return MalformedJsonResponse()

    with pytest.raises(LLMExplanationError):
        generate_explanation("forehand", FLAGGED_TABLE, api_key="fake-key", post_fn=fake_post)


class NoneContentResponse:
    """Simulates a 200 response where content field is None."""
    def __init__(self):
        self.status_code = 200
        self.text = ""

    def json(self):
        return {"choices": [{"message": {"content": None}}]}


def test_generate_explanation_raises_on_none_content():
    def fake_post(url, json, headers, timeout):
        return NoneContentResponse()

    with pytest.raises(LLMExplanationError):
        generate_explanation("forehand", FLAGGED_TABLE, api_key="fake-key", post_fn=fake_post)
