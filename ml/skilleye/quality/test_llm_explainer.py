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
