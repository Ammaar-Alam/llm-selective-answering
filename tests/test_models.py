from abstention.models import selected_features


def test_selected_features_can_exclude_subject_category() -> None:
    features = selected_features({"features": {"include_subject_category": False}})
    assert "subject_or_category" not in features
    assert "benchmark" in features
