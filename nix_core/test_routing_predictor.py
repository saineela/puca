from pathlib import Path

from routing_predictor import TinyRoutingPredictor


def test_untrained_predictor_abstains():
    predictor = TinyRoutingPredictor(dimensions=128, hidden=8)
    result = predictor.predict("maybe later")
    assert result["route"] == "unknown"
    assert result["abstained"] is True


def test_tiny_predictor_learns_balanced_routes():
    texts = [
        "hello", "tell me a joke", "what is the capital of australia",
        "how do I fix a faucet", "I am tired",
        "remember my sister is named Maya", "what is on my schedule tomorrow",
        "remind me to take medicine", "what do you remember about tea",
        "cancel my dentist appointment",
    ] * 3
    labels = [
        "chat", "chat", "chat", "chat", "chat",
        "knowledge", "knowledge", "knowledge", "knowledge", "knowledge",
    ] * 3
    predictor = TinyRoutingPredictor(dimensions=256, hidden=12, threshold=0.55)
    report = predictor.fit(texts, labels, epochs=100, validation_fraction=0.2)
    assert report["validation_examples"] > 0
    assert predictor.predict("remind me to take medicine")["route"] == "knowledge"
    assert predictor.predict("tell me a joke")["route"] in {"chat", "unknown"}


def test_artifact_round_trip(tmp_path: Path):
    predictor = TinyRoutingPredictor(dimensions=128, hidden=8, threshold=0.55)
    predictor.fit(
        ["hello", "hello there", "remember my birthday", "remember my age"],
        ["chat", "chat", "knowledge", "knowledge"],
        epochs=20,
        validation_fraction=0.5,
    )
    path = tmp_path / "router.npz"
    predictor.save(path)
    loaded = TinyRoutingPredictor.load(path)
    assert loaded is not None
    assert loaded.predict("hello")["route"] in {"chat", "unknown"}
    assert loaded.predict("remember my birthday")["route"] in {"knowledge", "unknown"}
