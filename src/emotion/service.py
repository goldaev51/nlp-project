# src/emotion/service.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any

import torch

from .baseline import load_emotion_baseline, predict_mood_baseline
from .transformer_model import load_emotion_transformer, predict_mood_transformer


ModeType = Literal["baseline", "transformer"]


@dataclass
class EmotionAnalyzer:
    mode: ModeType = "transformer"
    model_dir_baseline: str = "models/emotion_baseline"
    model_dir_transformer: str = "models/emotion_transformer"

    _baseline_clf: Any | None = None
    _tok: Any | None = None
    _transformer_model: Any | None = None
    _device: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "baseline":
            self._init_baseline()
        elif self.mode == "transformer":
            self._init_transformer()
        else:
            raise ValueError(f"Unsupported mode: {self.mode!r}")

    def _init_baseline(self) -> None:
        self._baseline_clf = load_emotion_baseline(self.model_dir_baseline)
        self._tok = None
        self._transformer_model = None
        self._device = None

    def _init_transformer(self) -> None:
        tok, model = load_emotion_transformer(self.model_dir_transformer)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = tok
        self._transformer_model = model.to(device)
        self._transformer_model.eval()
        self._baseline_clf = None
        self._device = device

    def analyze(self, text: str) -> Dict[str, Any]:
        if self.mode == "baseline":
            result = predict_mood_baseline(self._baseline_clf, text)
            result["probs"] = None
            return result

        return predict_mood_transformer(
            text=text,
            tokenizer=self._tok,
            model=self._transformer_model,
            device=self._device,
        )


def get_default_emotion_analyzer(
    mode: ModeType = "transformer",
) -> EmotionAnalyzer:
    return EmotionAnalyzer(mode=mode)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.emotion.service \"your text here\" [baseline|transformer]")
        sys.exit(1)

    text = sys.argv[1]
    mode: ModeType = "transformer"
    if len(sys.argv) >= 3:
        mode_arg = sys.argv[2].lower()
        if mode_arg in ("baseline", "transformer"):
            mode = mode_arg

    analyzer = get_default_emotion_analyzer(mode=mode)
    res = analyzer.analyze(text)

    print(f"Mode: {mode}")
    print(f"Text: {text!r}")
    print(f"Pred mood: {res['mood']} (id={res['mood_id']})")
    if res.get("probs") is not None:
        print("Probs:")
        for m, p in res["probs"].items():
            print(f"  {m}: {p:.4f}")
