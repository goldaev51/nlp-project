from typing import Dict, List, Optional

MOOD_TO_ID: Dict[str, int] = {
    "calm": 0,
    "annoyed": 1,
    "confused": 2,
}

ID_TO_MOOD: Dict[int, str] = {v: k for k, v in MOOD_TO_ID.items()}

EMOTION_TO_MOOD: Dict[str, str] = {
    "admiration": "calm",
    "amusement": "calm",
    "approval": "calm",
    "caring": "calm",
    "curiosity": "calm",
    "desire": "calm",
    "excitement": "calm",
    "gratitude": "calm",
    "joy": "calm",
    "love": "calm",
    "optimism": "calm",
    "pride": "calm",
    "realization": "calm",
    "relief": "calm",
    "surprise": "calm",
    "neutral": "calm",

    "anger": "annoyed",
    "annoyance": "annoyed",
    "disappointment": "annoyed",
    "disapproval": "annoyed",
    "disgust": "annoyed",
    "remorse": "annoyed",
    "sadness": "annoyed",
    "grief": "annoyed",

    "confusion": "confused",
    "embarrassment": "confused",
    "fear": "confused",
    "nervousness": "confused",
}


MOOD_PRIORITY: List[str] = ["annoyed", "confused", "calm"]


def choose_mood_for_emotions(emotions: List[str]) -> Optional[str]:
    moods = set()
    for e in emotions:
        mood = EMOTION_TO_MOOD.get(e)
        if mood is not None:
            moods.add(mood)

    if not moods:
        return None

    for mood in MOOD_PRIORITY:
        if mood in moods:
            return mood

    return None
