from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.emotion.service import get_default_emotion_analyzer, EmotionAnalyzer
from src.retrieval.tfidf_retriever import TfidfKBRetriever


@dataclass
class HelpdeskBot:
    emotion_analyzer: EmotionAnalyzer
    retriever: TfidfKBRetriever
    top_k: int = 3

    def answer(self, user_message: str) -> Dict[str, Any]:
        mood_info = self.emotion_analyzer.analyze(user_message)
        kb_hits = self.retriever.retrieve(user_message, top_k=self.top_k)
        reply_text = self._compose_reply(mood_info, kb_hits)

        return {
            "reply": reply_text,
            "mood": mood_info,
            "kb_hits": kb_hits,
        }

    @staticmethod
    def _compose_reply(
        mood_info: Dict[str, Any],
        kb_hits: List[Dict[str, Any]],
    ) -> str:
        mood = mood_info["mood"]

        if not kb_hits:
            if mood == "annoyed":
                prefix = (
                    "I'm sorry, I couldn't find a specific article matching your issue right now. "
                    "However, here are some general steps you can try:\n\n"
                )
            elif mood == "confused":
                prefix = (
                    "I understand this is confusing. I couldn't find an exact article, "
                    "but here are some general directions to check:\n\n"
                )
            else:
                prefix = (
                    "I couldn't find a direct match in the knowledge base. "
                    "Here are some general suggestions:\n\n"
                )

            generic = (
                "- Make sure you are using the correct username and tenant.\n"
                "- Try signing out and signing in again.\n"
                "- If the problem persists, contact your administrator or helpdesk.\n"
            )

            return prefix + generic

        primary = kb_hits[0]
        base_answer = primary["answer"]

        if mood == "annoyed":
            prefix = (
                "I'm sorry you're running into this issue. "
                "Let's go through the most relevant solution step by step:\n\n"
            )
        elif mood == "confused":
            prefix = (
                "I understand this can be confusing. "
                "Here's a clear explanation based on our documentation:\n\n"
            )
        else:
            prefix = "Here's the information you need:\n\n"

        extra_tips_parts: List[str] = []
        if len(kb_hits) > 1:
            extra_tips_parts.append("You might also find these topics helpful:")
            for h in kb_hits[1:]:
                extra_tips_parts.append(f"- {h['question']} (section: {h['section']})")
        extra_tips = "\n\n" + "\n".join(extra_tips_parts) if extra_tips_parts else ""

        if mood == "annoyed":
            postfix = (
                "\n\nIf this does not resolve the problem, please let me know what you tried "
                "so we can narrow it down further."
            )
        elif mood == "confused":
            postfix = (
                "\n\nIf any step is unclear, tell me which part is confusing, and I can break it "
                "down into smaller steps."
            )
        else:
            postfix = "\n\nIf you need more details or a different scenario, just ask."

        return prefix + base_answer + extra_tips + postfix


def get_default_helpdesk_bot(
    mode: str = "transformer",
    kb_path: Optional[str] = None,
    top_k: int = 3,
) -> HelpdeskBot:
    emotion_analyzer = get_default_emotion_analyzer(mode=mode)
    retriever = TfidfKBRetriever.from_csv(path=kb_path, max_features=20000)
    return HelpdeskBot(
        emotion_analyzer=emotion_analyzer,
        retriever=retriever,
        top_k=top_k,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.chatbot.core \"your question here\" "
            "[mode: transformer|baseline]"
        )
        sys.exit(1)

    user_message = sys.argv[1]
    mode = "transformer"
    if len(sys.argv) >= 3:
        mode_arg = sys.argv[2].lower()
        if mode_arg in ("transformer", "baseline"):
            mode = mode_arg

    bot = get_default_helpdesk_bot(mode=mode)
    result = bot.answer(user_message)

    print(f"Mode: {mode}")
    print(f"User: {user_message!r}\n")
    print("Reply:")
    print(result["reply"])

    print("\n--- Mood ---")
    print(result["mood"])

    print("\n--- KB hits ---")
    for i, h in enumerate(result["kb_hits"], start=1):
        print(f"\n[{i}] (score={h['score']:.4f})")
        print(f"Q: {h['question']}")
        print(f"Section: {h['section']}")
        print(f"Tags: {h.get('tags', '')}")
