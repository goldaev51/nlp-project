# src/app/streamlit_app.py

from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.chatbot.core import get_default_helpdesk_bot, HelpdeskBot  # noqa: E402


def init_bot() -> HelpdeskBot:
    if "helpdesk_bot" not in st.session_state:
        st.session_state.helpdesk_bot = get_default_helpdesk_bot(
            mode="transformer",
            kb_path=None,
            top_k=3,
        )
    return st.session_state.helpdesk_bot


def init_history() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, Any]] = []


def main() -> None:
    st.set_page_config(
        page_title="SecurePass Helpdesk Bot",
        page_icon="💬",
        layout="centered",
    )

    st.title("SecurePass Helpdesk Bot")
    st.write(
        "Ask questions about SecurePass SSO/MFA and get answers from the knowledge base. "
        "The bot also adapts its tone to your mood."
    )

    init_history()
    bot = init_bot()

    with st.sidebar:
        st.header("Settings")
        st.write("Emotion model: transformer (DistilBERT)")
        st.write("Retrieval: TF-IDF over KB")
        st.write("Top-K KB results: 3")

        if st.button("Clear chat history"):
            st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        user_message = st.text_area("Your question", height=80)
        submitted = st.form_submit_button("Send")

    if submitted and user_message.strip():
        user_text = user_message.strip()
        result = bot.answer(user_text)

        st.session_state.chat_history.append(
            {
                "user": user_text,
                "reply": result["reply"],
                "mood": result["mood"],
                "kb_hits": result["kb_hits"],
            }
        )

    for turn in reversed(st.session_state.chat_history):
        st.markdown("---")
        st.markdown(f"**Your message:** {turn['user']}")
        st.markdown(f"**Bot:** {turn['reply']}")

        with st.expander("Details (mood and KB hits)"):
            mood = turn["mood"]
            st.write(f"Mood: **{mood['mood']}** (id={mood['mood_id']})")
            if mood.get("probs") is not None:
                st.write("Probabilities:")
                for m, p in mood["probs"].items():
                    st.write(f"- {m}: {p:.3f}")

            st.write("KB hits:")
            for h in turn["kb_hits"]:
                st.write(
                    f"- [{h['section']}] {h['question']} "
                    f"(score={h['score']:.3f}, id={h['id']}, tags={h.get('tags', '')})"
                )


if __name__ == "__main__":
    main()
