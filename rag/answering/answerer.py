"""ANSWERING part 4: decide whether to call the LLM, and call it."""
from rag.answering import llm_client
from rag.answering.prompt import NO_ANSWER, build_messages
from rag.models import Hit


def answer(question: str, hits: list[Hit]):
    """Returns (answer_text, messages_sent_to_llm_or_None)."""
    useful = [h for h in hits if h.used]
    if not useful:                 # nothing relevant -> refuse without calling the LLM
        return NO_ANSWER, None
    messages = build_messages(question, useful)
    return llm_client.chat(messages), messages
