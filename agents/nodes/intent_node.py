"""Intent classification node — classifies query into Intent enum."""

from agents.state import GraphState
from models.documents import Intent
from utils.logger import get_logger

logger = get_logger(__name__)

_ANALYTICAL = [
    "how many", "count", "kitne", "kitni", "total questions",
    "list all", "table of", "marks analysis", "frequency",
    "most asked", "least asked", "repeat count", "year wise",
    "distribution", "statistics", "analytics",
]
_PYQ = ["pyq", "previous year", "purane question", "exam mein", "kitni baar",
        "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
_IMPORTANCE = ["important", "important topics", "kya padhein", "kitna important",
               "frequently", "baar baar"]
_SYLLABUS = ["syllabus", "unit", "kya kya aata hai", "course", "rgpv syllabus"]


def intent_node(state: GraphState) -> GraphState:
    """
    Classifies the user query intent using keyword heuristics.
    Sets state['intent'].
    """
    q = state["query"].lower()

    if any(k in q for k in _ANALYTICAL):
        intent = Intent.PYQ_ANALYTICS
    elif any(k in q for k in _PYQ):
        intent = Intent.PYQ_RETRIEVAL
    elif any(k in q for k in _IMPORTANCE):
        intent = Intent.TOPIC_IMPORTANCE
    elif any(k in q for k in _SYLLABUS):
        intent = Intent.SYLLABUS_LOOKUP
    else:
        intent = Intent.CONCEPT_EXPLANATION

    logger.info("[intent_node] query=%r → intent=%s", state["query"][:60], intent)
    return {**state, "intent": intent}
