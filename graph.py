import logging
from langgraph.graph import StateGraph, END
from app.agent.state import AgentState
from app.agent.nodes import (
    analyze_intent_node,
    literature_search_node,
    hypothesis_generation_node,
    experiment_design_node,
    synthesis_node
)

logger = logging.getLogger(__name__)

def route_intent(state: AgentState) -> str:
    """Determine the next node based on classified intent."""
    intent = state.get("intent", "general_inquiry")
    if intent == "literature_search":
        return "literature_search"
    elif intent == "hypothesis_generation":
        # First retrieve literature to ground the hypothesis generation
        return "literature_search"
    elif intent == "experiment_design":
        return "experiment_design"
    else:
        return "synthesize"

def route_after_literature(state: AgentState) -> str:
    """Route after literature retrieval: generate hypotheses or synthesize directly."""
    intent = state.get("intent", "general_inquiry")
    if intent == "hypothesis_generation":
        return "hypothesis_generation"
    return "synthesize"

def create_research_graph():
    """Build the compiled LangGraph workflow for scientific research assistance."""
    workflow = StateGraph(AgentState)

    # Register all nodes
    workflow.add_node("analyze_intent", analyze_intent_node)
    workflow.add_node("literature_search", literature_search_node)
    workflow.add_node("hypothesis_generation", hypothesis_generation_node)
    workflow.add_node("experiment_design", experiment_design_node)
    workflow.add_node("synthesize", synthesis_node)

    # Set entry point
    workflow.set_entry_point("analyze_intent")

    # Conditional routing from intent analysis
    workflow.add_conditional_edges(
        "analyze_intent",
        route_intent,
        {
            "literature_search": "literature_search",
            "experiment_design": "experiment_design",
            "synthesize": "synthesize"
        }
    )

    # Conditional routing after literature search
    workflow.add_conditional_edges(
        "literature_search",
        route_after_literature,
        {
            "hypothesis_generation": "hypothesis_generation",
            "synthesize": "synthesize"
        }
    )

    # Hypotheses flow to synthesis
    workflow.add_edge("hypothesis_generation", "synthesize")
    
    # Experiments flow to synthesis
    workflow.add_edge("experiment_design", "synthesize")

    # Final response terminates
    workflow.add_edge("synthesize", END)

    return workflow.compile()

# Global compiled graph
research_agent_graph = create_research_graph()
