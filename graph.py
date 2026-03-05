from langgraph.graph import StateGraph, END
from core.state import AgentState
from agents.planner import planner_node
from agents.executor import browser_execution_node
from agents.rag_analyzer import rag_analyzer_node

def build_graph():
    """Constructs the LangGraph state machine."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("browser_execution", browser_execution_node)
    workflow.add_node("rag_analyzer", rag_analyzer_node)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "browser_execution")
    workflow.add_edge("browser_execution", "rag_analyzer")
    workflow.add_edge("rag_analyzer", END)
    
    return workflow.compile()