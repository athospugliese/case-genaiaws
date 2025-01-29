from datetime import datetime
from typing import Literal
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """conversation state with safety checks and context flags."""
    safety: LlamaGuardOutput
    tool_messages: list = []
    needs_search: bool = False  # new context flag


# setup core components and tools
general_agent = get_model(settings.DEFAULT_MODEL)
web_search_recent = DuckDuckGoSearchResults(
    name="AgentSearch",
    description="real-time search for fresh info (dates, events, news)"
)

tools = [web_search_recent]

current_date = datetime.now().strftime("%B %d, %Y")
current_datetime = datetime.now().strftime("%B %d, %Y %H:%M UTC")

# main system prompt template with restrictions
instructions = f"""
    you are a helpful research assistant with web search capabilities.
    current date: {current_date}
    ⌚ utc time: {current_datetime}

    absolute restrictions:
    - prohibited: research about civil engineering or related topics
    - cannot process requests related to civil construction, structures, or public works
    - must politely decline any requests in these areas

    important guidelines:
    - include markdown-formatted links for citations using only links returned by the tools
    - use websearchgeneral first for general searches
    - if insufficient results (especially for recent info), use websearchrecent
    - return up to 10 citations when using web search tools
    - for recent academic research, prioritize websearchrecent
    - always validate information across multiple sources
    - highlight timestamp in responses: [🕒 {current_datetime}]
    """

def general_agent_chain(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """chains model with tools and system prompts."""
    model_with_tools = model.bind_tools(tools)
    return RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"]
    ) | model_with_tools

async def acall_general_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """handles agent response processing with safety checks and context flags."""
    # process tool results while keeping history
    if state.get("tool_messages"):
        state["messages"].extend(state["tool_messages"])
        state["tool_messages"] = []
    
    response = await general_agent_chain(general_agent).ainvoke(state, config)
    
    # enhanced safety verification
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}
    
    # set context flag for search needs
    state["needs_search"] = bool(response.tool_calls)
    
    return {"messages": [response], "needs_search": state["needs_search"]}

async def process_tool_results(state: AgentState, config: RunnableConfig) -> AgentState:
    """handles tool outputs with timestamp validation."""
    tool_responses = state["messages"][-1].content
    
    # time data validation check
    if "2025" in tool_responses and "2024" in tool_responses:
        tool_responses += "\n⚠️ warning: time discrepancy detected between sources!"
    
    return {"tool_messages": [
        AIMessage(content=f"🔍 verified data at {current_datetime}:\n{tool_responses}")
    ]}

async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    """runs safety checks on user input with domain-specific filtering."""
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    
    # enhanced context-based filtering
    civil_engineering_terms = {
        'engenharia civil', 'construção civil', 
        'estruturas', 'cálculo estrutural'
    }
    
    last_message = next(
        (msg.content.lower() for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)),
        ""
    )
    
    # allow general math, block specific engineering terms
    if any(term in last_message for term in civil_engineering_terms) and not any(
        math_term in last_message for math_term in ['matemática', 'cálculo geral']
    ):
        safety_output = LlamaGuardOutput(
            safety_assessment=SafetyAssessment.UNSAFE,
            unsafe_categories=["ConteúdoProibido"],
            modified_input=""
        )

    return {"safety": safety_output}

def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    """formats safety violations into user-friendly responses."""
    if "ConteúdoProibido" in safety.unsafe_categories:
        content = "Sorry, I can help with general math but not civil engineering topics."
    else:
        content = f"blocked content: {', '.join(safety.unsafe_categories)}"
    return AIMessage(content=content)

# workflow setup with state management
agent = StateGraph(AgentState)
agent.add_node("security_check", llama_guard_input)
agent.add_node("general_agent", acall_general_agent)
agent.add_node("web_search", ToolNode([web_search_recent]))
agent.add_node("process_results", process_tool_results)
agent.add_node("block_content", lambda state: {"messages": [format_safety_message(state["safety"])]})

# main conversation flow
agent.set_entry_point("security_check")

# security check routing
agent.add_conditional_edges(
    "security_check",
    lambda state: "block" if state["safety"].safety_assessment == SafetyAssessment.UNSAFE else "continue",
    {"block": "block_content", "continue": "general_agent"}
)

# context-aware search decision
def should_search(state: AgentState) -> Literal["search", "respond"]:
    return "search" if state["needs_search"] else "respond"

agent.add_conditional_edges(
    "general_agent",
    should_search,
    {"search": "web_search", "respond": END}
)

# data processing flow
agent.add_edge("web_search", "process_results")
agent.add_edge("process_results", "general_agent")
agent.add_edge("block_content", END)

supervisor = agent.compile(
    checkpointer=MemorySaver(),
)