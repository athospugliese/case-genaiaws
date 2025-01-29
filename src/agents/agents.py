from dataclasses import dataclass
from langgraph.graph.state import CompiledStateGraph
from agents.supervisor import supervisor
from schema import AgentInfo

DEFAULT_AGENT = "supervisor"


@dataclass
class Agent:
    """holds info about an agent like description and its workflow graph."""
    description: str
    graph: CompiledStateGraph


# registry of available agents (default is supervisor)
agents: dict[str, Agent] = {
    "supervisor": Agent(
        description="A support assistant with web search.", graph=supervisor
    ),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    """gets an agent's workflow graph by its id."""
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    """collects all agent info for UI display in the required format."""
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]