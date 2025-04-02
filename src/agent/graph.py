from typing import Any, TypedDict

from fastapi import FastAPI

from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary
from langserve import add_routes


class State(MessagesState):
    context: dict[str, Any]

def search(query: str):
    """Search the web for realtim information like weather forecasts."""
    return "The weather is sunny in New York, with a high of 104 degrees."

tools = [search]

model = ChatGroq(temperature=0, model_name="qwen-2.5-32b")

# async def call_model(state: State, config: RunnableConfig, *, store:BaseStore) -> dict:
    
builder = StateGraph
summarization_model = model.bind(max_tokens=128)

summarization_node = SummarizationNode(
    token_counter=model.get_num_tokens_from_messages,
    model=summarization_model,
    max_tokens=256,
    max_summary_tokens=128,
)

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, Any]

def call_model(state: LLMInputState):
    response = model.bind_tools(tools).invoke(state["summarized_messages"])
    return {"messages": [response]}

# Define a router that determines whether to execute tools or exit
def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "tools"

checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node("summarize_node", summarization_node)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("summarize_node")
builder.add_edge("summarize_node", "call_model")
builder.add_conditional_edges("call_model", should_continue, path_map=["tools", END])
# instead of returning to LLM after executing tools, we first return to the summarization node
builder.add_edge("tools", "summarize_node")
graph = builder.compile(checkpointer=checkpointer)

# Invoke the graph
# config = {"configurable": {"thread_id": "1"}}
# graph.invoke({"messages": "hi, i am bob"}, config)
# graph.invoke({"messages": "what's the weather in nyc this weekend"}, config)
# graph.invoke({"messages": "what's new on broadway?"}, config)

