import logging
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import OpenAIEmbeddings
from langchain.chains import LLMChain


from meta_buffer.meta_buffer import MetaBuffer
from problem_distiller.problem_distiller import ProblemDistiller
from reasoner.instantiated_reasoner import InstantiatedReasoner
from lats.lats_subgraph import LATSSubgraph
from problem_distiller.problem_distiller import ProblemInfo
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

llm = OpenAI(temperature=0.7)
prompt_template = PromptTemplate(input_variables=["text"], template="{text}")

llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Initialize framework components
meta_buffer = MetaBuffer(llm_chain, OpenAIEmbeddings())
problem_distiller = ProblemDistiller(llm_chain)
reasoner = InstantiatedReasoner()
lats = LATSSubgraph(llm_chain)


# Define StateGraph nodes
def distill_problem(state):
    """Distill the problem into a structured graph."""
    problem = state["problem"]
    try:
        problem_graph = problem_distiller.distill(problem)
    except Exception as e:
        logging.error(f"Error distilling problem: {e}, problem: {problem}")
        raise
    state["problem_graph"] = problem_graph.model_dump()
    return state


def analyze_with_lats(state):
    """Perform LATS-based logical analysis."""
    input_data = state["problem_graph"].get("distilled_info", {}).get("text", "")
    try:
        lats_analysis = lats.perform_task(input_data)
    except Exception as e:
        logging.error(f"Error performing LATS analysis: {e}, input_data: {input_data}")
        raise
    state["lats_analysis"] = lats_analysis
    return state


def retrieve_template(state):
    """Retrieve a suitable thought template."""
    problem_graph = state["problem_graph"]
    try:
        template = meta_buffer.retrieve_and_instantiate(
            problem_graph.get("distilled_info", {}).get("text", "")
        )
    except Exception as e:
        logging.error(f"Error retrieving template: {e}, problem_graph: {problem_graph}")
        raise
    if template is None or template == "No matching template found.":
        state["retrieval_success"] = False
    else:
        state["retrieval_success"] = True
        state["template"] = template
    return state


def instantiate_template(state):
    """Instantiate the retrieved thought template."""
    template = state["template"]
    problem_graph = state["problem_graph"]
    try:
        instantiated_graph = reasoner.instantiate(template, problem_graph)
    except Exception as e:
        logging.error(
            f"Error instantiating template: {e}, template: {template}, problem_graph: {problem_graph}"
        )
        raise
    state["instantiated_graph"] = instantiated_graph
    return state


def execute_reasoning(state):
    """Execute reasoning on the instantiated graph."""
    instantiated_graph = state["instantiated_graph"]
    try:
        solution = reasoner.execute(instantiated_graph)
    except Exception as e:
        logging.error(
            f"Error executing reasoning: {e}, instantiated_graph: {instantiated_graph}"
        )
        raise
    if solution == "No result available.":
        state["execution_success"] = False
    else:
        state["execution_success"] = True
        state["solution"] = solution
    return state


def distill_new_template(state):
    """Distill a new thought template from the solution."""
    problem_graph = state["problem_graph"]
    solution = state["solution"]
    try:
        distilled_template = meta_buffer.distill_template(problem_graph, solution)
    except Exception as e:
        logging.error(
            f"Error distilling new template: {e}, problem_graph: {problem_graph}, solution: {solution}"
        )
        raise
    state["distilled_template"] = distilled_template
    return state


def update_meta_buffer(state):
    """Update the MetaBuffer with the new template."""
    distilled_template = state["distilled_template"]
    try:
        meta_buffer.dynamic_update(distilled_template)
    except Exception as e:
        logging.error(
            f"Error updating MetaBuffer: {e}, distilled_template: {distilled_template}"
        )
        raise
    return state


# Retry logic and loop conditions
def should_retry_retrieval(state):
    """Retry template retrieval if it failed."""
    if not state.get("retrieval_success", True):
        retries = state.get("retrieval_retries", 0)
        if retries < 3:  # Retry up to 3 times
            state["retrieval_retries"] = retries + 1
            return "retrieve_template"
    if state.get("retrieval_success", False):
        return "instantiate_template"
    else:
        return END


def should_retry_execution(state):
    """Retry reasoning execution if it failed."""
    if not state.get("execution_success", True):
        retries = state.get("execution_retries", 0)
        if retries < 3:  # Retry up to 3 times
            state["execution_retries"] = retries + 1
            return "execute_reasoning"
    if state.get("execution_success", False):
        return "distill_new_template"
    else:
        return END


# Build the LangGraph
class StateSchema(TypedDict):
    problem: str
    solution: str
    problem_graph: ProblemInfo


builder = StateGraph(state_schema=StateSchema)

# Add nodes to the graph
builder.add_node("distill_problem", distill_problem)
builder.add_node("analyze_with_lats", analyze_with_lats)
builder.add_node("retrieve_template", retrieve_template)
builder.add_node("instantiate_template", instantiate_template)
builder.add_node("execute_reasoning", execute_reasoning)
builder.add_node("distill_new_template", distill_new_template)
builder.add_node("update_meta_buffer", update_meta_buffer)

# Define edges with conditional retries
builder.add_edge(START, "distill_problem")
builder.add_edge("distill_problem", "analyze_with_lats")
builder.add_edge("analyze_with_lats", "retrieve_template")
builder.add_conditional_edges(
    "retrieve_template",
    should_retry_retrieval,
    ["retrieve_template", "instantiate_template", END],
)
builder.add_edge("instantiate_template", "execute_reasoning")
builder.add_conditional_edges(
    "execute_reasoning",
    should_retry_execution,
    ["execute_reasoning", "distill_new_template", END],
)
builder.add_edge("distill_new_template", "update_meta_buffer")
builder.add_edge("update_meta_buffer", END)

# Compile and execute the graph
graph = builder.compile()

# Example usage
if __name__ == "__main__":
    initial_state = {
        "problem": "Calculate the area of a triangle with base 5 and height 10."
    }
    final_state = graph.invoke(initial_state)
    print("Solution:", final_state.get("solution", "No solution found."))
