from .thought_template import ThoughtTemplate
from utils.code_execution import extract_and_execute_code
from problem_distiller.problem_distiller import ProblemInfo


class InstantiatedReasoner:
    def instantiate(
        self, template: ThoughtTemplate, problem_info: ProblemInfo
    ) -> ThoughtTemplate:
        instantiated_graph = ThoughtTemplate(
            nodes=problem_info.nodes,
            edges=problem_info.edges,
            labels={**template.labels},
        )
        return instantiated_graph

    def execute(self, graph: ThoughtTemplate):
        result = graph.labels.get("output", "No result available.")
        return extract_and_execute_code(result)
