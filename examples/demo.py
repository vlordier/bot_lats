from lats.config import LATSConfig
from lats.search_agent import LATSAgent


def main():
    """Example usage of the LATS agent."""
    config = LATSConfig(
        max_search_depth=5,
        branching_factor=5,
        model_name="gpt-4"
    )

    agent = LATSAgent(config)
    
    # Run agent on example query
    result = agent.search("Write a research report on lithium pollution.")
    
    print(f"Solution found: {result['solution']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Steps taken: {result['num_steps']}")
    print(f"Complete solution: {result['is_complete']}")


if __name__ == "__main__":
    main()
