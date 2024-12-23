# LATS Overview

LATS (Language Agent Tree Search) is a general LLM agent search algorithm developed by Zhou et al. It combines reflection/evaluation and search (specifically Monte Carlo Tree Search) to achieve better overall task performance compared to similar techniques like ReACT, Reflexion, or Tree of Thoughts.

## How LATS Works

LATS operates by analyzing problems, retrieving relevant templates, and executing reasoning processes. It is designed to work seamlessly with language models to improve problem-solving capabilities.

## LATS Diagram

LATS operates through four main steps:

1. **Select**: Pick the best next actions based on the aggregate rewards from step (2). Either respond (if a solution is found or the max search depth is reached) or continue searching.
2. **Expand and Simulate**: Select the "best" 5 potential actions to take and execute them in parallel.
3. **Reflect + Evaluate**: Observe the outcomes of these actions and score the decisions based on reflection (and possibly external feedback).
4. **Backpropagate**: Update the scores of the root trajectories based on the outcomes.

Below is a high-level representation of LATS 

![LATS Overview](./lats.png)

## Implementation Details

The LATS implementation uses:

- Asynchronous parallel simulation of multiple trajectories
- Full MCTS with selection, expansion, simulation, and backpropagation
- Normalized evaluation scores for proper value propagation
- Configurable branching factor and exploration parameters

## Jupyter Notebook

For a detailed tutorial and examples, please refer to the [LATS Jupyter Notebook](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/lats/lats.ipynb).
