# Checkers Training Process
First, 64 games were generated using MCTS rollout with 100 expansions per move, and c=sqrt(2).
After it was observed that these games were taking a long time to be generated due to the rollouts, the number of expansions per move was reduced to 10.
