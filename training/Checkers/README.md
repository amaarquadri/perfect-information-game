# Checkers Training Process
First, 64 games were generated using MCTS rollout with 100 expansions per move, and c=sqrt(2).  
After it was observed that these games were taking a long time to be generated due to the length of the rollouts, it was decided to try training the network from scratch.  
A network with a 4x4 kernel size and 6 residual layers was created with random weights.  
Then the network was trained using reinforcement learning with 500 expansions per move, c=np.sqrt(2), and d=1.  
The replay buffer was initialized with the 64 MCTS rollout games, and allowed to grow to the full size of 1000 games as training occurred.  
A total of n games were played during this stage, and models were saved every 30 minutes.  
This graph shows the training process:  
