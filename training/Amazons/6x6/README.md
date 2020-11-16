# Amazons (6x6 Board) Training Process
First, 1000 games were generated using MCTS rollout with 500 expansions per move, and c=sqrt(2).  
Then a network with a 4x4 kernel size and 8 residual layers was trained using supervised learning on those games.  
The policy loss was given a weight of 50,000 and the value loss was given a weight of 1 in order to roughly equalize the two.  
Then the network was further trained using reinforcement learning with 500 expansions per move, c=0.1, and d=0.5.  
The replay buffer was a fixed size of 1000 games, and it was initialized with the 1000 MCTS rollout games.  
A total of ? games were played during this stage, and models were saved every 30 minutes.  
This graph shows the training process:  
