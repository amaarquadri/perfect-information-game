# Othello Training Process
First, games were generated using MCTS rollout with 100 expansions per move, and c=sqrt(2).  
Then a network with a 4x4 kernel size and 6 residual layers was trained using supervised learning on those games.  
Then the network was further trained using reinforcement learning with 500 expansions per move, c=np.sqrt(2), and d=1.  
A total of 7354 games were played during this stage, and models were saved every 30 minutes.  
This graph shows the training process:  
![Epoch Loss with 6 Residual Layers](/training/Othello/logs/training_process_4x4_6.png)  

The reinforcement learning process was repeated using the same MCTS rollout games with a network that had 10 residual layers instead of 6.  
A total of 2326 games were played during this stage before training was prematurely cut short.
This graphs shows the training process for this modified network:
![Epoch Loss with 10 Residual Layers](/training/Othello/logs/training_process_4x4_10.png)  

Interestingly, the value estimation seems to be better with the higher number of residual layers, but the policy seems to be worse.
It is not clear if this trend would continue as the training finishes.

Based on the learning curve, the following models (with the original 6 residual layers) were chosen for their respective difficulty levels:  
- Easy: model_reinforcement-1604045224.7152517 (0 hours into training)  
- Medium: model_reinforcement-1604092085.6274474 (13 hours into training)  
- Hard: model_reinforcement-1604228204.9054377 (2 days 3 hours into training)  
