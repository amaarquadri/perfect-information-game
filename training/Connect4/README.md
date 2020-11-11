First, games were generated using MCTS rollout with 100 expansions per move, and c=sqrt(2).  
Then a network with a 4x4 kernel size and 6 residual layers was trained using supervised learning on those games.  
Then the network was further trained using reinforcement learning with 500 expansions per move, c=np.sqrt(2), and d=1.  
A total of 13906 games were played during this stage, and models were saved every 30 minutes.  
This graph shows the training process:  
![Epoch Loss](/training/Connect4/logs/training_process.png)  

Based on the learning curve and playing against the agents, the following models were chosen for their respective difficulty levels:  
- Easy: model_reinforcement-1603784367.0827568 (0 hours into training)  
- Medium: model_reinforcement-1603813336.0729053 (8 hours into training)  
- Hard: model_reinforcement-1603854873.4800494 (19.5 hours into training)  
