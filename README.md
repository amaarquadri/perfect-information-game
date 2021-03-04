# Perfect Information Game
Creating 2D perfect information board games, and playing them with machine learning systems.

## Getting Started
- Ensure Python is installed
- Install requirements: \
`pip install -r requirements.txt`
- Play a game of Connect 4 against the ai: \
`python src/scripts/play_vs_ai.py`
- View games files that were generated during training: \
`python src/scripts/view_game_file.py`
- Switch the active game by uncommenting the corresponding line in `src/utils/active_game.py`
- For games with multiple versions, select the desired version by opening the corresponding file under `src/games/` and 
uncommenting the corresponding line that starts with `CONFIG = `

## How I Trained the Models
- [Connect 4](/training/Connect4)
- [Othello/Reversi](/training/Othello)
- [Amazons (6x6 Board)](/training/Amazons/6x6)

## Play Against Live Models on my Website
- Connect 4:  [Easy](https://www.amaarquadri.com/play?game=connect4&difficulty=easy&ai-time=1&log-stats=true), [Medium](https://www.amaarquadri.com/play?game=connect4&difficulty=medium&ai-time=1&log-stats=true), [Hard](https://www.amaarquadri.com/play?game=connect4&difficulty=hard&ai-time=1&log-stats=true)
- Othello: Coming Soon
- Amazons (6x6 Board): Coming Soon

## Resources I Used
- [How to Keep Improving When You're Better Than Any Teacher - Iterated Distillation and Amplification](https://youtu.be/v9M2Ho9I9Qo)
- [Multiple Policy Value Monte Carlo Tree Search](https://arxiv.org/pdf/1905.13521.pdf)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf)
- [Parallel Monte-Carlo Tree Search](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.159.4373&rep=rep1&type=pdf)
- [A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm](https://webdocs.cs.ualberta.ca/~mmueller/ps/enzenberger-mueller-acg12.pdf)
- [Time Management for Monte Carlo Tree Search](https://dke.maastrichtuniversity.nl/m.winands/documents/time_management_for_monte_carlo_tree_search.pdf)
- [Lessons From Alpha Zero (part 5): Performance Optimization](https://medium.com/oracledevs/lessons-from-alpha-zero-part-5-performance-optimization-664b38dc509e)
- [How much training data do you need?](https://medium.com/@malay.haldar/how-much-training-data-do-you-need-da8ec091e956)
- [Working with Numpy in Cython](https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html)
- [Chess Pieces Images](https://commons.wikimedia.org/wiki/Category:PNG_chess_pieces/Standard_transparent)
- [Chess test cases](https://www.chessprogramming.org/Perft_Results) and more [chess test cases](https://gist.github.com/peterellisjones/8c46c28141c162d1d8a0f0badbc9cff9)
