# Perfect Information Game
Creating 2D perfect information board games, and playing them with machine learning systems.

## Getting Started
- Install requirements: \
`pip install -r requirements.txt` \
`cd frontend` \
`npm install`
- Play a game of Connect 4 against the ai: \
`python src/scripts/play_vs_ai.py`
- View games files that were generated during training: \
`python src/scripts/view_game_file.py`
- Switch the active game by uncommenting the corresponding line in `src/utils/active_game.py`
- For games with multiple versions, select the desired version by opening the corresponding file under `src/games/` and 
uncommenting the corresponding line that starts with `CONFIG = `

## Relevant Resources
https://webdocs.cs.ualberta.ca/~mmueller/ps/enzenberger-mueller-acg12.pdf \
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.159.4373&rep=rep1&type=pdf \
https://arxiv.org/pdf/1905.13521.pdf \
https://medium.com/oracledevs/lessons-from-alpha-zero-part-5-performance-optimization-664b38dc509e \
https://medium.com/@malay.haldar/how-much-training-data-do-you-need-da8ec091e956 \
https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html \
https://dke.maastrichtuniversity.nl/m.winands/documents/time_management_for_monte_carlo_tree_search.pdf
