Creating 2D perfect information board games, and playing them with machine learning systems.

Training process:
raw_mcts_games: 500 rollouts per move, c=sqrt2, no network
policy0 and evaluation0 models: trained on raw_mcts_games data for 100 epochs each

Relevant Resources
https://webdocs.cs.ualberta.ca/~mmueller/ps/enzenberger-mueller-acg12.pdf
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.159.4373&rep=rep1&type=pdf
https://arxiv.org/pdf/1905.13521.pdf
https://medium.com/oracledevs/lessons-from-alpha-zero-part-5-performance-optimization-664b38dc509e
https://medium.com/@malay.haldar/how-much-training-data-do-you-need-da8ec091e956
https://cython.readthedocs.io/en/latest/src/tutorial/numpy.html