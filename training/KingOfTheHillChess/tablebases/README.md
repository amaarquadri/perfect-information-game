# King of the Hill Chess - Endgame Puzzles
King of the Hill Chess is a variant of chess where you can win by getting your king to one of the 4 central squares of the board (on top of the usual possibility of winning by checkmate).
All other chess rules apply as normal including stalemate, three-fold repetition, the fifty move rule etc.

While creating endgame tablebases for this variant of chess, I came across several interesting positions.

This position is black to play and draw:  
![Black to Play and Draw](/training/KingOfTheHillChess/tablebases/puzzle_1.png)  
<details><summary>Click to Show Solution</summary>
<p>

1. ...  KC7  
2. BF4+ KB6  
3. BE3+ KC7  

(draw by three-fold repetition)  
This is because anything other than repeating moves will lead to a win for the opposing side.  
For example:  
1. ... KxD8  
2. KB7 KD7  
3. BF4 KE6  
4. KC6 KF5  
5. KD5#  
</p>
</details>

This position is white to play. What is the optimal result and the best move?
![White to Play](/training/KingOfTheHillChess/tablebases/puzzle_2.png)
<details><summary>Click to Show Solution</summary>
<p>

1. ...  ND6

(draw by stalemate)  
White must settle for a draw because otherwise black will get to the center first.
</p>
</details>

This position is white to play and win.
![White to Play and Win](/training/KingOfTheHillChess/tablebases/puzzle_3.png)
<details><summary>Click to Show Solution</summary>
<p>

With optimal defense by black:
1. Nb5 Kd7
2. Bg8 Ke7
3. Nd4 Kd6
4. Nf3 Kc6
5. Kb8 Kb6
6. Kc8 Kc6
7. Kd8 Kd6
8. Ke8 Kc5
9. Ke7 Kc6
10. Ke6 Kc5
11. Ke5#
</p>
</details>

This similar position is white to play and win.
![White to Play and Win](/training/KingOfTheHillChess/tablebases/puzzle_4.png)
<details><summary>Click to Show Solution</summary>
<p>

With optimal defense by black:
1. Nd7 Kb5
2. Bg8 Kb4
3. Ne5 Kc5
4. Nf3 Kc6
5. Kb8 Kb6
6. Kc8 Kc6
7. Kd8 Kd6
8. Ke8 Kc6
9. Ke7 Kb6
10. Ke6 Ka6
11. Ke5#
</p>
</details>

This insane position is white to play and win in 28 moves!
![White to Play and Win in 28 Moves](/training/KingOfTheHillChess/tablebases/puzzle_5.png)
<details><summary>Click to Show Solution</summary>
<p>

With optimal defense by black:
1. Qd4+ Ke8
2. Kb7 Ke7
3. Kc7 Qh6
4. Qe5+ Qe6
5. Qg7+ Ke8
6. Qh7 Qd5
7. Kb6 Qd6+
8. Kb5 Qd5+
9. Kb4 Qd4+
10. Kb3 Kd8
11. Qf7 Qc5
12. Qc4 Qe5
13. Qc6 Qd4
14. Qe6 Qc5
15. Qf7 Qd4
16. Kc2 Qe3
17. Qh7 Ke8
18. Qg7 Qf3
19. Kd2 Kd8
20. Qc3 Qf4+
21. Qe3 Qg4
22. Kd3 Kd7
23. Qh6 Qb4
24. Qf6 Qg4
25. Ke3 Qc4
26. Qd4+ Qd5
27. Qxd5+ Kc7
28. Ke4#
</p>
</details>
