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