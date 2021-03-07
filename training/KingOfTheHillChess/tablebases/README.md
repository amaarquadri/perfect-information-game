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
