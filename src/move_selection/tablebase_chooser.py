from move_selection.move_chooser import MoveChooser
from tablebases.tablebase_manager import TablebaseManager
from move_selection.random_chooser import RandomMoveChooser


class TablebaseChooser(MoveChooser):
    def __init__(self, GameClass, backup_move_chooser=None, starting_position=None):
        super().__init__(GameClass, starting_position)
        self.backup_move_chooser = backup_move_chooser if backup_move_chooser is not None \
            else RandomMoveChooser(GameClass)
        self.tablebase_manager = TablebaseManager()

    def choose_move(self, return_distribution=False):
        if self.GameClass.is_over(self.position):
            raise Exception('Game Finished!')

        move, outcome, distance = self.tablebase_manager.query_position(self.position)

        if move is None:
            self.backup_move_chooser.position = self.position
            return self.backup_move_chooser.choose_move(return_distribution)
        else:
            print(f'Outcome: {outcome}, distance: {distance}')
            self.position = move
            return [move]
