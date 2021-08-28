from time import sleep
from perfect_information_game.move_selection import MoveChooser
from perfect_information_game.tablebases import TablebaseManager


class TablebaseChooser(MoveChooser):
    def __init__(self, GameClass, backup_move_chooser=None, starting_position=None, delay=1):
        super().__init__(GameClass, starting_position)
        self.backup_move_chooser = backup_move_chooser
        self.tablebase_manager = TablebaseManager(GameClass)
        self.delay = delay

    def choose_move(self, return_distribution=False):
        if self.GameClass.is_over(self.position):
            raise Exception('Game Finished!')

        move, outcome, distance = self.tablebase_manager.query_position(self.position)

        if move is None:
            if self.backup_move_chooser is None:
                raise ValueError('No backup move chooser configured!')
            self.backup_move_chooser.position = self.position
            move = self.backup_move_chooser.choose_move(return_distribution)
            self.position = move[-1]
            return move
        elif self.delay > 0:
            sleep(self.delay)

        print(f'Outcome: {outcome}, distance: {distance}')
        self.position = move
        return [move]

    def report_user_move(self, user_chosen_position):
        super().report_user_move(user_chosen_position)
        if self.backup_move_chooser is not None:
            self.backup_move_chooser.report_user_move(user_chosen_position)
