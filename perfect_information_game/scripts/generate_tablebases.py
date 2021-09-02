from perfect_information_game.tablebases import ChessTablebaseManager, ChessTablebaseGenerator
from perfect_information_game.utils import OptionalPool
from perfect_information_game.games import Chess as GameClass


def generate_tablebases(threads=12):
    manager = ChessTablebaseManager(GameClass)
    generator = ChessTablebaseGenerator(GameClass)

    with OptionalPool(threads) as pool:
        for piece_count in [2, 3, 4]:
            for descriptor in ChessTablebaseGenerator.generate_descriptors(piece_count):
                if descriptor in GameClass.DRAWING_DESCRIPTORS:
                    print(f'Skipping drawing descriptor: {descriptor}')
                    continue
                if descriptor in manager.available_tablebases:
                    print(f'Skipping existing descriptor: {descriptor}')
                    continue
                print(f'Generating tablebase for {descriptor}')
                generator.generate_tablebase(descriptor, pool)
                print(f'Completed {descriptor}')


if __name__ == '__main__':
    generate_tablebases()
