from perfect_information_game.tablebases import TablebaseGenerator
from perfect_information_game.utils import OptionalPool
from perfect_information_game.games import Chess as GameClass


def generate_tablebases(threads=12):
    generator = TablebaseGenerator(GameClass)

    TWO_MAN = 'Kk'
    THREE_MAN = 'KQk,KRk,KBk,KNk,KPk'
    FOUR_MAN_NO_ENEMY_NO_DUPLICATE = 'KQRk,KQBk,KQNk,KRBk,KRNk,KBNk'
    FOUR_MAN_NO_ENEMY = 'KQQk,KQRk,KQBk,KQNk,KQPk,KRRk,KRBk,KRNk,KRPk,KBBk,KBNk,KBPk,KNNk,KNPk,KPPk'
    FOUR_MAN_WITH_ENEMY = 'KQkq,KQkr,KQkb,KQkn,KQkp,KRkr,KRkb,KRkn,KRkp,KBkb,KBkn,KBkp,KNkn,KNkp,KPkp'
    with OptionalPool(threads) as pool:
        for section in [FOUR_MAN_WITH_ENEMY]:
            for descriptor in section.split(','):
                if descriptor in GameClass.DRAWING_DESCRIPTORS:
                    print(f'Skipping drawing descriptor: {descriptor}')
                    continue
                print(f'Generating tablebase for {descriptor}')
                generator.generate_tablebase(descriptor, pool)
                print(f'Completed {descriptor}')


if __name__ == '__main__':
    generate_tablebases()
