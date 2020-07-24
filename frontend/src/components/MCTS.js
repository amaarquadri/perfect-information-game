export default class MCTS {
    static chooseMove(GameClass, position, networkFunc) {
        const result = networkFunc([position])[0]
        const policy = result[0]

        const moves = GameClass.getPossibleMoves(position)
        return moves[this.argMax(policy)]
    }

    static argMax(arr) {
        const acc = arr.reduce((acc, val, index) => {
                if (val > acc.max) {
                    acc.index = index
                    acc.max = val
                }
                return acc
            },
            {
                index: -1,
                max: -Infinity
            })
        if (acc.index === -1) {
            throw new Error('acc is empty!')
        }
        return acc.index
    }
}
