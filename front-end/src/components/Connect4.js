export default class Connect4 {
    static ROWS = 6
    static COLUMNS = 7

    static getStartingState() {
        return [
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        ]
    }

    static toReactState(state) {
        return state.map((row, rowIndex) => row.map((square, columnIndex) => {
            return  {row: rowIndex, column: columnIndex,
                p1Piece: square[0] === 1,
                p2Piece: square[1] === 1,
                p1Turn: square[2] === 1}
        }))
    }

    static toTensorFlowState(reactState) {
        return reactState.map(row => row.map(square =>
            [square.p1Piece ? 1 : 0, square.p2Piece ? 1 : 0, square.p1Turn ? 1 : 0]))
    }

    static performUserMove(state, row, column) {
        //performs the user move on the given state,
        //if the selected move is illegal, then null will be returned
        if (state[0][column][0] === 1 || state[0][column][1] === 1) {
            return null
        }

        let targetRow = this.ROWS - 1
        while (state[targetRow][column][0] === 1 || state[targetRow][column][1] === 1) {
            targetRow -= 1
        }

        return state.map((rowData, rowIndex) => rowData.map((squareData, columnIndex) => {
            if (rowIndex === targetRow && columnIndex === column) {
                if (squareData[2] === 1) {
                    squareData[0] = 1
                }
                else {
                    squareData[1] = 1
                }
            }
            squareData[2] = (1 - squareData[2])
            return squareData
        }))
    }

    static isPlayer1Turn(state) {
        return state[0][0][2]
    }

    static getPossibleMoves(state) {
        let moves = []
        let isPlayer1Turn = this.isPlayer1Turn(state)
        for (let j = 0; j < this.COLUMNS; j++) {
            for (let i = this.ROWS - 1; i >= 0; i--) {
                if (state[i][j][0] === 0 && state[i][j][1] === 0) {
                    moves.push(state.map((rowData, rowIndex) => rowData.map((columnData, columnIndex) => {
                        if (rowIndex === i && columnIndex === j) {
                            if (isPlayer1Turn) {
                                columnData[0] = 1
                            }
                            else {
                                columnData[1] = 1
                            }
                        }
                        columnData[2] = isPlayer1Turn ? 0 : 1
                        return columnData
                    })))
                    break
                }
            }
        }
        return moves
    }

    static getLegalMoves(state) {
        return [0, 1, 2, 3, 4, 5, 6, 7].map(column =>
            (state[0][column][0] === 0 && state[0][column][1] === 0))
    }

    static copy(state) {
        let newState = this.getStartingState()
        for (let k = 0; k < 3; k++) {
            for (let i = 0; i < this.ROWS; i++) {
                for (let j = 0; j < this.COLUMNS; j++) {
                    newState[k][i][j] = state[k][i][j]
                }
            }
        }
        return newState
    }

    static nullMove(state) {
        let newState = this.getStartingState()
        for (let k = 0; k < 3; k++) {
            for (let i = 0; i < this.ROWS; i++) {
                for (let j = 0; j < this.COLUMNS; j++) {
                    newState[k][i][j] = k === 2 ? (1 - state[i][j][2]) : state[i][j][k]
                }
            }
        }
        return newState
    }
}