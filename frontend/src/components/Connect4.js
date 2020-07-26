export default class Connect4 {
    static ROWS = 6
    static COLUMNS = 7
    static MOVES = Connect4.COLUMNS

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
            return {
                row: rowIndex, column: columnIndex,
                p1Piece: square[0] === 1,
                p2Piece: square[1] === 1,
                p1Turn: square[2] === 1,
                highlight: false
            }
        }))
    }

    static toTensorFlowState(reactState) {
        return reactState.map(row => row.map(square =>
            [square.p1Piece ? 1 : 0, square.p2Piece ? 1 : 0, square.p1Turn ? 1 : 0]))
    }

    static logState(state) {
        console.log(state.map(rowData => rowData.map(squareData => {
            if (squareData[0] === 1) {
                return 1
            }
            if (squareData[1] === 1) {
                return -1
            }
            return 0
        })))
        console.log(this.isPlayer1Turn(state) ? 'P1 Turn' : 'P2 Turn')
    }

    static flatten(state) {
        return state.reduce(((acc, val) => acc.concat(Array.isArray(val) ? this.flatten(val) : val)), [])
    }

    static separateFlattenedPolicies(policies) {
        //Receives a flattened array with several policies, and outputs an array of arrays of flattened policies
        const acc = policies.reduce(((acc, val) => {
            acc.acc.push(val)
            if (acc.acc.length === this.MOVES) {
                acc.policies.push(acc.acc)
                acc.acc = []
            }
            return acc
        }), {
            policies: [],
            acc: []
        })
        if (acc.acc.length > 0) {
            throw new Error('policies.length is not a multiple of ' + this.MOVES)
        }
        return acc.policies
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
                } else {
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
        const moves = []
        const isPlayer1Turn = this.isPlayer1Turn(state)
        for (let j = 0; j < this.COLUMNS; j++) {
            for (let i = this.ROWS - 1; i >= 0; i--) {
                if (state[i][j][0] === 0 && state[i][j][1] === 0) {
                    moves.push(state.map((rowData, rowIndex) => rowData.map((columnData, columnIndex) => {
                        columnData = columnData.slice()
                        if (rowIndex === i && columnIndex === j) {
                            if (isPlayer1Turn) {
                                columnData[0] = 1
                            } else {
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
        // Returns flattened list of legal moves
        return [0, 1, 2, 3, 4, 5, 6].map(column =>
            (state[0][column][0] === 0 && state[0][column][1] === 0))
    }

    static checkWin(pieces) {
        // 4*6+3*7+2*12
        const wins = [
            // horizontal wins, grouped by row
            [[0, 0], [0, 1], [0, 2], [0, 3]],
            [[0, 1], [0, 2], [0, 3], [0, 4]],
            [[0, 2], [0, 3], [0, 4], [0, 5]],
            [[0, 3], [0, 4], [0, 5], [0, 6]],

            [[1, 0], [1, 1], [1, 2], [1, 3]],
            [[1, 1], [1, 2], [1, 3], [1, 4]],
            [[1, 2], [1, 3], [1, 4], [1, 5]],
            [[1, 3], [1, 4], [1, 5], [1, 6]],

            [[2, 0], [2, 1], [2, 2], [2, 3]],
            [[2, 1], [2, 2], [2, 3], [2, 4]],
            [[2, 2], [2, 3], [2, 4], [2, 5]],
            [[2, 3], [2, 4], [2, 5], [2, 6]],

            [[3, 0], [3, 1], [3, 2], [3, 3]],
            [[3, 1], [3, 2], [3, 3], [3, 4]],
            [[3, 2], [3, 3], [3, 4], [3, 5]],
            [[3, 3], [3, 4], [3, 5], [3, 6]],

            [[4, 0], [4, 1], [4, 2], [4, 3]],
            [[4, 1], [4, 2], [4, 3], [4, 4]],
            [[4, 2], [4, 3], [4, 4], [4, 5]],
            [[4, 3], [4, 4], [4, 5], [4, 6]],

            [[5, 0], [5, 1], [5, 2], [5, 3]],
            [[5, 1], [5, 2], [5, 3], [5, 4]],
            [[5, 2], [5, 3], [5, 4], [5, 5]],
            [[5, 3], [5, 4], [5, 5], [5, 6]],

            //vertical wins, grouped by column
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            [[1, 0], [2, 0], [3, 0], [4, 0]],
            [[2, 0], [3, 0], [4, 0], [5, 0]],

            [[0, 1], [1, 1], [2, 1], [3, 1]],
            [[1, 1], [2, 1], [3, 1], [4, 1]],
            [[2, 1], [3, 1], [4, 1], [5, 1]],

            [[0, 2], [1, 2], [2, 2], [3, 2]],
            [[1, 2], [2, 2], [3, 2], [4, 2]],
            [[2, 2], [3, 2], [4, 2], [5, 2]],

            [[0, 3], [1, 3], [2, 3], [3, 3]],
            [[1, 3], [2, 3], [3, 3], [4, 3]],
            [[2, 3], [3, 3], [4, 3], [5, 3]],

            [[0, 4], [1, 4], [2, 4], [3, 4]],
            [[1, 4], [2, 4], [3, 4], [4, 4]],
            [[2, 4], [3, 4], [4, 4], [5, 4]],

            [[0, 5], [1, 5], [2, 5], [3, 5]],
            [[1, 5], [2, 5], [3, 5], [4, 5]],
            [[2, 5], [3, 5], [4, 5], [5, 5]],

            [[0, 6], [1, 6], [2, 6], [3, 6]],
            [[1, 6], [2, 6], [3, 6], [4, 6]],
            [[2, 6], [3, 6], [4, 6], [5, 6]],

            //diagonal wins (\), grouped by row
            [[0, 0], [1, 1], [2, 2], [3, 3]],
            [[0, 1], [1, 2], [2, 3], [3, 4]],
            [[0, 2], [1, 3], [2, 4], [3, 5]],
            [[0, 3], [1, 4], [2, 5], [3, 6]],

            [[1, 0], [2, 1], [3, 2], [4, 3]],
            [[1, 1], [2, 2], [3, 3], [4, 4]],
            [[1, 2], [2, 3], [3, 4], [4, 5]],
            [[1, 3], [2, 4], [3, 5], [4, 6]],

            [[2, 0], [3, 1], [4, 2], [5, 3]],
            [[2, 1], [3, 2], [4, 3], [5, 4]],
            [[2, 2], [3, 3], [4, 4], [5, 5]],
            [[2, 3], [3, 4], [4, 5], [5, 6]],

            //diagonal wins (/), grouped by row
            [[0, 6], [1, 5], [2, 4], [3, 3]],
            [[0, 5], [1, 4], [2, 3], [3, 2]],
            [[0, 4], [1, 3], [2, 2], [3, 1]],
            [[0, 3], [1, 2], [2, 1], [3, 0]],

            [[1, 6], [2, 5], [3, 4], [4, 3]],
            [[1, 5], [2, 4], [3, 3], [4, 2]],
            [[1, 4], [2, 3], [3, 2], [4, 1]],
            [[1, 3], [2, 2], [3, 1], [4, 0]],

            [[2, 6], [3, 5], [4, 4], [5, 3]],
            [[2, 5], [3, 4], [4, 3], [5, 2]],
            [[2, 4], [3, 3], [4, 2], [5, 1]],
            [[2, 3], [3, 2], [4, 1], [5, 0]],
        ]
        return wins.some(win => win.every(requiredSquare =>
            pieces[requiredSquare[0]][requiredSquare[1]] === 1))
    }

    static isOver(state) {
        return this.checkWin(state.map(rowData => rowData.map(squareData => squareData[0]))) ||
            this.checkWin(state.map(rowData => rowData.map(squareData => squareData[1]))) ||
            state.every(rowData => rowData.every(squareData => (squareData[0] === 1 || squareData[1] === 1)))
    }

    static getWinner(state) {
        if (this.checkWin(state.map(rowData => rowData.map(squareData => squareData[0])))) {
            return 1
        }
        if (this.checkWin(state.map(rowData => rowData.map(squareData => squareData[1])))) {
            return -1
        }
        if (state.every(rowData => rowData.every(squareData => (squareData[0] === 1 || squareData[1] === 1)))) {
            return 0
        }
    }

    static stateEquals(firstState, secondState) {
        for (let i = 0; i < this.ROWS; i++) {
            for (let j = 0; j < this.COLUMNS; j++) {
                for (let k = 0; k < 3; k++) {
                    if (firstState[i][j][k] !== secondState[i][j][k]) {
                        return false
                    }
                }
            }
        }
        return true
    }

    static copy(state) {
        const newState = this.getStartingState()
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
        const newState = this.getStartingState()
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