import React, {Component} from "react"
import "bootstrap/dist/css/bootstrap.min.css"  // Gives access to all the classNames
import {Container, Row, Col} from 'react-bootstrap'
import * as tfjs from '@tensorflow/tfjs'
import Square from "./square.jsx"
import GameClass from "./Connect4.js"
import MCTS from "./MCTS.js"


export default class Board extends Component {
    state = {
        data: GameClass.toReactState(GameClass.getStartingState()),
        message: 'Yellow\'s Turn'
    }

    constructor(props) {
        super(props);
        this.handleClick = this.handleClick.bind(this)
        tfjs.loadLayersModel('/static/model.json')
            .then(model => {
                console.log('Model Loaded!')
                this.predict = (states) => {
                    const result = model.predict(tfjs.tensor(states))
                    const policies = GameClass.separateFlattenedPolicies(Array.from(result[0].dataSync()))
                    const values = Array.from(result[1].dataSync())

                    return policies
                        .map((policy, stateIndex) => {
                            const legalMoves = GameClass.getLegalMoves(states[stateIndex])
                            return policy.filter((move, moveIndex) => legalMoves[moveIndex])
                        })
                        .map((policy, stateIndex) => [policy, values[stateIndex]])
                }
            })
            .catch(error => {
                console.log('Error Loading Model! Error message: ')
                console.log(error)
            })
    }

    handleClick(row, column) {
        if (this.state.message.substring(0, 11) === 'Game Over: ') {
            return
        }

        const currentState = GameClass.toTensorFlowState(this.state.data);
        const userMove = GameClass.performUserMove(currentState, row, column)
        if (userMove !== null) {
            if (GameClass.isOver(userMove)) {
                this.setState({
                    data: this.getHighlight(this.state.data, GameClass.toReactState(userMove)),
                    message: 'Game Over: ' + this.getWinnerMessage(userMove)
                })
            } else {
                this.setState({
                    data: this.getHighlight(this.state.data, GameClass.toReactState(userMove)),
                    message: 'Ai\'s Turn'
                })
            }
        } else {
            this.setState({
                message: 'Invalid Move! Try Again!'
            })
        }
    }

    getWinnerMessage(state) {
        let winner
        switch (GameClass.getWinner(state)) {
            case 0:
                winner = 'Draw'
                break
            case 1:
                winner = 'You Win!'
                break
            case -1:
                winner = 'You Lose!'
                break
            default:
                winner = ''
        }
        return winner
    }

    getHighlight(oldReactState, newReactState) {
        return newReactState.map(rowData => rowData.map(squareData => {
            const oldSquareData = oldReactState[squareData.row][squareData.column]
            squareData.highlight = oldSquareData.p1Piece !== squareData.p1Piece ||
                oldSquareData.p2Piece !== squareData.p2Piece
            return squareData
        }))
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        const position = GameClass.toTensorFlowState(this.state.data);
        if (!GameClass.isPlayer1Turn(position)) {
            setTimeout(() => {
                MCTS.chooseMove(GameClass, position, this.predict).then(aiMove => {
                    if (GameClass.isOver(aiMove)) {
                        this.setState({
                            data: this.getHighlight(this.state.data, GameClass.toReactState(aiMove)),
                            message: 'Game Over: ' + this.getWinnerMessage(aiMove)
                        })
                    }
                    else {
                        this.setState({
                            data: this.getHighlight(this.state.data, GameClass.toReactState(aiMove)),
                            message: GameClass.isPlayer1Turn(aiMove) ? 'Your Turn' : 'Ai\'s Turn'
                        })
                    }
                }).catch(error => console.log(error))
            }, 100)
        }
    }

    render() {
        const noPad = {
            paddingRight: 0,
            paddingLeft: 0,
            marginRight: 0,
            marginLeft: 0
        }
        return (
            <React.Fragment>
                <Container style={noPad}>
                    {this.state.data.map(rowData => (
                        <Row className='show-grid' key={rowData[0].row} style={noPad}>
                            {rowData.map(squareData => (
                                <Col key={squareData.column} style={noPad}>
                                    <Square key={squareData.column} squareData={squareData}
                                            onClick={() => this.handleClick(squareData.row, squareData.column)}/>
                                </Col>
                            ))}
                        </Row>
                    ))}
                </Container>
                <p>{this.state.message}</p>
            </React.Fragment>
        )
    }
}
