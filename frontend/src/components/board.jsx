import React, {Component} from "react";
import Square from "./square";
import GameClass from "./Connect4"
import MCTS from "./MCTS"
import {Container, Row, Col} from 'react-bootstrap'
import * as tfjs from '@tensorflow/tfjs'


export default class Board extends Component {
    state = {
        data: GameClass.toReactState(GameClass.getStartingState()),
        message: 'Yellow\'s Turn'
    }

    constructor(props) {
        super(props);
        this.handleClick = this.handleClick.bind(this)
        tfjs.loadLayersModel('http://localhost:8000/connect4/static/model.json')
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
                    data: GameClass.toReactState(userMove),
                    message: 'Game Over: ' + this.getWinnerMessage(userMove)
                })
            } else {
                const aiMove = MCTS.chooseMove(GameClass, userMove, this.predict)
                if (GameClass.isOver(aiMove)) {
                    this.setState({
                        data: GameClass.toReactState(aiMove),
                        message: 'Game Over: ' + this.getWinnerMessage(aiMove)
                    })
                }
                else {
                    this.setState({
                        data: GameClass.toReactState(aiMove),
                        message: GameClass.isPlayer1Turn(aiMove) ? 'Your Turn' : 'Ai\'s Turn'
                    })
                }
            }
        } else {
            this.setState({
                message: this.state.data[0][0].p1Turn ? 'Invalid Move! Try Again!' :
                    'Invalid Move! Red\'s Turn'
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

    render() {
        return (
            <React.Fragment>
                <Container>
                    {this.state.data.map(rowData => (
                        <Row className='show-grid' key={rowData[0].row}>
                            {rowData.map(squareData => (
                                <Col key={squareData.column}>
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