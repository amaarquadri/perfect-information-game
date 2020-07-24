import React, {Component} from "react";
import Square from "./square";
import Connect4 from "./Connect4"
import {Container, Row, Col} from 'react-bootstrap'
import * as tfjs from '@tensorflow/tfjs'


export default class Board extends Component {
    state = {
        data: Connect4.toReactState(Connect4.getStartingState()),
        message: 'Yellow\'s Turn'
    }

    constructor(props) {
        super(props);
        this.handleClick = this.handleClick.bind(this)
        const model = tfjs.loadLayersModel('http://localhost:8000/connect4/static/model.json')
        this.predict = (state) => model.predict(tfjs.tensor3d(state))
    }

    handleClick(row, column) {
        if (this.state.message.substring(0, 11) === 'Game Over: ') {
            return
        }

        let move = Connect4.performUserMove(Connect4.toTensorFlowState(this.state.data), row, column)
        // console.log(this.predict(move))
        if (move !== null) {
            if (Connect4.isOver(move)) {
                let winner
                switch (Connect4.getWinner(move)) {
                    case 0:
                        winner = 'Draw'
                        break
                    case 1:
                        winner = 'Yellow Wins'
                        break
                    case -1:
                        winner = 'Red Wins'
                        break
                    default:
                        winner = ''
                }
                this.setState({data: Connect4.toReactState(move), message: 'Game Over: ' + winner})
            } else {
                this.setState({
                    data: Connect4.toReactState(move),
                    message: Connect4.isPlayer1Turn(move) ? 'Yellow\'s Turn' : 'Red\'s Turn'
                })
            }
        } else {
            this.setState({message: this.state.data[0][0].p1Turn ? 'Invalid Move! Yellow\'s Turn' :
                    'Invalid Move! Red\'s Turn'})
        }
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