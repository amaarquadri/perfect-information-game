import React, {Component} from "react";
import Square from "./square";
import Connect4 from "./Connect4"
import {Container, Row, Col} from 'react-bootstrap'


export default class Board extends Component {
    state = {
        data: Connect4.toReactState(Connect4.getStartingState())
    }

    constructor(props) {
        super(props);
        this.handleClick = this.handleClick.bind(this)
    }

    handleClick(row, column) {
        let move = Connect4.performUserMove(Connect4.toTensorFlowState(this.state.data), row, column)
        if (move !== null) {
            this.setState({data: Connect4.toReactState(move)})
        }
    }

    render() {
        return (
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
        )
    }
}