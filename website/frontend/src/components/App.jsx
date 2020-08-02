import React, {Component} from 'react';
import Board from "./board.jsx";

export default class App extends Component {
    render() {
        return (
            <React.Fragment>
                <h1>Connect 4</h1>
                <Board />
            </React.Fragment>
        )
    }
}
