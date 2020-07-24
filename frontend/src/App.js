import React, {Component} from 'react';
import Board from "./components/board";

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
