import React, {Component} from "react";

export default class Square extends Component {
    render() {
        return (
            <input onClick={this.props.onClick} type='image' src={this.getImgSrc()} alt=''/>
        )
    }

    getImgSrc() {
        const data = this.props.squareData
        if (data.p1Piece) {
            return 'http://localhost:8000/connect4/static/yellow_circle_dark_square.png'
        }
        else if (data.p2Piece) {
            return 'http://localhost:8000/connect4/static/red_circle_dark_square.png'
        }
        else {
            return 'http://localhost:8000/connect4/static/dark_square.png'
        }
    }
}