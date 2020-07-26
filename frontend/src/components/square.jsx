import React, {Component} from "react";

export default class Square extends Component {
    static PREFIX = 'http://localhost:8000/connect4/static/'

    render() {
        return (
            <input onClick={this.props.onClick} type='image' src={this.getImgSrc()} alt=''/>
        )
    }

    getImgSrc() {
        const data = this.props.squareData
        if (data.p1Piece) {
            return Square.PREFIX +
                (data.highlight? 'yellow_circle_dark_square_highlighted.png' : 'yellow_circle_dark_square.png')
        }
        else if (data.p2Piece) {
            return Square.PREFIX +
                (data.highlight? 'red_circle_dark_square_highlighted.png' : 'red_circle_dark_square.png')
        }
        else {
            return Square.PREFIX + 'dark_square.png'
        }
    }
}