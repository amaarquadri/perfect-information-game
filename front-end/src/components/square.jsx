import React, {Component} from "react";
import redCircle from './resources/red_circle_dark_square.png'
import yellowCircle from './resources/yellow_circle_dark_square.png'
import darkSquare from './resources/dark_square.png'

export default class Square extends Component {
    render() {
        return (
            <input onClick={this.props.onClick} type='image' src={this.getImgSrc()} alt=''/>
        )
    }

    getImgSrc() {
        const data = this.props.squareData
        if (data.p1Piece) {
            return yellowCircle
        }
        else if (data.p2Piece) {
            return redCircle
        }
        else {
            return darkSquare
        }
    }
}