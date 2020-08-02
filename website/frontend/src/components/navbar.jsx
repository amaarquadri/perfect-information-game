import React, {Component} from "react";
import "bootstrap/dist/css/bootstrap.min.css"  // Gives access to all the classNames

export default class Navbar extends Component {
    scrollToServices() {

    }

    render() {
        return (
            <nav className="navbar navbar-expand-lg navbar-light fixed-top py-3" id="mainNav">
                <div className="container">
                    <a className="navbar-brand js-scroll-trigger" href="#page-top">Start Bootstrap</a>
                    <button className="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse"
                            data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false"
                            aria-label="Toggle navigation"><span className="navbar-toggler-icon"/></button>
                    <div className="collapse navbar-collapse" id="navbarResponsive">
                        <ul className="navbar-nav ml-auto my-2 my-lg-0">
                            <li className="nav-item">
                                <a className="nav-link js-scroll-trigger" href="/connect4">About</a>
                            </li>
                            <li className="nav-item">
                                <a className="nav-link js-scroll-trigger" href="#services">Mechanical Design</a>
                            </li>
                            <li className="nav-item">
                                <a className="nav-link js-scroll-trigger" href="#portfolio">Software Engineering</a>
                            </li>
                            <li className="nav-item">
                                <a className="nav-link js-scroll-trigger" href="#contact">Contact</a>
                            </li>
                        </ul>
                    </div>
                </div>
            </nav>
        )
    }
}