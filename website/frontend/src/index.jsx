import React from "react";
import {render} from "react-dom";
import Navbar from "./components/navbar.jsx";
import "bootstrap/dist/css/bootstrap.min.css"

render(
    <React.StrictMode>
        <Navbar/>
    </React.StrictMode>,
    document.getElementById("app"));