import React from "react"
import {render} from "react-dom"
import App from "./components/App.jsx"
import "bootstrap/dist/css/bootstrap.min.css"

render(
    <React.StrictMode>
        <App/>
    </React.StrictMode>,
    document.getElementById("app"))