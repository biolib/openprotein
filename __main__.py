import runpy
import webbrowser
from dashboard import start_dashboard_server

# start web server
start_dashboard_server()

# open the performance dashboard in a browser
webbrowser.open("http://localhost:5000")

# run the PyMOL interpreter
runpy.run_module('pymol', run_name="__main__")
