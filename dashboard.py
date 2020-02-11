"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import logging
import threading
from subprocess import call
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

APP = Flask(__name__)
CORS = CORS(APP)
DATA = None


@APP.route('/graph', methods=['POST'])
def update_graph():
    global DATA
    DATA = request.json
    return jsonify({"result": "OK"})


@APP.route('/graph', methods=['GET'])
@cross_origin()
def get_graph():
    return jsonify(DATA)


@APP.route('/', methods=['GET'])
@cross_origin()
def index():
    return open("web/index.html", "r").read()


class GraphWebServer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        logging.basicConfig(filename="output/app.log", level=logging.DEBUG)
        APP.run(debug=False, host='0.0.0.0')


class FrontendWebServer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        call(["/bin/bash", "start_web_app.sh"])


def start_dashboard_server():
    flask_thread = GraphWebServer()
    flask_thread.start()
    front_end_thread = FrontendWebServer()
    front_end_thread.start()
