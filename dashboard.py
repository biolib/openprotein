# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import threading

app = Flask(__name__)
cors = CORS(app)
data = None

@app.route('/graph',methods=['POST'])
def update_graph():
    global data
    data = request.json
    return jsonify({"result":"OK"})

@app.route('/graph',methods=['GET'])
@cross_origin()
def get_graph():
    return jsonify(data)

@app.route('/',methods=['GET'])
@cross_origin()
def index():
    return open("web/index.html","r").read()

class graphWebServer (threading.Thread):
   def __init__(self):
      threading.Thread.__init__(self)
   def run(self):
       app.run(debug=False, host='0.0.0.0')


class frontendWebServer (threading.Thread):
   def __init__(self):
      threading.Thread.__init__(self)
   def run(self):
       from subprocess import call
       call(["/bin/bash", "start_web_app.sh"])

def start_dashboard_server():
    flask_thread = graphWebServer()
    flask_thread.start()
    front_end_thread = frontendWebServer()
    front_end_thread.start()
