import runpy
from flask import Flask, request, jsonify
import threading
import matplotlib.pyplot as plt
plt.ion()

from util import draw_plot
from drawnow import drawnow

app = Flask(__name__)

@app.route('/graph',methods=['POST'])
def update_graph():
    content = request.json
    validation_dataset_size = content['validation_dataset_size']
    sample_num = content['sample_num']
    train_loss_values = content['train_loss_values']
    validation_loss_values = content['validation_loss_values']
    fig = plt.figure()
    drawnow(draw_plot(fig, plt, validation_dataset_size, sample_num, train_loss_values, validation_loss_values))
    return jsonify({"result":"OK"})

class graphWebServer (threading.Thread):
   def __init__(self):
      threading.Thread.__init__(self)
   def run(self):
       app.run(debug=False, host='0.0.0.0')

flask_thread = graphWebServer()
flask_thread.start()

runpy.run_module('pymol', run_name="__main__")