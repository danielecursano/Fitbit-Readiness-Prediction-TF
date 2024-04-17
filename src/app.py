from flask import Flask, render_template, request
from model import Model
import numpy as np

model = Model("/Users/daniele/Desktop/fitbit/data/fitbit-model.h5", (1, 11))
app = Flask(__name__)

@app.route("/")
def index(output=None, evaluate_flag=0, evaluate_output=None, alert=None):
    summary = []
    model.model.summary(print_fn=lambda x: summary.append(x))
    summary = "\n".join(summary)
    return render_template("home.html", summary=summary, output=output, evaluate_output=evaluate_output, alert=alert, evaluate_flag=evaluate_flag)

@app.route("/predict", methods=["POST"])
def predict():
    arr = np.array([float(request.form[f'float{i}']) for i in range(1, 12)]).reshape(model.input_shape)
    return index(output=str(model.predict(arr)).replace("[", "").replace("]", ""))

if __name__ == "__main__":
    app.run(debug=1)