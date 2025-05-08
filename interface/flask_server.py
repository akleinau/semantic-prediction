from flask import Flask, request
from flask_cors import CORS

from predict import predict, InputClass

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=['POST'])
def predict_request():
    data = request.get_json()
    input = InputClass(data["input"])
    prediction = predict(input)
    return prediction

@app.route("/get_input_params")
def get_input_params():
    return {"params": [
        {"id": "meanage", "label": "Mean Age", "type": "slider", "min": 15, "max": 80, "value": 50},
        {"id": "proportionfemale", "label": "Proportion female", "type": "slider", "min": 0, "max": 100, "value": 50},
        {"id": "meantobacco", "label": "Mean number of times tobacco used", "type": "slider", "min": 1, "max": 30, "value": 10},
        {"id":"followup", "label": "Follow up (weeks)", "type": "slider", "min": 4, "max": 60, "value": 26},
        {"id": "patientrole", "label": "Patient role?", "type": "checkbox", "value": 0},
        {"id": "verification", "label": "Biochemical verification", "type": "checkbox", "value": 0},
        {"id": "outcome", "label": "Outcome", "type": "select", "choices":
            ["Abstinence: Continuous ", "Abstinence: Point Prevalence "], "value": "Abstinence: Continuous "},
        ],
        "interventions": [
        {"id": "intervention", "label": "Intervention", "type": "select", "choices": [], "value": []},
        {"id": "delivery", "label": "Delivery", "type": "select", "choices": [], "value": []},
        {"id": "source", "label": "Source", "type": "select", "choices": [], "value": []}
    ]}

@app.route("/hello_world")
def hello_world():
    return {"text":"Hello, World!"}


