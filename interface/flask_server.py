from flask import Flask, request
from predict import predict, InputClass

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    input = InputClass(data.input)
    return predict(input)


