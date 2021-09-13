from flask import Flask, request, jsonify
import os
import json

from ..constants import MODELS_DIR_PATH, HYPERPARAMS_DIR_PATH
from ..utils.file import get_random_file_name


app = Flask(__name__)

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = "my-secret-key"
app.config["SECRET_TYPE"] = "filesystem"

@app.route("/", methods=["GET"])
def index():

    if request.method == "GET":
        return jsonify({"status": "ok"})

@app.route("/build-model", methods=["POST"])
def build_model():

    if request.method == "POST":
        model_data = request.get_json()
        
        file_name = get_random_file_name()
        file_path = os.path.join(MODELS_DIR_PATH, file_name)

        with open(file_path, "w") as f:
            json.dump(model_data, f, indent=4)

        return jsonify({"status": "success", "file_path": file_path})

@app.route("/build-hyperparams", methods=["POST"])
def build_hyperparams():

    if request.method == "POST":
        hyperparams_data = request.get_json()

        file_name = get_random_file_name()
        file_path = os.path.join(HYPERPARAMS_DIR_PATH, file_name)

        with open(file_path, "w") as f:
            json.dump(hyperparams_data, f, indent=4)

        return jsonify({"status": "success", "file_path": file_path})