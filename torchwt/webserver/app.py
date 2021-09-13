from flask import Flask, request, jsonify
import time
import os
import random
import hashlib
import json

from ..constants import MODELS_DIR_PATH


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
        data = request.get_json()
        
        file_name = str(round(time.time()) + random.randint(90, 100))
        file_name = hashlib.sha512(file_name.encode()).hexdigest() + ".json"
    
        file_path = os.path.join(MODELS_DIR_PATH, file_name)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

        return jsonify({"status": "success", "file_path": file_path})