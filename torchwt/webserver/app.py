from flask import Flask, request, jsonify


app = Flask(__name__)

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = "my-secret-key"
app.config["SECRET_TYPE"] = "filesystem"


@app.route("/build-model", methods=["POST"])
def build_model():

    if request.method == "POST":
        data = request.get_json()
        