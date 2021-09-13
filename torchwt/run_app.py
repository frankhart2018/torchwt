import os
import argparse

from torchwt.webserver.app import app
from torchwt.constants import MODELS_DIR_PATH, HYPERPARAMS_DIR_PATH


def run_app():
    parser = argparse.ArgumentParser(description="Run the torchwt server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run the server in debug mode")

    args = parser.parse_args()

    os.makedirs(MODELS_DIR_PATH, exist_ok=True)
    os.makedirs(HYPERPARAMS_DIR_PATH, exist_ok=True)
    app.run(port=args.port, debug=args.debug)