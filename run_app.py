import os

from torchwt.webserver.app import app
from torchwt.constants import MODELS_DIR_PATH, HYPERPARAMS_DIR_PATH


if __name__ == '__main__':
    os.makedirs(MODELS_DIR_PATH, exist_ok=True)
    os.makedirs(HYPERPARAMS_DIR_PATH, exist_ok=True)
    app.run(debug=True)