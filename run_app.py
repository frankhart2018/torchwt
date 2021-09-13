import os

from torchwt.webserver.app import app
from torchwt.constants import TORCHWT_DIR_PATH, MODELS_DIR_PATH


if __name__ == '__main__':
    os.makedirs(TORCHWT_DIR_PATH, exist_ok=True)
    os.makedirs(MODELS_DIR_PATH, exist_ok=True)
    app.run(debug=True)