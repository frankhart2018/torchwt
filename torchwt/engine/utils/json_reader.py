import json
import os

from .errors import *


class JSONReader:
    @staticmethod
    def read_json(json_file_path, error_class, error_message):
        if not os.path.exists(json_file_path):
            raise error_class(message=error_message)

        json_data = ""

        with open(json_file_path, "r") as file:
            json_data = json.load(file)

        return json_data