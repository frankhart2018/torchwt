import json

from .errors import *


class JSONReader:
    @staticmethod
    def read_json(json_file_path):
        json_data = ""

        with open(json_file_path, "r") as file:
            json_data = json.load(file)

        return json_data