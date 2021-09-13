import random
import hashlib
import time


def get_random_file_name():
    file_name = str(round(time.time()) + random.randint(90, 100))
    file_name = hashlib.sha512(file_name.encode()).hexdigest() + ".json"

    return file_name