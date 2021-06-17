class CustomException(Exception):
    def __init__(self, message):
        self.__message = message

        super().__init__(self.__message)

    @property
    def message(self):
        return self.__message

class ModelFileNotFound(CustomException):
    def __init__(self, message):
        super().__init__(message)