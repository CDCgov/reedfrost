from . import model
from .data import Data
from .view import View


class Controller:
    def __init__(self, data_yaml, ui_yaml, text_yaml):
        self.data = Data(self, data_yaml, text_yaml)
        self.view = View(self, ui_yaml)
        self.view.page_config()
        self.run_model()
        self.view.display()

    def get_data(self, key):
        return self.data.get(key)

    def set_data(self, key, value):
        self.data.set(key, value)

    def run_model(self):
        self.set_data("results", False)
        model.run_model(self)
        self.set_data("results", True)

    def format_string(self, str):
        return self.data.format_string(str)

    def get_default(self, key):
        try:
            return self.data.get_default(key)
        except KeyError:
            return self.data.get_default(f"code_{key}")

    def eval_code(self, code):
        return eval(code, {}, {"app": self})
