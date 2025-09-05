import importlib.util
import sys

from .data import Data
from .view import View


class Controller:
    def __init__(
        self, data_yaml, ui_yaml, text_yaml, code_file, model, data=Data, view=View
    ):
        self.data = data(self, data_yaml, text_yaml)
        self.view = view(self, ui_yaml)
        self.load_code(code_file)
        self.view.initialize()
        self.model = model
        self.run_model()
        self.view.display()

    def get_data(self, key):
        return self.data.get(key)

    def set_data(self, key, value):
        self.data.set(key, value)

    def run_model(self):
        self.model.run_model(self)

    def format_string(self, str):
        return self.data.format_string(str)

    def get_default(self, key):
        return self.data.get_default(key)

    def run_code(self, name):
        return self.code[name](self)

    def load_code(self, code_file):
        spec = importlib.util.spec_from_file_location("code", code_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["code"] = module
        spec.loader.exec_module(module)
        self.code = module.code
