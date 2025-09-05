import os
from collections import UserDict

import yaml


class TextHelper(UserDict):
    def __init__(self, initialdata=None, yaml_file=None):
        # Default yaml path
        default_yaml = os.path.join(
            os.path.dirname(__file__), "app_assets", "app_text.yaml"
        )

        # Allow calling TextHelper(yaml_file_path) as before
        if isinstance(initialdata, str) and yaml_file is None:
            yaml_file = initialdata
            initialdata = None

        if yaml_file is None:
            yaml_file = default_yaml

        if initialdata is None:
            initialdata = {}

        super().__init__(initialdata)
        self.yaml_file = yaml_file

        with open(yaml_file) as f:
            app_text = yaml.safe_load(f) or {}
            for key, value in app_text.items():
                self.data[str(key).strip()] = str(value).strip()

    def __setitem__(self, key, value):
        if key in self.data:
            raise KeyError(f"Key '{key}' already exists.")
        super().__setitem__(key, value)

    def __getitem__(self, key):
        """
        Retrieve a string formatted with anything defined in the dictionary.
        If the key is missing, return an empty string to keep callers robust.
        """
        text = self.data[key]
        return text.format(**self.data)

    def format(self, s, d={}):
        d.update(self.data)
        return s.format(**d)
