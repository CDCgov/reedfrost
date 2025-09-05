import streamlit as st
import yaml

from .texthelper import TextHelper


class Data:
    def __init__(self, controller, data_yaml, text_yaml):
        self.controller = controller
        with open(data_yaml, "r") as file:
            self.defaults = yaml.safe_load(file)
        self.initialize_data(self.defaults)
        self.text_helper = TextHelper(text_yaml)

    def initialize_data(self, data):
        self.data = data

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value

    def get_default(self, key):
        return self.defaults.get(key)

    def format_string(self, str):
        return self.text_helper.format(str, self.data)


class StreamlitData(Data):
    def initialize_data(self, data):
        st.session_state.app_data = data

    def get(self, key):
        try:
            return st.session_state[key]
        except KeyError:
            return st.session_state.app_data[key]

    def set(self, key, value):
        st.session_state.app_data[key] = value

    def format_string(self, str):
        return self.text_helper.format(str, st.session_state.app_data)
