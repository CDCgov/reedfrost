from typing import Callable

import streamlit as st


class Controller:
    def __init__(
        self,
        app: Callable,
        components: list[dict],
        getters: list[dict],
        initial_data: dict | None = None,
    ):
        self.app = app
        self.components = components
        self.getters = getters

        if initial_data is not None:
            for key, value in initial_data.items():
                self.set(key, value, overwrite=False)

        # validate components
        for x in components:
            assert isinstance(x, dict)
            # all components have a key
            assert {"key", "type", "func"}.issubset(x.keys())
            assert x["type"] in {"input", "special_input", "output"}
            assert callable(x["func"])

        # all keys should be unique
        keys = set(x["key"] for x in components)
        assert len(keys) == len(
            components
        ), f"There are {len(keys)} keys for {len(components)} components"

        # validate getters
        for x in getters:
            assert isinstance(x, dict)
            assert {"key", "getter"}.issubset(x.keys())
            assert callable(x["getter"])

    def set(self, key: str, value, overwrite: bool = True) -> None:
        if overwrite or key not in st.session_state:
            st.session_state[key] = value

    def get(self, key: str):
        # if there is a getter for this key, use it
        if getter := self._get_by_key(self.getters, key):
            return getter["getter"](self)
        else:
            return st.session_state[key]

    def ensure(self, key: str) -> None:
        self.set(key, self.get(key))

    def place(self, key: str) -> None:
        # get the component with the given key
        item = self._get_by_key(self.components, key)
        assert item is not None

        kwargs = {k: v for k, v in item.items() if k not in {"type", "func"}}

        if item["type"] in {"input", "special_input"}:
            args = [self] if item["type"] == "special_input" else []
            item["func"](*args, **kwargs)
        elif item["type"] == "output":
            item["func"](self)
        else:
            raise RuntimeError(f"Unknown component type: {item['type']}")

    def run(self):
        self.app(self)

    @staticmethod
    def _get_by_key(lst: list[dict], key: str, default=None) -> dict | None:
        elts = [x for x in lst if x["key"] == key]

        match len(elts):
            case 0:
                return default
            case 1:
                return elts[0]
            case _:
                raise RuntimeError(f"Multiple elements with key {key}: {elts}")
