from typing import Callable


class Controller:
    def __init__(self, app: Callable, components: list[dict], getters: list[dict]):
        self.app = app
        self.components = components
        self.getters = getters
        self.state = {}

        # validate components
        for x in components:
            assert isinstance(x, dict)
            # all components have a key
            assert {"key", "type"}.issubset(x.keys())

            # input components have a setter function
            match x["type"]:
                case "input":
                    assert "setter" in x
                    assert callable(x["setter"])
                case "output":
                    assert "func" in x
                    assert callable(x["func"])
                case _:
                    raise RuntimeError(f"Unknown component type: {x['type']}")

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

    def set(self, key: str, value) -> None:
        self.state[key] = value

    def get(self, key: str):
        # if there is a getter for this key, use it
        if getter := self._get_by_key(self.getters, key):
            return getter["getter"](self)
        else:
            assert key in self.state or key == "results", f"Unknown key: {key}"
            return self.state[key]

    def place(self, key: str):
        # get the component with the given key
        item = self._get_by_key(self.components, key)

        match item:
            case {"type": "input", "setter": setter}:
                value = setter(self)
                self.set(key, value)
            case {"type": "output", "func": func}:
                func(self)
            case _:
                raise RuntimeError(f"Unknown component: {item}")

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
