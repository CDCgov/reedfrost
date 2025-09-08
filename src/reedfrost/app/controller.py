from reedfrost.app.model import get_results


class Controller:
    def __init__(self, ui, inputs):
        self.state = {}
        self.ui = ui
        self.inputs = inputs

        # validate inputs
        for x in inputs:
            assert "key" in x and "setter" in x
            assert callable(x["setter"])

        # all keys should be unique
        keys = set(x["key"] for x in inputs)
        assert len(keys) == len(
            inputs
        ), f"There are {len(keys)} keys for {len(inputs)} inputs"

    def set(self, key: str, value) -> None:
        self.state[key] = value

    def get(self, key: str):
        # if "results" are requested, run the model on the available state
        if key == "results" and "results" not in self.state:
            self.state["results"] = get_results(self.state)

        return self.state.get(key, None)

    def place_input(self, key: str):
        # get the input with the given key
        item = next(x for x in self.inputs if x["key"] == key)

        # if the value is a callable, call it with self as argument
        if callable(item["setter"]):
            value = item["setter"](self)
        else:
            value = item["setter"]

        self.set(key, value)

    def run(self):
        self.ui(self)
