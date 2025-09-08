from reedfrost.app.model import get_results


class Controller:
    def __init__(self, app, components):
        self.state = {}
        self.app = app
        self.components = components

        # validate components
        for x in components:
            # all components have a key
            assert "key" in x
            # input components have a setter function
            if "type" in x and x["type"] == "input":
                assert "setter" in x
                assert callable(x["setter"])

        # all keys should be unique
        keys = set(x["key"] for x in components)
        assert len(keys) == len(
            components
        ), f"There are {len(keys)} keys for {len(components)} components"

    def set(self, key: str, value) -> None:
        self.state[key] = value

    def get(self, key: str):
        # if "results" are requested, run the model on the available state
        if key == "results" and "results" not in self.state:
            self.state["results"] = get_results(self.state)

        return self.state.get(key, None)

    def place(self, key: str):
        # get the component with the given key
        item = next(x for x in self.components if x["key"] == key)

        if "type" in item and item["type"] == "input":
            value = item["setter"](self)
            self.set(key, value)

    def run(self):
        self.app(self)
