from reedfrost.app.model import get_results


class Controller:
    def __init__(self, ui):
        self.state = {}
        self.ui = ui

    def set(self, key: str, value) -> None:
        self.state[key] = value

    def get(self, key: str):
        # if "results" are requested, run the model on the available state
        if key == "results" and "results" not in self.state:
            self.state["results"] = get_results(self.state)

        return self.state.get(key, None)

    def run(self):
        self.ui(self)
