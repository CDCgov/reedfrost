from reedfrost.app.controller import Controller
from reedfrost.app.model import GETTERS, INITIAL_STATE
from reedfrost.app.ui import COMPONENTS, app


def main():
    Controller(
        app=app, components=COMPONENTS, getters=GETTERS, initial_state=INITIAL_STATE
    ).run()


if __name__ == "__main__":
    main()
