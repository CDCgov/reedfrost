from reedfrost.app.controller import Controller
from reedfrost.app.model import GETTERS, INITIAL_DATA
from reedfrost.app.ui import COMPONENTS, app


def main(initial_data=INITIAL_DATA):
    Controller(
        app=app, components=COMPONENTS, getters=GETTERS, initial_data=initial_data
    ).run()


if __name__ == "__main__":
    main()
