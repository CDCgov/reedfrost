from reedfrost.app.controller import Controller
from reedfrost.app.model import GETTERS, INITIAL_DATA
from reedfrost.app.ui import COMPONENTS, app


def main():
    Controller(
        app=app, components=COMPONENTS, getters=GETTERS, initial_data=INITIAL_DATA
    ).run()


if __name__ == "__main__":
    main()
