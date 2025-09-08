from reedfrost.app.controller import Controller
from reedfrost.app.model import GETTERS
from reedfrost.app.ui import COMPONENTS, app


def main():
    Controller(app=app, components=COMPONENTS, getters=GETTERS).run()


if __name__ == "__main__":
    main()
