from reedfrost.app.controller import Controller
from reedfrost.app.ui import app, components

if __name__ == "__main__":
    Controller(app=app, components=components).run()
