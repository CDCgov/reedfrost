from pathlib import Path

import streamlit.web.cli

if __name__ == "__main__":
    streamlit.web.cli.main_run([str(Path(__file__).parent / "app.py")])
