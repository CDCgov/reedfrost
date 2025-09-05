from mvc import Controller, StreamlitData, StreamlitView
from reedfrost_streamlit_app import model


class ReedFrostView(StreamlitView):
    def handle_trajectories_chart(self):
        self.run_code("trajectories_chart")

    def handle_theoretical_chart(self):
        self.run_code("theoretical_chart")


root = "src/reedfrost/bbb_app/"


def main(
    data_yaml=f"{root}reedfrost_streamlit_app/data.yaml",
    ui_yaml=f"{root}reedfrost_streamlit_app/ui.yaml",
    text_yaml=f"{root}reedfrost_streamlit_app/text.yaml",
    code_file=f"{root}reedfrost_streamlit_app/code.py",
):
    Controller(
        data_yaml,
        ui_yaml,
        text_yaml,
        code_file,
        model,
        data=StreamlitData,
        view=ReedFrostView,
    )


if __name__ == "__main__":
    main()
