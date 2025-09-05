import polars as pl
from mvc import Controller, View
from reedfrost_streamlit_app import model

root = "src/reedfrost/bbb_app/"


class TextView(View):
    def handle_set_page_config(self, _, __):
        pass

    def display(self):
        super().display()
        var = input("Change variable (Q to quit): ")
        if var.upper() == "Q":
            return
        val = input("New value? ")
        try:
            val = int(val)
        except ValueError:
            pass
        self.controller.set_data(var, val)
        self.controller.run_model()
        self.display()

    def handle_title(self, _, v):
        print("#############################################")
        print("#", self.controller.format_string(v).upper())
        print("#############################################")

    def handle_header(self, _, v):
        print(self.controller.format_string(v).upper())

    def handle_divider(self):
        print("---------------------------------------------")

    def handle_sidebar(self, _, v):
        print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
        self.process_component(v)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    def handle_page_link(self, _, v):
        label = self.controller.format_string(v["label"])
        page = self.controller.format_string(v["page"])
        print(f"-- {label}: <{page}>")

    def handle_select_slider(self, k, v):
        self.handle_slider(k, v)

    def handle_slider(self, k, v):
        print(f"<-- {v['key']} = {v['value']}")

    def handle_segmented_control(self, k, v):
        print(
            f"<-- {v['key']} = {self.controller.format_string(v['value'])} : {v['options']}"
        )

    def handle_expander(self, k, v):
        children = v.pop("children", [])
        self.handle_header(k, v["label"])
        self.process_component(children)

    def handle_empty(self, k, v):
        self.process_component(v)

    def handle_number_input(self, k, v):
        self.handle_slider(k, v)

    def handle_columns(self, k, v):
        children = v.pop("children", [])
        self.process_component(children)

    def handle_column(self, k, v):
        self.process_component(v)

    def handle_metric(self, k, v):
        print(
            f"--> {self.controller.format_string(v['label'])} = {self.run_code(v['value'][6:])}"
        )

    def handle_trajectories_chart(self):
        print(
            f"--> {self.controller.get_data("peak_traj").group_by("peak_y").agg(pl.col("iter").count().alias("count")).sort("peak_y", descending=True)}"
        )

    def handle_theoretical_chart(self):
        print("--> not implemented")


def main(
    data_yaml=f"{root}reedfrost_streamlit_app/data.yaml",
    ui_yaml=f"{root}reedfrost_streamlit_app/ui.yaml",
    text_yaml=f"{root}reedfrost_streamlit_app/text.yaml",
    code_file=f"{root}reedfrost_streamlit_app/code.py",
):
    Controller(data_yaml, ui_yaml, text_yaml, code_file, model, view=TextView)


if __name__ == "__main__":
    main()
