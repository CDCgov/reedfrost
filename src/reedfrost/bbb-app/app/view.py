import streamlit as st
import yaml

from . import charts


class View:
    def __init__(self, controller, ui_yaml):
        with open(ui_yaml) as f:
            self.ui = yaml.safe_load(f)
        self.controller = controller

    def page_config(self):
        if "set_page_config" in self.ui[0]:
            self.process_component(self.ui[0])

    def display(self):
        if "set_page_config" in self.ui[0]:
            self.process_component(self.ui[1:])
        else:
            self.process_component(self.ui)

    def process_component(self, comp):
        if isinstance(comp, list):
            for item in comp:
                self.process_component(item)
        elif isinstance(comp, dict) and len(comp) == 1:
            for k, v in comp.items():
                if "key" in v:
                    default_name = self.default_name.get(k, "value")
                    v[default_name] = self.controller.get_default(v["key"])
                if hasattr(self, f"handle_{k}"):
                    getattr(self, f"handle_{k}")(k, v)
                else:
                    self.make_st_call(k, v)
        elif isinstance(comp, str):
            if hasattr(self, f"handle_{comp}"):
                getattr(self, f"handle_{comp}")()
            else:
                self.make_st_call(comp, {})
        else:
            raise ValueError(self.yaml_format)

    def make_st_call(self, k, v):
        args = []
        kwargs = {}
        # single parameter
        if not isinstance(v, dict):
            if isinstance(v, str):
                v = self.controller.format_string(v)
            args.append(v)
        # multiple parameters with processing to handle inputs
        else:
            keys_to_remove = []
            entries_to_add = {}
            for kk in v.keys():
                if kk.startswith("code_"):
                    entries_to_add[kk[5:]] = self.handle_code(kk, v[kk])
                    keys_to_remove.append(kk)
                elif isinstance(v[kk], str):
                    v[kk] = self.controller.format_string(v[kk])
            for kk in keys_to_remove:
                v.pop(kk)
            v.update(entries_to_add)
            for kk in v.keys():
                kwargs[kk] = v[kk]

        try:
            return getattr(st, k)(*args, **kwargs)
        except Exception as e:
            raise type(e)(f"Error processing {k} component: {args}, {kwargs}") from e

    def handle_code(self, _, code):
        return self.controller.eval_code(code)

    def handle_columns(self, k, v):
        children = v.pop("children", [])
        col_var = v.pop("key", "cols")
        self.columns[col_var] = self.make_st_call(k, v)
        for i, child in enumerate(children):
            with self.columns[col_var][i]:
                self.process_component(child["column"])

    def handle_empty(self, _, v):
        with st.empty():
            self.process_component(v)

    def handle_expander(self, k, v):
        children = v.pop("children", [])
        with self.make_st_call(k, v):
            self.process_component(children)

    def handle_if(self, _, block):
        if self.handle_code(_, block.get("condition")):
            self.process_component(block.get("then"))
        else:
            # optional else by passing empty set of components
            # if the else key isn't found
            self.process_component(block.get("else", []))

    def handle_sidebar(self, _, v):
        with st.sidebar:
            self.process_component(v)

    def handle_trajectories_chart(self):
        charts.trajectories_chart(self.controller)

    def handle_theoretical_chart(self):
        charts.theoretical_chart(self.controller)

    default_name = {"segmented_control": "default"}

    columns = {}

    yaml_format = """
    Invalid YAML format - expecting:
    - comp0        # for component with zero parameters
    - comp1 param  # for component with only one parameter
    - comp2        # for component with multiple parameters
      param1
      param2
    - comp3        # for component with zero params + subcomponents (e.g., sidebar)
      - comp4
        param1
        param2
      - comp5
        param1
        param2
    - comp4        # for component with parameters + subcomponents (e.g., expander)
      param1
      param2
      children
        - comp6
          param1
          param2
        - comp7
          param1
          param2
    """
