from types import ModuleType
from typing import Any, Callable

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

type Inputs = dict[str, Any]


class Component:
    def __init__(self, exec: Callable):
        """
        Define a general component that can be called in Streamlit
        """
        self.exec = exec

    def __call__(self, inputs: Inputs) -> Any:
        return self.exec(inputs)


class StreamlitComponent(Component):
    def __init__(self, method_name: str, kwargs_template: dict):
        """
        Define a Streamlit component, without actually executing it
        """
        assert method_name in dir(st)

        self.method_name = method_name
        self.kwargs_template = kwargs_template

    def __call__(self, inputs: Inputs, c: DeltaGenerator | ModuleType = st):
        kwargs = {
            k: v(inputs) if callable(v) else v for k, v in self.kwargs_template.items()
        }
        return getattr(c, self.method_name)(**kwargs)


class Inputter:
    def __init__(self):
        """
        Track streamlit input components and the values they return
        """
        self.components = {}
        self.inputs = {}

    def register_component(self, key: str, component: Component):
        assert key not in self.components
        self.components[key] = component

    def place_component(self, key: str, c: DeltaGenerator | ModuleType = st):
        """
        Args:
            key: Component ID
            c: If a DeltaGenerator, the component will be placed in that container.
                Default is `st`, which will place in the app relative to where it
                is called in the script.
        """
        component = self.components[key]
        if isinstance(component, StreamlitComponent):
            self.inputs[key] = component(self.inputs, c=c)
        elif isinstance(component, Component):
            self.inputs[key] = component(self.inputs)
        else:
            raise RuntimeError()

    def inset_input_value(self, key: str, value):
        assert key not in self.inputs
        self.inputs[key] = value


def register_inputs() -> Inputter:
    inputter = Inputter()
    inputter.register_component(
        "n",
        StreamlitComponent(
            "slider",
            {
                "label": "Population size",
                "min_value": 1,
                "max_value": 100,
                "step": 1,
                "value": 10,
            },
        ),
    )

    # user input is in proportions, but we get the integer number
    inputter.register_component(
        "n_immune",
        StreamlitComponent(
            "select_slider",
            {
                "label": "Proportion initially immune",
                # values are from 0 to N-1, leaving space for at least 1 infected
                "options": lambda p: range(0, p["n"]),
                "value": 0,
                "format_func": lambda p: lambda x: f"{x / p['n']:.0%}",
            },
        ),
    )

    inputter.register_component(
        "brn",
        StreamlitComponent(
            "slider",
            {
                "label": "Basic reproduction number",
                "min_value": 0.0,
                "max_value": lambda p: min(15.0, float(p["n"])),
                "step": 0.1,
                "format": "%.1f",
                "value": lambda p: min(1.5, float(p["n"])),
            },
        ),
    )

    inputter.register_component(
        "model",
        StreamlitComponent(
            "segmented_control",
            {
                "label": "Model",
                "options": ["Reed-Frost", "Enko", "Greenwood"],
                "default": "Reed-Frost",
            },
        ),
    )

    inputter.register_component(
        "result_type",
        StreamlitComponent(
            "segmented_control",
            {
                "label": "Results type",
                "options": ["Trajectories", "Theoretical"],
                "default": "Trajectories",
            },
        ),
    )

    inputter.register_component(
        "metric",
        StreamlitComponent(
            "segmented_control",
            {
                "label": "Infections metric",
                "options": ["Cumulative", "Incident"],
                "default": "Cumulative",
            },
        ),
    )

    inputter.register_component("n_infected", Component(_exec_n_infected))

    inputter.register_component(
        "n_simulations",
        StreamlitComponent(
            "slider",
            {
                "label": "No. simulations",
                "min_value": 5,
                "max_value": 250,
                "step": 1,
                "value": 100,
            },
        ),
    )

    inputter.register_component(
        "seed",
        StreamlitComponent(
            "number_input",
            {
                "label": "Random seed",
                "min_value": 0,
                "max_value": 2**32 - 1,
                "step": 1,
                "value": 42,
            },
        ),
    )

    return inputter


def _exec_n_infected(inputs: dict) -> int:
    # need special handling for the case where everyone is immune but 1,
    # because streamlit sliders must have a range
    if inputs["n"] - inputs["n_immune"] == 1:
        st.text("No. initially infected: 1")
        return 1
    else:
        return st.slider(
            "No. initially infected",
            min_value=1,
            max_value=inputs["n"] - inputs["n_immune"],
            step=1,
            value=1,
        )
