# Reed-Frost BBB App

This directory contains a general MVC package that is driven by a YAML
configuration for the UI, text, and default data with supporting files providing
the model and the app-specific code components.

An example Streamlit application for exploring chain binomial epidemic models
(Reed-Frost, Greenwood, Enko) based on Scott Olesen's reedfrost package is
provided, run `make local`.

A toy example of a text-based view is provided to show the flexibility of the
package, run `make local_text`.

## Directory Structure

```
bbb_app/
├── mvc/
│   ├── controller.py   # Orchestrates data, model, and view
│   ├── data.py         # Loads and manages data from YAML
│   ├── view.py         # Renders UI from YAML
│   ├── texthelper.py   # Manage text in one loaction
│   └── __init__py      # Module setup
├── reedfrost_streamlit_app/
│   ├── data.yaml       # Default values for model parameters
│   ├── text.yaml       # All user-facing text/labels
│   ├── ui.yaml         # UI layout and component structure
|   ├── model.py        # Simulation/model logic
|   ├── code.py         # Code snippets and custom UI logic
│   └── __init__py      # So can import model easier
└── main.py             # App entry point
```

## YAML-Driven UI

- **ui.yaml**: Describes the UI layout and components (sliders, selectors, sidebars, etc). Uses a simple DSL with support for containers, conditionals, and custom handlers.
- **data.yaml**: Provides default values for all model parameters (e.g., population size, R0).
- **text.yaml**: Centralizes all user-facing text, labels, and titles for easy editing and localization.

### Example: UI Component in `ui.yaml`
```yaml
sidebar:
  - slider:
    key: n
    label: "{n_label}"
    min: 1
    max: 100
    step: 1
  - select_slider:
      key: n_immune
      label: "{prop_immune_label}"
      options: _code  # looks up n_immune_options in code.py code dict
      format_func: _code # looks up n_immune_format_func in code.py code dict
```
- `{n_label}` and `{prop_immune_label}` are replaced at runtime with
  values from `text.yaml`.
- Default values are loaded from `data.yaml`.

## Component Interaction

The app follows a Model-View-Controller (MVC) pattern:

```
+-----------+        +-----------+        +------------+
|  View     | <----> |Controller | <----> | Data Model |
+-----------+        +-----------+        +------------+
     ^                    ^                     ^
     |                    |                     |
     |                    |                     |
     |                    |                     |
     |                    |                     |
     |                    |                     |
     |                    |                     |
     |                    |                     |
     |                    |                     |
     v                    v                     v
  ui.yaml              model.py             data.yaml
  code.py          provides isolation,
  renders UI          runs model           controls data
```

- **View** (`view.py`): Reads `ui.yaml` and with `StreamlitView` renders Streamlit components. Handles custom logic via `handle_*` methods.
- **Controller** (`controller.py`): Mediates between the view and the model/data and isolates them. Provides `get_data`, `set_data`, and formatting utilities.
- **Model** (`model.py`): Runs simulations and calculations based on current parameters.
- **YAML files**: Provide structure, defaults, and text for the UI and model.

## How It Works

1. **Startup**: `main.py` loads the controller, which loads YAML files and initializes the app.
2. **UI Rendering**: The view reads `ui.yaml`, replacing placeholders with values from `text.yaml` and `data.yaml`.
3. **User Input**: User changes values in the UI; the controller or streamlit directly updates the data.
4. **Model Execution**: The model runs simulations using current parameters from the controller.
5. **Results Display**: The view displays results, charts, and tables as defined in `ui.yaml`.

## Extending the App

- Add new UI elements by editing `ui.yaml` and, if needed, subclassing a `View` and adding new `handle_*` methods.
- Add new model logic in `model.py`.
- Add or change default values in `data.yaml`.
- Change labels or help text in `text.yaml`.
