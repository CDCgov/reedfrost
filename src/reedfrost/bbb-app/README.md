# Reed-Frost BBB App

This directory contains a modular Streamlit application for exploring chain binomial epidemic models (Reed-Frost, Greenwood, Enko) with a YAML-driven UI and configuration system.

## Directory Structure

```
bbb-app/
├── app/
│   ├── controller.py   # Orchestrates data, model, and view
│   ├── data.py         # Loads and manages data from YAML
│   ├── model.py        # Simulation/model logic
│   ├── charts.py       # Custom charts and UI logic
│   ├── view.py         # Renders UI from YAML
│   ├── texthelper.py   # Manage text in one loaction
│   └── __init__py      # Module setup
├── resources/
│   ├── data.yaml       # Default values for model parameters
│   ├── text.yaml       # All user-facing text/labels
│   └── ui.yaml         # UI layout and component structure
└── main.py             # App entry point
```

## YAML-Driven UI

- **ui.yaml**: Describes the UI layout and components (sliders, selectors, sidebars, etc). Uses a simple DSL with support for containers, conditionals, and custom handlers.
- **data.yaml**: Provides default values for all model parameters (e.g., population size, iymmunit, R0).
- **text.yaml**: Centralizes all user-facing text, labels, and titles for easy editing and localization.

### Example: UI Component in `ui.yaml`
```yaml
sidebar:
  n:
    type: slider
    label: {n_label}
    min: 1
    max: 100
    step: 1
  brn:
    type: slider
    label: {brn_label}
    min: 0.0
    max: 15.0
    step: 0.1
```
- `{n_label}` and `{brn_label}` are replaced at runtime with values from `text.yaml`.
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
                  provides isolation,
  renders UI          runs model           controls data
```

- **View** (`view.py`): Reads `ui.yaml` and renders Streamlit components. Handles custom logic via `handle_*` methods.
- **Controller** (`controller.py`): Mediates between the view and the model/data and isolates them. Provides `get_data`, `set_data`, and formatting utilities.
- **Model** (`model.py`): Runs simulations and calculations based on current parameters.
- **YAML files**: Provide structure, defaults, and text for the UI and model.

## How It Works

1. **Startup**: `main.py` loads the controller, which loads YAML files and initializes the view and model.
2. **UI Rendering**: The view reads `ui.yaml`, replacing placeholders with values from `text.yaml` and `data.yaml`.
3. **User Input**: User changes values in the UI; the controller or streamlit directly updates the data.
4. **Model Execution**: The model runs simulations using current parameters from the controller.
5. **Results Display**: The view displays results, charts, and tables as defined in `ui.yaml`.

## Extending the App

- Add new UI elements by editing `ui.yaml` and, if needed, adding new `handle_*` methods in `view.py`.
- Add new model logic in `model.py`.
- Add or change default values in `data.yaml`.
- Change labels or help text in `text.yaml`.
