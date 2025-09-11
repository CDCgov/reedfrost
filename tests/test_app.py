import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Exception as StreamlitException


@pytest.mark.filterwarnings(
    r"ignore:\s+Deprecated since `altair=5.5.0`. Use altair.theme instead."
)
def test_app_from_file():
    """Succeed with default parameters"""
    # Cf. https://docs.streamlit.io/develop/api-reference/app-testing
    at = AppTest.from_file("src/reedfrost/app/__init__.py")
    at.run()
    assert not at.exception


def test_app_default():
    """Succeed with default parameters"""

    # confusingly, the .from_function() tests need to have all their imports
    def f():
        import reedfrost.app

        reedfrost.app.main()

    at = AppTest.from_function(f)
    at.run()
    assert not at.exception


def test_app_different():
    """Succeed with somewhat different parameters"""

    def f():
        import reedfrost.app
        from reedfrost.app.model import INITIAL_DATA

        initial_data = INITIAL_DATA | {"n": 20, "n_infected": 2, "brn": 2.5}

        reedfrost.app.main(initial_data=initial_data)

    at = AppTest.from_function(f)
    at.run()
    assert not at.exception


def test_app_fail():
    """Fail if n = 0"""

    def f():
        import reedfrost.app
        from reedfrost.app.model import INITIAL_DATA

        initial_data = INITIAL_DATA | {"n": 0}

        reedfrost.app.main(initial_data=initial_data)

    at = AppTest.from_function(f)
    at.run()
    e = at.exception[0]

    assert isinstance(e, StreamlitException)
    assert "n >= 1" in e.stack_trace[-1]
