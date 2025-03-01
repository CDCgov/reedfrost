import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.filterwarnings(
    r"ignore:\s+Deprecated since `altair=5.5.0`. Use altair.theme instead."
)
def test_app():
    # Cf. https://docs.streamlit.io/develop/api-reference/app-testing
    at = AppTest.from_file("src/reedfrost/app.py")
    at.run()
    assert not at.exception
