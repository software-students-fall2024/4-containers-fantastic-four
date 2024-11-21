"""
Unit tests for the app.py module, which performs flower classification using a ResNet50 model.
This script tests each function independently to achieve a minimum of 80% code coverage.
"""

import os
import io
import pytest
import torch
from PIL import Image
import torchvision.models
from app import app
from app import load_flower_names, load_model, transform_image, predict_plant

# Constants for data paths
DATA_PATH = os.path.join("data", "flowers-102", "jpg", "image_00001.jpg")
FLOWER_NAMES_PATH = os.path.join("data", "flower_to_name.json")


@pytest.fixture
def mock_open(monkeypatch):
    """Fixture to mock the open function for loading flower names."""

    def mock_file_open(*_args, **_kwargs):
        """Mock file open function to return JSON-like flower names."""

        class MockFile:
            """Mock file object that mimics open()."""

            # pylint: disable=too-few-public-methods
            def read(self):
                """Return bytes-like JSON data for flower names."""
                return '{"1": "Rose", "2": "Tulip"}'.encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return MockFile()

    monkeypatch.setattr("builtins.open", mock_file_open)


@pytest.mark.usefixtures("mock_open")
def test_load_flower_names():
    """Test loading flower names from a JSON file."""
    flower_names = load_flower_names()
    assert flower_names == {"1": "Rose", "2": "Tulip"}


@pytest.fixture
def mock_resnet50(monkeypatch):
    """Fixture to mock torchvision.models.resnet50 with a simplified model."""

    class MockModel:
        """Mock ResNet50 model with a modified final layer."""

        # pylint: disable=too-few-public-methods
        def __init__(self):
            self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(64, 102)  # Mock final layer for 102 classes

        def forward(self, x):
            """Mock forward pass simulating a ResNet-like structure."""
            x = self.conv1(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)  # Flatten before the fully connected layer
            return self.fc(x)

        def load_state_dict(self, _):
            """Mock method for loading state dictionary."""
            pass  # pylint: disable=unnecessary-pass

        def eval(self):
            """Mock method to set the model to evaluation mode."""
            return self

        def to(self, _device):
            """Mock method to move the model to a specified device."""
            return self

    monkeypatch.setattr(
        torchvision.models, "resnet50", lambda *_args, **_kwargs: MockModel()
    )


@pytest.mark.usefixtures("mock_resnet50")
def test_load_model():
    """Test loading and initializing the ResNet50 model."""
    model = load_model()
    assert model is not None


@pytest.fixture
def mock_image_open(monkeypatch):
    """Fixture to mock PIL.Image.open to return a sample image for testing."""
    sample_image = Image.new("RGB", (224, 224), color="red")  # Create a mock red image
    monkeypatch.setattr(Image, "open", lambda _path: sample_image)


@pytest.mark.usefixtures("mock_image_open")
def test_transform_image():
    """Test transforming an image to a tensor for model input."""
    tensor = transform_image(DATA_PATH)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)


@pytest.mark.usefixtures("mock_open", "mock_image_open", "mock_resnet50")
def test_predict_plant(monkeypatch):
    """Test predicting the plant name using the model and flower names dictionary."""
    # Load mocked flower names
    flower_names = load_flower_names()  # Using the mocked flower names data

    # Create a mock model that produces predictable outputs
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 2),  # Reduced output size for test purposes
    )
    model.eval()

    with torch.no_grad():
        # Set weights and biases to produce predictable outputs
        model[3].weight.fill_(
            0.1
        )  # Fill weights with values that will produce predictable results
        model[3].bias[0] = 0.5  # Bias for "Rose"
        model[3].bias[1] = 1.0  # Bias for "Tulip"

    # Mock the global variables in app.py
    monkeypatch.setattr("app.TRAINED_MODEL", model)
    monkeypatch.setattr("app.FLOWER_CLASS_NAMES", flower_names)

    # Call predict_plant with only the image_path argument
    result = predict_plant(DATA_PATH)
    # Assert the expected result
    assert result == "Tulip"


@pytest.fixture
def create_flask_test_app():
    """Create a test Flask app."""
    app.testing = True
    return app


def test_predict_route(request, monkeypatch):
    """Test the /predict route."""

    # Explicitly retrieve the fixture without passing it as a parameter
    flask_test_app = request.getfixturevalue("create_flask_test_app")

    def mock_predict_plant(_):
        """Mock predict_plant to return a valid result."""
        return "Mocked Flower"

    monkeypatch.setattr("app.predict_plant", mock_predict_plant)

    with flask_test_app.test_client() as client:
        data = {"image": (io.BytesIO(b"dummy image data"), "image.jpg")}
        response = client.post(
            "/predict", data=data, content_type="multipart/form-data"
        )
        assert response.status_code == 200
        assert response.get_json() == {"plant_name": "Mocked Flower"}
