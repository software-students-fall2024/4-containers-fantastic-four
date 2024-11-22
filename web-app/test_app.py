"""
Extended test suite for the Flask application using MagicMock.
"""

import os
import pytest
from unittest.mock import patch, mock_open, MagicMock
from app import create_app, decode_photo, save_photo, process_photo
from werkzeug.security import check_password_hash, generate_password_hash
from bson import ObjectId

@pytest.fixture
def app_fixture():
    """
    Create and configure a new app instance for testing with mocked db.
    """
    with patch("app.pymongo.MongoClient") as mock_mongo_client:
        # Mock database and collection
        mock_db = MagicMock()
        mock_mongo_client.return_value = {os.getenv("MONGO_DBNAME"): mock_db}
        app = create_app()
        app.config.update(
            {
                "TESTING": True,
                "SECRET_KEY": "testsecretkey",
            }
        )
        yield app, mock_db

@pytest.fixture
def client(app_fixture):
    """
    A test client for the app.
    """
    app, mock_db = app_fixture
    return app.test_client()

def test_home_page(client):
    """Test the home page."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome to PlatifyAI!" in response.data
    assert b'<a href="/login">Log in</a>' in response.data
    assert b'<a href="/signup">Sign up</a>' in response.data

def test_signup_get(client):
    """Test GET request to signup page."""
    response = client.get("/signup")
    assert response.status_code == 200
    assert b"Sign Up" in response.data

def test_upload_get(client):
    """Test GET request to upload page."""
    response = client.get("/upload")
    assert response.status_code == 200
    assert b"Upload" in response.data

def test_upload_post_no_photo(client):
    """Test POST request to upload without photo data."""
    response = client.post("/upload", data={})
    assert response.status_code == 400
    assert b"No photo data received" in response.data

def test_decode_photo_valid():
    """Test decode_photo with valid data."""
    valid_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA"
    decoded = decode_photo(valid_data)
    assert isinstance(decoded, bytes)

def test_decode_photo_invalid():
    """Test decode_photo with invalid data."""
    invalid_data = "invaliddata"
    with pytest.raises(ValueError) as excinfo:
        decode_photo(invalid_data)
    assert "Invalid photo data" in str(excinfo.value)

def test_save_photo():
    """Test save_photo function."""
    photo_binary = b"test binary data"
    uploads_dir = os.path.join("static", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    filepath, filename = save_photo(photo_binary)
    assert os.path.exists(filepath)
    assert filename.endswith(".png")

    # Cleanup
    os.remove(filepath)

def test_new_entry_no_id(client):
    """Test new_entry route with no ID provided."""
    response = client.get("/new_entry")
    assert response.status_code == 400
    assert b"No entry ID provided" in response.data

def test_process_photo(app_fixture, client):
    """
    Test process_photo function with mocked dependencies.
    Steps:
    1. Mock the MongoDB connection.
    2. Mock the HTTP POST request to the ML client.
    3. Mock the file open operation.
    4. Set up a test filepath and filename.
    5. Set the session's username.
    6. Call the process_photo function.
    7. Assert that the requests.post call was made correctly.
    8. Assert that the response from the ML client was processed and stored in MongoDB.
    """
    app, mock_db = app_fixture
    with app.test_request_context():
        with patch.dict("flask.session", {"username": "testuser"}):
            with patch("builtins.open", mock_open(read_data=b"mock_image_data")) as mock_file:
                with patch("requests.post") as mock_post:
                    # Step 2: Mock HTTP POST request to ML client
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"plant_name": "Rose"}
                    mock_response.raise_for_status = MagicMock()
                    mock_post.return_value = mock_response

                    # Step 4: Set up test filepath and filename
                    test_filepath = "test_photo.png"
                    test_filename = "test_photo.png"

                    # Step 6: Call the process_photo function
                    process_photo(test_filepath, test_filename)

                    # Step 7: Assert that the requests.post call was made correctly
                    mock_post.assert_called_once_with(
                        "http://ml-client:3001/predict",
                        files={"image": (test_filename, mock_file(), "image/png")},
                        timeout=10,
                    )

                    # Step 8: Assert the response from the ML client is processed and stored in MongoDB
                    mock_db.predictions.insert_one.assert_called_once_with(
                        {
                            "photo": test_filename,
                            "filepath": test_filepath,
                            "plant_name": "Rose",
                            "user": "testuser",
                        }
                    )


def test_history_page(client):
    """Test the history page."""
    response = client.get("/history")
    assert response.status_code == 302  # Redirect to login since user is not logged in

def test_uploaded_file(client):
    """Test the uploaded file route."""
    uploads_dir = os.path.join("static", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    test_file_path = os.path.join(uploads_dir, "test_photo.png")
    with open(test_file_path, "wb") as f:
        f.write(b"test file content")

    response = client.get("/uploads/test_photo.png")
    assert response.status_code == 404  # Assuming the file should not exist in this test

    # Cleanup
    os.remove(test_file_path)

def test_custom_404_error(client):
    """Test custom 404 error handler."""
    response = client.get("/nonexistent_route")
    assert response.status_code == 404


def test_upload_post_success(app_fixture, client):
    """Test successful photo upload."""
    app, mock_db = app_fixture
    with patch("app.decode_photo") as mock_decode_photo, \
         patch("app.save_photo") as mock_save_photo, \
         patch("app.process_photo") as mock_process_photo:
        
        mock_decode_photo.return_value = b"decoded_image_data"
        mock_save_photo.return_value = ("uploads/test_photo.png", "test_photo.png")
        mock_process_photo.return_value = None  # Assuming process_photo doesn't return anything

        response = client.post("/upload", data={"photo": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA"}, follow_redirects=False)
        assert response.status_code == 302  # Redirect to results
        mock_decode_photo.assert_called_once()
        mock_save_photo.assert_called_once_with(b"decoded_image_data")
        mock_process_photo.assert_called_once_with("uploads/test_photo.png", "test_photo.png")

def test_upload_post_error(app_fixture, client):
    """Test photo upload with processing error."""
    app, mock_db = app_fixture
    with patch("app.decode_photo") as mock_decode_photo, \
         patch("app.save_photo") as mock_save_photo, \
         patch("app.process_photo") as mock_process_photo:
        
        mock_decode_photo.side_effect = ValueError("Invalid photo data")

        response = client.post("/upload", data={"photo": "invalid_data"}, follow_redirects=False)
        assert response.status_code == 500
        assert b"Error processing the photo" in response.data
        mock_decode_photo.assert_called_once()
        mock_save_photo.assert_not_called()
        mock_process_photo.assert_not_called()

def test_new_entry_get_missing_id(client):
    """Test accessing new_entry without providing an entry ID."""
    response = client.get("/new_entry")
    assert response.status_code == 400
    assert b"No entry ID provided" in response.data

def test_env_variables():
    """Test that environment variables are properly set."""
    assert os.getenv("MONGO_URI") is not None
    assert os.getenv("MONGO_DBNAME") is not None

def test_home_logged_in(client, app_fixture):
    """Test the home route when a user is logged in."""
    app, mock_db = app_fixture
    mock_db.plants.find.return_value = [{"user": "testuser", "_id": ObjectId(), "name": "Plant 1"}]
    with client.session_transaction() as session:
        session["username"] = "testuser"
    response = client.get("/")
    assert response.status_code == 200
    assert b"testuser" in response.data

def test_history_logged_in(client, app_fixture):
    """Test the history route when a user is logged in."""
    app, mock_db = app_fixture
    mock_db.predictions.find.return_value = [{"photo": "test_photo.png"}]
    with client.session_transaction() as session:
        session["username"] = "testuser"
    response = client.get("/history")
    assert response.status_code == 200
    assert b"test_photo.png" in response.data


def test_login_valid_user(client, app_fixture):
    """Test login with a valid user."""
    app, mock_db = app_fixture
    mock_db.users.find_one.return_value = {
        "username": "testuser",
        "password": generate_password_hash("password"),
    }
    response = client.post(
        "/login", data={"username": "testuser", "password": "password"}
    )
    assert response.status_code == 302  # Redirect to home

def test_logout(client):
    """Test logout."""
    with client.session_transaction() as session:
        session["username"] = "testuser"
    response = client.get("/logout")
    assert response.status_code == 302  # Redirect to home

def test_signup_new_user(client, app_fixture):
    """Test signup with a new user."""
    app, mock_db = app_fixture
    mock_db.users.find_one.return_value = None
    mock_db.users.insert_one.return_value = None
    response = client.post(
        "/signup", data={"username": "newuser", "password": "password"}
    )
    assert response.status_code == 302  # Redirect to home

def test_results_valid(client, app_fixture):
    """Test results route with a valid result."""
    app, mock_db = app_fixture
    mock_db.predictions.find_one.return_value = {"photo": "test_photo.png", "plant_name": "Rose"}
    response = client.get("/results/test_photo.png")
    assert response.status_code == 200
    assert b"Rose" in response.data

def test_delete_entry(client, app_fixture):
    """Test deleting an entry."""
    app, mock_db = app_fixture
    mock_entry_id = ObjectId()  # Use a valid ObjectId
    mock_db.predictions.delete_one.return_value = None
    with client.session_transaction() as session:
        session["username"] = "testuser"
    response = client.post(f"/delete/{mock_entry_id}")
    assert response.status_code == 302  # Redirect to history

def test_new_entry_get(client, app_fixture):
    """Test new_entry GET with a valid ID."""
    app, mock_db = app_fixture
    mock_entry_id = ObjectId()  # Use a valid ObjectId
    mock_db.plants.find_one.return_value = {"_id": mock_entry_id, "photo": "test_photo.png", "name": "Rose"}
    response = client.get(f"/new_entry?new_entry_id={mock_entry_id}")
    assert response.status_code == 200
    assert b"test_photo.png" in response.data

def test_new_entry_post(client, app_fixture):
    """Test new_entry POST with a valid ID."""
    app, mock_db = app_fixture
    mock_entry_id = ObjectId()  # Use a valid ObjectId
    with client.session_transaction() as session:
        session["username"] = "testuser"
    response = client.post(
        f"/new_entry?new_entry_id={mock_entry_id}", data={"instructions": "Water daily"}
    )
    assert response.status_code == 302  # Redirect to home
    

def test_handle_error(client):
    """Test the handle_error function."""
    response = client.get("/nonexistent_route")
    assert response.status_code == 404
