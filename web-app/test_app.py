"""
Extended test suite for the Flask application without pymongo.
"""

import os
import pymongo
import pytest
from app import create_app, decode_photo, save_photo

# import pymongo


@pytest.fixture
def app():  # pylint: disable=redefined-outer-name
    """Create and configure a new app instance for testing."""
    # Mock environment variables
    # os.environ["MONGO_URI"] = "mongodb://mockdb:27017/"
    # os.environ["MONGO_DBNAME"] = "test"

    app = create_app()  # pylint: disable=redefined-outer-name
    app.config.update(
        {
            "TESTING": True,
            "SECRET_KEY": "testsecretkey",
        }
    )
    return app


@pytest.fixture
def client(app):  # pylint: disable=redefined-outer-name
    """A test client for the app."""
    return app.test_client()


def test_home_page(client):  # pylint: disable=redefined-outer-name
    """Test the home page."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Welcome to PlatifyAI!" in response.data
    assert b'<a href="/login">Log in</a>' in response.data
    assert b'<a href="/signup">Sign up</a>' in response.data


def test_login_get(client):  # pylint: disable=redefined-outer-name
    """Test the login page displays correctly"""
    response = client.get("/login")
    assert response.status_code == 200
    assert b"Username" in response.data
    assert b"Password" in response.data


def test_login_post(client):  # pylint: disable=redefined-outer-name
    """Test the login form results post correctly"""
    response = client.post("/login", data={"username": "test_user"})
    assert response.status_code == 302
    assert response.headers["Location"] == "/?user=test_user"


def test_home_with_user(client):  # pylint: disable=redefined-outer-name
    """Test home route with user input"""
    connection = pymongo.MongoClient(os.getenv("MONGO_URI"))
    db_name = os.getenv("MONGO_DBNAME")
    # print(connection.list_database_names())
    db = connection[db_name].plants
    db.delete_many({"user": "test_user"})
    db.insert_many(
        [
            {"_id": "9999", "photo": "photo1", "name": "rose", "user": "test_user"},
            {"_id": "2", "photo": "photo2", "name": "lily", "user": "test_user"},
        ]
    )

    response = client.get("/?user=test_user")
    assert response.status_code == 200
    assert b"test_user" in response.data


def test_signup_get(client):  # pylint: disable=redefined-outer-name
    """Test GET request to signup page."""
    response = client.get("/signup")
    assert response.status_code == 200
    assert b"Sign Up" in response.data


def test_signup_post(client):  # pylint: disable=redefined-outer-name
    """Test the signup form results post correctly"""
    connection = pymongo.MongoClient(os.getenv("MONGO_URI"))
    # make config to identify test
    db = connection[os.getenv("MONGO_DBNAME")].users
    response = client.post("/signup", data={"username": "new_user", "password": "pass"})
    assert response.status_code == 302
    assert response.headers["Location"] == "/?user=new_user"
    # make sure new user was added properly to db
    user = db.find_one({"username": "new_user"})
    assert user is not None
    db.delete_one({"username": "new_user"})


def test_upload_get(client):  # pylint: disable=redefined-outer-name
    """Test GET request to upload page."""
    response = client.get("/upload")
    assert response.status_code == 200
    assert b"Upload" in response.data


def test_upload_post_no_photo(client):  # pylint: disable=redefined-outer-name
    """Test POST request to upload without photo data."""
    response = client.post("/upload", data={})
    assert response.status_code == 400
    assert b"No photo data received" in response.data


def test_decode_photo_valid():  # pylint: disable=redefined-outer-name
    """Test decode_photo with valid data."""
    valid_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA"
    decoded = decode_photo(valid_data)
    assert isinstance(decoded, bytes)


def test_decode_photo_invalid():  # pylint: disable=redefined-outer-name
    """Test decode_photo with invalid data."""
    invalid_data = "invaliddata"
    with pytest.raises(ValueError) as excinfo:
        decode_photo(invalid_data)
    assert "Invalid photo data" in str(excinfo.value)


def test_save_photo():  # pylint: disable=redefined-outer-name
    """Test save_photo function."""
    photo_binary = b"test binary data"
    uploads_dir = os.path.join("static", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    filepath, filename = save_photo(photo_binary)
    assert os.path.exists(filepath)
    assert filename.endswith(".png")

    # Cleanup
    os.remove(filepath)


def test_new_entry_no_id(client):  # pylint: disable=redefined-outer-name
    """Test new_entry route with no ID provided."""
    response = client.get("/new_entry")
    assert response.status_code == 400
    assert b"No entry ID provided" in response.data


def test_get_new_entry_valid(client):  # pylint: disable=redefined-outer-name
    """Test new_entry route with a valid ID provided."""
    connection = pymongo.MongoClient(os.getenv("MONGO_URI"))
    db_name = os.getenv("MONGO_DBNAME")
    # print(connection.list_database_names())
    db = connection[db_name].plants
    db.delete_one({"name": "test_plant"})
    db.insert_one({"photo": "test_file", "name": "test_plant"})
    doc_id = db.find_one({"plant_name": "test_plant"})["_id"]
    response = client.get(f"/new_entry?new_entry_id={doc_id}")
    assert response.status_code == 200
    assert b"test_file" in response.data


def test_post_new_entry_valid(client):  # pylint: disable=redefined-outer-name
    """Test new_entry posts correctly with valid provided."""
    connection = pymongo.MongoClient(os.getenv("MONGO_URI"))
    db_name = os.getenv("MONGO_DBNAME")
    # print(connection.list_database_names())
    db = connection[db_name].plants
    db.delete_one({"name": "test_plant"})
    db.insert_one({"photo": "test_file", "name": "test_plant"})
    doc_id = db.find_one({"plant_name": "test_plant"})["_id"]
    response = client.post(
        f"/new_entry?new_entry_id={doc_id}", data={"instructions": "water daily"}
    )
    assert response.status_code == 302


def test_results(client):  # pylint: disable=redefined-outer-name
    """Test results route"""
    connection = pymongo.MongoClient(os.getenv("MONGO_URI"))
    db_name = os.getenv("MONGO_DBNAME")
    # print(connection.list_database_names())
    db = connection[db_name].predictions
    db.insert_one({"photo": "test_file", "plant_name": "test_plant"})
    # found = db.find({"plant_name":"test_plant"})[0]
    response = client.get("/results/test_file")
    print(response)
    assert response.status_code == 200
    assert b"test_plant" in response.data


def test_results_none(client):  # pylint: disable=redefined-outer-name
    """Test results route with invalid input"""
    response = client.get("/results/unknown_file")
    print(response)
    assert response.status_code == 404
