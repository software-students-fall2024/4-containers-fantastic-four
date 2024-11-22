"""
This module sets up the Flask application for the Plant Identifier project.
"""

import os
import base64
import uuid

from bson import ObjectId
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from flask import (
    Flask,
    flash,
    request,
    render_template,
    redirect,
    make_response,
    session,
    url_for,
)
import pymongo
import requests

load_dotenv()


def create_app():
    """Initializes and configures the Flask app."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit
    app.secret_key = os.getenv("SECRET_KEY")

    # Debugging: Print environment variables
    print(f"MONGO_URI: {os.getenv('MONGO_URI')}")
    print(f"MONGO_DBNAME: {os.getenv('MONGO_DBNAME')}")

    mongo_uri = os.getenv("MONGO_URI")
    mongo_dbname = os.getenv("MONGO_DBNAME")

    if not mongo_uri:
        raise ValueError("MONGO_URI is not set in the environment variables.")
    if not mongo_dbname:
        raise ValueError("MONGO_DBNAME is not set in the environment variables.")

    connection = pymongo.MongoClient(mongo_uri)
    db = connection[mongo_dbname]

    register_routes(app, db)

    return app


def register_routes(app, db):
    """Registers all the routes for the Flask app."""
    register_home_routes(app, db)
    register_auth_routes(app, db)
    register_entry_routes(app, db)


def register_home_routes(app, db):
    """Register routes for the home page and history."""

    @app.route("/")
    def home():
        username = session.get("username")
        if username:
            user_entries = list(db.plants.find({"user": username}))
            recent_entries = (
                user_entries[-3:] if len(user_entries) > 3 else user_entries
            )
            return render_template(
                "home.html", user=username, user_entries=recent_entries
            )
        return render_template("home.html", user=None)

    @app.route("/history")
    def history():
        username = session.get("username")
        if not username:
            return redirect(url_for("login"))
        user_results = list(db.predictions.find({"user": username}))
        return render_template("history.html", results=user_results)

    @app.route("/delete/<entry_id>", methods=["POST"])
    def delete_entry(entry_id):
        """Delete an entry by ID."""
        username = session.get("username")
        if not username:
            return redirect(url_for("login"))
        db.predictions.delete_one({"_id": ObjectId(entry_id), "user": username})
        flash("Entry deleted successfully", "success")
        return redirect(url_for("history"))


def register_auth_routes(app, db):
    """Register authentication-related routes."""

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form["username"]
            password = request.form["password"]
            user = db.users.find_one({"username": username})
            if not user or not check_password_hash(user["password"], password):
                flash("Invalid username or password!", "error")
                return render_template("login.html")
            session["username"] = username
            return redirect(url_for("home"))
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.pop("username", None)
        return redirect(url_for("home"))

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "POST":
            username = request.form["username"]
            password = request.form["password"]
            if db.users.find_one({"username": username}):
                flash("Username already exists", "error")
                return render_template("signup.html")
            hashed_password = generate_password_hash(password)
            db.users.insert_one({"username": username, "password": hashed_password})
            session["username"] = username
            return redirect(url_for("home"))
        return render_template("signup.html")


def register_entry_routes(app, db):
    """Register routes for entry management."""

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "POST":
            photo_data = request.form.get("photo")
            if not photo_data:
                return handle_error("No photo data received", 400)
            try:
                photo_binary = decode_photo(photo_data)
                filepath, filename = save_photo(photo_binary)
                process_photo(filepath, filename)
            except (
                ValueError,
                IOError,
                requests.RequestException,
                pymongo.errors.PyMongoError,
            ) as error:
                print(f"Error processing file: {error}")
                return handle_error("Error processing the photo", 500)
            return redirect(url_for("results", filename=filename))
        return render_template("upload.html")

    @app.route("/results/<filename>")
    def results(filename):
        result = db.predictions.find_one({"photo": filename})
        if result:
            return render_template("results.html", result=result)
        return handle_error("Result not found", 404)

    @app.route("/new_entry", methods=["GET", "POST"])
    def new_entry():
        new_entry_id = request.args.get("new_entry_id")
        if not new_entry_id:
            return handle_error("No entry ID provided", 400)
        entry_id = ObjectId(new_entry_id)
        if request.method == "POST":
            instructions = request.form["instructions"]
            db.plants.update_one(
                {"_id": entry_id}, {"$set": {"instructions": instructions}}
            )
            return redirect(url_for("home", user=session.get("username")))
        document = db.plants.find_one({"_id": entry_id})
        if not document:
            return handle_error("Entry not found", 404)
        photo = document.get("photo")
        name = document.get("name")
        return render_template(
            "new-entry.html", photo=photo, name=name, new_entry_id=new_entry_id
        )


def decode_photo(photo_data):
    """Decodes base64 photo data."""
    try:
        photo_data = photo_data.split(",")[1]  # Remove the data URL prefix
        return base64.b64decode(photo_data)
    except (IndexError, ValueError) as error:
        raise ValueError("Invalid photo data") from error


def save_photo(photo_binary):
    """Saves the decoded photo to the uploads directory."""
    uploads_dir = os.path.join("static", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(uploads_dir, filename)

    with open(filepath, "wb") as file_handle:
        file_handle.write(photo_binary)
    print(f"File saved successfully: {filepath}")

    return filepath, filename


def process_photo(filepath, filename):
    """Sends the photo to the ML client and saves the prediction to MongoDB."""
    ml_client_url = "http://ml-client:3001/predict"
    with open(filepath, "rb") as file_handle:
        files = {"image": (filename, file_handle, "image/png")}
        response = requests.post(
            ml_client_url, files=files, timeout=10
        )  # Added timeout
        response.raise_for_status()
        result = response.json()
        plant_name = result.get("plant_name", "Unknown")
        res = {
            "photo": filename,
            "filepath": filepath,
            "plant_name": plant_name,
            "user": session.get("username"),
        }

        # Save the result to the database
        db = get_db()
        db.predictions.insert_one(res)
        print(f"Inserted prediction into MongoDB: {res}")


def handle_error(message, status_code):
    """Handles errors by returning a response with a message and status code."""
    return make_response(message, status_code)


def get_db():
    """Retrieves the database connection from the Flask app context."""
    return pymongo.MongoClient(os.getenv("MONGO_URI"))[os.getenv("MONGO_DBNAME")]


if __name__ == "__main__":
    APP = create_app()
    APP.run(host="0.0.0.0", port=5000)
