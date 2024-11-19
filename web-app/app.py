"""
This module sets up the Flask application for the Plant Identifier project.
"""

import base64
import os
from bson import ObjectId
import mongomock
# from bson import ObjectId
from dotenv import load_dotenv
from flask import (
    Flask,
    request,
    render_template,
    redirect,
    make_response,
    session,
    url_for,
)
import pymongo

# from werkzeug.utils import secure_filename

load_dotenv()


def create_app(test_config=None):
    """Initializes and configures the Flask app"""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit
    app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

    if test_config:
        app.config.update(test_config)

    # Initialize the database connection
    if 'DB_CLIENT' in app.config:
        app.db = app.config['DB_CLIENT']
    else:
        connection = pymongo.MongoClient(os.getenv('MONGO_URI'))
        app.db = connection[os.getenv("MONGO_DBNAME")]
    
    # connection = pymongo.MongoClient(os.getenv("MONGO_URI"))


    @app.route("/")
    def home():
        if request.args.get("user"):
            user_entries = list(app.db.plants.find({"user":request.args.get('user')}))
            if len(user_entries) > 3:
                new_entries = [user_entries[-1], user_entries[-2], user_entries[-3]]
                user_entries = new_entries
            return render_template(
                "home.html", user=request.args.get("user"), user_entries=user_entries
            )
        return render_template("home.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form["username"]
            # password = request.form["password"]

            session["username"] = username

            return redirect(url_for("home", user=username))

        return render_template("login.html")

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "POST":
            username = request.form["username"]
            password = request.form["password"]

            app.db.users.insert_one({"username": username, "password": password})
            session["username"] = username

            return redirect(url_for("home", user=username))

        return render_template("signup.html")

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "POST":
            plant_photo = request.form["photo"]

            # convert photo data to jpg file
            photo_data = plant_photo.split(",")[1]
            with open("photo.jpg", "wb") as jpg_file:
                jpg_file.write(base64.b64decode(photo_data))

            # get plant name based on ml module that identifies plant types
            # in the meantime use 'placeholder'
            plant_name = "placeholder"

            # post to database
            plant_data = {
                "photo": plant_photo,
                "name": plant_name,
                "user": session["username"],
            }
            new_entry = app.db.plants.insert_one(plant_data)
            new_entry_id = new_entry.inserted_id
            return redirect(url_for("new_entry", new_entry_id=new_entry_id))
        return render_template("upload.html")

    @app.route("/new_entry", methods=["GET", "POST"])
    def new_entry():
        new_entry_id = request.args.get("new_entry_id")
        entry = ObjectId(new_entry_id)
        if request.method == "POST":
            instructions = request.form["instructions"]
            app.db.plants.update_one(
                {"_id": entry}, {"$set": {"instructions": instructions}}
            )
            return redirect(url_for("home", user=session["username"]))
        document = app.db.plants.find_one({"_id": entry})
        photo = document["photo"]
        name = document["name"]
        return render_template(
            "new-entry.html", photo=photo, name=name, new_entry_id=new_entry_id
        )

    @app.route("/results/<filename>")
    def results(filename):
        result = app.db.identifications.find_one({"filename": filename})
        if result:
            return render_template("results.html", result=result)
        return make_response("Result not found", 404)

    @app.route("/history")
    def history():
        all_results = list(app.db.plants.find({"user": session["username"]}))
        return render_template("history.html", all_results=all_results)

    return app


if __name__ == "__main__":
    FLASK_PORT = os.getenv("FLASK_PORT", "3000")
    application = create_app()
    application.run(port=FLASK_PORT)
