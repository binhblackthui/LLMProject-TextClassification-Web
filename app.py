from flask import Flask, request, render_template, session, redirect, url_for, flash
from flask_session import Session
from datetime import datetime, timedelta
import jwt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pymongo import MongoClient, ASCENDING, DESCENDING
import os
from dotenv import load_dotenv
import logging
from functools import lru_cache
import bleach
from werkzeug.security import generate_password_hash, check_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.urandom(24).hex())
Session(app)

# Load environment variables
load_dotenv()

# Connect to MongoDB
mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://binhblackthui:123@sentimentcluster.7hyqvuy.mongodb.net/sentiment_db?retryWrites=true&w=majority&appName=SentimentCluster")
try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)  # Add timeout to avoid hanging
    db = client["sentiment_db"]  # Match the database name in Atlas
    users_collection = db["users"]      # Collection for users
    predictions_collection = db["predictions"]  # Collection for predictions
    # Test connection
    client.admin.command("ping")
    logger.info("Connected to MongoDB successfully")
except Exception as e:
    logger.error(f"MongoDB connection error: {str(e)}")
    raise Exception(f"Cannot connect to MongoDB: {str(e)}")

# Create index for collections
try:
    users_collection.create_index("username", unique=True)
    predictions_collection.create_index([("username", ASCENDING), ("timestamp", DESCENDING)])
    logger.info("Indexes created successfully")
except Exception as e:
    logger.warning(f"Index creation error, might already exist: {str(e)}")

# Load model and tokenizer
try:
    model_path = "./sentiment_model/checkpoint-3000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading error: {str(e)}")
    raise Exception(f"Cannot load model: {str(e)}")

# Emotion prediction function
@lru_cache(maxsize=100)
def predict_emotion(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()
        emotions = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
        logger.info(f"Emotion prediction for '{text[:20]}...': {emotions[predicted_label]}")
        return emotions[predicted_label], probs[0].detach().numpy()
    except Exception as e:
        logger.error(f"Emotion prediction error: {str(e)}")
        return "error", [0.0] * 6

# JWT creation function
def create_jwt(username):
    payload = {
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")
    return token

# JWT verification function
def verify_jwt(token):
    try:
        payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT has expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid JWT")
        return None

# Register route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        if not username or not password:
            flash("Please fill in all fields!", "error")
            return redirect(url_for("register"))
        
        if len(username) < 4 or len(password) < 6:
            flash("Username must be at least 4 characters, password at least 6 characters!", "error")
            return redirect(url_for("register"))
        
        if users_collection.find_one({"username": username}):
            flash("Username already exists!", "error")
            return redirect(url_for("register"))
        
        hashed_password = generate_password_hash(password)
        try:
            users_collection.insert_one({
                "username": username,
                "password": hashed_password,
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            })
            logger.info(f"Registration successful for user {username}")
            flash("Registration successful! Please log in.", "success")
        except Exception as e:
            logger.error(f"Error saving user: {str(e)}")
            flash("Error during registration, please try again!", "error")
            return redirect(url_for("register"))
        return redirect(url_for("login"))
    
    return render_template("register.html")

# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user["password"], password):
            token = create_jwt(username)
            session["jwt"] = token
            logger.info(f"User {username} logged in successfully")
            return redirect(url_for("index"))
        else:
            logger.warning(f"Login failed for user {username}")
            flash("Invalid username or password!", "error")
            return render_template("login.html")
    return render_template("login.html")

# Logout route
@app.route("/logout")
def logout():
    session.pop("jwt", None)
    flash("Logged out successfully!", "success")
    logger.info("User logged out")
    return redirect(url_for("login"))

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    token = session.get("jwt")
    if not token or not verify_jwt(token):
        flash("Please log in!", "error")
        return redirect(url_for("login"))
    
    username = verify_jwt(token)["username"]
    
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not text:
            flash("Please enter text!", "error")
            return redirect(url_for("index"))
        if len(text) > 1000:
            flash("Text is too long! Maximum 1000 characters.", "error")
            return redirect(url_for("index"))
        
        text = bleach.clean(text)
        emotion, probabilities = predict_emotion(text)
        if emotion == "error":
            flash("Error processing text!", "error")
            return redirect(url_for("index"))
        
        probabilities = [f"{p:.2%}" for p in probabilities]
        now = datetime.now().strftime("%H:%M:%S")
        
        prediction = {
            "username": username,
            "text": text,
            "emotion": emotion,
            "probabilities": probabilities,
            "timestamp": now
        }
        try:
            predictions_collection.insert_one(prediction)
            logger.info(f"Prediction successful for user {username}")

        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return redirect(url_for("index"))
        return redirect(url_for("index"))
    
    predictions = list(predictions_collection.find({"username": username}).sort("timestamp", DESCENDING).limit(50))
    return render_template("index.html", predictions=predictions)

# MongoDB connection test route
@app.route("/test-db")
def test_db():
    try:
        client.admin.command("ping")
        return "MongoDB connection successful!"
    except Exception as e:
        logger.error(f"MongoDB connection test error: {str(e)}")
        return f"Connection error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)