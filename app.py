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

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo Flask app
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", os.urandom(24).hex())
Session(app)

# Tải biến môi trường
load_dotenv()

# Kết nối MongoDB
mongo_uri = os.getenv("MONGO_URI", "mongodb+srv://binhblackthui:123@sentimentcluster.7hyqvuy.mongodb.net/sentiment_db?retryWrites=true&w=majority&appName=SentimentCluster")
try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)  # Thêm timeout để tránh treo
    db = client["sentiment_db"]  # Khớp với database name trong Atlas
    users_collection = db["users"]      # Collection cho người dùng
    predictions_collection = db["predictions"]  # Collection cho dữ liệu dự đoán
    # Kiểm tra kết nối
    client.admin.command("ping")
    logger.info("Kết nối MongoDB thành công")
except Exception as e:
    logger.error(f"Lỗi kết nối MongoDB: {str(e)}")
    raise Exception(f"Không thể kết nối đến MongoDB: {str(e)}")

# Tạo index cho collection
try:
    users_collection.create_index("username", unique=True)
    predictions_collection.create_index([("username", ASCENDING), ("timestamp", DESCENDING)])
    logger.info("Tạo index thành công")
except Exception as e:
    logger.warning(f"Lỗi tạo index, có thể đã tồn tại: {str(e)}")

# Tải mô hình và tokenizer
try:
    model_path = "./sentiment_model/checkpoint-3000"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    logger.info("Tải mô hình thành công")
except Exception as e:
    logger.error(f"Lỗi tải mô hình: {str(e)}")
    raise Exception(f"Không thể tải mô hình: {str(e)}")

# Hàm dự đoán cảm xúc
@lru_cache(maxsize=100)
def predict_emotion(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()
        emotions = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
        logger.info(f"Dự đoán cảm xúc cho '{text[:20]}...': {emotions[predicted_label]}")
        return emotions[predicted_label], probs[0].detach().numpy()
    except Exception as e:
        logger.error(f"Lỗi dự đoán cảm xúc: {str(e)}")
        return "error", [0.0] * 6

# Hàm tạo JWT
def create_jwt(username):
    payload = {
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")
    return token

# Hàm xác thực JWT
def verify_jwt(token):
    try:
        payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT đã hết hạn")
        return None
    except jwt.InvalidTokenError:
        logger.warning("JWT không hợp lệ")
        return None

# Route đăng ký
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        if not username or not password:
            flash("Vui lòng nhập đầy đủ thông tin!", "error")
            return redirect(url_for("register"))
        
        if len(username) < 4 or len(password) < 6:
            flash("Tên đăng nhập tối thiểu 4 ký tự, mật khẩu tối thiểu 6 ký tự!", "error")
            return redirect(url_for("register"))
        
        if users_collection.find_one({"username": username}):
            flash("Tên đăng nhập đã tồn tại!", "error")
            return redirect(url_for("register"))
        
        hashed_password = generate_password_hash(password)
        try:
            users_collection.insert_one({
                "username": username,
                "password": hashed_password,
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            })
            logger.info(f"Đăng ký thành công cho người dùng {username}")
            flash("Đăng ký thành công! Vui lòng đăng nhập.", "success")
        except Exception as e:
            logger.error(f"Lỗi lưu người dùng: {str(e)}")
            flash("Lỗi khi đăng ký, hãy thử lại!", "error")
            return redirect(url_for("register"))
        return redirect(url_for("login"))
    
    return render_template("register.html")

# Route đăng nhập
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user["password"], password):
            token = create_jwt(username)
            session["jwt"] = token
            logger.info(f"Người dùng {username} đăng nhập thành công")
            return redirect(url_for("index"))
        else:
            logger.warning(f"Đăng nhập thất bại cho người dùng {username}")
            flash("Sai tên đăng nhập hoặc mật khẩu!", "error")
            return render_template("login.html")
    return render_template("login.html")

# Route đăng xuất
@app.route("/logout")
def logout():
    session.pop("jwt", None)
    flash("Đăng xuất thành công!", "success")
    logger.info("Người dùng đã đăng xuất")
    return redirect(url_for("login"))

# Route chính
@app.route("/", methods=["GET", "POST"])
def index():
    token = session.get("jwt")
    if not token or not verify_jwt(token):
        flash("Vui lòng đăng nhập!", "error")
        return redirect(url_for("login"))
    
    username = verify_jwt(token)["username"]
    
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if not text:
            flash("Vui lòng nhập văn bản!", "error")
            return redirect(url_for("index"))
        if len(text) > 1000:
            flash("Văn bản quá dài! Tối đa 1000 ký tự.", "error")
            return redirect(url_for("index"))
        
        text = bleach.clean(text)
        emotion, probabilities = predict_emotion(text)
        if emotion == "error":
            flash("Lỗi khi xử lý văn bản!", "error")
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
            logger.info(f"Dự đoán thành công cho người dùng {username}")

        except Exception as e:
            logger.error(f"Lỗi lưu dự đoán: {str(e)}")
            return redirect(url_for("index"))
        return redirect(url_for("index"))
    
    predictions = list(predictions_collection.find({"username": username}).sort("timestamp", DESCENDING).limit(50))
    return render_template("index.html", predictions=predictions)

# Route kiểm tra kết nối MongoDB
@app.route("/test-db")
def test_db():
    try:
        client.admin.command("ping")
        return "Kết nối MongoDB thành công!"
    except Exception as e:
        logger.error(f"Lỗi kiểm tra kết nối MongoDB: {str(e)}")
        return f"Lỗi kết nối: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)