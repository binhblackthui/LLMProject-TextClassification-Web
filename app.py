from flask import Flask, request, render_template, session
from flask_session import Session
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Tải mô hình và tokenizer
model_path = "./sentiment_model/checkpoint-3000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Hàm dự đoán cảm xúc
def predict_emotion(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()
        emotions = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
        return emotions[predicted_label], probs[0].detach().numpy()
    except Exception as e:
        return "error", [0.0] * 6

@app.route("/", methods=["GET", "POST"])
def index():
    if not session.get("messages"):
        session["messages"] = []
    
    if request.method == "POST":
        text = request.form["text"]
        if text.strip() == "":
            return render_template("index.html", error="Vui lòng nhập văn bản!", messages=session["messages"])
        if len(text) > 1000:
            return render_template("index.html", error="Văn bản quá dài!", messages=session["messages"])
        emotion, probabilities = predict_emotion(text)
        if emotion == "error":
            return render_template("index.html", error="Lỗi khi xử lý văn bản!", messages=session["messages"])
        probabilities = [f"{p:.2%}" for p in probabilities]
        now = datetime.now().strftime("%H:%M:%S")
        session["messages"].append({
            "text": text,
            "emotion": emotion,
            "probabilities": probabilities,
            "timestamp": now
        })
        session.modified = True
        return render_template("index.html", messages=session["messages"])
    return render_template("index.html", messages=session["messages"])

if __name__ == "__main__":
    app.run(debug=True)