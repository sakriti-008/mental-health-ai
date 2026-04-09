from flask import Flask, render_template, request
import pickle
import datetime
import csv
import os

app = Flask(__name__)

# Load ML model (IMPORTANT PATH FIX)
model = pickle.load(open("models/text_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Chatbot replies
def chatbot_reply(prediction):
    if prediction.lower() == "happy":
        return "That's great to hear! Keep smiling 😊"
    elif prediction.lower() == "sad":
        return "I'm here for you. Things will get better 💙"
    elif prediction.lower() == "angry":
        return "Take a deep breath. Try to relax 🧘"
    else:
        return "Stay positive and take care of yourself 🌟"

# Save history to CSV
def save_history(text, prediction, confidence):
    file_exists = os.path.isfile("history.csv")

    with open("history.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Time", "Text", "Prediction", "Confidence"])

        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            text,
            prediction,
            str(confidence) + "%"
        ])

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    confidence = 0
    reply = ""
    color = "black"

    if request.method == "POST":
        text = request.form.get("text", "")

        if text.strip() == "":
            return render_template("index.html", error="Please enter some text")

        # Transform text
        vec = vectorizer.transform([text])

        # Prediction
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec).max()

        prediction = pred
        confidence = round(prob * 100, 2)

        # Chatbot reply
        reply = chatbot_reply(prediction)

        # Color logic
        if prediction.lower() == "happy":
            color = "green"
        elif prediction.lower() == "sad":
            color = "red"
        elif prediction.lower() == "angry":
            color = "orange"
        else:
            color = "blue"

        # Save history
        save_history(text, prediction, confidence)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        reply=reply,
        color=color
    )

# Run app (for local testing only)
if __name__ == "__main__":
    app.run(debug=True)
