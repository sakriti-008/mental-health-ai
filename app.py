import tkinter as tk
from tkinter import filedialog
from utils.text_predict import predict_text_emotion
from utils.image_predict import predict_image_emotion
from utils.fusion import fuse_emotions
from utils.responses import get_response
from utils.graph import show_graph
from utils.voice_input import get_voice_input
import datetime


def run_app():

    def analyze():
        text = entry.get()
        img = file_path.get()

        text_emotion = predict_text_emotion(text)
        image_emotion = predict_image_emotion(img)

        final = fuse_emotions(text_emotion, image_emotion)
        response = get_response(final)

        with open("history.txt", "a") as f:
            f.write(f"{datetime.datetime.now()} - {final}\n")

        result_label.config(text=f"Emotion: {final}\n{response}")

    def upload():
        path = filedialog.askopenfilename()
        file_path.set(path)

    def use_voice():
        text = get_voice_input()
        entry.delete(0, tk.END)
        entry.insert(0, text)

    # GUI window
    root = tk.Tk()
    root.title("Mental Health AI")
    root.geometry("400x400")
    root.configure(bg="#1e1e2f")

    title = tk.Label(root, text="Mental Health AI", font=("Arial", 16, "bold"), bg="#1e1e2f", fg="white")
    title.pack(pady=10)

    entry = tk.Entry(root, width=35, font=("Arial", 12))
    entry.pack(pady=10)

    file_path = tk.StringVar()

    tk.Button(root, text="Upload Image", bg="#4CAF50", fg="white", command=upload).pack(pady=5)
    tk.Button(root, text="🎤 Speak", bg="#2196F3", fg="white", command=use_voice).pack(pady=5)
    tk.Button(root, text="Analyze", bg="#FF9800", fg="white", command=analyze).pack(pady=5)
    tk.Button(root, text="📊 Show Graph", bg="#9C27B0", fg="white", command=show_graph).pack(pady=5)

    result_label = tk.Label(root, text="", font=("Arial", 12), bg="#1e1e2f", fg="white")
    result_label.pack(pady=20)

    root.mainloop()