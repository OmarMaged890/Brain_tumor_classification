import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
import cv2

model_path = r"C:\Users\LENOVO\AMIT AI\Amit-1\myenv\Scripts\Final Project BME\classification\Brain-Tumor-MRI-Classification-main\model-22-0.98-0.10.h5"
model = load_model(model_path)

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (200, 200))  
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Classify image function
def classify_image():
    if not app.selected_image_path:
        messagebox.showerror("Error", "Please upload an image first.")
        return

    img = preprocess_image(app.selected_image_path)
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    app.result_label.config(
        text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%", fg="#004d00"
    )

# Upload image function
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        app.selected_image_path = file_path
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        app.image_label.config(image=img_tk)
        app.image_label.image = img_tk
        app.result_label.config(text="")

# Main App Class
class BrainTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Classification")
        self.root.geometry("450x600")
        self.root.configure(bg="#f2f2f2")
        self.root.resizable(False, False)

        self.selected_image_path = None

        # Title
        tk.Label(
            root, text="Brain Tumor Classifier", font=("Helvetica", 20, "bold"), bg="#f2f2f2", fg="#333"
        ).pack(pady=20)

        # Frame for image display
        self.image_frame = tk.Frame(root, width=260, height=260, bg="#ddd", bd=2, relief=tk.GROOVE)
        self.image_frame.pack(pady=10)
        self.image_label = tk.Label(self.image_frame, bg="#ddd")
        self.image_label.pack()

        # Buttons
        tk.Button(
            root,
            text="Upload MRI Image",
            command=upload_image,
            bg="#007acc",
            fg="white",
            font=("Arial", 12),
            padx=10,
            pady=5,
            width=20
        ).pack(pady=10)

        tk.Button(
            root,
            text="Classify Image",
            command=classify_image,
            bg="#28a745",
            fg="white",
            font=("Arial", 12),
            padx=10,
            pady=5,
            width=20
        ).pack(pady=5)

        # Result label
        self.result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f2f2f2", fg="#333")
        self.result_label.pack(pady=20)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()
