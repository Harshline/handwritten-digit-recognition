import tkinter as tk
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load the saved model in HDF5 format
model = tf.keras.models.load_model('digit_recognition_model.h5')

# Create the drawing pad GUI
class DigitRecognizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.canvas = tk.Canvas(self, width=200, height=200, bg="white")
        self.canvas.grid(row=0, column=0, pady=2, padx=2)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.button_predict = tk.Button(self, text="Predict", command=self.predict_digit)
        self.button_predict.grid(row=1, column=0, pady=2)
        
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=2, column=0, pady=2)

        # Create a blank image to store the drawing
        self.image = Image.new("L", (200, 200), 255)  # 'L' mode for grayscale, white background

    def paint(self, event):
        # Draw on the canvas and update the image
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.image.paste(0, [event.x, event.y, event.x + 5, event.y + 5])  # Draw on the image

    def clear_canvas(self):
        # Clear the canvas and reset the image
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), 255)

    def predict_digit(self):
        # Resize the image to 28x28 pixels (MNIST input size)
        img = self.image.resize((28, 28))
        # Invert the image colors to match the MNIST dataset (black digit on white background)
        img = ImageOps.invert(img)
        # Convert the image to a numpy array
        img = np.array(img)
        # Reshape to fit the model input (1, 28, 28, 1)
        img = img.reshape(1, 28, 28, 1)
        # Normalize pixel values (0 to 1)
        img = img / 255.0
        
        # Predict the digit using the model
        prediction = model.predict([img])
        digit = np.argmax(prediction)
        
        # Display the predicted digit
        print(f"Predicted digit: {digit}")

# Run the app
if __name__ == "__main__":
    app = DigitRecognizerApp()
    app.mainloop()