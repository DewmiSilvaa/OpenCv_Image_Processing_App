import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation App")
        self.root.configure(background='#2e2e2e')

        # Disable maximize button
        self.root.resizable(False, False)

        self.image = None
        self.processed_image = None

        # UI Elements
        self.canvas = Canvas(root, width=600, height=600, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

        self.open_button = Button(root, text="Open Image", command=self.open_image, bg='#4CAF50', fg='white')
        self.open_button.grid(row=1, column=0, padx=5, pady=5, sticky=W+E)

        self.save_button = Button(root, text="Save Image", command=self.save_image, bg='#4CAF50', fg='white')
        self.save_button.grid(row=1, column=1, padx=5, pady=5, sticky=W+E)

        self.reset_button = Button(root, text="Reset Image", command=self.reset_image, bg='#f44336', fg='white')
        self.reset_button.grid(row=1, column=2, padx=5, pady=5, sticky=W+E)

        self.threshold_button = Button(root, text="Threshold", command=self.apply_threshold, bg='#2196F3', fg='white')
        self.threshold_button.grid(row=1, column=3, padx=5, pady=5, sticky=W+E)

        self.edge_button = Button(root, text="Edge Detection", command=self.apply_edge_detection, bg='#2196F3', fg='white')
        self.edge_button.grid(row=1, column=4, padx=5, pady=5, sticky=W+E)

        self.region_button = Button(root, text="Region Growing", command=self.apply_region_growing, bg='#FFC107', fg='black')
        self.region_button.grid(row=2, column=0, padx=5, pady=5, sticky=W+E)

        self.kmeans_button = Button(root, text="K-means", command=self.apply_kmeans, bg='#FFC107', fg='black')
        self.kmeans_button.grid(row=2, column=1, padx=5, pady=5, sticky=W+E)

        self.contour_button = Button(root, text="Contour Detection", command=self.apply_contour_detection, bg='#9C27B0', fg='white')
        self.contour_button.grid(row=2, column=2, padx=5, pady=5, sticky=W+E)

        self.exit_button = Button(root, text="Exit", command=root.quit, bg='#9C27B0', fg='white')
        self.exit_button.grid(row=2, column=3, columnspan=2, padx=5, pady=5, sticky=W+E)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                self.processed_image = self.image.copy()  # Initialize processed_image with the opened image
                self.display_image(self.image)
            else:
                messagebox.showerror("Error", "Failed to open image")

    def save_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                try:
                    # Check if the image is a NumPy array (OpenCV format) or a PIL Image
                    if isinstance(self.processed_image, np.ndarray):
                        cv2.imwrite(file_path, self.processed_image)
                    elif isinstance(self.processed_image, Image.Image):
                        self.processed_image.save(file_path)
                    else:
                        messagebox.showerror("Error", "Unsupported image format")
                        return

                    messagebox.showinfo("Image Saved", "The image has been saved successfully.")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image: {str(e)}")
            else:
                messagebox.showerror("Error", "No file selected")
        else:
            messagebox.showerror("Error", "No image to save")

    def reset_image(self):
        if self.image is not None:
            self.processed_image = self.image.copy()  # Reset processed_image
            self.display_image(self.image)
        else:
            messagebox.showerror("Error", "No image loaded")

    def display_image(self, img):
        img = self.resize_image(img, 600, 600)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=NW, image=img)
        self.canvas.image = img

    def resize_image(self, img, max_width, max_height):
        height, width = img.shape[:2]
        scaling_factor = min(max_width/width, max_height/height)
        if scaling_factor < 1:
            img = cv2.resize(img, (int(width * scaling_factor), int(height * scaling_factor)), interpolation=cv2.INTER_AREA)
        return img

    def apply_threshold(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, self.processed_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            self.display_image(self.processed_image)
        else:
            messagebox.showerror("Error", "No image loaded")

    def apply_edge_detection(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.Canny(gray, 100, 200)
            self.display_image(self.processed_image)
        else:
            messagebox.showerror("Error", "No image loaded")

    def apply_region_growing(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, self.processed_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            self.display_image(self.processed_image)
        else:
            messagebox.showerror("Error", "No image loaded")

    def apply_kmeans(self):
        if self.image is not None:
            Z = self.image.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 8
            _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            self.processed_image = centers[labels.flatten()]
            self.processed_image = self.processed_image.reshape((self.image.shape))
            self.display_image(self.processed_image)
        else:
            messagebox.showerror("Error", "No image loaded")

    def apply_contour_detection(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.processed_image = self.image.copy()
            cv2.drawContours(self.processed_image, contours, -1, (0, 255, 0), 2)  # Draw contours in green
            self.display_image(self.processed_image)
        else:
            messagebox.showerror("Error", "No image loaded")

if __name__ == "__main__":
    root = Tk()
    app = ImageSegmentationApp(root)
    root.mainloop()
