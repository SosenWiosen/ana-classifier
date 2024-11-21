import os
import json
import tkinter as tk
from PIL import Image, ImageEnhance, ImageTk
import tkinter.messagebox as messagebox


class ImageCropper:
    def __init__(self, root, callback):
        self.root = root
        self.callback = callback
        self.root.title("Image Cropper")
        self.root.geometry("1000x1000")

        # Top frame for buttons
        self.frame = tk.Frame(root)
        self.frame.pack(side=tk.TOP, fill=tk.X)

        self.crop_button = tk.Button(self.frame, text="Crop Image", command=self.crop_image, state=tk.DISABLED)
        self.crop_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.finalize_button = tk.Button(self.frame, text="Finalize Crop", command=self.finalize_crop, state=tk.DISABLED)
        self.finalize_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.info_label = tk.Label(self.frame, text="")
        self.info_label.pack(side=tk.LEFT, padx=5, pady=5)

        # Brightness slider
        self.brightness_slider = tk.Scale(
            self.frame, from_=0.1, to=10.0, resolution=0.1, orient=tk.HORIZONTAL, label="Brightness", command=self.adjust_brightness
        )
        self.brightness_slider.set(1.0)  # Default value (no brightness change)
        self.brightness_slider.pack(side=tk.RIGHT, padx=5, pady=5)

        # Canvas for displaying the image
        self.canvas = tk.Canvas(root, cursor="cross", bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Mouse events for cropping
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None
        self.start_x = self.start_y = 0
        self.crop_coords = None
        self.image = self.displayed_image = self.cropped_image = None
        self.enhanced_image = None  # To store the brightness-adjusted image
    def center_window_on_primary_display(self, width=800, height=600):
        # Get the screen dimensions of the primary display
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate the x, y position to center the window
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        
        # Set the geometry of the window
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def open_image(self, filepath):
        self.cleanup()
        try:
            self.image = Image.open(filepath)
            self.display_image(self.image)
            self.crop_button.config(state=tk.NORMAL)
            self.finalize_button.config(state=tk.NORMAL)  # Enable finalize without crop
            self.info_label.config(text=f"Original Image: {self.image.width}x{self.image.height}")
        except Exception as e:
            self.info_label.config(text=f"Error opening image: {e}")
            self.crop_button.config(state=tk.DISABLED)
            self.finalize_button.config(state=tk.DISABLED)

    def display_image(self, image):
        self.canvas.delete("all")
        self.root.update_idletasks()
        max_width, max_height = self.root.winfo_width(), self.root.winfo_height()
        self.displayed_image = image.copy()
        self.displayed_image.thumbnail((max_width, max_height), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.displayed_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def cleanup(self):
        if self.image:
            self.image.close()
        if self.cropped_image:
            self.cropped_image.close()

        self.image = None
        self.displayed_image = None
        self.cropped_image = None
        self.tk_image = None
        self.canvas.delete("all")

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_mouse_drag(self, event):
        cur_x, cur_y = event.x, event.y
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        """Handle mouse button release and adjust cropping coordinates."""
        if not self.image:
            return

        # Ensure the crop rectangle doesn't exceed the displayed image bounds
        canvas_width = self.tk_image.width()
        canvas_height = self.tk_image.height()

        # Clamp the crop coordinates to the displayed bounds
        x1, y1, x2, y2 = self.canvas.coords(self.rect)
        x1 = max(0, min(canvas_width, x1))
        y1 = max(0, min(canvas_height, y1))
        x2 = max(0, min(canvas_width, x2))
        y2 = max(0, min(canvas_height, y2))
        self.crop_coords = (x1, y1, x2, y2)

        if self.crop_coords:
            self.finalize_button.config(state=tk.NORMAL)

    def crop_image(self):
        """Perform cropping while ensuring the crop coordinates are valid."""
        if self.crop_coords and self.image:
            # Get the crop coordinates (clamped to the displayed image)
            x1, y1, x2, y2 = [int(coord) for coord in self.crop_coords]
            scale_x = self.image.width / self.displayed_image.width
            scale_y = self.image.height / self.displayed_image.height

            # Scale crop coordinates to the original image dimensions
            x1, x2 = max(0, int(x1 * scale_x)), max(0, int(x2 * scale_x))
            y1, y2 = max(0, int(y1 * scale_y)), max(0, int(y2 * scale_y))

            # Ensure the final crop coordinates are within the original image's dimensions
            x1, x2 = min(self.image.width, x1), min(self.image.width, x2)
            y1, y2 = min(self.image.height, y1), min(self.image.height, y2)

            # Perform the crop
            self.cropped_image = self.image.crop((x1, y1, x2, y2))
            self.display_image(self.cropped_image)

    def adjust_brightness(self, value):
        """Adjust the brightness of the displayed image."""
        if self.image:
            brightness_factor = float(value)
            enhancer = ImageEnhance.Brightness(self.image)
            self.enhanced_image = enhancer.enhance(brightness_factor)
            self.display_image(self.enhanced_image)

    def finalize_crop(self):
        # Return the original image if no cropping was done
        if self.cropped_image:
            image_to_return = self.cropped_image
        else:
            image_to_return = self.image

        if image_to_return:
            self.callback(image_to_return)
            self.root.destroy()


# Example usage:
def save_image(image):
    # Example callback that saves the image to disk
    filepath = "cropped_image.png"
    image.save(filepath)
    print(f"Image saved to {filepath}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropper(root, save_image)
    app.open_image("path_to_your_image.jpg")  # Replace with your actual image path
    root.mainloop()