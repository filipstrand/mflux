from pathlib import Path

import cv2
import numpy as np


class MaskCreator:
    BRUSH_SIZES = {"1": 2, "2": 4, "3": 8, "4": 16, "5": 32, "6": 48, "7": 96, "8": 192, "9": 384}

    def __init__(self, image_path: Path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Could not open or find the image: {image_path}")

        self.mask_output_path = image_path.with_name(f"{image_path.stem}_mask").with_suffix(".png")

        # Create a window and set up display image
        self.window_name = "MFlux Inpaint Mask Creator - Draw with mouse or trackpad (hot keys: (s)ave, (r)eset, (q)uit"
        self.display_image = self.original_image.copy()

        # Create a blank mask the same size as the image
        self.mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)

        # Set up drawing parameters
        self.drawing = False
        self.brush_size = self.BRUSH_SIZES["5"]
        self.last_point = None

        self.overlay = np.zeros_like(self.original_image)

        # Update display every N drawing events - lower is more responsive
        self.update_frequency = 1
        self.event_counter = 0

        # Show the initial display
        self.update_display()

    def mouse_callback(self, event, x, y, flags, param):
        # Start drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (x, y)
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
            self.event_counter += 1
            if self.event_counter % self.update_frequency == 0:
                self.update_display()

        # Continue drawing
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Use thickness based on brush size for smoother lines
            if self.last_point:  # Ensure we have a last point
                # Draw a line between the last point and current point
                cv2.line(self.mask, self.last_point, (x, y), 255, self.brush_size * 2)
                # Also draw a circle at the current point to avoid gaps in fast movements
                cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
            self.last_point = (x, y)
            self.event_counter += 1
            if self.event_counter % self.update_frequency == 0:
                self.update_display()

        # Stop drawing
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.update_display()  # Always update display when stopping

    def update_display(self):
        # Create a copy of the original image
        self.display_image = self.original_image.copy()

        # Create a colored overlay for the mask (semi-transparent red)
        self.overlay[:] = 0  # Reset overlay
        self.overlay[self.mask > 0] = [0, 0, 255]  # Red overlay

        # Apply the overlay
        alpha = 0.5  # Transparency level
        cv2.addWeighted(self.overlay, alpha, self.display_image, 1 - alpha, 0, self.display_image)

        # Draw brush size indicator in the corner
        text = f"Brush Size: {self.brush_size} (Hotkeys 1-9: change brush size) "
        cv2.putText(
            self.display_image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,  # line type: anti-aliased
        )

        # Display the result with OpenCV's high GUI priority
        cv2.imshow(self.window_name, self.display_image)
        cv2.waitKey(1)  # Process events to force display update

    def save_mask(self, output_path):
        cv2.imwrite(output_path, self.mask)
        print(f"Mask saved to {output_path}")

    def reset_mask(self):
        self.mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        self.update_display()

    def set_brush_size(self, size_key):
        if size_key in self.BRUSH_SIZES:
            self.brush_size = self.BRUSH_SIZES[size_key]
            print(f"Brush size {size_key}: {self.brush_size}")
            self.update_display()

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF
            key_char = chr(key) if key < 128 else ""

            # Check for brush size hotkeys (1-5)
            if key_char in self.BRUSH_SIZES:
                self.set_brush_size(key_char)

            # Save mask (press 's')
            elif key == ord("s"):
                self.save_mask(self.mask_output_path)

            # Reset mask (press 'r')
            elif key == ord("r"):
                self.reset_mask()
                print("Mask reset")

            # Quit (press 'q' or ESC)
            elif key == ord("q") or key == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create binary mask image from source image to use as the complementary --masked-image-path arg."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image")
    args = parser.parse_args()

    try:
        mask_creator = MaskCreator(args.image_path)
        mask_creator.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:  # noqa
        print(f"An unexpected error occurred: {e}")
    except KeyboardInterrupt:
        pass
