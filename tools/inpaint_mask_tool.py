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
        self.window_name = "MFlux Inpaint Mask Creator - Tools: (b)rush, (e)rase, (r)ectangle, s(q)uare, (o)val, (t)riangle | (u)ndo, (s)ave, (esc) quit"
        self.display_image = self.original_image.copy()

        # Create a blank mask the same size as the image
        self.mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)

        # Set up drawing parameters
        self.drawing = False
        self.brush_size = self.BRUSH_SIZES["5"]
        self.last_point = None

        # Tool modes
        self.tool_mode = "brush"  # brush, erase, rectangle, square, oval, triangle
        self.shape_start_point = None

        # Undo history
        self.mask_history = []
        self.max_history = 50

        self.overlay = np.zeros_like(self.original_image)

        # Update display every N drawing events - lower is more responsive
        self.update_frequency = 1
        self.event_counter = 0

        # Show the initial display
        self.update_display()

    def save_to_history(self):
        self.mask_history.append(self.mask.copy())
        if len(self.mask_history) > self.max_history:
            self.mask_history.pop(0)

    def undo(self):
        if self.mask_history:
            self.mask = self.mask_history.pop()
            self.update_display()
            print("Undo completed")
        else:
            print("Nothing to undo")

    def mouse_callback(self, event, x, y, flags, param):
        if self.tool_mode in ["brush", "erase"]:
            self._handle_brush(event, x, y)
        else:
            self._handle_shape(event, x, y)

    def _handle_brush(self, event, x, y):
        # Determine color based on tool mode
        color = 0 if self.tool_mode == "erase" else 255

        # Start drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            self.save_to_history()
            self.drawing = True
            self.last_point = (x, y)
            cv2.circle(self.mask, (x, y), self.brush_size, color, -1)
            self.event_counter += 1
            if self.event_counter % self.update_frequency == 0:
                self.update_display()

        # Continue drawing
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Use thickness based on brush size for smoother lines
            if self.last_point:  # Ensure we have a last point
                # Draw a line between the last point and current point
                cv2.line(self.mask, self.last_point, (x, y), color, self.brush_size * 2)
                # Also draw a circle at the current point to avoid gaps in fast movements
                cv2.circle(self.mask, (x, y), self.brush_size, color, -1)
            self.last_point = (x, y)
            self.event_counter += 1
            if self.event_counter % self.update_frequency == 0:
                self.update_display()

        # Stop drawing
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.update_display()  # Always update display when stopping

    def _handle_shape(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.save_to_history()
            self.shape_start_point = (x, y)
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Show preview of shape
            self.update_display()
            self._draw_shape_preview(self.shape_start_point, (x, y))
            cv2.imshow(self.window_name, self.display_image)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            if self.shape_start_point:
                self._draw_shape_on_mask(self.shape_start_point, (x, y))
                self.shape_start_point = None
                self.update_display()

    def _get_shape_params(self, start, end):
        """Calculate shape parameters based on tool mode."""
        if self.tool_mode == "rectangle":
            return {"type": "rectangle", "start": start, "end": end}
        elif self.tool_mode == "square":
            # Force square aspect ratio
            width = abs(end[0] - start[0])
            height = abs(end[1] - start[1])
            size = max(width, height)
            # Preserve direction of drag
            end_x = start[0] + size if end[0] > start[0] else start[0] - size
            end_y = start[1] + size if end[1] > start[1] else start[1] - size
            return {"type": "rectangle", "start": start, "end": (end_x, end_y)}
        elif self.tool_mode == "oval":
            center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            axes = (abs(end[0] - start[0]) // 2, abs(end[1] - start[1]) // 2)
            return {"type": "ellipse", "center": center, "axes": axes}
        elif self.tool_mode == "triangle":
            return {"type": "polygon", "points": self._get_triangle_points(start, end)}
        return None

    def _draw_shape_preview(self, start, end):
        overlay = self.display_image.copy()
        color = (0, 0, 255)  # Red preview

        params = self._get_shape_params(start, end)
        if not params:
            return

        if params["type"] == "rectangle":
            cv2.rectangle(overlay, params["start"], params["end"], color, -1)
        elif params["type"] == "ellipse" and params["axes"][0] > 0 and params["axes"][1] > 0:
            cv2.ellipse(overlay, params["center"], params["axes"], 0, 0, 360, color, -1)
        elif params["type"] == "polygon":
            cv2.fillPoly(overlay, [params["points"]], color)

        cv2.addWeighted(overlay, 0.3, self.display_image, 0.7, 0, self.display_image)

    def _draw_shape_on_mask(self, start, end):
        params = self._get_shape_params(start, end)
        if not params:
            return

        if params["type"] == "rectangle":
            cv2.rectangle(self.mask, params["start"], params["end"], 255, -1)
        elif params["type"] == "ellipse" and params["axes"][0] > 0 and params["axes"][1] > 0:
            cv2.ellipse(self.mask, params["center"], params["axes"], 0, 0, 360, 255, -1)
        elif params["type"] == "polygon":
            cv2.fillPoly(self.mask, [params["points"]], 255)

    def _get_triangle_points(self, start, end):
        # Create isosceles triangle
        base_center = ((start[0] + end[0]) // 2, end[1])
        apex = (base_center[0], start[1])
        left = (start[0], end[1])
        right = (end[0], end[1])
        return np.array([apex, left, right], np.int32)

    def update_display(self):
        # Create a copy of the original image
        self.display_image = self.original_image.copy()

        # Create a colored overlay for the mask (semi-transparent red)
        self.overlay[:] = 0  # Reset overlay
        self.overlay[self.mask > 0] = [0, 0, 255]  # Red overlay

        # Apply the overlay
        alpha = 0.5  # Transparency level
        cv2.addWeighted(self.overlay, alpha, self.display_image, 1 - alpha, 0, self.display_image)

        # Draw tool info in the corner
        mode_display = {
            "brush": "Brush",
            "erase": "Erase",
            "rectangle": "Rectangle",
            "square": "Square",
            "oval": "Oval",
            "triangle": "Triangle",
        }
        text = f"Tool: {mode_display.get(self.tool_mode, self.tool_mode)} | "
        if self.tool_mode in ["brush", "erase"]:
            text += f"Size: {self.brush_size} (1-9) | "
        text += f"History: [{len(self.mask_history)}]"

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

            # Check for brush size hotkeys (1-9)
            if key_char in self.BRUSH_SIZES and self.tool_mode in ["brush", "erase"]:
                self.set_brush_size(key_char)

            # Tool selection
            elif key == ord("b"):
                self.tool_mode = "brush"
                print("Switched to brush tool")
                self.update_display()
            elif key == ord("e"):
                self.tool_mode = "erase"
                print("Switched to erase tool")
                self.update_display()
            elif key == ord("r"):
                self.tool_mode = "rectangle"
                print("Switched to rectangle tool")
                self.update_display()
            elif key == ord("q"):
                self.tool_mode = "square"
                print("Switched to square tool")
                self.update_display()
            elif key == ord("o"):
                self.tool_mode = "oval"
                print("Switched to oval tool")
                self.update_display()
            elif key == ord("t"):
                self.tool_mode = "triangle"
                print("Switched to triangle tool")
                self.update_display()

            # Undo (press 'u')
            elif key == ord("u"):
                self.undo()

            # Save mask (press 's')
            elif key == ord("s"):
                self.save_mask(self.mask_output_path)

            # Quit (press ESC)
            elif key == 27:
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
