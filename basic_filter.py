from PIL import Image, ImageFilter, ImageOps, ImageDraw
import numpy as np


def apply_filters(image_path):
    try:
        img = Image.open(image_path).convert("RGB")

        # 1. Edge Detection
        edge_img = img.filter(ImageFilter.FIND_EDGES)
        edge_img.save("output_edge_detected.jpg")

        # 2. Sharpening
        sharpened_img = img.filter(ImageFilter.SHARPEN)
        sharpened_img.save("output_sharpened.jpg")

        # 3. Embossing
        embossed_img = img.filter(ImageFilter.EMBOSS)
        embossed_img.save("output_embossed.jpg")

        # 4. TV Static
        static_img = tv_static(img)
        static_img.save("output_tv_static.jpg")

        # 5. Bright Center Spotlight Gradient with radius 2
        spotlight_img = spotlight_center_gradient(img, radius=2)

        # 6. Draw microphone on bottom half
        final_img = draw_microphone(spotlight_img)
        final_img.save("output_spotlight_with_microphone.jpg")
        final_img.show(title="Spotlight with Microphone")

        print("All filtered images saved.")

    except Exception as e:
        print(f"Error: {e}")


def tv_static(img, noise_level=0.2):
    gray = ImageOps.grayscale(img)
    gray_np = np.array(gray).astype(np.uint8)
    noise = np.random.choice([0, 255], size=gray_np.shape, p=[
                             0.5, 0.5]).astype(np.uint8)
    mask = np.random.rand(*gray_np.shape) < noise_level
    gray_np[mask] = noise[mask]
    return Image.fromarray(gray_np, mode='L')


def spotlight_center_gradient(img, radius=40):
    """Applies a radial gradient: bright center fading to dark edges."""
    w, h = img.size
    img_np = np.array(img).astype(np.float32)

    cx, cy = w // 2, h // 2
    max_dist = np.sqrt(cx**2 + cy**2)

    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    gradient = 1 - np.clip(dist / (max_dist / radius), 0, 1)

    gradient = gradient[:, :, np.newaxis]
    spotlighted = img_np * gradient
    spotlighted = np.clip(spotlighted, 0, 255).astype(np.uint8)

    return Image.fromarray(spotlighted)


def draw_microphone(img):
    """Draw a simple stylized microphone on the bottom half of the image."""
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Microphone dimensions relative to image size
    mic_width = w // 8
    mic_height = h // 4
    mic_x_center = w // 2
    mic_y_bottom = h - h // 10

    # Microphone base rectangle (handle)
    base_top = mic_y_bottom - mic_height // 2
    base_left = mic_x_center - mic_width // 4
    base_right = mic_x_center + mic_width // 4
    base_bottom = mic_y_bottom

    # Draw mic base (rectangle)
    draw.rectangle([base_left, base_top, base_right,
                   base_bottom], fill="gray", outline="black")

    # Microphone head (circle above base)
    head_radius = mic_width // 2
    head_center = (mic_x_center, base_top - head_radius)
    draw.ellipse([head_center[0] - head_radius, head_center[1] - head_radius,
                  head_center[0] + head_radius, head_center[1] + head_radius],
                 fill="silver", outline="black")

    # Mic grill lines (horizontal lines on head)
    grill_lines = 5
    line_spacing = (head_radius * 2) / (grill_lines + 1)
    for i in range(1, grill_lines + 1):
        y = head_center[1] - head_radius + i * line_spacing
        draw.line([(head_center[0] - head_radius, y),
                  (head_center[0] + head_radius, y)], fill="black", width=1)

    # Mic stand line
    stand_top = base_bottom
    stand_bottom = mic_y_bottom + h // 20
    draw.line([(mic_x_center, stand_top), (mic_x_center,
              stand_bottom)], fill="black", width=3)

    return img


if __name__ == "__main__":
    image_path = "My_Beautiful_Luna2.jpg"  # Replace with your image path
    apply_filters(image_path)
