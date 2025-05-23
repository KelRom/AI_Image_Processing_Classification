import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.applications import MobileNetV2

# Suppress TF logs
tf.get_logger().setLevel('ERROR')

# Load pre-trained model
model = MobileNetV2(weights="imagenet")


# ========== Grad-CAM ==========

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# ========== Display Grad-CAM ==========

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)

    plt.imshow(cv2.cvtColor(superimposed_img.astype("uint8"), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Grad-CAM")
    plt.show()


# ========== Heatmap Region Extraction ==========

def get_heatmap_bbox(heatmap, threshold=0.6):
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    _, thresh_map = cv2.threshold(
        np.uint8(255 * heatmap_resized), int(255 * threshold), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest_contour)
    return None


# ========== Occlusion ==========

def apply_occlusion(original_img, bbox, method="black"):
    x, y, w, h = bbox
    occluded_img = original_img.copy()

    if method == "black":
        occluded_img[y:y+h, x:x+w] = 0
    elif method == "blur":
        roi = occluded_img[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (21, 21), 0)
        occluded_img[y:y+h, x:x+w] = roi
    elif method == "pixelate":
        roi = occluded_img[y:y+h, x:x+w]
        temp = cv2.resize(roi, (16, 16), interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        occluded_img[y:y+h, x:x+w] = roi

    return occluded_img


# ========== Display & Save Occluded Variants ==========

def display_occlusions(image_path, heatmap, model):
    original = cv2.imread(image_path)
    original = cv2.resize(original, (224, 224))
    bbox = get_heatmap_bbox(heatmap)
    if bbox is None:
        print("No significant heatmap region found.")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for method in ["black", "blur", "pixelate"]:
        occluded = apply_occlusion(original, bbox, method)

        # Save occluded image
        occluded_filename = f"{base_name}_occluded_{method}.jpg"
        cv2.imwrite(occluded_filename, occluded)
        print(f"Saved: {occluded_filename}")

        # Prepare for prediction
        img_array = cv2.cvtColor(
            occluded, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Show
        plt.imshow(cv2.cvtColor(occluded, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Occlusion: {method.capitalize()}")
        plt.show()

        # Print predictions
        print(f"Top-3 Predictions after {method} occlusion:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")
        print()


# ========== Main Classification Flow ==========

def classify_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        heatmap = make_gradcam_heatmap(img_array, model, 'Conv_1')
        save_and_display_gradcam(image_path, heatmap)

        display_occlusions(image_path, heatmap, model)

    except Exception as e:
        print(f"Error processing image: {e}")


# ========== Entry Point ==========

if __name__ == "__main__":
    image_path = "My_Beautiful_Luna2.jpg"
    classify_image(image_path)
