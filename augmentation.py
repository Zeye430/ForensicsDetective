import cv2
import random
import numpy as np
from pathlib import Path

def add_gaussian_noise(image, sigma: float = None):

    if sigma is None:
        sigma = random.uniform(5, 20)

    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def apply_jpeg_compression(image, quality: int = None):

    if quality is None:
        quality = random.randint(20, 80)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise RuntimeError("JPEG encoding failed")

    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return decoded

def downsample_dpi(image, target_dpi: int = None):

    if target_dpi is None:
        target_dpi = random.choice([150, 72])

    if target_dpi not in (150, 72):
        raise ValueError("target_dpi must be 150 or 72")

    h, w = image.shape
    scale = target_dpi / 300.0

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored

def random_crop(image, pct_range: tuple[float, float] = (0.01, 0.03)):

    h, w = image.shape

    top = int(h * random.uniform(*pct_range))
    bottom = int(h * random.uniform(*pct_range))
    left = int(w * random.uniform(*pct_range))
    right = int(w * random.uniform(*pct_range))

    cropped = image[top:h - bottom, left:w - right]

    if cropped.size == 0:
        raise RuntimeError("Cropping removed entire image")

    restored = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return restored

def reduce_bit_depth(image):

    reduced = (image // 16) * 16
    return reduced.astype(np.uint8)
    

def load_image(path):

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    return image

def save_image(image, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise RuntimeError(f"Failed to save image: {output_path}")

def augment_image(image):

    return {
        "gaussian_noise": add_gaussian_noise(image),
        "jpeg_compression": apply_jpeg_compression(image),
        "downsample_dpi": downsample_dpi(image),
        "random_crop": random_crop(image),
        "reduce_bit_depth": reduce_bit_depth(image)
    }


def process_directory(input_dir, output_dir):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    image_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    print(f"Found {len(image_files)} images in {input_dir}")

    for i, image_path in enumerate(image_files, 1):
        
        image = load_image(image_path)

        save_image(image, output_dir / "original" / image_path.name)

        augmented_images = augment_image(image)
        for aug_name, aug_image in augmented_images.items():
            save_image(aug_image, output_dir / aug_name / image_path.name)
        
        if i % 25 == 0 or i == len(image_files):
            print(f"Processed {i}/{len(image_files)} images")

def main():
    print("starting augmentation...")

    process_directory("word_pdfs_png", "augmented_images/word")
    process_directory("google_docs_pdfs_png", "augmented_images/google")
    process_directory("python_pdfs_png", "augmented_images/python")


if __name__ == "__main__":
    main()