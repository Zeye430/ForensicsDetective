import cv2
import random
import numpy as np
from pathlib import Path


def add_gaussian_noise(image, sigma=None):
    if sigma is None:
        sigma = random.uniform(5, 20)
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_jpeg_compression(image, quality=None):
    if quality is None:
        quality = random.randint(20, 80)
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def downsample_dpi(image, target_dpi=None):
    if target_dpi is None:
        target_dpi = random.choice([150, 72])
    h, w = image.shape
    scale = target_dpi / 300.0
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def random_crop(image, pct_range=(0.01, 0.03)):
    h, w = image.shape
    top    = int(h * random.uniform(*pct_range))
    bottom = int(h * random.uniform(*pct_range))
    left   = int(w * random.uniform(*pct_range))
    right  = int(w * random.uniform(*pct_range))
    cropped = image[top:h - bottom, left:w - right]
    if cropped.size == 0:
        raise RuntimeError("crop removed entire image")
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def reduce_bit_depth(image):
    return ((image // 16) * 16).astype(np.uint8)


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to load: {path}")
    return img


def save_image(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"failed to save: {output_path}")


def augment_image(image):
    return {
        "gaussian_noise":   add_gaussian_noise(image),
        "jpeg_compression": apply_jpeg_compression(image),
        "downsample_dpi":   downsample_dpi(image),
        "random_crop":      random_crop(image),
        "reduce_bit_depth": reduce_bit_depth(image),
    }


def process_directory(input_dir, output_dir):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    image_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    print(f"found {len(image_files)} images in {input_dir}")

    for i, img_path in enumerate(image_files, 1):
        image = load_image(img_path)
        save_image(image, output_dir / "original" / img_path.name)
        for aug_name, aug_img in augment_image(image).items():
            save_image(aug_img, output_dir / aug_name / img_path.name)
        if i % 25 == 0 or i == len(image_files):
            print(f"  {i}/{len(image_files)}")


def main():
    process_directory("word_pdfs_png",        "augmented_images/word")
    process_directory("google_docs_pdfs_png", "augmented_images/google")
    process_directory("python_pdfs_png",      "augmented_images/python")


if __name__ == "__main__":
    main()
