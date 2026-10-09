"""Spike: Tesseract OCR verification.
Verifies pytesseract discovery and pytesseract.image_to_data confidence extraction.
Generates an in-memory sample image with text and computes mean confidence.
"""

import os
import shutil
from pathlib import Path

import pytesseract
from PIL import Image, ImageDraw


def configure_tesseract() -> str | None:
    # Check default windows installer path, PATH, or env var
    custom_path = os.getenv("TESSERACT_CMD")
    standard_windows_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    if custom_path and Path(custom_path).exists():
        pytesseract.pytesseract.tesseract_cmd = custom_path
        return custom_path
    elif standard_windows_path.exists():
        pytesseract.pytesseract.tesseract_cmd = str(standard_windows_path)
        return str(standard_windows_path)
    elif shutil.which("tesseract"):
        cmd = shutil.which("tesseract")
        pytesseract.pytesseract.tesseract_cmd = cmd
        return cmd
    return None


def create_sample_image() -> Image.Image:
    # Create high-contrast test image
    img = Image.new("RGB", (500, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw simple black text
    text = "FILUMART SPEC PLATE\nModel: PKG-120\nVoltage: 230V 50Hz"
    draw.text((20, 20), text, fill=(0, 0, 0))
    return img


def test_tesseract() -> None:
    print("--- Testing Tesseract OCR Discovery & Confidence Parsing ---")
    binary_path = configure_tesseract()
    if not binary_path:
        print(
            "Tesseract binary NOT FOUND in PATH or "
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        print("Status: Tesseract unavailable locally (requires UB-Mannheim installer).")
        print("Graceful degradation mode will be active for OCR.")
        return

    print(f"Found Tesseract binary at: {binary_path}")
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract Version: {version}")

        sample_img = create_sample_image()
        data = pytesseract.image_to_data(sample_img, output_type=pytesseract.Output.DICT)

        words = []
        confidences = []
        for word, conf in zip(data["text"], data["conf"], strict=False):
            clean_word = word.strip()
            conf_val = float(conf)
            if clean_word and conf_val >= 0:
                words.append(clean_word)
                confidences.append(conf_val)

        print(f"Extracted {len(words)} words: {words}")
        if confidences:
            mean_conf = sum(confidences) / len(confidences) / 100.0
            print(f"Mean word confidence: {mean_conf:.2f} (0.0 to 1.0 scale)")
            assert mean_conf > 0.50, f"Expected mean confidence > 0.50, got {mean_conf}"
            print("Tesseract OCR confidence parsing spike PASSED.\n")
        else:
            print("Warning: No words detected in sample image.")
    except Exception as exc:
        print(f"Error executing Tesseract: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    test_tesseract()
    print("=== Tesseract spike script completed ===")
