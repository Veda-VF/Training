# Screenshot Analyzer Tool (TOPIC: Multi-Modal ai)

# Install dependencies:
# python -m pip install opencv-python pytesseract pillow
# You also need to install the Tesseract OCR engine separately; installing pytesseract alone is not enough. 
# On Windows, install Tesseract OCR and note its executable location. The script below uses that path by default if it exists. If yours is different, update TESSERACT_PATH

import os
import re
import json
from datetime import datetime

import cv2
import pytesseract
from pytesseract import Output
from PIL import Image

# -------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------

# Change this path if Tesseract is installed elsewhere.
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.isfile(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# -------------------------------------------------
# 2. LOAD AND VALIDATE THE IMAGE
# -------------------------------------------------

def load_screenshot(file_path):
    """Load the image and return its dimensions and OpenCV data."""

    if not os.path.isfile(file_path):
        raise FileNotFoundError("The image file does not exist.")

    image = cv2.imread(file_path)

    if image is None:
        raise ValueError("Could not read the image. Check its format.")

    height, width = image.shape[:2]

    return image, width, height

# -------------------------------------------------
# 3. EXTRACT TEXT AND ITS POSITION USING OCR
# -------------------------------------------------

def extract_text(image):
    """Extract visible text and its bounding boxes."""

    # Convert to grayscale to help OCR.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tesseract returns each detected word and its coordinates.
    data = pytesseract.image_to_data(
        gray,
        output_type=Output.DICT,
        config="--psm 11"
    )

    extracted = []

    for i, word in enumerate(data["text"]):
        word = word.strip()

        if not word:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if confidence < 0:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        extracted.append({
            "text": word,
            "bounding_box": {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            },
            "center": {
                "x": x + w // 2,
                "y": y + h // 2
            },
            "confidence": round(confidence, 2)
        })

    return extracted


# -------------------------------------------------
# 4. CLASSIFY THE SCREENSHOT
# -------------------------------------------------

def classify_screenshot(image, text_items):
    """
    Use OCR keywords and image features to estimate
    screenshot categories. Multiple labels are possible.
    """

    full_text = " ".join(
        item["text"] for item in text_items
    ).lower()

    categories = []
    evidence = {}

    # Common programming keywords and syntax.
    code_patterns = [
        r"\bdef\b", r"\bclass\b", r"\bimport\b",
        r"\breturn\b", r"\bfunction\b", r"\bconst\b",
        r"\bpublic\b", r"\bprint\b", r"\bSELECT\b",
        r"=>", r"\{\}", r";"
    ]

    code_score = sum(
        bool(re.search(pattern, full_text, re.IGNORECASE))
        for pattern in code_patterns
    )

    if code_score >= 2:
        categories.append("code")
        evidence["code"] = (
            f"Found {code_score} programming-related patterns."
        )

    # Browser and webpage clues.
    webpage_words = [
        "https://", "www.", "sign in", "sign up",
        "home", "about us", "contact", "search",
        "privacy policy"
    ]

    webpage_matches = [
        word for word in webpage_words if word in full_text
    ]

    if len(webpage_matches) >= 2 or "https://" in full_text:
        categories.append("webpage")
        evidence["webpage"] = webpage_matches

    # Chart clues from visible labels.
    chart_words = [
        "axis", "legend", "revenue", "sales",
        "percentage", "average", "total", "trend",
        "series", "count"
    ]

    chart_matches = [
        word for word in chart_words if word in full_text
    ]

    # Detect substantial straight lines that might be axes or gridlines.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=3.14159 / 180,
        threshold=80,
        minLineLength=max(30, min(image.shape[:2]) // 5),
        maxLineGap=10
    )

    line_count = 0 if lines is None else len(lines)

    if len(chart_matches) >= 2 and line_count >= 2:
        categories.append("chart")
        evidence["chart"] = {
            "chart_related_words": chart_matches,
            "long_lines_detected": line_count
        }

    # UI clues: control labels and application terminology.
    ui_words = [
        "button", "submit", "cancel", "save",
        "settings", "menu", "login", "password",
        "username", "email", "dashboard", "profile",
        "search", "next", "back", "upload", "home", "contact"
    ]

    ui_matches = [
        word for word in ui_words if word in full_text
    ]

    if len(ui_matches) >= 2:
        categories.append("UI")
        evidence["UI"] = {
            "interface_terms": ui_matches
        }

    # A fallback prevents an empty category list.
    if not categories:
        categories.append("unknown_or_general_screenshot")
        evidence["classification"] = (
            "No strong category indicators were detected."
        )

    return {
        "categories": categories,
        "evidence": evidence,
        "classification_method": "OCR keyword and image-feature heuristics"
    }


# -------------------------------------------------
# 5. IDENTIFY LIKELY UI ELEMENTS
# -------------------------------------------------

def identify_ui_elements(text_items, image_width, image_height):
    """
    Infer possible UI controls from visible text.
    These are guesses, not verified semantic elements.
    """

    elements = []

    button_labels = {
        "submit", "save", "cancel", "login", "sign in",
        "sign up", "search", "next", "back", "send",
        "continue", "delete", "upload", "download",
        "ok", "close", "apply"
    }

    input_hints = {
        "email", "password", "username", "search",
        "enter name", "enter email"
    }

    navigation_labels = {
        "home", "about", "contact", "settings",
        "profile", "dashboard", "menu"
    }

    for item in text_items:
        text = item["text"].strip()
        lower = text.lower()
        box = item["bounding_box"]

        element_type = None

        if lower in button_labels:
            element_type = "possible_button"

        elif lower in input_hints or lower.startswith("enter "):
            element_type = "possible_input_field_or_placeholder"

        elif lower in navigation_labels:
            element_type = "possible_navigation_element"

        elif lower in {"☰", "≡", "⋮", "..."}:
            element_type = "possible_menu_control"

        if element_type:
            elements.append({
                "type": element_type,
                "visible_label": text,
                "bounding_box": box,
                "confidence": "heuristic"
            })

    # Detect large rectangular outlines as possible controls.
    # This can also detect cards, panels, and other containers.
    image = None

    return elements


# -------------------------------------------------
# 6. GENERATE ACCESSIBILITY-FRIENDLY ALT TEXT
# -------------------------------------------------

def generate_alt_text(classification, text_items, width, height):
    """Create a short description from the detected content."""

    categories = classification["categories"]

    # Keep the most readable text snippets.
    visible_text = [
        item["text"] for item in text_items[:12]
    ]

    if "code" in categories:
        description = "Screenshot containing programming code."
    elif "chart" in categories:
        description = "Screenshot containing a possible chart or graph."
    elif "webpage" in categories:
        description = "Screenshot of a webpage."
    elif "UI" in categories:
        description = "Screenshot of an application interface."
    else:
        description = "Screenshot containing visible text or other content."

    description += f" Image dimensions: {width} by {height} pixels."

    if visible_text:
        description += " Visible text includes: " + ", ".join(
            visible_text[:8]
        ) + "."

    return description


# -------------------------------------------------
# 7. ANALYZE THE SCREENSHOT
# -------------------------------------------------

def analyze_screenshot(file_path):
    image, width, height = load_screenshot(file_path)

    # OCR extracts text and positions.
    text_items = extract_text(image)

    # Identify likely screenshot category/categories.
    classification = classify_screenshot(image, text_items)

    # Only infer UI controls when the screenshot looks like a UI
    # or webpage. These remain heuristic predictions.
    ui_elements = []

    if "UI" in classification["categories"] or \
       "webpage" in classification["categories"]:
        ui_elements = identify_ui_elements(
            text_items, width, height
        )

    alt_text = generate_alt_text(
        classification, text_items, width, height
    )

    report = {
        "file": os.path.basename(file_path),
        "analysis_time": datetime.now().isoformat(timespec="seconds"),
        "image": {
            "width": width,
            "height": height,
            "format": os.path.splitext(file_path)[1].lower()
        },
        "classification": classification,
        "text_extraction": {
            "method": "Tesseract OCR",
            "word_count": len(text_items),
            "items": text_items
        },
        "ui_elements": ui_elements,
        "accessibility": {
            "alt_text": alt_text,
            "note": (
                "Automatically generated alt text should be reviewed "
                "for accuracy and accessibility."
            )
        }
    }

    return report


# -------------------------------------------------
# 8. MAIN PROGRAM
# -------------------------------------------------

def main():
    print("Screenshot Analysis Tool")

    file_path = input(
        "Enter the screenshot image path: "
    ).strip().strip('"')

    try:
        report = analyze_screenshot(file_path)

    except FileNotFoundError as error:
        print("File error:", error)
        return

    except pytesseract.TesseractNotFoundError:
        print(
            "Error: Tesseract OCR was not found. Install Tesseract "
            "and update TESSERACT_PATH."
        )
        return

    except Exception as error:
        print("Could not analyze the screenshot:", error)
        return

    # Display a readable summary.
    print("\n========== ANALYSIS REPORT ==========")
    print("Image:", report["file"])
    print("Dimensions:", report["image"]["width"],
          "x", report["image"]["height"])
    print("Categories:",
          ", ".join(report["classification"]["categories"]))
    print("Words extracted:",
          report["text_extraction"]["word_count"])
    print("Likely UI elements:", len(report["ui_elements"]))
    print("Alt text:", report["accessibility"]["alt_text"])

    print("\nVisible text preview:")
    for item in report["text_extraction"]["items"][:15]:
        print(
            f"- {item['text']} "
            f"at {item['bounding_box']}"
        )

    # Ask before creating a JSON file.
    answer = input("\nSave the full report as JSON? (yes/no): ").strip().lower()

    if answer in {"yes", "y"}:
        default_name = os.path.splitext(file_path)[0] + "_report.json"
        output_path = input(
            f"Output JSON path [{default_name}]: "
        ).strip().strip('"')

        if not output_path:
            output_path = default_name

        if not output_path.lower().endswith(".json"):
            output_path += ".json"

        try:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=2, ensure_ascii=False)

            print("Report saved to:", output_path)

        except OSError as error:
            print("Could not save the report:", error)

    else:
        print("JSON report was not saved.")

    print("=====================================")


if __name__ == "__main__":
    main()
