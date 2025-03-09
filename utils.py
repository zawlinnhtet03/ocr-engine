import streamlit as st
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import streamlit as st
import io
import pytesseract
import re
import fitz
from docx import Document
import tempfile
import os
from base64 import b64encode
from sklearn.decomposition import PCA

# Define maximum file size (5MB)
MAX_FILE_SIZE = 200 * 1024 * 1024 

def enhance_image_for_ocr(image):
    """
    Enhanced preprocessing pipeline for text recognition with focus on clarity
    """
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Determine the number of channels
        if len(img_array.shape) == 2:  # Grayscale image (1 channel)
            gray = img_array
        elif len(img_array.shape) == 3:  # RGB or RGBA image (3 or 4 channels)
            if img_array.shape[2] == 4:  # RGBA image
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            # Enhance contrast before converting to grayscale
            img_array = cv2.convertScaleAbs(img_array, alpha=1.5, beta=0)  # Increase contrast
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported image format: {img_array.shape} channels")

        # Ensure the image data type is uint8
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Apply PCA for dimensionality reduction (works with 2D grayscale)
        pca = PCA(n_components=0.95)  # Retain 95% of variance
        flat_gray = gray.flatten().reshape(1, -1)
        pca_result = pca.fit_transform(flat_gray)
        gray = pca.inverse_transform(pca_result).reshape(gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased clipLimit for better contrast
        enhanced = clahe.apply(gray)

        # Add bilateral filtering to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Sharpen the image
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # Apply adaptive thresholding instead of Otsu's for better handling of varying contrast
        binary = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Remove small noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Convert back to PIL Image
        processed_image = Image.fromarray(cleaned)
        print(f"Enhanced image shape: {np.array(processed_image).shape}, mode: {processed_image.mode}")  # Debug print
        return processed_image

    except Exception as e:
        st.error(f"Error during image preprocessing: {str(e)}")
        return None

def initialize_ocr_engine():
    """
    Initialize OCR processing engine with optimal parameters
    """
    try:
        # Initialize backend OCR engine
        api_key = st.secrets["api_key"]
        genai.configure(api_key=api_key)
        engine = genai.GenerativeModel('gemini-2.0-flash')
        return engine
    except Exception as e:
        st.error(f"Error initializing OCR engine: {str(e)}")
        return None

def perform_ocr_recognition(image, engine):
    """
    Perform OCR text recognition using advanced computer vision techniques
    """
    try:
        if engine is None:
            return "Error: OCR engine initialization failed"

        # Prepare image for OCR processing
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()

        # Perform OCR text extraction
        # Note: Using advanced ML-based OCR for improved accuracy
        result = engine.generate_content([
            "Extract text from this document image, focus on accuracy and character recognition. Return only the extracted text.",
            {'mime_type': 'image/png', 'data': img_byte_array}
        ])

        if result and result.text:
            return result.text.strip()
        else:
            return "No text detected in image"

    except Exception as e:
        error_msg = str(e)
        if "deprecated" in error_msg.lower():
            return "Error: OCR engine needs updating. Please check for software updates."
        return f"OCR processing error: {error_msg}"
  
def detect_equations_with_gemini(image):
    """Use Gemini to detect equations in the image"""
    if not initialize_ocr_engine():
        return "API key not configured"

    try:
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Configure Gemini
        model = genai.GenerativeModel('gemini-2.0-flash')

        # Create the image part for the model
        image_parts = [
            {
                "mime_type": "image/png",
                "data": b64encode(img_byte_arr).decode('utf-8')
            }
        ]

        # Prompt for equation detection
        # prompt = """
        # Please analyze this image and extract any mathematical equations or matrices you find.
        # Return the matrices in a format like [[a b c], [d e f]] or the equations in correct format.
        # If no equations or matrices are found, return 'No equations or matrices detected'.
        # """

        prompt = """
        Please analyze this image and extract any mathematical equations or matrices you find.
        Return only the equations in understandable latex format.
        If no equations are found, return 'No equations detected'.
        """

        # Generate response
        response = model.generate_content([prompt, image_parts[0]])
        equations = response.text.strip()
        return equations if equations else "No equations detected"

    except Exception as e:
        return f"Error detecting equations: {str(e)}"

def check_file_size(file_bytes):
    """Check if file size is within limits"""
    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(f"File size exceeds maximum limit of {MAX_FILE_SIZE/1024/1024:.1f}MB")
    return True

def extract_text_from_image(image):
    """Extract text from PIL Image object"""
    # Convert to numpy array
    image_np = np.array(image)

    # Check the number of channels and convert to grayscale if needed
    if len(image_np.shape) == 2:  # Grayscale image (1 channel)
        gray = image_np
    elif len(image_np.shape) == 3:  # RGB or RGBA image (3 or 4 channels)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unsupported image format: {image_np.shape} channels")

    # Apply noise reduction
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=1)

    # Ensure the image data type is uint8
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Extract text using Tesseract OCR with optimized settings
    text = pytesseract.image_to_string(gray, config='--psm 6 --oem 3')  # PSM 6 for single uniform block, OEM 3 for default Tesseract
    print(f"Raw Tesseract output: '{text.strip()}'")  # Debug print

    # Use Gemini for equation detection if API is configured
    if initialize_ocr_engine():
        equations = detect_equations_with_gemini(image)
    else:
        # Fallback to Tesseract for equation detection with number whitelist
        equations = pytesseract.image_to_string(
            gray,
            config='--psm 6 -c tessedit_char_whitelist=0123456789+-*/()=xyz{}[]'
        )
        print(f"Raw Tesseract equation output: '{equations.strip()}'")  # Debug print

    return {
        'text': text.strip(),
        'equations': equations.strip()
    }

def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF bytes"""
    temp_path = None
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            temp_path = tmp.name

        # Open and process the PDF
        with fitz.open(temp_path) as pdf_document:
            text_blocks = []
            for page in pdf_document:
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))  # Sort by y, then x coordinate
                text_blocks.extend(block[4] for block in blocks)

        return {'text': '\n'.join(text_blocks), 'equations': ''}

    finally:
        # Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def extract_text_from_docx(docx_bytes):
    """Extract text from DOCX bytes"""
    doc = Document(io.BytesIO(docx_bytes))
    # Preserve paragraphs and their order
    text_blocks = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():  # Skip empty paragraphs
            text_blocks.append(paragraph.text.strip())
    return {'text': '\n\n'.join(text_blocks), 'equations': ''}

# def extract_text_from_txt(txt_bytes):
#     """Extract text from TXT bytes"""
#     text = txt_bytes.decode("utf-8")
#     # Preserve line breaks and paragraphs
#     text_blocks = [block.strip() for block in text.split('\n\n')]
#     return {'text': '\n\n'.join(text_blocks), 'equations': ''}

def extract_relevant_sentences(text, keywords):
    """Extract sentences containing keywords with context"""
    if not text or not keywords:
        return []

    # Split text into paragraphs first
    paragraphs = text.split('\n\n')
    matches = []

    for para_idx, paragraph in enumerate(paragraphs, 1):
        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)

        for sent_idx, sentence in enumerate(sentences):
            if any(word.lower() in sentence.lower() for word in keywords):
                # Add paragraph context
                context = f"Paragraph {para_idx}, Sentence {sent_idx + 1}"
                matches.append({
                    'context': context,
                    'sentence': sentence.strip(),
                    'keywords': [word for word in keywords if word.lower() in sentence.lower()]
                })

    return matches if matches else [{"context": "No matches", "sentence": "No relevant sentences found.", "keywords": []}]

def process_uploaded_file(file_bytes, filename):
    """Process uploaded file based on its type"""
    filename = filename.lower()

    try:
        # Check file size
        check_file_size(file_bytes)

        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(io.BytesIO(file_bytes))
            return extract_text_from_image(image)
        elif filename.endswith('.pdf'):
            return extract_text_from_pdf(file_bytes)
        elif filename.endswith('.docx'):
            return extract_text_from_docx(file_bytes)
        # elif filename.endswith('.txt'):
        #     return extract_text_from_txt(file_bytes)
        else:
            return {"text": "Unsupported file format", "equations": ""}

    except ValueError as ve:
        return {"text": str(ve), "equations": ""}
    except Exception as e:
        return {"text": f"Error processing {filename}: {str(e)}", "equations": ""}

def extract_text_with_camera(image):
    """Extract text from an image using Gemini API."""
    try:
        # Convert PIL Image to bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # Prepare the image for Gemini (Gemini expects a file-like object or bytes)
        model = genai.GenerativeModel('gemini-1.5-flash')  # Use the appropriate vision model
        prompt = "Extract all text from this image. Returned only the extracted text."
        
        # Send image and prompt to Gemini
        response = model.generate_content([prompt, {"mime_type": "image/png", "data": image_bytes}])
        
        # Return the extracted text
        return response.text
    except Exception as e:
        return f"Error extracting text: {str(e)}"
