import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
from utils import (
    enhance_image_for_ocr,
    initialize_ocr_engine,
    perform_ocr_recognition,
    process_uploaded_file,
    extract_relevant_sentences,
    extract_text_with_camera,
    MAX_FILE_SIZE
)

# Page configuration
st.set_page_config(
    page_title="Text Extraction & Document Scanner",
    page_icon="🔍",
)

# if st.button("Clear All", key="clear_all_btn"):
#     reset_session_state()
#     st.experimental_rerun()

# Load custom CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state for storing images and results
def init_session_state():
    st.session_state.setdefault('image', None)  # HTR original image
    st.session_state.setdefault('processed_image', None)  # HTR processed image
    st.session_state.setdefault('htr_extracted_text', "")  # HTR extracted text
    st.session_state.setdefault('htr_extracted_equations', "")  # HTR extracted equations
    st.session_state.setdefault('file_type', None)  # OCR file type
    st.session_state.setdefault('uploaded_image', None)  # OCR uploaded image
    st.session_state.setdefault('ocr_extracted_text', "")  # OCR extracted text
    st.session_state.setdefault('ocr_extracted_equations', "")  # OCR extracted equations
    st.session_state.setdefault('camera_active', True)  # SCANNER camera state
    st.session_state.setdefault('captured_image', None)  # SCANNER captured image
    st.session_state.setdefault('scanner_extracted_text', "")  # SCANNER extracted text
    st.session_state.setdefault('scanner_extracted_equations', "")  # SCANNER extracted equations

# def reset_session_state():
#     st.session_state.image = None
#     st.session_state.processed_image = None
#     st.session_state.htr_extracted_text = ""
#     st.session_state.htr_extracted_equations = ""
#     st.session_state.file_type = None
#     st.session_state.uploaded_image = None
#     st.session_state.ocr_extracted_text = ""
#     st.session_state.ocr_extracted_equations = ""
#     st.session_state.camera_active = True
#     st.session_state.captured_image = None
#     st.session_state.scanner_extracted_text = ""
#     st.session_state.scanner_extracted_equations = ""

init_session_state()

# Title and description
st.title("🔍 Text Extraction & Document Scanner")
st.markdown("""
Extract text from images, handwritten notes or scanned documents using our advanced OCR and document scanning tools.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["OCR", "HTR", "SCANNER"])

with tab1:
    st.header("📄 Text Extractor & Document Scanner")
    st.markdown("""
    Extract text from various document formats.
    Supported formats:
    - 📸 Images (PNG, JPG)
    - 📄 PDF documents
    - 📎 Word documents (DOCX)
    """)

    st.warning(f"""
    📢 **File Size Limit: {MAX_FILE_SIZE/1024/1024:.1f}MB**

    Recommended file sizes for best performance:
    - Images (PNG, JPG): 2-4MB
    - PDF documents: 1-5MB
    - Word documents (DOCX): 1-3MB
    - Text files (TXT): < 1MB
    """)

    st.subheader("📤 File Upload")
    uploaded_file = st.file_uploader(
        f"Choose a file (Max size: {MAX_FILE_SIZE/1024/1024:.1f}MB)",
        type=['png', 'jpg', 'jpeg', 'pdf', 'docx'],
        key="scan_upload"
    )

    if uploaded_file:
        st.session_state.file_type = uploaded_file.type
        file_bytes = uploaded_file.getvalue()
        file_size = len(file_bytes) / (1024 * 1024)

        if file_size > MAX_FILE_SIZE / (1024 * 1024):
            st.error(f"File size ({file_size:.1f}MB) exceeds the maximum limit of {MAX_FILE_SIZE/1024/1024:.1f}MB")
        else:
            if uploaded_file.type.startswith('image'):
                try:
                    image = Image.open(io.BytesIO(file_bytes))
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    st.session_state.uploaded_image = image
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")

            with st.spinner("Processing file..."):
                result = process_uploaded_file(file_bytes, uploaded_file.name)
                st.session_state.ocr_extracted_text = result['text']
                st.session_state.ocr_extracted_equations = result['equations']

    st.subheader("🔍 Text Search")
    col3, col4 = st.columns([3, 1])

    with col3:
        keywords = st.text_input(
            "Enter keywords (comma-separated)",
            help="Enter words to search for in the extracted text",
            key="scan_keywords"
        )

    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        clear_btn = st.button("Clear Results", key="scan_clear_btn")
        if clear_btn:
            st.session_state.ocr_extracted_text = ""
            st.session_state.ocr_extracted_equations = ""
            st.session_state.file_type = None
            st.session_state.uploaded_image = None

    if st.session_state.ocr_extracted_text or st.session_state.ocr_extracted_equations:
        st.subheader("📄 Extracted Content")
        st.markdown("### Text Content")
        st.markdown(
            f"""
            <div style="
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
                background-color: white;
                height: 300px;
                overflow-y: auto;
                font-family: monospace;
                white-space: pre-wrap;
                line-height: 1.4;
            ">{st.session_state.ocr_extracted_text}</div>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.ocr_extracted_equations and st.session_state.ocr_extracted_equations != "No equations detected":
            st.markdown("### Detected Equations")
            st.latex(st.session_state.ocr_extracted_equations)
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 10px;
                    background-color: white;
                    max-height: 200px;
                    overflow-y: auto;
                    font-family: monospace;
                    white-space: pre-wrap;
                    line-height: 1.4;
                ">{st.session_state.ocr_extracted_equations}</div>
                """,
                unsafe_allow_html=True
            )

        if keywords:
            st.subheader("🎯 Matching Results")
            keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
            matches = extract_relevant_sentences(
                st.session_state.ocr_extracted_text,
                keyword_list
            )
            for match in matches:
                st.markdown(
                    f"""
                    <div style="
                        margin: 10px 0;
                        padding: 15px;
                        background-color: #f8f9fa;
                        border: 1px solid #dee2e6;
                        border-radius: 5px;
                    ">
                        <div style="color: #6c757d; font-size: 0.9em; margin-bottom: 5px;">
                            {match['context']}
                        </div>
                        <div style="margin: 10px 0;">
                            {match['sentence']}
                        </div>
                        <div style="color: #28a745; font-size: 0.9em;">
                            Keywords found: {', '.join(match['keywords'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        col5, col6 = st.columns(2)
        with col5:
            if st.button("📋 Copy Text", key="ocr_copy_text_btn"):
                st.code(st.session_state.ocr_extracted_text)
                st.success("Text copied to clipboard!")
        with col6:
            if st.session_state.ocr_extracted_equations and st.button("📋 Copy Equations", key="ocr_copy_equations_btn"):
                st.code(st.session_state.ocr_extracted_equations)
                st.success("Equations copied to clipboard!")

with tab2:
    st.header("✍️ Advanced Handwritten Text Recognition")
    st.markdown("""
    Upload an image of handwritten text.
    """)

    st.subheader("📤 File Upload")
    uploaded_file1 = st.file_uploader(
        f"Choose a file (Max size: {MAX_FILE_SIZE/1024/1024:.1f}MB)",
        type=['png', 'jpg', 'jpeg'],
        key="scan_upload1"
    )

    if uploaded_file1 is not None:
        try:
            st.session_state.image = Image.open(uploaded_file1)
        except Exception as e:
            st.error(f"Error opening image: {str(e)}")

    if st.session_state.image is not None:
        st.markdown("### Image Preview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Original Image")
            st.image(st.session_state.image, caption="Original Image", use_container_width=True)

        process_image_clicked = st.button("Process Image", key="process_btn_advanced", help="Enhance the image for OCR")
        extract_text_clicked = st.button("Extract Text", key="extract_btn_advanced", help="Perform OCR on the processed image")

        if process_image_clicked:
            with st.spinner("Enhancing image quality..."):
                try:
                    st.session_state.processed_image = enhance_image_for_ocr(st.session_state.image)
                    if st.session_state.processed_image is not None:
                        st.success("Image processing completed!")
                    else:
                        st.error("Image preprocessing failed. Please try with a clearer image.")
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

        if st.session_state.processed_image is not None:
            with col2:
                st.markdown("#### Preprocessed Image")
                st.image(st.session_state.processed_image, caption="Enhanced for OCR", use_container_width=True)

            if extract_text_clicked:
                with st.spinner("Performing text extraction..."):
                    try:
                        ocr_engine = initialize_ocr_engine()  # Re-initialize for safety
                        if ocr_engine is None:
                            st.error("OCR engine initialization failed. Please check your API key.")
                        else:
                            extracted_text = perform_ocr_recognition(st.session_state.processed_image, ocr_engine)
                            st.session_state.htr_extracted_text = extracted_text
                            st.markdown("### Extracted Text")
                            if "Error" in extracted_text:
                                st.error(extracted_text)
                            else:
                                st.success("Text extraction completed!")
                                st.markdown(f"""
                                <div class="result-section">
                                    <p style='font-size: 18px;'>{st.session_state.htr_extracted_text}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"An error occurred during text extraction: {str(e)}")

with tab3:
    st.header("📸 Document Scanner")
    st.markdown("""
    Capture a document using your camera to extract text.  
    **Tip**: Use your device's back camera for better quality (switch cameras if needed).
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
            <style>
                /* Force camera input to fill the container and increase height */
                .stCamera {
                    width: 100% !important;
                    max-width: 100% !important;
                    height: auto !important;
                    min-height: 70vh !important; /* Increased to make it taller */
                    max-height: 90vh !important; /* Increased to allow more vertical space */
                    margin: 0 !important;
                    padding: 0 !important;
                    display: block !important;
                    box-sizing: border-box !important;
                }
                video {
                    width: 100% !important;
                    max-width: 100% !important;
                    height: auto !important;
                    min-height: 70vh !important; /* Match minimum height */
                    max-height: 90vh !important; /* Match maximum height */
                    object-fit: cover !important; /* Ensure the video fills the space */
                    border-radius: 5px; /* Match preview styling */
                }
                /* Ensure the full-screen content applies to both camera and preview */
                .full-screen-content {
                    width: 100% !important;
                    max-width: 100% !important;
                    padding: 0 !important; /* Remove padding to maximize space */
                    margin: 0 !important;
                    text-align: center;
                    box-sizing: border-box !important;
                }
                /* Target Streamlit's internal camera wrapper (class may vary by version) */
                .st-emotion-cache-1j7x7c6, /* Adjust this class based on inspection */
                .st-ca {
                    width: 100% !important;
                    max-width: 100% !important;
                    margin: 0 !important;
                    padding: 0 !important;
                    min-height: 70vh !important;
                }
                .result-box {
                    width: 100% !important;
                    max-width: 100% !important;
                    padding: 15px;
                    background-color: #f9f9f9;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    overflow-y: auto;
                    font-family: monospace;
                    white-space: pre-wrap;
                    line-height: 1.5;
                    font-size: 14px;
                    margin: 10px 0;
                    max-height: 300px;
                }
            </style>
        """, unsafe_allow_html=True)

        if st.session_state.camera_active:
            st.subheader("Capture Image")
            st.markdown("<div class='full-screen-content' style='min-height: 70vh;'>", unsafe_allow_html=True)  # Inline style for debugging
            camera_image = st.camera_input(
                "Take a picture",
                key="scanner_camera",
                help="Point your camera at the document and capture. Use the back camera for best results."
            )
            st.markdown("</div>", unsafe_allow_html=True)
            if camera_image is not None:
                st.session_state.captured_image = Image.open(camera_image)
                st.session_state.camera_active = False

        if st.session_state.captured_image is not None and not st.session_state.camera_active:
            st.subheader("Preview")
            st.markdown("<div class='full-screen-content'>", unsafe_allow_html=True)
            st.image(st.session_state.captured_image, caption="Captured Document", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Extract Text", key="extract_scanner_btn", help="Extract text from the captured image", use_container_width=True):
                with st.spinner("Extracting text..."):
                    try:
                        extracted_text = extract_text_with_camera(st.session_state.captured_image)
                        st.session_state.scanner_extracted_text = extracted_text
                        st.session_state.scanner_extracted_equations = ""
                    except Exception as e:
                        st.error(f"Error extracting text: {str(e)}")

            if st.session_state.scanner_extracted_text:
                st.subheader("📄 Extracted Text")
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                if "Error" in st.session_state.scanner_extracted_text:
                    st.error(st.session_state.scanner_extracted_text)
                else:
                    st.success("Text extraction completed!")
                    st.write(st.session_state.scanner_extracted_text)
                st.markdown("</div>", unsafe_allow_html=True)

            if st.button("Retake Photo", key="retake_btn", help="Capture a new image", use_container_width=True):
                st.session_state.captured_image = None
                st.session_state.scanner_extracted_text = ""
                st.session_state.camera_active = True

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by Advanced Computer Vision & OCR Technology</p>
</div>
""", unsafe_allow_html=True)
