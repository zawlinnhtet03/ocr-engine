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
    MAX_FILE_SIZE
)

# Initialize OCR engine
ocr_engine = initialize_ocr_engine()

# Page configuration
st.set_page_config(
    page_title="Text Extraction & Document Scanner",
    page_icon="🔍",
    # layout="wide"
)

# Load custom CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize session state for storing images and results
def init_session_state():
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'extracted_equations' not in st.session_state:
        st.session_state.extracted_equations = ""
    if 'file_type' not in st.session_state:
        st.session_state.file_type = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None

# Define the toggle_camera function
def toggle_camera():
    """Toggle camera state"""
    st.session_state.camera_on = not st.session_state.camera_on

init_session_state()

# Title and description
st.title("🔍 Text Extraction & Document Scanner")
st.markdown("""
Extract text from images, handwritten notes, or scanned documents using our advanced OCR and document scanning tools.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["HTR", "OCR", "SCANNER"])

# Tab 2: Advanced HWT
with tab1:
    st.header("Advanced Handwritten Text Recognition")
    st.markdown("""
    Upload an image of handwritten text or use your camera to capture text.
    Our advanced algorithms will process and extract the text content.
    """)

    # Input method selection
    # input_method = st.radio("Choose input method:", ["Upload Image", "Use Camera"], key="advanced_hwt_input")

    st.subheader("📤 File Upload")
    uploaded_file1 = st.file_uploader(
        f"Choose a file (Max size: {MAX_FILE_SIZE/1024/1024:.1f}MB)",
        type=['png', 'jpg', 'jpeg', 'pdf', 'docx'],
        key="scan_upload1"
    )
    # Image input handling
    if uploaded_file1 is not None:
        # uploaded_file = st.file_uploader("Upload text image", type=["jpg", "jpeg", "png", 'pdf', 'docx'], key="advanced_hwt_upload")
        # if uploaded_file is not None:
            try:
                st.session_state.image = Image.open(uploaded_file1)
            except Exception as e:
                st.error(f"Error opening image: {str(e)}")
                
    # else:  # Camera input
    #     camera_input = st.camera_input("Take a picture", key="advanced_hwt_camera")
    #     if camera_input is not None:
    #         try:
    #             st.session_state.image = Image.open(camera_input)
    #         except Exception as e:
    #             st.error(f"Error capturing image: {str(e)}")

    # Process image and display results
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
                    # Enhance image for OCR
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
                        # Perform OCR
                        extracted_text = perform_ocr_recognition(st.session_state.processed_image, ocr_engine)

                        # Display results
                        st.markdown("### Extracted Text")
                        if "Error" in extracted_text:
                            st.error(extracted_text)
                        else:
                            st.success("Text extraction completed!")
                            st.markdown(f"""
                            <div class="result-section">
                                <p style='font-size: 18px;'>{extracted_text}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"An error occurred during text extraction: {str(e)}")

# Tab 3: Scan
with tab2:
    st.header("Text Extractor & Document Scanner")
    st.markdown("""
    Extract text from various document formats or capture using your camera.
    Supported formats:
    - 📸 Images (PNG, JPG)
    - 📄 PDF documents
    - 📎 Word documents (DOCX)
    """)

    # Add a more prominent size limit warning
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
        file_bytes = uploaded_file.getvalue()  # Read file once
        file_size = len(file_bytes) / (1024 * 1024)  # Size in MB

        if file_size > MAX_FILE_SIZE/(1024*1024):
            st.error(f"File size ({file_size:.1f}MB) exceeds the maximum limit of {MAX_FILE_SIZE/1024/1024:.1f}MB")
        else:
            # Display image if it's an image file
            if uploaded_file.type.startswith('image'):
                try:
                    image = Image.open(io.BytesIO(file_bytes))
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    st.session_state.uploaded_image = image
                except Exception as e:
                    st.error(f"Error displaying image: {str(e)}")

            with st.spinner("Processing file..."):
                result = process_uploaded_file(file_bytes, uploaded_file.name)
                st.session_state.extracted_text = result['text']
                st.session_state.extracted_equations = result['equations']

    # Keyword search section
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
            st.session_state.extracted_text = ""
            st.session_state.extracted_equations = ""
            st.session_state.file_type = None
            st.session_state.uploaded_image = None

    # Results section
    if st.session_state.extracted_text or st.session_state.extracted_equations:
        st.subheader("📄 Extracted Content")

        # Display extracted text with better visibility and scrolling
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
            ">{st.session_state.extracted_text}</div>
            """,
            unsafe_allow_html=True
        )

        # Display equations if found
        if st.session_state.extracted_equations and st.session_state.extracted_equations != "No equations detected":
            st.markdown("### Detected Equations")
            st.latex(st.session_state.extracted_equations)
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
                ">{st.session_state.extracted_equations}</div>
                """,
                unsafe_allow_html=True
            )

        # Display keyword matches if any
        if keywords:
            st.subheader("🎯 Matching Results")
            keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
            matches = extract_relevant_sentences(
                st.session_state.extracted_text,
                keyword_list
            )

            matches_container = st.container()
            with matches_container:
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

        # Copy buttons
        col5, col6 = st.columns(2)
        with col5:
            if st.button("📋 Copy Text", key="scan_copy_text_btn"):
                st.code(st.session_state.extracted_text)
                st.success("Text copied to clipboard!")

        with col6:
            if st.session_state.extracted_equations and st.button("📋 Copy Equations", key="scan_copy_equations_btn"):
                st.code(st.session_state.extracted_equations)
                st.success("Equations copied to clipboard!")
                
with tab3:
    st.subheader("📸 Camera Capture")
    camera_btn = st.button(
        "Toggle Camera",
        on_click=toggle_camera,
        help="Turn camera on/off",
        key="scan_camera_toggle"
    )

    if st.session_state.camera_on:
        video_capture = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        capture_btn = st.button("Capture Image", key="scan_capture_btn")

        try:
            ret, frame = video_capture.read()
            if ret:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB")

                if capture_btn:
                    # Convert to PIL Image
                    pil_image = Image.fromarray(rgb_frame)
                    st.session_state.uploaded_image = pil_image
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    result = process_uploaded_file(
                        buf.getvalue(),
                        "camera_capture.png"
                    )
                    st.session_state.extracted_text = result['text']
                    st.session_state.extracted_equations = result['equations']
                    st.session_state.camera_on = False
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
        finally:
            video_capture.release()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by Advanced Computer Vision & OCR Technology</p>
</div>
""", unsafe_allow_html=True)