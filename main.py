import streamlit as st
import numpy as np
from PIL import Image
import io
from utils import (
    enhance_image_for_ocr,
    initialize_ocr_engine,
    perform_ocr_recognition,
    process_uploaded_file,
    extract_relevant_sentences,
    extract_text_with_camera,
    translate_text,
    MAX_FILE_SIZE
)

# Page configuration
st.set_page_config(
    page_title="Text Extraction & Document Scanner",
    page_icon="🔍",
)

# Load custom CSS (commented out for debugging)
# with open("assets/styles.css") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def init_session_state():
    st.session_state.setdefault('ocr_image', None)  # OCR original image
    st.session_state.setdefault('htr_image', None)  # HTR original image
    st.session_state.setdefault('processed_image', None)  # HTR processed image
    st.session_state.setdefault('htr_extracted_text', "")  # HTR extracted text
    st.session_state.setdefault('htr_extracted_equations', "")  # HTR extracted equations
    st.session_state.setdefault('file_type', None)  # OCR file type
    st.session_state.setdefault('ocr_extracted_text', "")  # OCR extracted text
    st.session_state.setdefault('ocr_extracted_equations', "")  # OCR extracted equations
    st.session_state.setdefault('camera_active', True)  # SCANNER camera state
    st.session_state.setdefault('captured_image', None)  # SCANNER captured image
    st.session_state.setdefault('scanner_extracted_text', "")  # SCANNER extracted text
    st.session_state.setdefault('scanner_extracted_equations', "")  # SCANNER extracted equations

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
                    st.session_state.ocr_image = image
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
            st.session_state.ocr_image = None

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
                height: 500px;
                overflow-y: auto;
                font-family: monospace;
                white-space: pre-wrap;
                line-height: 1.4;
            ">{st.session_state.ocr_extracted_text or 'No text detected'}</div>
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
            st.session_state.htr_image = Image.open(uploaded_file1)
            print(f"HTR: Loaded image with shape {np.array(st.session_state.htr_image).shape}")  # Debug print
        except Exception as e:
            st.error(f"Error opening image: {str(e)}")

    # Add Clear Results button
    col5, col6 = st.columns([3, 1])
    with col5:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer for alignment
    with col6:
        st.markdown("<br>", unsafe_allow_html=True)
        clear_btn_htr = st.button("Clear Results", key="htr_clear_btn")
        if clear_btn_htr:
            st.session_state.htr_image = None
            st.session_state.processed_image = None
            st.session_state.htr_extracted_text = ""
            st.session_state.htr_extracted_equations = ""

    if st.session_state.htr_image is not None:
        st.markdown("### Image Preview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Original Image")
            st.image(st.session_state.htr_image, caption="Original Image", use_container_width=True)

        process_image_clicked = st.button("Process Image", key="process_btn_advanced", help="Enhance the image for OCR")
        extract_text_clicked = st.button("Extract Text", key="extract_btn_advanced", help="Perform OCR on the processed image")

        if process_image_clicked:
            with st.spinner("Enhancing image quality..."):
                try:
                    print(f"HTR: Enhancing image with shape {np.array(st.session_state.htr_image).shape}")  # Debug print
                    st.session_state.processed_image = enhance_image_for_ocr(st.session_state.htr_image)
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
                            extracted_text = perform_ocr_recognition(st.session_state.htr_image, ocr_engine)
                            st.session_state.htr_extracted_text = extracted_text
                            st.markdown("### Extracted Text")
                            if "Error" in extracted_text:
                                st.error(extracted_text)
                            else:
                                st.success("Text extraction completed!")
                                st.markdown(
                                    f"""
                                    <div style="
                                        border: 1px solid #ccc;
                                        border-radius: 5px;
                                        padding: 15px;
                                        background-color: #f9f9f9;
                                        max-height: 300px;
                                        overflow-y: auto;
                                        font-family: monospace;
                                        white-space: pre-wrap;
                                        line-height: 1.5;
                                        font-size: 18px;
                                        margin: 10px 0;
                                    ">
                                        <p>{st.session_state.htr_extracted_text}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    except Exception as e:
                        st.error(f"An error occurred during text extraction: {str(e)}")

# with tab3:
#     st.header("📸 Document Scanner")
#     st.markdown("""
#     Capture a document using your camera to extract text.  
#     **Tip**: Use your device's back camera for better quality (switch cameras if needed).
#     """, unsafe_allow_html=True)

#     # Container for better mobile layout
#     with st.container():
#         # Add inline CSS to control camera input and preview size
#         st.markdown("""
#             <style>
#                 /* Force camera input to fill the container and match preview height */
#                 .stCamera {
#                     width: 100% !important;
#                     max-width: 100% !important;
#                     height: auto !important;
#                     min-height: 70vh !important; /* Increased to match preview height */
#                     max-height: 80vh !important; /* Cap height to avoid overflow */
#                     margin: 0 !important;
#                     padding: 0 !important;
#                     display: block !important;
#                     box-sizing: border-box !important;
#                 }
#                 video {
#                     width: 100% !important;
#                     max-width: 100% !important;
#                     height: auto !important;
#                     min-height: 70vh !important; /* Match minimum height */
#                     max-height: 80vh !important; /* Match maximum height */
#                     object-fit: contain !important; /* Preserve aspect ratio to avoid distortion */
#                     border-radius: 5px; /* Match preview styling */
#                 }
#                 /* Ensure the full-screen content applies to both camera and preview */
#                 .full-screen-content {
#                     width: 100% !important;
#                     max-width: 100% !important;
#                     padding: 10px !important; /* Add padding for better appearance */
#                     margin: 0 !important;
#                     text-align: center;
#                     box-sizing: border-box !important;
#                     min-height: 70vh !important; /* Ensure the container height matches */
#                 }
#                 .st-emotion-cache-1j7x7c6 { /* Target Streamlit's internal camera wrapper */
#                     width: 100% !important;
#                     max-width: 100% !important;
#                     margin: 0 !important;
#                     padding: 0 !important;
#                     min-height: 70vh !important; /* Ensure wrapper height matches */
#                 }
#                 .result-box {
#                     width: 100% !important;
#                     max-width: 100% !important;
#                     padding: 15px;
#                     background-color: #f9f9f9;
#                     border: 1px solid #ccc;
#                     border-radius: 5px;
#                     overflow-y: auto;
#                     font-family: monospace;
#                     white-space: pre-wrap;
#                     line-height: 1.5;
#                     font-size: 14px;
#                     margin: 10px 0;
#                     max-height: 300px;
#                 }
#             </style>
#         """, unsafe_allow_html=True)

#         st.subheader("Capture Image")
#         # Add padding and center the camera input using the full-screen-content class
#         st.markdown("<div class='full-screen-content'>", unsafe_allow_html=True)
#         camera_image = st.camera_input(
#             "Take a picture",
#             key="scanner_camera",
#             help="Point your camera at the document and capture. Use the back camera for best results."
#         )
#         st.markdown("</div>", unsafe_allow_html=True)

#         # Preview and action buttons in a responsive layout
#         if camera_image is not None:
#             # Convert BytesIO to PIL Image
#             image = Image.open(camera_image)
#             st.session_state.captured_image = image  # Store in session state for consistency

#             # Display preview in a centered column
#             col1, col2 = st.columns([3, 1])  # 3 parts for preview, 1 for spacer
#             with col1:
#                 st.subheader("Preview")
#                 st.markdown("<div class='full-screen-content'>", unsafe_allow_html=True)
#                 st.image(image, caption="Captured Document", use_container_width=True)
#                 st.markdown("</div>", unsafe_allow_html=True)

#             # Action buttons in a separate container for mobile tapability
#             with st.container():
#                 st.markdown("<div style='padding: 15px; text-align: center;'>", unsafe_allow_html=True)
#                 if st.button("Extract Text", key="extract_scanner_btn", help="Extract text from the captured image", use_container_width=True):
#                     with st.spinner("Extracting text..."):
#                         # Extract text using Gemini
#                         extracted_text = extract_text_with_camera(image)
                        
#                         # Store in session state
#                         st.session_state.scanner_extracted_text = extracted_text
#                         st.session_state.scanner_extracted_equations = ""  # Gemini might not separate equations

#                         # Display results
#                         st.subheader("📄 Extracted Text")
#                         if "Error" in extracted_text:
#                             st.error(extracted_text)
#                         else:
#                             st.success("Text extraction completed!")
#                             st.markdown(
#                                 f"""
#                                 <div class="result-box">
#                                     {extracted_text}
#                                 </div>
#                                 """,
#                                 unsafe_allow_html=True
#                             )
#                 # Add a Retake Photo button to clear the captured image
#                 if st.button("Retake Photo", key="retake_btn", help="Capture a new image", use_container_width=True):
#                     st.session_state.captured_image = None
#                     st.session_state.scanner_extracted_text = ""
#                     st.session_state.scanner_extracted_equations = ""
#                 st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.header("📸 Document Scanner")
    st.markdown("""
    Capture a document using your camera to extract text.  
    **Tip**: Use your device's back camera for better quality (switch cameras if needed).
    """, unsafe_allow_html=True)

    # Container for better mobile layout
    with st.container():
        st.subheader("Capture Image")
        # Add padding and center the camera input
        st.markdown("<div style='padding: 10px; text-align: center;'>", unsafe_allow_html=True)
        camera_image = st.camera_input(
            "Take a picture",
            key="scanner_camera",
            help="Point your camera at the document and capture. Use the back camera for best results."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Preview and action buttons in a responsive layout
        if camera_image is not None:
            # Convert BytesIO to PIL Image
            image = Image.open(camera_image)

            # Display preview in a centered column
            col1, col2 = st.columns([3, 1])  # 3 parts for preview, 1 for spacer
            with col1:
                st.subheader("Preview")
                st.image(image, caption="Captured Document", use_container_width=True)

            # Action button in a separate column for mobile tapability
            with st.container():
                st.markdown("<div style='padding: 15px; text-align: center;'>", unsafe_allow_html=True)
                if st.button("Extract Text", key="extract_scanner_btn", help="Extract text from the captured image", use_container_width=True):
                    with st.spinner("Extracting text..."):
                        # Extract text using Gemini
                        extracted_text = extract_text_with_camera(image)
                        
                        # Store in session state
                        st.session_state.extracted_text = extracted_text
                        st.session_state.extracted_equations = ""  # Gemini might not separate equations

                        # Translate the extracted text to English
                        translated_text = translate_text(extracted_text)

                        # Display results
                        st.subheader("📄 Extracted Text")
                        if "Error" in extracted_text:
                            st.error(extracted_text)
                        else:
                            st.success("Text extraction completed!")
                            st.markdown(
                                f"""
                                <div style="
                                    border: 1px solid #ccc;
                                    border-radius: 5px;
                                    padding: 15px;
                                    background-color: #f9f9f9;
                                    max-height: 300px;
                                    overflow-y: auto;
                                    font-family: monospace;
                                    white-space: pre-wrap;
                                    line-height: 1.5;
                                    font-size: 14px;
                                    margin: 10px 0;
                                ">{extracted_text}</div>
                                """,
                                unsafe_allow_html=True
                            )

                        # Display translated text
                        st.subheader("🌐 Translated Text (English)")
                        if "Error" in translated_text:
                            st.error(translated_text)
                        else:
                            st.success("Translation completed!")
                            st.markdown(
                                f"""
                                <div style="
                                    border: 1px solid #ccc;
                                    border-radius: 5px;
                                    padding: 15px;
                                    background-color: #f9f9f9;
                                    max-height: 300px;
                                    overflow-y: auto;
                                    font-family: monospace;
                                    white-space: pre-wrap;
                                    line-height: 1.5;
                                    font-size: 14px;
                                    margin: 10px 0;
                                ">{translated_text}</div>
                                """,
                                unsafe_allow_html=True
                            )
                st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Powered by Advanced Computer Vision & OCR Technology</p>
    </div>
    """, unsafe_allow_html=True)
