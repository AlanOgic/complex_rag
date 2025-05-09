"""
OCR processing utilities for extracting text from images and PDF documents.
"""

import logging
import os
import tempfile
import asyncio
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import base64

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Handles OCR processing to extract text from images and PDF documents.
    
    Features:
    - PDF text extraction with OCR fallback
    - Image text extraction
    - Table detection and processing
    - Layout preservation
    - Multi-language support
    """
    
    def __init__(self,
                 ocr_engine: str = "tesseract",
                 languages: List[str] = ["eng"],
                 detect_tables: bool = True,
                 preserve_layout: bool = True,
                 confidence_threshold: float = 60.0,
                 temp_dir: Optional[str] = None):
        """
        Initialize the OCR processor.
        
        Args:
            ocr_engine: OCR engine to use ('tesseract', 'pytesseract', 'easyocr', 'azure')
            languages: Languages to use for OCR
            detect_tables: Whether to detect and process tables
            preserve_layout: Whether to preserve document layout
            confidence_threshold: Minimum confidence threshold for OCR results
            temp_dir: Directory for temporary files
        """
        self.ocr_engine = ocr_engine
        self.languages = languages
        self.detect_tables = detect_tables
        self.preserve_layout = preserve_layout
        self.confidence_threshold = confidence_threshold
        self.temp_dir = temp_dir or os.environ.get("TEMP_DIR", "/tmp")
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize OCR engine
        self._ocr_engine = None
    
    async def initialize(self) -> bool:
        """
        Initialize the OCR engine.
        
        Returns:
            True if initialization was successful
        """
        try:
            if self.ocr_engine == "tesseract":
                # Check if tesseract is installed
                import shutil
                tesseract_path = shutil.which("tesseract")
                
                if not tesseract_path:
                    logger.error("Tesseract not found. Please install Tesseract OCR.")
                    return False
                
                # Verify languages are installed
                result = await asyncio.create_subprocess_exec(
                    "tesseract", "--list-langs",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()
                
                available_langs = stdout.decode().strip().split("\n")[1:]
                missing_langs = [lang for lang in self.languages if lang not in available_langs]
                
                if missing_langs:
                    logger.warning(f"Missing Tesseract languages: {missing_langs}")
                
                logger.info(f"Initialized Tesseract OCR engine with languages: {self.languages}")
                return True
                
            elif self.ocr_engine == "pytesseract":
                try:
                    import pytesseract
                    from PIL import Image
                    
                    # Try to get tesseract version
                    version = pytesseract.get_tesseract_version()
                    logger.info(f"Initialized PyTesseract OCR engine v{version}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error initializing PyTesseract: {e}")
                    return False
                
            elif self.ocr_engine == "easyocr":
                try:
                    import easyocr
                    
                    # Initialize reader (this may take some time the first run)
                    self._ocr_engine = await asyncio.to_thread(
                        easyocr.Reader, self.languages
                    )
                    
                    logger.info(f"Initialized EasyOCR engine with languages: {self.languages}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error initializing EasyOCR: {e}")
                    return False
                
            elif self.ocr_engine == "azure":
                try:
                    from azure.ai.formrecognizer import DocumentAnalysisClient
                    from azure.core.credentials import AzureKeyCredential
                    
                    # Get Azure credentials from environment variables
                    endpoint = os.environ.get("AZURE_FORM_RECOGNIZER_ENDPOINT")
                    key = os.environ.get("AZURE_FORM_RECOGNIZER_KEY")
                    
                    if not endpoint or not key:
                        logger.error("Azure Form Recognizer credentials not found")
                        return False
                    
                    # Initialize client
                    self._ocr_engine = DocumentAnalysisClient(
                        endpoint=endpoint,
                        credential=AzureKeyCredential(key)
                    )
                    
                    logger.info("Initialized Azure Form Recognizer engine")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error initializing Azure Form Recognizer: {e}")
                    return False
                
            else:
                logger.error(f"Unsupported OCR engine: {self.ocr_engine}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing OCR processor: {e}")
            return False
    
    async def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process an image file with OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with OCR results
        """
        if not self._ocr_engine and not await self.initialize():
            return {"text": "", "error": "Failed to initialize OCR engine"}
        
        try:
            image_path = str(image_path)
            
            if self.ocr_engine == "tesseract":
                return await self._process_with_tesseract(image_path)
                
            elif self.ocr_engine == "pytesseract":
                return await self._process_with_pytesseract(image_path)
                
            elif self.ocr_engine == "easyocr":
                return await self._process_with_easyocr(image_path)
                
            elif self.ocr_engine == "azure":
                return await self._process_with_azure(image_path)
                
            else:
                return {"text": "", "error": f"Unsupported OCR engine: {self.ocr_engine}"}
                
        except Exception as e:
            logger.error(f"Error processing image with OCR: {e}")
            return {"text": "", "error": str(e)}
    
    async def process_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a PDF file, extracting text and applying OCR when needed.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            pdf_path = str(pdf_path)
            
            # Try to extract text directly first
            direct_text = await self._extract_pdf_text_direct(pdf_path)
            
            # If direct extraction yields substantial text, use it
            if direct_text and len(direct_text) > 100:
                result = {
                    "text": direct_text,
                    "source": "direct_extraction",
                    "pages": await self._count_pdf_pages(pdf_path),
                    "ocr_applied": False,
                }
                
                # Check if pdf has images that might need OCR
                has_images = await self._check_pdf_has_images(pdf_path)
                
                if has_images:
                    # OCR might be needed for image content
                    result["has_images"] = True
                    
                    # TODO: Implement selective OCR for pages with low text content
                
                return result
            
            # If direct extraction failed or yielded little text, apply OCR
            return await self._process_pdf_with_ocr(pdf_path)
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"text": "", "error": str(e)}
    
    async def _extract_pdf_text_direct(self, pdf_path: str) -> str:
        """
        Extract text directly from PDF without OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            # Try different PDF parsers
            
            # First try: PyPDF2
            try:
                from PyPDF2 import PdfReader
                
                reader = await asyncio.to_thread(PdfReader, pdf_path)
                text = ""
                
                for page in reader.pages:
                    page_text = await asyncio.to_thread(page.extract_text)
                    text += page_text + "\n\n"
                
                if text.strip():
                    return text
                    
            except Exception as e:
                logger.debug(f"PyPDF2 extraction failed: {e}")
            
            # Second try: pdfminer.six
            try:
                from pdfminer.high_level import extract_text as pdfminer_extract_text
                
                text = await asyncio.to_thread(pdfminer_extract_text, pdf_path)
                
                if text.strip():
                    return text
                    
            except Exception as e:
                logger.debug(f"pdfminer.six extraction failed: {e}")
            
            # Third try: pdfplumber
            try:
                import pdfplumber
                
                text = ""
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = await asyncio.to_thread(page.extract_text)
                        if page_text:
                            text += page_text + "\n\n"
                
                if text.strip():
                    return text
                    
            except Exception as e:
                logger.debug(f"pdfplumber extraction failed: {e}")
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    async def _count_pdf_pages(self, pdf_path: str) -> int:
        """
        Count the number of pages in a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of pages
        """
        try:
            from PyPDF2 import PdfReader
            
            reader = await asyncio.to_thread(PdfReader, pdf_path)
            return len(reader.pages)
            
        except Exception as e:
            logger.error(f"Error counting PDF pages: {e}")
            return 0
    
    async def _check_pdf_has_images(self, pdf_path: str) -> bool:
        """
        Check if a PDF contains images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if the PDF contains images
        """
        try:
            import fitz  # PyMuPDF
            
            doc = await asyncio.to_thread(fitz.open, pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = await asyncio.to_thread(page.get_images)
                
                if image_list:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking for images in PDF: {e}")
            # Assume there might be images if we can't check
            return True
    
    async def _process_pdf_with_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF file with OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with OCR results
        """
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            
            doc = await asyncio.to_thread(fitz.open, pdf_path)
            text_results = []
            
            with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_dir:
                # Process each page
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Convert page to image
                    pix = await asyncio.to_thread(page.get_pixmap)
                    img_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                    await asyncio.to_thread(pix.save, img_path)
                    
                    # Process with OCR
                    page_result = await self.process_image(img_path)
                    
                    if "text" in page_result and page_result["text"]:
                        text_results.append(page_result["text"])
            
            return {
                "text": "\n\n".join(text_results),
                "source": "ocr",
                "pages": len(doc),
                "ocr_applied": True,
                "ocr_engine": self.ocr_engine
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF with OCR: {e}")
            return {"text": "", "error": str(e)}
    
    async def _process_with_tesseract(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image with Tesseract OCR CLI.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with OCR results
        """
        try:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, dir=self.temp_dir) as temp_file:
                output_path = temp_file.name
            
            # Prepare language parameter
            lang_param = "+".join(self.languages)
            
            # Run tesseract command
            cmd = ["tesseract", image_path, output_path.replace(".txt", ""), "-l", lang_param]
            
            if self.detect_tables:
                cmd.extend(["--psm", "6"])
            else:
                cmd.extend(["--psm", "3"])
            
            # Add additional parameters
            if self.detect_tables:
                cmd.extend(["-c", "tessedit_prefer_line_breaks=0"])
            
            if self.preserve_layout:
                cmd.extend(["-c", "textonly_pdf=1"])
            
            # Execute the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return {
                    "text": "",
                    "error": f"Tesseract error: {stderr.decode()}"
                }
            
            # Read the output file
            async with aiofiles.open(output_path, "r") as f:
                text = await f.read()
            
            # Clean up
            try:
                os.unlink(output_path)
            except Exception:
                pass
            
            return {
                "text": text,
                "source": "tesseract",
                "languages": self.languages
            }
            
        except Exception as e:
            logger.error(f"Error processing with Tesseract: {e}")
            return {"text": "", "error": str(e)}
    
    async def _process_with_pytesseract(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image with PyTesseract.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with OCR results
        """
        try:
            import pytesseract
            from PIL import Image
            
            # Open image
            image = await asyncio.to_thread(Image.open, image_path)
            
            # Process with pytesseract
            config = ""
            if self.detect_tables:
                config += "--psm 6"
            else:
                config += "--psm 3"
            
            if self.preserve_layout:
                config += " -c textonly_pdf=1"
            
            # Perform OCR
            text = await asyncio.to_thread(
                pytesseract.image_to_string,
                image,
                lang="+".join(self.languages),
                config=config
            )
            
            # Get confidence
            data = await asyncio.to_thread(
                pytesseract.image_to_data,
                image,
                lang="+".join(self.languages),
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [float(conf) for conf in data["conf"] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": text,
                "source": "pytesseract",
                "languages": self.languages,
                "confidence": avg_confidence
            }
            
        except Exception as e:
            logger.error(f"Error processing with PyTesseract: {e}")
            return {"text": "", "error": str(e)}
    
    async def _process_with_easyocr(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image with EasyOCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with OCR results
        """
        try:
            # Ensure the engine is initialized
            if not self._ocr_engine:
                import easyocr
                self._ocr_engine = await asyncio.to_thread(
                    easyocr.Reader, self.languages
                )
            
            # Process the image
            result = await asyncio.to_thread(
                self._ocr_engine.readtext,
                image_path,
                detail=1,
                paragraph=self.preserve_layout
            )
            
            # Extract text and confidence
            text_parts = []
            confidences = []
            
            for detection in result:
                bbox, text, confidence = detection
                
                if confidence >= self.confidence_threshold / 100.0:
                    text_parts.append(text)
                    confidences.append(confidence)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Format text based on layout option
            if self.preserve_layout:
                full_text = " ".join(text_parts)
            else:
                full_text = "\n".join(text_parts)
            
            return {
                "text": full_text,
                "source": "easyocr",
                "languages": self.languages,
                "confidence": avg_confidence * 100.0,
                "word_count": len(" ".join(text_parts).split())
            }
            
        except Exception as e:
            logger.error(f"Error processing with EasyOCR: {e}")
            return {"text": "", "error": str(e)}
    
    async def _process_with_azure(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image with Azure Form Recognizer.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with OCR results
        """
        try:
            # Read image file
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Process with Azure
            poller = await asyncio.to_thread(
                self._ocr_engine.begin_analyze_document,
                "prebuilt-read",
                image_data
            )
            
            # Get result (this will wait for completion)
            result = await asyncio.to_thread(poller.result)
            
            # Extract text
            text = ""
            for page in result.pages:
                for line in page.lines:
                    text += line.content + "\n"
                text += "\n"
            
            return {
                "text": text,
                "source": "azure_form_recognizer",
                "confidence": 90.0,  # Azure doesn't provide confidence scores in this API
                "pages": len(result.pages),
                "word_count": sum(len(page.words) for page in result.pages)
            }
            
        except Exception as e:
            logger.error(f"Error processing with Azure Form Recognizer: {e}")
            return {"text": "", "error": str(e)}
    
    async def close(self) -> None:
        """Clean up resources."""
        # Most OCR engines don't need explicit cleanup
        pass


# Import here for asynchronous file operations
import aiofiles