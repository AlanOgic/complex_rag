"""
Document quality assessment and enhancement utilities.

This module provides tools for assessing and improving document quality
before chunking and indexing.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)


class DocumentQualityProcessor:
    """
    Handles document quality assessment and enhancement.
    
    Features:
    - Content quality scoring
    - Noise detection and filtering
    - Structure analysis
    - Language detection
    - Content deduplication
    - Entity identification
    """
    
    def __init__(self,
                 min_content_length: int = 50,
                 max_noise_ratio: float = 0.3,
                 language_detection_threshold: float = 0.8,
                 deduplication_threshold: float = 0.85):
        """
        Initialize the document quality processor.
        
        Args:
            min_content_length: Minimum content length to consider valid
            max_noise_ratio: Maximum ratio of noise to content
            language_detection_threshold: Threshold for language detection confidence
            deduplication_threshold: Threshold for content deduplication similarity
        """
        self.min_content_length = min_content_length
        self.max_noise_ratio = max_noise_ratio
        self.language_detection_threshold = language_detection_threshold
        self.deduplication_threshold = deduplication_threshold
        
        # Cached language detection model
        self._language_model = None
    
    def analyze_document_quality(self, content: str) -> Dict[str, Any]:
        """
        Analyze document quality and return metrics.
        
        Args:
            content: Document content text
            
        Returns:
            Dictionary of quality metrics
        """
        if not content or len(content.strip()) < self.min_content_length:
            return {
                "quality_score": 0.0,
                "is_valid": False,
                "issues": ["Empty or insufficient content"],
                "noise_ratio": 1.0,
                "language": "unknown",
                "language_confidence": 0.0,
                "content_density": 0.0,
                "has_structure": False,
            }
        
        # Calculate basic metrics
        content_length = len(content)
        noise_ratio = self._calculate_noise_ratio(content)
        language, language_confidence = self._detect_language(content)
        structure_quality = self._analyze_structure(content)
        information_density = self._calculate_information_density(content)
        has_duplicates, duplication_ratio = self._check_for_duplicates(content)
        
        # Identify issues
        issues = []
        
        if noise_ratio > self.max_noise_ratio:
            issues.append(f"High noise ratio ({noise_ratio:.2f})")
        
        if language_confidence < self.language_detection_threshold:
            issues.append(f"Low language confidence ({language_confidence:.2f})")
        
        if has_duplicates and duplication_ratio > 0.3:
            issues.append(f"High content duplication ({duplication_ratio:.2f})")
        
        if information_density < 0.4:
            issues.append(f"Low information density ({information_density:.2f})")
        
        # Calculate overall quality score (weighted average of metrics)
        noise_score = 1.0 - noise_ratio
        language_score = language_confidence
        structure_score = structure_quality
        density_score = information_density
        duplication_score = 1.0 - duplication_ratio if has_duplicates else 1.0
        
        quality_score = 0.25 * noise_score + 0.2 * language_score + 0.2 * structure_score + 0.2 * density_score + 0.15 * duplication_score
        
        # Determine if document is valid for indexing
        is_valid = quality_score >= 0.5 and len(issues) < 3
        
        return {
            "quality_score": quality_score,
            "is_valid": is_valid,
            "issues": issues,
            "noise_ratio": noise_ratio,
            "language": language,
            "language_confidence": language_confidence,
            "content_density": information_density,
            "has_structure": structure_quality > 0.5,
            "duplication_ratio": duplication_ratio if has_duplicates else 0.0,
            "metrics": {
                "noise_score": noise_score,
                "language_score": language_score,
                "structure_score": structure_score,
                "density_score": density_score,
                "duplication_score": duplication_score,
            }
        }
    
    def enhance_document(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance document quality by cleaning and restructuring content.
        
        Args:
            content: Document content text
            
        Returns:
            Tuple of (enhanced_content, enhancement_info)
        """
        if not content or len(content.strip()) < self.min_content_length:
            return content, {"changes": [], "improved": False}
        
        # Initialize tracking
        original_content = content
        changes = []
        
        # Process the content
        content = self._remove_noise(content)
        if content != original_content:
            changes.append("Removed noise")
        
        content = self._normalize_whitespace(content)
        if content != original_content and "Removed noise" not in changes:
            changes.append("Normalized whitespace")
        
        content = self._deduplicate_content(content)
        if content != original_content and "Removed noise" not in changes and "Normalized whitespace" not in changes:
            changes.append("Deduplicated content")
        
        # Calculate improvement
        original_quality = self.analyze_document_quality(original_content)
        enhanced_quality = self.analyze_document_quality(content)
        
        improvement = enhanced_quality["quality_score"] - original_quality["quality_score"]
        
        enhancement_info = {
            "changes": changes,
            "improved": improvement > 0.1,
            "original_quality": original_quality["quality_score"],
            "enhanced_quality": enhanced_quality["quality_score"],
            "improvement": improvement
        }
        
        return content, enhancement_info
    
    def _calculate_noise_ratio(self, content: str) -> float:
        """
        Calculate the ratio of noise to content.
        
        Args:
            content: Document content text
            
        Returns:
            Noise ratio (0.0-1.0)
        """
        # Consider patterns that indicate noise
        noise_patterns = [
            r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',  # Email addresses
            r'https?://\S+',  # URLs
            r'\{\{.*?\}\}',  # Template markers
            r'<[^>]*>',  # HTML tags
            r'\[\[.*?\]\]',  # Wiki-style links
            r'\[.*?\]\(.*?\)',  # Markdown links
            r'^\s*[#*=-]{3,}\s*$',  # Separator lines
            r'^\s*>\s.*$',  # Quote blocks
            r'^\s*\d+\.\s+.*$',  # Numbered lists (in some contexts)
            r'^\s*[-*+]\s+.*$',  # Bullet lists (in some contexts)
            r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)',  # IP addresses
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # Dates
            r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?',  # Times
        ]
        
        # Get total content length
        total_length = len(content)
        if total_length == 0:
            return 1.0  # Empty content is 100% noise
        
        # Find noise matches
        noise_text = content
        for pattern in noise_patterns:
            noise_text = re.sub(pattern, '', noise_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Calculate noise ratio
        noise_characters = total_length - len(noise_text)
        return noise_characters / total_length
    
    def _detect_language(self, content: str) -> Tuple[str, float]:
        """
        Detect the language of the document.
        
        Args:
            content: Document content text
            
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            # Use langdetect if available
            from langdetect import detect, DetectorFactory
            from langdetect.lang_detect_exception import LangDetectException
            
            # Set seed for consistent results
            DetectorFactory.seed = 0
            
            try:
                # Get a sample of the text for efficiency
                sample = content[:min(5000, len(content))]
                language = detect(sample)
                
                # Simplified confidence calculation
                confidence = 0.85  # Base confidence
                
                # Adjust confidence based on content length
                if len(content) < 200:
                    confidence *= 0.7
                elif len(content) < 500:
                    confidence *= 0.85
                
                return language, confidence
                
            except LangDetectException:
                return "unknown", 0.0
                
        except ImportError:
            # Fallback to simple English detection
            english_words = {
                'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
                'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
                'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
                'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what'
            }
            
            words = re.findall(r'\b\w+\b', content.lower())
            if not words:
                return "unknown", 0.0
            
            # Count English words
            english_count = sum(1 for word in words if word in english_words)
            confidence = english_count / min(len(words), 100)
            
            if confidence > 0.3:
                return "en", confidence
            else:
                return "unknown", confidence
    
    def _analyze_structure(self, content: str) -> float:
        """
        Analyze document structure quality.
        
        Args:
            content: Document content text
            
        Returns:
            Structure quality score (0.0-1.0)
        """
        lines = content.split('\n')
        if not lines:
            return 0.0
        
        # Check for sections/headers
        header_pattern = re.compile(r'^\s*#+\s+.+$|^\s*.+\s*\n\s*[-=]+\s*$')
        headers = sum(1 for line in lines if header_pattern.match(line))
        
        # Check for paragraphs
        paragraphs = 0
        current_paragraph = []
        
        for line in lines:
            if line.strip():
                current_paragraph.append(line)
            elif current_paragraph:
                paragraphs += 1
                current_paragraph = []
        
        if current_paragraph:
            paragraphs += 1
        
        # Check for lists
        list_pattern = re.compile(r'^\s*(?:[-*+]|\d+\.)\s+.+$')
        list_items = sum(1 for line in lines if list_pattern.match(line))
        
        # Check for balanced sections
        if headers > 0:
            avg_lines_per_section = len(lines) / (headers + 1)
            section_balance = min(1.0, 10.0 / avg_lines_per_section)
        else:
            section_balance = 0.0
        
        # Calculate structure score
        has_headers = headers > 0
        has_paragraphs = paragraphs > 1
        has_lists = list_items > 3
        
        score = 0.0
        components = 0
        
        if has_headers:
            score += 0.3
            components += 1
        
        if has_paragraphs:
            score += 0.4
            components += 1
        
        if has_lists:
            score += 0.2
            components += 1
        
        if section_balance > 0:
            score += 0.1 * section_balance
            components += 1
        
        return score / max(1, components)
    
    def _calculate_information_density(self, content: str) -> float:
        """
        Calculate information density of the document.
        
        Args:
            content: Document content text
            
        Returns:
            Information density score (0.0-1.0)
        """
        # Count unique words and their frequencies
        words = re.findall(r'\b\w+\b', content.lower())
        if not words:
            return 0.0
        
        word_counts = Counter(words)
        unique_words = len(word_counts)
        
        # Calculate entropy as an information density measure
        total_words = len(words)
        entropy = 0.0
        
        for count in word_counts.values():
            probability = count / total_words
            entropy -= probability * np.log2(probability)
        
        # Normalize entropy to 0-1 range (typical English text has entropy of ~9-11 bits per word)
        max_entropy = np.log2(unique_words) if unique_words > 0 else 1.0
        normalized_entropy = min(1.0, entropy / max_entropy)
        
        # Calculate unique word ratio
        unique_ratio = unique_words / total_words if total_words > 0 else 0.0
        
        # Combine metrics for final density score
        density_score = 0.7 * normalized_entropy + 0.3 * unique_ratio
        
        return density_score
    
    def _check_for_duplicates(self, content: str) -> Tuple[bool, float]:
        """
        Check for duplicate content in the document.
        
        Args:
            content: Document content text
            
        Returns:
            Tuple of (has_duplicates, duplication_ratio)
        """
        # Check for exact paragraph duplication
        paragraphs = re.split(r'\n\s*\n', content)
        if len(paragraphs) <= 1:
            return False, 0.0
        
        paragraph_counts = Counter(p.strip() for p in paragraphs if p.strip())
        
        # Calculate duplication ratio
        total_paragraphs = len(paragraph_counts)
        duplicate_paragraphs = sum(count - 1 for count in paragraph_counts.values() if count > 1)
        
        duplication_ratio = duplicate_paragraphs / total_paragraphs if total_paragraphs > 0 else 0.0
        
        has_duplicates = duplication_ratio > 0.1
        
        return has_duplicates, duplication_ratio
    
    def _remove_noise(self, content: str) -> str:
        """
        Remove noise from document content.
        
        Args:
            content: Document content text
            
        Returns:
            Cleaned content
        """
        # Remove repeated line breaks
        cleaned = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove boilerplate text
        boilerplate_patterns = [
            r'(?i)confidentiality notice.*?end of notice',
            r'(?i)this email and any attachments.*?intended recipient',
            r'(?i)copyright © \d{4}.*?rights reserved',
            r'(?i)this message contains information which may be confidential'
        ]
        
        for pattern in boilerplate_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
        
        # Remove ASCII art
        ascii_art_pattern = r'(?:\s*[^\w\s]{3,}[\s\w]*[^\w\s]{3,}\s*){3,}'
        cleaned = re.sub(ascii_art_pattern, '', cleaned)
        
        return cleaned
    
    def _normalize_whitespace(self, content: str) -> str:
        """
        Normalize whitespace in document content.
        
        Args:
            content: Document content text
            
        Returns:
            Content with normalized whitespace
        """
        # Replace multiple spaces with single space
        normalized = re.sub(r' {2,}', ' ', content)
        
        # Normalize line breaks
        normalized = re.sub(r'\r\n|\r', '\n', normalized)
        
        # Remove trailing whitespace
        normalized = re.sub(r' +$', '', normalized, flags=re.MULTILINE)
        
        return normalized
    
    def _deduplicate_content(self, content: str) -> str:
        """
        Remove duplicate paragraphs from document content.
        
        Args:
            content: Document content text
            
        Returns:
            Deduplicated content
        """
        paragraphs = re.split(r'\n\s*\n', content)
        
        # Keep track of unique paragraphs
        unique_paragraphs = []
        seen_paragraphs = set()
        
        for p in paragraphs:
            stripped = p.strip()
            if stripped and stripped not in seen_paragraphs:
                unique_paragraphs.append(p)
                seen_paragraphs.add(stripped)
        
        # Reconstruct content
        return '\n\n'.join(unique_paragraphs)