"""
Document Processing Service
Handles parsing and chunking of various document formats (PDF, DOCX, CSV, JSON).
"""

from typing import List, Dict, Any
import tiktoken
from unstructured.partition.auto import partition
from pathlib import Path


def parse_document(file_path: str) -> str:
    """
    Parse any document type and return extracted text.
    Uses Unstructured.io to handle PDF, DOCX, CSV, JSON, and other formats.

    Args:
        file_path: Path to the document file

    Returns:
        str: Extracted text content from the document

    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: If parsing fails
    """
    # Verify file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Use Unstructured.io's auto partition to handle any file type
        elements = partition(filename=file_path)

        # Combine all elements into a single text string
        text = "\n\n".join([str(el) for el in elements])

        return text

    except Exception as e:
        raise Exception(f"Failed to parse document {file_path}: {str(e)}")


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding_name: str = "cl100k_base"  # GPT-4 encoding
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks based on token count.

    Args:
        text: The text to chunk
        chunk_size: Maximum tokens per chunk (default: 512)
        overlap: Number of overlapping tokens between chunks (default: 50)
        encoding_name: Tokenizer encoding to use (default: cl100k_base for GPT-4)

    Returns:
        List of dictionaries containing:
            - text: The chunk text
            - chunk_index: Index of the chunk
            - token_count: Number of tokens in the chunk
            - start_char: Starting character position
            - end_char: Ending character position
    """
    # Initialize tokenizer
    try:
        tokenizer = tiktoken.get_encoding(encoding_name)
    except Exception:
        # Fallback to default encoding
        tokenizer = tiktoken.encoding_for_model("gpt-4")

    # Encode the entire text
    tokens = tokenizer.encode(text)

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        # Get chunk tokens
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]

        # Decode back to text
        chunk_text = tokenizer.decode(chunk_tokens)

        # Calculate character positions (approximate)
        if chunks:
            # For subsequent chunks, use the previous end position
            start_char = chunks[-1]['end_char'] - (overlap * 4)  # Rough estimate
            start_char = max(0, start_char)
        else:
            start_char = 0

        end_char = start_char + len(chunk_text)

        # Create chunk metadata
        chunk_data = {
            'text': chunk_text,
            'chunk_index': len(chunks),
            'token_count': len(chunk_tokens),
            'start_char': start_char,
            'end_char': end_char
        }

        chunks.append(chunk_data)

        # Move to next chunk with overlap
        start_idx += (chunk_size - overlap)

        # Break if we've reached the end
        if end_idx >= len(tokens):
            break

    return chunks


def get_document_stats(file_path: str) -> Dict[str, Any]:
    """
    Get statistics about a document.

    Args:
        file_path: Path to the document

    Returns:
        Dictionary with document statistics
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Parse document
    text = parse_document(file_path)

    # Get token count
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    tokens = tokenizer.encode(text)

    return {
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "file_type": path.suffix,
        "character_count": len(text),
        "token_count": len(tokens),
        "estimated_chunks_512": (len(tokens) // 512) + 1
    }
