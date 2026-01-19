"""RAG (Retrieval-Augmented Generation) utilities."""
import hashlib
import logging
from typing import List, Tuple
from pathlib import Path

from src.vectordb import VectorDB
from src.llm import LLMClient


logger = logging.getLogger(__name__)


def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Hexadecimal hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def chunk_text(text: str, min_chunk_size: int = 80, max_chunk_size: int = 250) -> List[str]:
    """
    Split text into chunks by paragraphs with size constraints.
    
    Args:
        text: Text to split
        min_chunk_size: Minimum chunk size in characters
        max_chunk_size: Maximum chunk size in characters
        
    Returns:
        List of text chunks
    """
    # Split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # Check if paragraph looks like a section header (ends with colon)
        is_section_header = para.rstrip().endswith(':')
        
        # If paragraph itself is too long, split it further
        if len(para) > max_chunk_size:
            # Save current chunk if exists
            if current_chunk and len(current_chunk.strip()) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long paragraph by sentences (approximation)
            sentences = para.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                if len(current_chunk) + len(sent) <= max_chunk_size:
                    current_chunk += sent + " "
                else:
                    if current_chunk and len(current_chunk.strip()) >= min_chunk_size:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + " "
            
            # Save remaining
            if current_chunk and len(current_chunk.strip()) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        else:
            # If adding this paragraph would exceed max size OR it's a new section, save current chunk
            will_exceed = len(current_chunk) + len(para) > max_chunk_size
            is_new_section = is_section_header and current_chunk
            
            if current_chunk and (will_exceed or is_new_section):
                if len(current_chunk.strip()) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += para + "\n\n"
            else:
                current_chunk = para + "\n\n"
    
    # Save last chunk
    if current_chunk and len(current_chunk.strip()) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    
    return chunks


def index_company_data(
    company_txt_path: str,
    db: VectorDB,
    llm_client: LLMClient
) -> None:
    """
    Index company data from text file into vector database.
    
    Args:
        company_txt_path: Path to company.txt
        db: Vector database instance
        llm_client: LLM client for embeddings
    """
    logger.info(f"Reading company data from {company_txt_path}")
    
    # Read file
    with open(company_txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Chunk text
    chunks = chunk_text(text)
    logger.info(f"Split into {len(chunks)} chunks")
    
    # Get embeddings
    logger.info("Generating embeddings...")
    embeddings = llm_client.get_embeddings_batch(chunks)
    
    # Clear database and add chunks
    db.clear_all()
    for chunk, embedding in zip(chunks, embeddings):
        db.add_chunk(chunk, embedding)
    
    # Save file hash
    file_hash = compute_file_hash(company_txt_path)
    db.set_meta("company_txt_hash", file_hash)
    
    logger.info(f"Indexed {len(chunks)} chunks into database")


def needs_reindexing(company_txt_path: str, db: VectorDB) -> bool:
    """
    Check if company data needs reindexing.
    
    Args:
        company_txt_path: Path to company.txt
        db: Vector database instance
        
    Returns:
        True if reindexing is needed
    """
    # Check if database is empty
    if db.count_chunks() == 0:
        logger.info("Database is empty, indexing needed")
        return True
    
    # Check file hash
    stored_hash = db.get_meta("company_txt_hash")
    if not stored_hash:
        logger.info("No hash found in database, indexing needed")
        return True
    
    current_hash = compute_file_hash(company_txt_path)
    if stored_hash != current_hash:
        logger.info("File hash changed, reindexing needed")
        return True
    
    logger.info("Database is up to date")
    return False


def retrieve_context(
    query: str,
    db: VectorDB,
    llm_client: LLMClient,
    top_k: int = 3
) -> Tuple[List[str], float]:
    """
    Retrieve relevant context chunks for a query.
    
    Args:
        query: User query
        db: Vector database instance
        llm_client: LLM client for embeddings
        top_k: Number of top chunks to retrieve
        
    Returns:
        Tuple of (list of chunks, best similarity score)
    """
    # Get query embedding
    query_embedding = llm_client.get_embedding(query)
    
    # Search in database
    results = db.search(query_embedding, top_k=top_k)
    
    if not results:
        return [], 0.0
    
    chunks = [text for text, score in results]
    best_score = results[0][1]
    
    logger.info(f"Retrieved {len(chunks)} chunks, best score: {best_score:.4f}")
    
    return chunks, best_score
