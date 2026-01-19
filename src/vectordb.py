"""Vector database implementation using SQLite."""
import sqlite3
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class VectorDB:
    """SQLite-based vector database for storing and searching embeddings."""
    
    def __init__(self, db_path: str):
        """
        Initialize vector database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_db()
    
    def _ensure_db_dir(self) -> None:
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL
                )
            """)
            
            # Embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id INTEGER PRIMARY KEY,
                    vec BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
                )
            """)
            
            # Metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            
            conn.commit()
    
    def get_meta(self, key: str) -> Optional[str]:
        """
        Get metadata value by key.
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM meta WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def set_meta(self, key: str, value: str) -> None:
        """
        Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, value)
            )
            conn.commit()
    
    def clear_all(self) -> None:
        """Clear all data from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM embeddings")
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM meta")
            conn.commit()
        logger.info("Database cleared")
    
    def add_chunk(self, text: str, embedding: np.ndarray) -> int:
        """
        Add a text chunk with its embedding.
        
        Args:
            text: Text content
            embedding: Embedding vector (numpy array)
            
        Returns:
            Chunk ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert chunk
            cursor.execute("INSERT INTO chunks (text) VALUES (?)", (text,))
            chunk_id = cursor.lastrowid
            
            # Insert embedding
            vec_bytes = embedding.astype(np.float32).tobytes()
            cursor.execute(
                "INSERT INTO embeddings (chunk_id, vec, dim) VALUES (?, ?, ?)",
                (chunk_id, vec_bytes, len(embedding))
            )
            
            conn.commit()
            return chunk_id
    
    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """
        Get all chunks with their embeddings.
        
        Returns:
            List of tuples (chunk_id, text, embedding)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.id, c.text, e.vec, e.dim
                FROM chunks c
                JOIN embeddings e ON c.id = e.chunk_id
            """)
            
            results = []
            for row in cursor.fetchall():
                chunk_id, text, vec_bytes, dim = row
                embedding = np.frombuffer(vec_bytes, dtype=np.float32)
                results.append((chunk_id, text, embedding))
            
            return results
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Search for most similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (text, similarity_score) sorted by score descending
        """
        all_data = self.get_all_embeddings()
        
        if not all_data:
            return []
        
        # Calculate cosine similarities
        query_norm = np.linalg.norm(query_embedding)
        similarities = []
        
        for chunk_id, text, embedding in all_data:
            # Cosine similarity
            emb_norm = np.linalg.norm(embedding)
            if emb_norm == 0 or query_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_embedding, embedding) / (query_norm * emb_norm)
            
            similarities.append((text, float(similarity)))
        
        # Sort by similarity descending and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def count_chunks(self) -> int:
        """
        Get total number of chunks in database.
        
        Returns:
            Number of chunks
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
