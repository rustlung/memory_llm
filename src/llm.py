"""OpenAI API wrapper for chat completions and embeddings."""
import logging
from typing import List, Dict
from openai import OpenAI
import numpy as np


logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper for OpenAI API."""
    
    def __init__(self, api_key: str, chat_model: str, embed_model: str):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            chat_model: Model name for chat completions
            embed_model: Model name for embeddings
        """
        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=text
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=texts
            )
            embeddings = [
                np.array(item.embedding, dtype=np.float32)
                for item in response.data
            ]
            return embeddings
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {e}")
            raise
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate chat completion.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
    
    def chat_completion_with_context(
        self,
        user_message: str,
        history: List[Dict[str, str]] = None,
        system_prompt: str = None
    ) -> str:
        """
        Generate chat completion with conversation history.
        
        Args:
            user_message: Current user message
            history: Previous conversation messages
            system_prompt: System prompt (optional)
            
        Returns:
            Generated response text
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add history
        if history:
            messages.extend(history)
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        return self.chat_completion(messages)
    
    def rag_completion(
        self,
        user_question: str,
        context_chunks: List[str]
    ) -> str:
        """
        Generate response using RAG (Retrieval-Augmented Generation).
        
        Args:
            user_question: User's question
            context_chunks: Retrieved relevant text chunks
            
        Returns:
            Generated response text
        """
        # Strict system prompt for RAG
        system_prompt = """Ты — помощник компании ТехноСервис. 
Отвечай ТОЛЬКО на основе предоставленного контекста о компании.
Если ответа нет в контексте — честно скажи: "К сожалению, в базе знаний компании нет информации по этому вопросу."
Не выдумывай информацию. Отвечай кратко и по делу."""
        
        # Format context
        context_text = "\n\n---\n\n".join(context_chunks)
        
        user_prompt = f"""Контекст о компании:
{context_text}

Вопрос пользователя: {user_question}

Ответ на основе контекста:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.chat_completion(messages, temperature=0.3)
