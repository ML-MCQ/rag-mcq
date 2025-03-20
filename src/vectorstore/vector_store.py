"""
Vector Store module for storing and retrieving document embeddings.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from configLoader import load_config

# Configure logging
logger = logging.getLogger(__name__)

config = load_config()


class VectorStore:
    """
    A class for managing vector storage of document embeddings.

    This class handles storing document embeddings in a vector store for efficient
    semantic search and retrieval.
    """

    def __init__(
        self,
        embedding_model_name: str = config["embeddingModel"]["EMBEDDING_MODEL"],
        vector_store_path: str = config["vectorStore"]["VECTOR_STORE_PATH"],
    ):
        """
        Initialize the vector store with embedding model and storage path.

        Args:
            embedding_model_name: Name of the Hugging Face model to use for embeddings
            vector_store_path: Path to store the vector database
        """
        self.embedding_model_name = embedding_model_name
        self.vector_store_path = vector_store_path
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store = None

        # Create the vector store directory if it doesn't exist
        os.makedirs(vector_store_path, exist_ok=True)

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents in the vector store.

        Args:
            documents: List of document objects with text and metadata
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return

        try:
            # Create a FAISS vector store from the documents
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Successfully indexed {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise

    def save_vector_store(self, index_name: str = "default") -> None:
        """
        Save the vector store to disk.

        Args:
            index_name: Name of the index to save
        """
        if self.vector_store is None:
            logger.warning("No vector store available to save")
            return

        try:
            # Create the save path
            save_path = os.path.join(self.vector_store_path, index_name)

            # Save the FAISS vector store
            self.vector_store.save_local(save_path)
            logger.info(f"Vector store saved to {save_path}")

            # Also export visualization data
            self.export_visualization_data(index_name)
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise

    def export_visualization_data(self, index_name: str = "default") -> None:
        """
        Export vector store data for visualization purposes.

        Args:
            index_name: Name of the index to export
        """
        if self.vector_store is None:
            logger.warning("No vector store available to export")
            return

        try:
            # Prepare visualization data
            docs = self.vector_store.docstore._dict.values()

            # Get embeddings - this is a bit hacky as we're accessing internal FAISS structure
            embeddings = []
            if (
                hasattr(self.vector_store, "index")
                and self.vector_store.index is not None
            ):
                # Get number of vectors
                num_vectors = self.vector_store.index.ntotal
                # Get dimension of vectors
                dim = self.vector_store.index.d

                # Create a numpy array to store the embeddings
                temp_embeddings = np.zeros((num_vectors, dim), dtype=np.float32)

                # Extract embeddings from FAISS index
                for i in range(num_vectors):
                    self.vector_store.index.reconstruct(i, temp_embeddings[i])

                # Convert to list for JSON serialization
                embeddings = temp_embeddings.tolist()

            # Create visualization data structure
            visualization_data = {
                "documents": [],
                "embeddings_available": len(embeddings) > 0,
            }

            # If embeddings are available, include them
            if visualization_data["embeddings_available"]:
                visualization_data["embeddings"] = embeddings

            # Add document data
            for i, doc in enumerate(docs):
                doc_data = {
                    "id": i,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                visualization_data["documents"].append(doc_data)

            # Save visualization data to file
            viz_path = os.path.join(
                self.vector_store_path, f"{index_name}_visualization.json"
            )
            with open(viz_path, "w") as f:
                json.dump(visualization_data, f, indent=2)

            logger.info(f"Vector store visualization data exported to {viz_path}")
        except Exception as e:
            logger.error(f"Error exporting visualization data: {str(e)}")
            # Log but don't raise, as this is a non-critical operation

    def load_vector_store(self, index_name: str = "default") -> bool:
        """
        Load a vector store from disk.

        Args:
            index_name: Name of the index to load

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            # Create the load path
            load_path = os.path.join(self.vector_store_path, index_name)

            if not os.path.exists(load_path):
                logger.warning(f"Vector store path {load_path} does not exist")
                return False

            # Load the FAISS vector store
            self.vector_store = FAISS.load_local(load_path, self.embeddings)
            logger.info(f"Vector store loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a similarity search against the vector store.

        Args:
            query: The query text to search for
            k: Number of relevant documents to return

        Returns:
            List of relevant documents with similarity scores
        """
        if self.vector_store is None:
            logger.warning("No vector store available for search")
            return []

        try:
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(query, k=k)

            # Format results for easier consumption
            formatted_results = []
            for doc, score in results:
                formatted_results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": score,
                    }
                )

            return formatted_results
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []