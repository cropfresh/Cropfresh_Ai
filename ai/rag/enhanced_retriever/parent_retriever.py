"""
Parent Document Retriever
"""

from typing import Optional
import uuid
import numpy as np
from loguru import logger

from .models import DocumentNode, RetrievalResult, RetrievalStrategy, EnhancedRetrieverConfig


class ParentDocumentRetriever:
    """
    Parent Document Retriever.
    
    Strategy:
    1. Store both small chunks and their parent documents
    2. Search using small chunks (better precision)
    3. Return parent documents (better context)
    
    This gives the best of both worlds: precise matching
    and complete context.
    
    Usage:
        retriever = ParentDocumentRetriever(embedding_manager)
        
        # Add documents
        await retriever.add_documents(documents)
        
        # Retrieve with parent context
        results = await retriever.retrieve("query", top_k=5)
    """
    
    def __init__(
        self,
        embedding_manager,
        config: Optional[EnhancedRetrieverConfig] = None,
    ):
        """Initialize parent document retriever."""
        self.embedding_manager = embedding_manager
        self.config = config or EnhancedRetrieverConfig()
        
        # Storage
        self.parent_nodes: dict[str, DocumentNode] = {}
        self.child_nodes: dict[str, DocumentNode] = {}
        self.child_to_parent: dict[str, str] = {}
        
        logger.info("ParentDocumentRetriever initialized")
    
    async def add_documents(self, documents: list) -> int:
        """
        Add documents with parent-child splitting.
        
        Args:
            documents: List of documents
            
        Returns:
            Number of child chunks created
        """
        total_children = 0
        
        for doc in documents:
            text = doc.text if hasattr(doc, 'text') else str(doc)
            doc_id = doc.id if hasattr(doc, 'id') else str(uuid.uuid4())[:8]
            
            # Create parent chunks
            parent_chunks = self._split_into_parents(text)
            
            for p_idx, parent_text in enumerate(parent_chunks):
                parent_id = f"{doc_id}_p{p_idx}"
                
                parent_node = DocumentNode(
                    id=parent_id,
                    text=parent_text,
                    source_doc_id=doc_id,
                    node_type="parent",
                    metadata={"parent_index": p_idx},
                )
                self.parent_nodes[parent_id] = parent_node
                
                # Create child chunks from parent
                child_chunks = self._split_into_children(parent_text)
                
                for c_idx, child_text in enumerate(child_chunks):
                    child_id = f"{parent_id}_c{c_idx}"
                    
                    # Generate embedding for child
                    embedding = self.embedding_manager.embed_query(child_text)
                    
                    child_node = DocumentNode(
                        id=child_id,
                        text=child_text,
                        parent_id=parent_id,
                        embedding=embedding,
                        source_doc_id=doc_id,
                        node_type="chunk",
                        metadata={"child_index": c_idx},
                    )
                    
                    self.child_nodes[child_id] = child_node
                    self.child_to_parent[child_id] = parent_id
                    parent_node.children_ids.append(child_id)
                    total_children += 1
        
        logger.info(f"Added {len(self.parent_nodes)} parents, {total_children} children")
        return total_children
    
    def _split_into_parents(self, text: str) -> list[str]:
        """Split text into parent-sized chunks."""
        chunk_size = self.config.parent_chunk_size
        chunks = []
        
        paragraphs = text.split('\n\n')
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) <= chunk_size:
                current += "\n\n" + para if current else para
            else:
                if current:
                    chunks.append(current)
                current = para
        
        if current:
            chunks.append(current)
        
        return chunks
    
    def _split_into_children(self, text: str) -> list[str]:
        """Split parent text into child-sized chunks."""
        chunk_size = self.config.child_chunk_size
        chunks = []
        
        sentences = text.replace('. ', '.\n').split('\n')
        current = ""
        
        for sent in sentences:
            if len(current) + len(sent) <= chunk_size:
                current += " " + sent if current else sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
        
        if current:
            chunks.append(current.strip())
        
        return chunks
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_parents: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve using small chunks, return parent documents.
        
        Args:
            query: Search query
            top_k: Number of results
            return_parents: Whether to return parent docs or just child matches
            
        Returns:
            RetrievalResult with parent documents
        """
        import time
        start_time = time.time()
        
        # Embed query
        query_embedding = np.array(self.embedding_manager.embed_query(query))
        
        # Search child nodes
        scored = []
        for child_id, child in self.child_nodes.items():
            if child.embedding:
                child_vec = np.array(child.embedding)
                similarity = np.dot(query_embedding, child_vec) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(child_vec) + 1e-8
                )
                scored.append((child, similarity))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        top_children = scored[:top_k * 2]  # Get more for parent dedup
        
        # Get unique parents
        seen_parents = set()
        result_nodes = []
        parent_docs = []
        
        for child, score in top_children:
            if return_parents:
                parent_id = self.child_to_parent.get(child.id)
                if parent_id and parent_id not in seen_parents:
                    parent = self.parent_nodes.get(parent_id)
                    if parent:
                        result_nodes.append(parent)
                        parent_docs.append(parent.text)
                        seen_parents.add(parent_id)
                        
                        if len(result_nodes) >= top_k:
                            break
            else:
                result_nodes.append(child)
                if len(result_nodes) >= top_k:
                    break
        
        return RetrievalResult(
            nodes=result_nodes,
            strategy_used=RetrievalStrategy.PARENT_DOCUMENT,
            parent_documents=parent_docs,
            num_unique_sources=len(set(n.source_doc_id for n in result_nodes)),
            avg_similarity=sum(s for _, s in top_children[:top_k]) / top_k if top_children else 0,
            processing_time_ms=(time.time() - start_time) * 1000,
        )
