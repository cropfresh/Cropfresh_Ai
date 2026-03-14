"""
RAPTOR Tree Index for multi-level abstraction retrieval.
"""

from typing import Optional

import numpy as np
from loguru import logger

from src.rag.raptor_pkg.models import (
    ClusterInfo,
    RAPTORConfig,
    RAPTORNode,
    RAPTORTreeStats,
)


class RAPTORIndex:
    """
    RAPTOR Hierarchical Tree Index for enhanced retrieval.
    
    Builds a multi-level tree from documents:
    - Level 0 (Leaf): Original document chunks
    - Level 1+: Cluster summaries at increasing abstraction
    - Top Level: High-level document summaries
    
    Supports two retrieval strategies:
    1. Collapsed: Search all levels simultaneously
    2. Tree Traversal: Start from root, expand relevant subtrees
    """
    
    # Summarization prompt
    SUMMARIZATION_PROMPT = """You are creating a summary for a group of related text chunks.
These chunks are from agricultural knowledge documents about farming, crops, and markets in India.

Please create a concise, informative summary that:
1. Captures the main topics and key information
2. Preserves important facts, numbers, and recommendations
3. Is self-contained and understandable without the original chunks
4. Is 2-3 paragraphs maximum

Chunks to summarize:
{chunks}

Summary:"""
    
    def __init__(
        self,
        embedding_manager,
        llm=None,
        knowledge_base=None,
        config: Optional[RAPTORConfig] = None,
    ):
        """
        Initialize RAPTOR index.
        
        Args:
            embedding_manager: For generating embeddings
            llm: For summarization (optional, uses simple extractive if None)
            knowledge_base: For storing nodes in Qdrant (optional)
            config: RAPTOR configuration
        """
        self.embedding_manager = embedding_manager
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.config = config or RAPTORConfig()
        
        # Tree structure
        self.nodes: dict[str, RAPTORNode] = {}
        self.levels: dict[int, list[str]] = {}  # Level -> list of node IDs
        self.root_ids: list[str] = []
        
        # Index state
        self._is_built = False
        self._build_stats: Optional[RAPTORTreeStats] = None
        
        logger.info("RAPTORIndex initialized")
    
    async def build_tree(
        self,
        documents: list,
        force_rebuild: bool = False,
    ) -> RAPTORTreeStats:
        """
        Build RAPTOR tree from documents.
        """
        if self._is_built and not force_rebuild:
            logger.info("RAPTOR tree already built, skipping")
            return self._build_stats
        
        logger.info(f"Building RAPTOR tree from {len(documents)} documents")
        
        # Reset state
        self.nodes = {}
        self.levels = {}
        self.root_ids = []
        
        # Step 1: Create leaf nodes from documents
        leaf_nodes = await self._create_leaf_nodes(documents)
        self.levels[0] = [node.id for node in leaf_nodes]
        
        for node in leaf_nodes:
            self.nodes[node.id] = node
        
        logger.info(f"Created {len(leaf_nodes)} leaf nodes")
        
        # Step 2: Build tree levels recursively
        current_level = 0
        current_nodes = leaf_nodes
        
        while len(current_nodes) > 1 and current_level < self.config.max_levels:
            # Cluster current level nodes
            clusters = await self._cluster_nodes(current_nodes)
            
            if len(clusters) == 0:
                break
            
            # Create summary nodes for each cluster
            summary_nodes = []
            for cluster in clusters:
                if len(cluster.node_ids) >= self.config.min_cluster_size:
                    summary_node = await self._create_summary_node(
                        cluster,
                        level=current_level + 1,
                    )
                    summary_nodes.append(summary_node)
                    self.nodes[summary_node.id] = summary_node
            
            if not summary_nodes:
                break
            
            current_level += 1
            self.levels[current_level] = [node.id for node in summary_nodes]
            current_nodes = summary_nodes
            
            logger.info(f"Level {current_level}: {len(summary_nodes)} nodes")
            
            # Check for convergence
            if len(summary_nodes) == 1 and self.config.stop_at_single_cluster:
                break
        
        # Mark root nodes
        self.root_ids = self.levels.get(current_level, [])
        
        # Calculate stats
        self._build_stats = self._calculate_stats()
        self._is_built = True
        
        logger.info(f"RAPTOR tree built: {self._build_stats.total_nodes} nodes, "
                   f"{self._build_stats.max_depth} levels")
        
        return self._build_stats
    
    async def _create_leaf_nodes(self, documents: list) -> list[RAPTORNode]:
        """Create leaf nodes from document chunks."""
        
        leaf_nodes = []
        
        for doc_idx, doc in enumerate(documents):
            # Get text content
            text = doc.text if hasattr(doc, 'text') else str(doc)
            doc_id = doc.id if hasattr(doc, 'id') else str(doc_idx)
            
            # Split into chunks using built-in splitter
            chunks = self._split_text(text)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_manager.embed_query(chunk_text)
                
                node = RAPTORNode(
                    text=chunk_text,
                    level=0,
                    embedding=embedding,
                    source_doc_ids=[doc_id],
                    chunk_indices=[chunk_idx],
                    metadata={
                        "doc_title": doc.metadata.get("title", "") if hasattr(doc, 'metadata') else "",
                        "doc_source": doc.metadata.get("source", "") if hasattr(doc, 'metadata') else "",
                    }
                )
                leaf_nodes.append(node)
        
        return leaf_nodes
    
    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks with overlap.
        
        Uses paragraph and sentence boundaries for cleaner splits.
        """
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph keeps us under limit
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Save current chunk if substantial
                if len(current_chunk) >= chunk_size // 4:
                    chunks.append(current_chunk)
                
                # Handle paragraph that's too long
                if len(para) > chunk_size:
                    # Split by sentences
                    sentences = para.replace('. ', '.\n').split('\n')
                    current_chunk = ""
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        
                        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                else:
                    current_chunk = para
        
        # Don't forget last chunk
        if current_chunk and len(current_chunk) >= chunk_size // 4:
            chunks.append(current_chunk)
        
        return chunks
    
    async def _cluster_nodes(
        self,
        nodes: list[RAPTORNode],
    ) -> list[ClusterInfo]:
        """
        Cluster nodes using UMAP + GMM.
        """
        if len(nodes) < self.config.min_cluster_size:
            return []
        
        # Get embeddings as numpy array
        embeddings = np.array([node.embedding for node in nodes])
        
        # Determine number of clusters
        n_clusters = max(2, int(len(nodes) * self.config.n_clusters_ratio))
        n_clusters = min(n_clusters, len(nodes) // self.config.min_cluster_size)
        
        if n_clusters < 2:
            return []
        
        try:
            # Optional UMAP dimensionality reduction
            if self.config.use_umap and len(nodes) > 20:
                try:
                    import umap
                    
                    n_components = min(self.config.umap_n_components, len(nodes) - 1)
                    n_neighbors = min(self.config.umap_n_neighbors, len(nodes) - 1)
                    
                    reducer = umap.UMAP(
                        n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=0.0,
                        metric='cosine',
                        random_state=42,
                    )
                    reduced_embeddings = reducer.fit_transform(embeddings)
                except ImportError:
                    logger.warning("UMAP not available, using raw embeddings")
                    reduced_embeddings = embeddings
            else:
                reduced_embeddings = embeddings
            
            # GMM clustering
            from sklearn.mixture import GaussianMixture
            
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type='full',
                random_state=42,
                max_iter=100,
            )
            cluster_labels = gmm.fit_predict(reduced_embeddings)
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, using simple grouping")
            # Simple fallback: group by chunks
            cluster_labels = [i % n_clusters for i in range(len(nodes))]
        
        # Build cluster info
        clusters = []
        for cluster_id in range(n_clusters):
            node_ids = [
                nodes[i].id for i in range(len(nodes))
                if cluster_labels[i] == cluster_id
            ]
            
            if len(node_ids) >= self.config.min_cluster_size:
                # Calculate centroid
                cluster_embeddings = embeddings[cluster_labels == cluster_id]
                centroid = cluster_embeddings.mean(axis=0).tolist()
                
                clusters.append(ClusterInfo(
                    cluster_id=cluster_id,
                    node_ids=node_ids,
                    centroid=centroid,
                ))
        
        return clusters
    
    async def _create_summary_node(
        self,
        cluster: ClusterInfo,
        level: int,
    ) -> RAPTORNode:
        """
        Create a summary node for a cluster.
        """
        # Get child nodes
        child_nodes = [self.nodes[nid] for nid in cluster.node_ids]
        
        # Combine texts for summarization
        combined_text = "\n\n---\n\n".join([node.text for node in child_nodes])
        
        # Generate summary
        if self.llm is not None:
            summary_text = await self._llm_summarize(combined_text)
        else:
            summary_text = self._extractive_summarize(child_nodes)
        
        # Generate embedding for summary
        embedding = self.embedding_manager.embed_query(summary_text)
        
        # Collect source document IDs from children
        all_doc_ids = []
        all_chunk_indices = []
        for node in child_nodes:
            all_doc_ids.extend(node.source_doc_ids)
            all_chunk_indices.extend(node.chunk_indices)
        
        # Create summary node
        summary_node = RAPTORNode(
            text=summary_text,
            level=level,
            embedding=embedding,
            children=cluster.node_ids,
            source_doc_ids=list(set(all_doc_ids)),
            chunk_indices=list(set(all_chunk_indices)),
            metadata={
                "cluster_id": cluster.cluster_id,
                "num_children": len(child_nodes),
            }
        )
        
        # Update children to point to parent
        for child_id in cluster.node_ids:
            self.nodes[child_id].parent = summary_node.id
        
        return summary_node
    
    async def _llm_summarize(self, text: str) -> str:
        """Generate summary using LLM."""
        try:
            prompt = self.SUMMARIZATION_PROMPT.format(chunks=text[:4000])  # Truncate to fit context
            
            # Use LLM to generate summary
            response = await self.llm.agenerate([prompt])
            summary = response.generations[0][0].text.strip()
            
            # Truncate if too long
            if len(summary) > self.config.max_summary_length * 4:  # Rough char estimate
                summary = summary[:self.config.max_summary_length * 4] + "..."
            
            return summary
            
        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return self._extractive_summarize_from_text(text)
    
    def _extractive_summarize(self, nodes: list[RAPTORNode]) -> str:
        """
        Simple extractive summarization fallback.
        
        Takes key sentences from each node.
        """
        summaries = []
        max_per_node = 2  # Max sentences per node
        
        for node in nodes[:5]:  # Limit to first 5 nodes
            sentences = node.text.split('. ')
            key_sentences = sentences[:max_per_node]
            summaries.append('. '.join(key_sentences))
        
        return ' '.join(summaries)[:self.config.max_summary_length * 4]
    
    def _extractive_summarize_from_text(self, text: str) -> str:
        """Extractive summarization from raw text."""
        sentences = text.split('. ')
        # Take every Nth sentence to get a summary
        step = max(1, len(sentences) // 5)
        key_sentences = sentences[::step][:5]
        return '. '.join(key_sentences)
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "collapsed",
        levels: Optional[list[int]] = None,
    ) -> list[RAPTORNode]:
        """
        Retrieve relevant nodes from RAPTOR tree.
        """
        if not self._is_built:
            logger.warning("RAPTOR tree not built, returning empty results")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_query(query)
        query_vec = np.array(query_embedding)
        
        if strategy == "collapsed":
            return self._collapsed_retrieval(query_vec, top_k, levels)
        elif strategy == "tree":
            return await self._tree_traversal_retrieval(query_vec, top_k)
        elif strategy == "mixed":
            # Combine both strategies
            collapsed = self._collapsed_retrieval(query_vec, top_k, levels)
            tree_results = await self._tree_traversal_retrieval(query_vec, top_k)
            
            # Merge and deduplicate
            seen_ids = set()
            merged = []
            for node in collapsed + tree_results:
                if node.id not in seen_ids:
                    merged.append(node)
                    seen_ids.add(node.id)
            
            return merged[:top_k]
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
    
    def _collapsed_retrieval(
        self,
        query_vec: np.ndarray,
        top_k: int,
        levels: Optional[list[int]] = None,
    ) -> list[RAPTORNode]:
        """
        Collapsed retrieval: search all levels simultaneously.
        """
        candidates = []
        
        for level, node_ids in self.levels.items():
            if levels is not None and level not in levels:
                continue
            
            for node_id in node_ids:
                node = self.nodes[node_id]
                if node.embedding:
                    # Cosine similarity
                    node_vec = np.array(node.embedding)
                    similarity = np.dot(query_vec, node_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(node_vec) + 1e-8
                    )
                    candidates.append((node, similarity))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, _ in candidates[:top_k]]
    
    async def _tree_traversal_retrieval(
        self,
        query_vec: np.ndarray,
        top_k: int,
    ) -> list[RAPTORNode]:
        """
        Tree traversal retrieval: start from root, expand relevant subtrees.
        """
        if not self.root_ids:
            return self._collapsed_retrieval(query_vec, top_k, None)
        
        # Start with root nodes
        to_explore = list(self.root_ids)
        relevant_nodes = []
        
        while to_explore and len(relevant_nodes) < top_k * 2:
            # Score current nodes
            scored = []
            for node_id in to_explore:
                node = self.nodes.get(node_id)
                if node and node.embedding:
                    node_vec = np.array(node.embedding)
                    similarity = np.dot(query_vec, node_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(node_vec) + 1e-8
                    )
                    scored.append((node, similarity))
            
            # Sort and keep top candidates
            scored.sort(key=lambda x: x[1], reverse=True)
            top_nodes = scored[:max(2, top_k // 2)]
            
            # Add to results
            for node, score in top_nodes:
                if node not in relevant_nodes:
                    relevant_nodes.append(node)
            
            # Expand children of top nodes
            to_explore = []
            for node, _ in top_nodes:
                to_explore.extend(node.children)
        
        return relevant_nodes[:top_k]
    
    def _calculate_stats(self) -> RAPTORTreeStats:
        """Calculate tree statistics."""
        total_nodes = len(self.nodes)
        nodes_per_level = {level: len(ids) for level, ids in self.levels.items()}
        max_depth = max(self.levels.keys()) if self.levels else 0
        
        # Calculate average children
        nodes_with_children = [n for n in self.nodes.values() if n.children]
        avg_children = (
            sum(len(n.children) for n in nodes_with_children) / len(nodes_with_children)
            if nodes_with_children else 0
        )
        
        # Average summary length (for non-leaf nodes)
        summaries = [n for n in self.nodes.values() if n.level > 0]
        avg_summary = (
            sum(len(n.text) for n in summaries) // len(summaries)
            if summaries else 0
        )
        
        # Unique source documents
        all_doc_ids = set()
        for node in self.nodes.values():
            all_doc_ids.update(node.source_doc_ids)
        
        return RAPTORTreeStats(
            total_nodes=total_nodes,
            nodes_per_level=nodes_per_level,
            max_depth=max_depth,
            total_documents=len(all_doc_ids),
            avg_children_per_node=avg_children,
            avg_summary_length=avg_summary,
        )
    
    def get_stats(self) -> Optional[RAPTORTreeStats]:
        """Get tree statistics."""
        return self._build_stats
    
    def get_node_by_id(self, node_id: str) -> Optional[RAPTORNode]:
        """Get a specific node by ID."""
        return self.nodes.get(node_id)
    
    def get_ancestors(self, node_id: str) -> list[RAPTORNode]:
        """Get all ancestors of a node (path to root)."""
        ancestors = []
        current_id = node_id
        
        while current_id:
            node = self.nodes.get(current_id)
            if node and node.parent:
                parent = self.nodes.get(node.parent)
                if parent:
                    ancestors.append(parent)
                current_id = node.parent
            else:
                break
        
        return ancestors
    
    def get_descendants(self, node_id: str) -> list[RAPTORNode]:
        """Get all descendants of a node (subtree)."""
        descendants = []
        to_visit = [node_id]
        
        while to_visit:
            current_id = to_visit.pop()
            node = self.nodes.get(current_id)
            if node:
                for child_id in node.children:
                    child = self.nodes.get(child_id)
                    if child:
                        descendants.append(child)
                        to_visit.append(child_id)
        
        return descendants
    
    def visualize_tree(self, max_text_len: int = 50) -> str:
        """Create ASCII visualization of the tree."""
        lines = ["RAPTOR Tree Structure", "=" * 40]
        
        for level in sorted(self.levels.keys(), reverse=True):
            node_ids = self.levels[level]
            lines.append(f"\nLevel {level} ({len(node_ids)} nodes):")
            lines.append("-" * 30)
            
            for node_id in node_ids[:5]:  # Limit display
                node = self.nodes[node_id]
                text_preview = node.text[:max_text_len] + "..." if len(node.text) > max_text_len else node.text
                text_preview = text_preview.replace('\n', ' ')
                lines.append(f"  [{node_id}] {text_preview}")
            
            if len(node_ids) > 5:
                lines.append(f"  ... and {len(node_ids) - 5} more nodes")
        
        return "\n".join(lines)


# Factory function
def create_raptor_index(
    embedding_manager,
    llm=None,
    knowledge_base=None,
    config: Optional[RAPTORConfig] = None,
) -> RAPTORIndex:
    """
    Create a configured RAPTOR index.
    
    Args:
        embedding_manager: For generating embeddings
        llm: For summarization (optional)
        knowledge_base: For storing nodes (optional)
        config: RAPTOR configuration
        
    Returns:
        RAPTORIndex instance
    """
    return RAPTORIndex(
        embedding_manager=embedding_manager,
        llm=llm,
        knowledge_base=knowledge_base,
        config=config,
    )
