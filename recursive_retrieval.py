"""
Recursive Retrieval Skill for Hermes

Hierarchical context traversal across memory layers:
- L0 (Shared): Global context accessible to all agents
- L1 (Agent): Agent-specific memory and patterns
- L2 (Session): Session-local working context
"""

from __future__ import annotations

import json
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


@dataclass
class ContextNode:
    """A single node in the context tree."""
    layer: str  # 'shared', 'agent', 'session'
    key: str
    value: Any
    metadata: dict = field(default_factory=dict)
    relevance_score: float = 1.0
    children: list[ContextNode] = field(default_factory=list)
    
    def __repr__(self) -> str:
        return f"ContextNode(layer={self.layer}, key={self.key}, relevance={self.relevance_score:.2f})"


@dataclass  
class TraversalConfig:
    """Configuration for context tree traversal."""
    max_depth: int = 3
    min_relevance: float = 0.0
    include_patterns: list[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: list[str] = field(default_factory=list)
    layer_priority: list[str] = field(default_factory=lambda: ["session", "agent", "shared"])
    merge_strategy: str = "overlay"  # "overlay" or "combine"


class ContextTree:
    """
    Hierarchical context tree for memory traversal.
    
    Traversal order: Shared (L0) → Agent (L1) → Session (L2)
    Each layer can override values from previous layers.
    """
    
    def __init__(
        self,
        shared_memory: dict | None = None,
        agent_memory: dict | None = None,
        session_memory: dict | None = None,
        agent_id: str = "default"
    ):
        self.agent_id = agent_id
        self.layers: dict[str, dict] = {
            "shared": shared_memory or {},
            "agent": agent_memory or {},
            "session": session_memory or {},
        }
        self.root = ContextNode("root", "root", {})
        self._build_tree()
    
    def _build_tree(self) -> None:
        """Build the hierarchical tree from memory layers."""
        # Build in order: shared → agent → session
        for layer in ["shared", "agent", "session"]:
            layer_data = self.layers[layer]
            self._add_layer_to_tree(layer, layer_data)
    
    def _add_layer_to_tree(self, layer: str, data: dict, parent: Optional[ContextNode] = None) -> None:
        """Recursively add a layer's data to the tree."""
        if parent is None:
            parent = self.root
        
        for key, value in data.items():
            node = ContextNode(
                layer=layer,
                key=key,
                value=value,
                metadata={"layer_index": ["shared", "agent", "session"].index(layer)}
            )
            
            # If value is a nested dict, recurse
            if isinstance(value, dict):
                self._add_layer_to_tree(layer, value, node)
            
            parent.children.append(node)
    
    def traverse(
        self,
        config: TraversalConfig | None = None,
        callback: Callable[[ContextNode], None] | None = None
    ) -> Iterator[ContextNode]:
        """
        Traverse the context tree with configurable depth and filtering.
        
        Args:
            config: Traversal configuration
            callback: Optional callback for each node
            
        Yields:
            ContextNode: Nodes in traversal order
        """
        config = config or TraversalConfig()
        visited = set()
        
        def _traverse_node(
            node: ContextNode,
            depth: int,
            layer_accumulator: dict[str, Any]
        ) -> Iterator[ContextNode]:
            if depth > config.max_depth:
                return
            
            if node.key in visited:
                return
            
            # Check relevance filtering
            if node.relevance_score < config.min_relevance:
                return
            
            # Check include/exclude patterns
            if not self._matches_patterns(node.key, config.include_patterns, config.exclude_patterns):
                return
            
            visited.add(node.key)
            
            # Accumulate context by layer
            if node.layer in config.layer_priority:
                layer_accumulator[node.key] = {
                    "value": node.value,
                    "layer": node.layer,
                    "relevance": node.relevance_score
                }
            
            if callback:
                callback(node)
            
            yield node
            
            # Traverse children
            for child in node.children:
                yield from _traverse_node(child, depth + 1, layer_accumulator)
        
        layer_data: dict[str, Any] = {}
        for child in self.root.children:
            yield from _traverse_node(child, 0, layer_data)
    
    def _matches_patterns(
        self,
        key: str,
        include: list[str],
        exclude: list[str]
    ) -> bool:
        """Check if key matches include patterns and doesn't match exclude."""
        # Must match at least one include pattern
        included = any(fnmatch.fnmatch(key, pat) for pat in include)
        # Must not match any exclude pattern
        excluded = any(fnmatch.fnmatch(key, pat) for pat in exclude)
        return included and not excluded
    
    def get_merged_context(self, config: TraversalConfig | None = None) -> dict[str, Any]:
        """
        Get the fully merged context from all layers.
        
        Returns context with layer precedence applied:
        session > agent > shared
        """
        config = config or TraversalConfig()
        return merge_context(
            self.layers["shared"],
            self.layers["agent"],
            self.layers["session"],
            strategy=config.merge_strategy
        )
    
    def query(self, key: str, default: Any = None) -> Any:
        """
        Query a specific key, returning the highest-layer value.
        
        Search order: session → agent → shared
        """
        for layer in ["session", "agent", "shared"]:
            if key in self.layers[layer]:
                return self.layers[layer][key]
        return default
    
    def get_layer(self, layer: str) -> dict:
        """Get a specific memory layer."""
        return self.layers.get(layer, {}).copy()
    
    def set_session_context(self, data: dict) -> None:
        """Update the session context layer."""
        self.layers["session"] = data
        self._build_tree()


def merge_context(
    shared: dict[str, Any],
    agent: dict[str, Any],
    session: dict[str, Any],
    strategy: str = "overlay"
) -> dict[str, Any]:
    """
    Merge three context layers with configurable strategy.
    
    Layers (in precedence order, highest first):
    - session (L2): Working session context
    - agent (L1): Agent-specific memory
    - shared (L0): Global shared context
    
    Args:
        shared: L0 shared context
        agent: L1 agent context
        session: L2 session context
        strategy: "overlay" (later layers override) or "combine" (merge nested)
        
    Returns:
        Merged context dictionary
    """
    if strategy == "overlay":
        # Simple overlay: session > agent > shared
        result = shared.copy()
        result.update(agent)
        result.update(session)
        return result
    
    elif strategy == "combine":
        # Deep merge for nested structures
        def deep_merge(base: Any, overlay: Any) -> Any:
            if isinstance(base, dict) and isinstance(overlay, dict):
                merged = base.copy()
                for key, value in overlay.items():
                    if key in merged and isinstance(merged[key], dict):
                        merged[key] = deep_merge(merged[key], value)
                    else:
                        merged[key] = value
                return merged
            return overlay
        
        result = deep_merge(shared, agent)
        result = deep_merge(result, session)
        return result
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")


def traverse(
    memory_source: dict[str, dict],
    agent_id: str = "default",
    config: TraversalConfig | None = None
) -> ContextTree:
    """
    Create and traverse a context tree from memory source.
    
    Args:
        memory_source: Dict with keys 'shared', 'agent', 'session'
        agent_id: Identifier for the agent
        config: Optional traversal configuration
        
    Returns:
        ContextTree instance
    """
    tree = ContextTree(
        shared_memory=memory_source.get("shared", {}),
        agent_memory=memory_source.get("agent", {}),
        session_memory=memory_source.get("session", {}),
        agent_id=agent_id
    )
    return tree


# Convenience functions for episodic memory integration
def from_episodic_memory(
    episodic_data: dict,
    agent_id: str = "default"
) -> ContextTree:
    """
    Create ContextTree from episodic memory format.
    
    Expected format:
    {
        "shared": {...},
        "agents": {
            "agent_id": {...}
        },
        "sessions": {
            "session_id": {...}
        }
    }
    """
    return ContextTree(
        shared_memory=episodic_data.get("shared", {}),
        agent_memory=episodic_data.get("agents", {}).get(agent_id, {}),
        session_memory=episodic_data.get("sessions", {}).get("current", {}),
        agent_id=agent_id
    )


def extract_relevant_context(
    tree: ContextTree,
    query_keys: list[str],
    config: TraversalConfig | None = None
) -> dict[str, Any]:
    """
    Extract only relevant context entries matching query keys.
    
    Args:
        tree: ContextTree to search
        query_keys: List of keys to extract (supports wildcards)
        config: Traversal configuration
        
    Returns:
        Filtered context dictionary
    """
    config = config or TraversalConfig()
    config.include_patterns = query_keys
    
    merged = tree.get_merged_context(config)
    
    # Filter by patterns
    result = {}
    for key, value in merged.items():
        if any(fnmatch.fnmatch(key, pat) for pat in query_keys):
            result[key] = value
    
    return result
