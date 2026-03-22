"""
Unit tests for recursive retrieval skill.
"""

import json
import sys
import unittest
from io import StringIO
from pathlib import Path

from recursive_retrieval import (
    ContextTree,
    ContextNode,
    TraversalConfig,
    merge_context,
    traverse,
    extract_relevant_context,
)
from cli import create_parser, load_memory_source


class TestContextNode(unittest.TestCase):
    """Test ContextNode dataclass."""
    
    def test_basic_creation(self):
        """Test creating a basic context node."""
        node = ContextNode(layer="shared", key="test_key", value="test_value")
        self.assertEqual(node.layer, "shared")
        self.assertEqual(node.key, "test_key")
        self.assertEqual(node.value, "test_value")
        self.assertEqual(node.relevance_score, 1.0)
    
    def test_node_with_metadata(self):
        """Test node with metadata."""
        node = ContextNode(
            layer="agent",
            key="pref",
            value={"theme": "dark"},
            metadata={"version": "1.0"}
        )
        self.assertEqual(node.metadata["version"], "1.0")


class TestMergeContext(unittest.TestCase):
    """Test merge_context function."""
    
    def test_overlay_strategy(self):
        """Test overlay merge strategy."""
        shared = {"key1": "shared_value", "key2": "shared_value2"}
        agent = {"key2": "agent_value"}
        session = {"key1": "session_value"}
        
        result = merge_context(shared, agent, session, strategy="overlay")
        
        self.assertEqual(result["key1"], "session_value")  # session overrides
        self.assertEqual(result["key2"], "agent_value")     # agent overrides shared
    
    def test_combine_strategy_simple(self):
        """Test combine strategy with simple values."""
        shared = {"a": 1}
        agent = {"b": 2}
        session = {"c": 3}
        
        result = merge_context(shared, agent, session, strategy="combine")
        
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 2)
        self.assertEqual(result["c"], 3)
    
    def test_combine_strategy_nested(self):
        """Test combine strategy with nested dicts."""
        shared = {"config": {"theme": "light", "lang": "en"}}
        agent = {"config": {"theme": "dark"}}
        session = {"config": {"lang": "fr"}}
        
        result = merge_context(shared, agent, session, strategy="combine")
        
        # Deep merge: session's lang + agent's theme (agent applied before session)
        self.assertIn("config", result)
        self.assertEqual(result["config"]["theme"], "dark")
        self.assertEqual(result["config"]["lang"], "fr")
    
    def test_invalid_strategy(self):
        """Test invalid merge strategy raises error."""
        with self.assertRaises(ValueError):
            merge_context({}, {}, {}, strategy="invalid")


class TestContextTree(unittest.TestCase):
    """Test ContextTree class."""
    
    def setUp(self):
        """Set up mock memory structure."""
        self.shared_memory = {
            "system_prompt": "You are a helpful assistant",
            "config": {"max_tokens": 1000, "model": "gpt-4"},
            "global_setting": "shared_value"
        }
        self.agent_memory = {
            "agent_name": "TestAgent",
            "agent_pref": {"verbosity": "low"},
            "config": {"model": "gpt-3.5"}  # Overrides shared
        }
        self.session_memory = {
            "session_id": "sess_123",
            "temp_data": {"working": True},
            "agent_pref": {"verbosity": "high"}  # Overrides agent
        }
    
    def test_tree_creation(self):
        """Test creating a context tree."""
        tree = ContextTree(
            shared_memory=self.shared_memory,
            agent_memory=self.agent_memory,
            session_memory=self.session_memory
        )
        
        self.assertEqual(tree.agent_id, "default")
        self.assertIn("shared", tree.layers)
        self.assertIn("agent", tree.layers)
        self.assertIn("session", tree.layers)
    
    def test_query_priority(self):
        """Test query respects layer priority."""
        tree = ContextTree(
            shared_memory=self.shared_memory,
            agent_memory=self.agent_memory,
            session_memory=self.session_memory
        )
        
        # config exists in shared and agent - agent should win
        self.assertEqual(tree.query("config"), self.agent_memory["config"])
        
        # session_id only in session
        self.assertEqual(tree.query("session_id"), "sess_123")
        
        # non-existent key
        self.assertIsNone(tree.query("non_existent"))
    
    def test_query_default(self):
        """Test query with default value."""
        tree = ContextTree(
            shared_memory=self.shared_memory,
            agent_memory=self.agent_memory,
            session_memory=self.session_memory
        )
        
        result = tree.query("non_existent", default="default_value")
        self.assertEqual(result, "default_value")
    
    def test_get_layer(self):
        """Test getting specific layer."""
        tree = ContextTree(
            shared_memory=self.shared_memory,
            agent_memory=self.agent_memory,
            session_memory=self.session_memory
        )
        
        shared = tree.get_layer("shared")
        self.assertEqual(shared["system_prompt"], "You are a helpful assistant")
        
        # Should return copy
        shared["new_key"] = "new_value"
        self.assertNotIn("new_key", tree.layers["shared"])
    
    def test_get_merged_context_overlay(self):
        """Test merged context with overlay strategy."""
        tree = ContextTree(
            shared_memory=self.shared_memory,
            agent_memory=self.agent_memory,
            session_memory=self.session_memory
        )
        
        config = TraversalConfig(merge_strategy="overlay")
        merged = tree.get_merged_context(config)
        
        # With overlay strategy, agent's config completely replaces shared's config
        # (full dict replacement, not key-by-key merge)
        self.assertEqual(merged["config"]["model"], "gpt-3.5")
        # max_tokens is NOT preserved because agent's config doesn't have it
        self.assertNotIn("max_tokens", merged["config"])
    
    def test_traversal_config(self):
        """Test traversal with configuration."""
        tree = ContextTree(
            shared_memory=self.shared_memory,
            agent_memory=self.agent_memory,
            session_memory=self.session_memory
        )
        
        config = TraversalConfig(
            max_depth=2,
            min_relevance=0.5,
            include_patterns=["config*"]
        )
        
        nodes = list(tree.traverse(config))
        
        # Should find config node
        config_nodes = [n for n in nodes if n.key == "config"]
        self.assertTrue(len(config_nodes) > 0)
    
    def test_set_session_context(self):
        """Test updating session context."""
        tree = ContextTree(
            shared_memory=self.shared_memory,
            agent_memory=self.agent_memory,
            session_memory=self.session_memory
        )
        
        new_session = {"new_key": "new_value"}
        tree.set_session_context(new_session)
        
        self.assertEqual(tree.query("new_key"), "new_value")


class TestTraverseFunction(unittest.TestCase):
    """Test traverse convenience function."""
    
    def test_traverse_from_dict(self):
        """Test traverse function with dict input."""
        memory_source = {
            "shared": {"global": "value"},
            "agent": {"agent": "data"},
            "session": {"session": "info"}
        }
        
        tree = traverse(memory_source, agent_id="test_agent")
        
        self.assertEqual(tree.agent_id, "test_agent")
        self.assertEqual(tree.query("global"), "value")


class TestExtractRelevantContext(unittest.TestCase):
    """Test extract_relevant_context function."""
    
    def test_extract_with_wildcards(self):
        """Test extracting with wildcard patterns."""
        memory_source = {
            "shared": {"config_theme": "dark", "config_lang": "en", "other": "value"},
            "agent": {},
            "session": {}
        }
        
        tree = traverse(memory_source)
        result = extract_relevant_context(tree, ["config*"])
        
        self.assertIn("config_theme", result)
        self.assertIn("config_lang", result)
        self.assertNotIn("other", result)


class TestCLI(unittest.TestCase):
    """Test CLI functionality."""
    
    def test_parser_creation(self):
        """Test CLI parser creation."""
        parser = create_parser()
        self.assertIsNotNone(parser)
    
    def test_load_memory_from_stringio(self):
        """Test loading memory from file-like object."""
        test_data = {"shared": {"key": "value"}}
        
        # Simulate stdin
        old_stdin = sys.stdin
        sys.stdin = StringIO(json.dumps(test_data))
        
        result = load_memory_source("-")
        self.assertEqual(result, test_data)
        
        sys.stdin = old_stdin


class TestPatternMatching(unittest.TestCase):
    """Test pattern matching in traversal."""
    
    def test_include_patterns(self):
        """Test include pattern filtering."""
        tree = ContextTree(
            shared_memory={"alpha_key": 1, "beta_key": 2, "gamma": 3},
            agent_memory={},
            session_memory={}
        )
        
        config = TraversalConfig(include_patterns=["*key"])
        nodes = list(tree.traverse(config))
        
        keys = [n.key for n in nodes]
        self.assertIn("alpha_key", keys)
        self.assertIn("beta_key", keys)
        self.assertNotIn("gamma", keys)
    
    def test_exclude_patterns(self):
        """Test exclude pattern filtering."""
        tree = ContextTree(
            shared_memory={"include_this": 1, "exclude_that": 2, "other": 3},
            agent_memory={},
            session_memory={}
        )
        
        config = TraversalConfig(
            include_patterns=["*"],
            exclude_patterns=["exclude*"]
        )
        nodes = list(tree.traverse(config))
        
        keys = [n.key for n in nodes]
        self.assertIn("include_this", keys)
        self.assertNotIn("exclude_that", keys)


class TestEpisodicMemoryIntegration(unittest.TestCase):
    """Test integration with episodic memory format."""
    
    def test_from_episodic_memory(self):
        """Test loading from episodic memory structure."""
        episodic_data = {
            "shared": {"global_config": "value"},
            "agents": {
                "agent_1": {"agent_specific": "data_1"},
                "agent_2": {"agent_specific": "data_2"}
            },
            "sessions": {
                "current": {"session_data": "active"},
                "archived": {"session_data": "old"}
            }
        }
        
        from recursive_retrieval import from_episodic_memory
        tree = from_episodic_memory(episodic_data, agent_id="agent_1")
        
        self.assertEqual(tree.query("global_config"), "value")
        self.assertEqual(tree.query("agent_specific"), "data_1")
        self.assertEqual(tree.query("session_data"), "active")


def create_mock_memory_file(tmp_path) -> Path:
    """Create a mock memory file for testing."""
    memory_data = {
        "shared": {"system": "prompt"},
        "agent": {"agent_name": "Test"},
        "session": {"session_id": "123"}
    }
    
    file_path = tmp_path / "test_memory.json"
    with open(file_path, "w") as f:
        json.dump(memory_data, f)
    
    return file_path


if __name__ == "__main__":
    unittest.main()
