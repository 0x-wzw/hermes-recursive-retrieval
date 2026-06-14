# Recursive Retrieval Skill

Hierarchical context traversal for Hermes memory layers.

## Overview

This skill implements recursive context retrieval across three memory layers:

- **L0 (Shared)**: Global context accessible to all agents
- **L1 (Agent)**: Agent-specific memory and patterns
- **L2 (Session)**: Session-local working context

The traversal accumulates context from L0 → L1 → L2, with higher layers overriding lower ones.

## Installation

```bash
# Clone the repository
git clone https://github.com/0x-wzw/hermes-recursive-retrieval.git
cd hermes-recursive-retrieval

# Install as module
pip install -e .
```

## Core Components

### ContextTree

The main class for hierarchical context traversal.

```python
from recursive_retrieval import ContextTree, TraversalConfig

# Create tree from memory layers
tree = ContextTree(
    shared_memory={"global_setting": "value"},
    agent_memory={"agent_name": "MyAgent"},
    session_memory={"session_id": "abc123"}
)

# Query a key (respects layer priority: session > agent > shared)
value = tree.query("key_name")

# Get fully merged context
merged = tree.get_merged_context()
```

### Traversal Configuration

Configure traversal depth, filtering, and merge strategy:

```python
config = TraversalConfig(
    max_depth=3,                    # Maximum traversal depth
    min_relevance=0.5,              # Minimum relevance score
    include_patterns=["config*"],   # Wildcard patterns to include
    exclude_patterns=["temp*"],     # Wildcard patterns to exclude
    merge_strategy="overlay"        # "overlay" or "combine"
)

# Traverse with config
for node in tree.traverse(config):
    print(f"{node.layer}: {node.key} = {node.value}")
```

### Merge Strategies

- **overlay**: Higher layers completely override lower layers (default)
- **combine**: Deep merge for nested dictionaries

```python
from recursive_retrieval import merge_context

# Overlay strategy
result = merge_context(shared, agent, session, strategy="overlay")

# Combine strategy (deep merge)
result = merge_context(shared, agent, session, strategy="combine")
```

## CLI Usage

### Basic Commands

```bash
# Merge all layers and output
cat memory.json | python cli.py

# Query a specific key
python cli.py --query "agent_name" -m memory.json

# Extract specific layer
python cli.py --layer agent -m memory.json

# Use combine strategy for nested merging
python cli.py --strategy combine -m memory.json
```

### Advanced Filtering

```bash
# Include/exclude patterns
python cli.py \
    --include "config*" \
    --include "preference*" \
    --exclude "*temp*" \
    --exclude "*cache*" \
    -m memory.json

# Traversal with depth and relevance limits
python cli.py traverse \
    --max-depth 2 \
    --min-relevance 0.8 \
    --limit 50 \
    -m memory.json
```

### Command Reference

| Command | Description |
|---------|-------------|
| `merge` | Merge layers with custom priority |
| `traverse` | List all nodes in traversal order |
| (default) | Query or full merge |

### Options

| Option | Description |
|--------|-------------|
| `-m, --memory` | Memory source JSON (default: stdin) |
| `-a, --agent-id` | Agent identifier |
| `-o, --output` | Output file (default: stdout) |
| `--max-depth` | Maximum traversal depth |
| `--min-relevance` | Minimum relevance score filter |
| `--strategy` | Merge strategy: overlay/combine |
| `--include` | Include patterns (repeatable) |
| `--exclude` | Exclude patterns (repeatable) |
| `--query` | Query specific key |
| `--layer` | Extract only specific layer |
| `--format` | Output format: json/compact |
| `-v, --verbose` | Verbose output |

## Episodic Memory Integration

Integrates with the episodic-memory skill:

```python
from recursive_retrieval import from_episodic_memory

# Load from episodic memory structure
episodic_data = {
    "shared": {"global": "context"},
    "agents": {
        "agent_1": {"agent_data": "value"}
    },
    "sessions": {
        "current": {"session_data": "active"}
    }
}

tree = from_episodic_memory(episodic_data, agent_id="agent_1")
```

## Memory Structure

Expected memory JSON structure:

```json
{
  "shared": {
    "system_prompt": "...",
    "global_config": {...}
  },
  "agent": {
    "agent_name": "MyAgent",
    "preferences": {...}
  },
  "session": {
    "session_id": "...",
    "working_context": {...}
  }
}
```

## Running Tests

```bash
# Run all tests
python -m pytest test_recursive.py -v

# Run specific test class
python -m pytest test_recursive.py::TestContextTree -v

# Run with coverage
python -m pytest test_recursive.py --cov=recursive_retrieval
```

## API Reference

### Classes

- `ContextTree`: Hierarchical context tree
- `ContextNode`: Individual context node
- `TraversalConfig`: Traversal configuration

### Functions

- `traverse(memory_source, agent_id, config)`: Create tree from dict
- `merge_context(shared, agent, session, strategy)`: Merge layers
- `from_episodic_memory(data, agent_id)`: Load from episodic format
- `extract_relevant_context(tree, query_keys, config)`: Filtered extraction

## License

MIT License - See LICENSE file
