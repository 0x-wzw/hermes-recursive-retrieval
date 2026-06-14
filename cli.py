#!/usr/bin/env python3
"""
Recursive Retrieval CLI
Command-line interface for hierarchical context traversal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from recursive_retrieval import (
    ContextTree,
    TraversalConfig,
    merge_context,
    traverse,
)


def load_memory_source(path: str) -> dict:
    """Load memory source from JSON file or stdin."""
    if path == "-":
        return json.load(sys.stdin)
    
    with open(path, "r") as f:
        return json.load(f)


def save_output(data: dict, path: str | None) -> None:
    """Save output to file or stdout."""
    output = json.dumps(data, indent=2, default=str)
    if path:
        with open(path, "w") as f:
            f.write(output)
        print(f"Output written to: {path}", file=sys.stderr)
    else:
        print(output)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="recursive-retrieval",
        description="Hierarchical context traversal for Hermes memory layers"
    )
    
    parser.add_argument(
        "-m", "--memory",
        default="-",
        help="Path to memory source JSON (default: stdin)"
    )
    
    parser.add_argument(
        "-a", "--agent-id",
        default="default",
        help="Agent identifier (default: default)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum traversal depth (default: 3)"
    )
    
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=0.0,
        help="Minimum relevance score filter (default: 0.0)"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["overlay", "combine"],
        default="overlay",
        help="Merge strategy: overlay (override) or combine (deep merge)"
    )
    
    parser.add_argument(
        "--include",
        action="append",
        default=["*"],
        help="Include patterns (can be repeated, default: *)"
    )
    
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude patterns (can be repeated)"
    )
    
    parser.add_argument(
        "--query",
        help="Query a specific key (supports wildcards)"
    )
    
    parser.add_argument(
        "--layer",
        choices=["shared", "agent", "session", "all"],
        default="all",
        help="Extract only specific layer (default: all)"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "compact"],
        default="json",
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output with traversal details"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # merge command
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge specific layers and output result"
    )
    merge_parser.add_argument(
        "--layers",
        default="session,agent,shared",
        help="Comma-separated layer order (highest priority first)"
    )
    
    # traverse command
    traverse_parser = subparsers.add_parser(
        "traverse",
        help="Traverse context tree and list nodes"
    )
    traverse_parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum nodes to output (default: 100)"
    )
    
    return parser


def cmd_merge(args: argparse.Namespace) -> None:
    """Execute merge command."""
    memory = load_memory_source(args.memory)
    
    tree = traverse(memory, args.agent_id)
    
    config = TraversalConfig(
        max_depth=args.max_depth,
        min_relevance=args.min_relevance,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        merge_strategy=args.strategy
    )
    
    result = tree.get_merged_context(config)
    
    # Filter by layer if specified
    if args.layer != "all":
        layer_data = tree.get_layer(args.layer)
        # Intersection with merged context
        result = {k: v for k, v in result.items() if k in layer_data}
    
    output = {"merged_context": result}
    
    if args.verbose:
        output["config"] = {
            "strategy": args.strategy,
            "max_depth": args.max_depth,
            "agent_id": args.agent_id
        }
    
    save_output(output, args.output)


def cmd_traverse(args: argparse.Namespace) -> None:
    """Execute traverse command."""
    memory = load_memory_source(args.memory)
    
    tree = traverse(memory, args.agent_id)
    
    config = TraversalConfig(
        max_depth=args.max_depth,
        min_relevance=args.min_relevance,
        include_patterns=args.include,
        exclude_patterns=args.exclude
    )
    
    nodes = []
    count = 0
    for node in tree.traverse(config):
        if count >= args.limit:
            break
        nodes.append({
            "layer": node.layer,
            "key": node.key,
            "value": node.value,
            "relevance": node.relevance_score,
            "child_count": len(node.children)
        })
        count += 1
    
    output = {
        "nodes": nodes,
        "total_traversed": count
    }
    
    if args.verbose:
        output["config"] = {
            "max_depth": args.max_depth,
            "min_relevance": args.min_relevance,
            "include_patterns": args.include,
            "exclude_patterns": args.exclude
        }
    
    save_output(output, args.output)


def cmd_default(args: argparse.Namespace) -> None:
    """Execute default command (query or full merge)."""
    memory = load_memory_source(args.memory)
    
    tree = traverse(memory, args.agent_id)
    
    config = TraversalConfig(
        max_depth=args.max_depth,
        min_relevance=args.min_relevance,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        merge_strategy=args.strategy
    )
    
    # Query mode
    if args.query:
        result = tree.query(args.query)
        output = {
            "query": args.query,
            "result": result
        }
    else:
        # Full merge mode
        merged = tree.get_merged_context(config)
        output = {"merged_context": merged}
        
        if args.verbose:
            output["layers"] = {
                "shared": tree.get_layer("shared"),
                "agent": tree.get_layer("agent"),
                "session": tree.get_layer("session")
            }
    
    if args.format == "compact":
        print(json.dumps(output))
    else:
        save_output(output, args.output)


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        if args.command == "merge":
            cmd_merge(args)
        elif args.command == "traverse":
            cmd_traverse(args)
        else:
            cmd_default(args)
        return 0
    
    except FileNotFoundError as e:
        print(f"Error: Memory file not found: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
