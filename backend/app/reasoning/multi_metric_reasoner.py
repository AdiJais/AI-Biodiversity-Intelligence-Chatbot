"""
Multi-Metric Reasoning Agent.

This is the core differentiator the brief calls out explicitly: "no
single-variable answers." Given the set of known variable keys for a site,
this module walks data/variable_interactions.json to find the chains of
relationships that connect the user's variables to biodiversity outcomes,
and returns them as plain-language statements plus the KB evidence ids that
back each edge - so multi-metric reasoning is never just asserted, it's
traceable to the same evidence layer as the recommendations.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.knowledge.loader import load_interaction_graph


def relevant_edges(known_variable_keys: List[str]) -> List[Dict[str, Any]]:
    graph = load_interaction_graph()
    known = set(known_variable_keys)
    edges = []
    for edge in graph["edges"]:
        if edge["from"] in known or edge["to"] in known:
            edges.append(edge)
    return edges


def build_chains(known_variable_keys: List[str], max_hops: int = 3) -> List[List[Dict[str, Any]]]:
    """
    Builds short causal chains (up to max_hops edges) starting from any
    known variable, so a two- or three-variable input can surface something
    like: rainfall -> soil_moisture -> species_richness.
    """
    graph = load_interaction_graph()
    edges_by_from = {}
    for e in graph["edges"]:
        edges_by_from.setdefault(e["from"], []).append(e)

    chains: List[List[Dict[str, Any]]] = []
    known = set(known_variable_keys)

    def dfs(path: List[Dict[str, Any]], current_var: str, depth: int, visited: set):
        if depth >= max_hops:
            return
        for edge in edges_by_from.get(current_var, []):
            if edge["to"] in visited:
                continue
            new_path = path + [edge]
            chains.append(new_path)
            dfs(new_path, edge["to"], depth + 1, visited | {edge["to"]})

    for var in known:
        dfs([], var, 0, {var})

    # Keep chains that start from a variable the user actually gave us,
    # de-duplicate by the tuple of edge ids, prefer longer chains first.
    seen = set()
    unique_chains = []
    for c in sorted(chains, key=len, reverse=True):
        key = tuple(e["id"] for e in c)
        if key not in seen:
            seen.add(key)
            unique_chains.append(c)
    return unique_chains[:6]


def summarize_relationships(known_variable_keys: List[str]) -> List[str]:
    """Human-readable one-liners describing which of the user's variables interact and how."""
    chains = build_chains(known_variable_keys)
    lines = []
    for chain in chains:
        if not chain:
            continue
        path_vars = [chain[0]["from"]] + [e["to"] for e in chain]
        arrow = " -> ".join(path_vars)
        mechanism_text = " Then: ".join(e["relationship"] for e in chain)
        lines.append(f"{arrow}: {mechanism_text}")
    return lines


def all_supporting_kb_ids(known_variable_keys: List[str]) -> List[str]:
    ids: List[str] = []
    for edge in relevant_edges(known_variable_keys):
        for kb_id in edge.get("supporting_kb", []):
            if kb_id not in ids:
                ids.append(kb_id)
    return ids
