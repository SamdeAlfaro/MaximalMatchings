import random
from collections import defaultdict
import networkx as nx

# ______________________ Generate graph ______________________

def generate_random_graph(num_vertices, max_degree, num_edges=None, max_edges=None):
    if max_degree >= num_vertices:
        raise ValueError("max_degree must be less than the number of vertices for a simple graph.")
    
    if num_edges and max_edges:
        raise ValueError("Specify only one of num_edges or max_edges, not both.")

    adjacency = defaultdict(set)

    edge_limit = max_edges if max_edges is not None else num_edges

    attempts = 0
    max_attempts = num_vertices * max_degree * 2

    while attempts < max_attempts and (edge_limit is None or sum(len(neighbors) for neighbors in adjacency.values()) // 2 < edge_limit):
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)

        if u == v:
            attempts += 1
            continue

        if v in adjacency[u] or u in adjacency[v]:
            attempts += 1
            continue  # no duplicate edges

        if len(adjacency[u]) >= max_degree or len(adjacency[v]) >= max_degree:
            attempts += 1
            continue  # respect max degree

        adjacency[u].add(v)
        adjacency[v].add(u)

    num_edges_generated = sum(len(neighbors) for neighbors in adjacency.values()) // 2  # undirected graph

    total_degree = sum(len(neighbors) for neighbors in adjacency.values())
    avg_degree = total_degree / num_vertices if num_vertices > 0 else 0

    return {
        "graph": {k: sorted(list(vs)) for k, vs in adjacency.items()},
        "num_edges": num_edges_generated,
        "avg_degree": avg_degree
    }

# ______________________ ROG-MM ______________________

def random_order_greedy_maximal_matching(graph):
    edges = []
    for u, neighbors in graph.items():
        for v in neighbors:
            if u < v:
                edges.append((u, v))
    
    random.shuffle(edges)
    
    matching = set()
    matched_vertices = set()

    for u, v in edges:
        if u not in matched_vertices and v not in matched_vertices:
            matching.add((u, v))
            matched_vertices.add(u)
            matched_vertices.add(v)

    return matching

# ______________________ Optimal Matching ______________________

def optimal_matching(graph):
    G = nx.Graph()
    
    for u, neighbors in graph.items():
        for v in neighbors:
            if u < v:
                G.add_edge(u, v)
    
    matching = nx.max_weight_matching(G, maxcardinality=True, weight=None) # Blossom alg

    return matching


if __name__ == "__main__":
    num_vertices = 10
    max_degree = 3
    num_edges = 7
    result = generate_random_graph(num_vertices, max_degree, num_edges=num_edges)
    
    for node, neighbors in result['graph'].items():
        print(f"{node}: {neighbors}")
    
    print(f"Number of vertices: {num_vertices}")
    print(f"Number of edges: {result['num_edges']}")
    print(f"Average degree: {result['avg_degree']:.2f}")

    graph = result['graph']

    rogmatching = random_order_greedy_maximal_matching(graph)

    print("Maximal matching:", rogmatching)

    optmatching = optimal_matching(graph)

    print("Optimal Matching:", optmatching)




