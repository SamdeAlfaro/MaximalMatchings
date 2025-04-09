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

    # Calculate the limit on number of edges
    edge_limit = max_edges if max_edges is not None else num_edges

    # Try to add edges until the number of edges reaches the limit or all degrees are within the limit
    attempts = 0
    max_attempts = num_vertices * max_degree * 2  # avoid infinite loop

    while attempts < max_attempts and (edge_limit is None or sum(len(neighbors) for neighbors in adjacency.values()) // 2 < edge_limit):
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)

        if u == v:
            attempts += 1
            continue  # no self-loops

        if v in adjacency[u] or u in adjacency[v]:
            attempts += 1
            continue  # no duplicate edges

        if len(adjacency[u]) >= max_degree or len(adjacency[v]) >= max_degree:
            attempts += 1
            continue  # respect max degree

        # Add edge
        adjacency[u].add(v)
        adjacency[v].add(u)

    # Calculate number of edges
    num_edges_generated = sum(len(neighbors) for neighbors in adjacency.values()) // 2  # undirected graph

    # Calculate average degree
    total_degree = sum(len(neighbors) for neighbors in adjacency.values())
    avg_degree = total_degree / num_vertices if num_vertices > 0 else 0

    # Convert sets to sorted lists for prettier output
    return {
        "graph": {k: sorted(list(vs)) for k, vs in adjacency.items()},
        "num_edges": num_edges_generated,
        "avg_degree": avg_degree
    }

# ______________________ ROG-MM ______________________

def random_order_greedy_maximal_matching(graph):
    # List of edges in the graph
    edges = []
    for u, neighbors in graph.items():
        for v in neighbors:
            if u < v:  # Ensure each edge is only considered once (undirected graph)
                edges.append((u, v))
    
    # Randomly shuffle the edges to create a random order
    random.shuffle(edges)
    
    # Initialize the matching set and a set to keep track of matched vertices
    matching = set()
    matched_vertices = set()

    # Process edges in random order
    for u, v in edges:
        # If neither u nor v is in the matching, add the edge to the matching
        if u not in matched_vertices and v not in matched_vertices:
            matching.add((u, v))
            matched_vertices.add(u)
            matched_vertices.add(v)

    return matching

# ______________________ Optimal Matching ______________________

def optimal_matching(graph):
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add edges to the graph
    for u, neighbors in graph.items():
        for v in neighbors:
            if u < v:  # Prevent adding duplicate edges (undirected graph)
                G.add_edge(u, v)
    
    # Find the maximum matching using the Blossom algorithm
    matching = nx.max_weight_matching(G, maxcardinality=True, weight=None)

    return matching


# ______________________ Example usage ______________________

if __name__ == "__main__":
    num_vertices = 10
    max_degree = 3
    num_edges = 7  # specify the exact number of edges you want
    result = generate_random_graph(num_vertices, max_degree, num_edges=num_edges)
    
    # Print the graph
    for node, neighbors in result['graph'].items():
        print(f"{node}: {neighbors}")
    
    # Print number of edges and average degree
    print(f"Number of vertices: {num_vertices}")
    print(f"Number of edges: {result['num_edges']}")
    print(f"Average degree: {result['avg_degree']:.2f}")

    # Get the graph from the result
    graph = result['graph']

    # Call the matching algorithm with the generated graph
    rogmatching = random_order_greedy_maximal_matching(graph)

    # Print the matching result
    print("Maximal matching:", rogmatching)

    optmatching = optimal_matching(graph)

    print("Optimal Matching:", optmatching)




