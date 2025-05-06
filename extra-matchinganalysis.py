import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

# Generate d-regular graph
def generate_d_regular_graph(n, d):
    while True:
        G = nx.random_regular_graph(d, n)
        if nx.is_connected(G):
            return G

def rogmm_matching(graph):
    matching = set()
    vertices = list(graph.keys())
    random.shuffle(vertices)

    matched = set()
    for u in vertices:
        if u not in matched:
            for v in graph[u]:
                if v not in matched:
                    matching.add((u, v))
                    matched.add(u)
                    matched.add(v)
                    break

    return matching

def run_experiment(n, min_d, max_d, steps, num_trials=10):
    ratios = []

    for d in range(min_d, max_d + 1, steps):
        matched_fractions = []
        
        for _ in range(num_trials):
            G = generate_d_regular_graph(n, d)
            
            graph_adj_list = {node: list(G.neighbors(node)) for node in G.nodes}
            
            matching = rogmm_matching(graph_adj_list)
            
            matched_nodes = set()
            for u, v in matching:
                matched_nodes.add(u)
                matched_nodes.add(v)

            matched_fraction = len(matched_nodes) / n
            matched_fractions.append(matched_fraction)

        avg_matched_fraction = sum(matched_fractions) / num_trials
        ratios.append((d/n, avg_matched_fraction))

        print(f"Graph with d={d} and n={n}: Avg Matched Fraction = {avg_matched_fraction:.3f}")

    return ratios

# Plot the relationship between d/n and matched fraction
def plot_results(ratios):
    d_n_values, matched_fractions = zip(*ratios)
    
    plt.figure(figsize=(10, 6))
    plt.plot(d_n_values, matched_fractions, marker='o', linestyle='-', color='b')
    plt.xlabel("d/n (degree / number of nodes)")
    plt.ylabel("Average Matched Fraction")
    plt.title("RGO-MM: Matched Fraction vs d/n")
    plt.grid(True)
    plt.show()

def run_additionallowdeg_experiment(start_n=32, max_n=4096, c=10, num_trials=10):
    ratios = []
    n = start_n

    while n <= max_n:
        d_values = sorted(set([
            2, 3, 4, 5,
            max(1, int(np.log(n))),
            max(1, int(np.sqrt(n))),
            max(1, int(n / c))
        ]))

        for d in d_values:
            if d >= n or (d * n) % 2 != 0:
                continue  # skip invalid degree configs

            matched_fractions = []
            for _ in range(num_trials):
                G = generate_d_regular_graph(n, d)
                graph_adj = {node: list(G.neighbors(node)) for node in G.nodes}
                matching = rogmm_matching(graph_adj)

                matched_nodes = {u for edge in matching for u in edge}
                matched_fraction = len(matched_nodes) / n
                matched_fractions.append(matched_fraction)

            avg_matched_fraction = sum(matched_fractions) / num_trials
            ratios.append([n, d, d/n, avg_matched_fraction])
            print(f"n={n}, d={d}, d/n={d/n:.4f}, avg match frac={avg_matched_fraction:.4f}")

        n *= 2  # double n

    return ratios

def save_results_to_csv(ratios, filename="matching_results.csv"):
    # Write the results to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["n", "d", "d/n", "Average Matched Fraction"])
        writer.writerows(ratios)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    n = 100  # Nodes in the graph
    min_d = 2  # Min degree
    max_d = 80  # Max degree
    steps = 10  # Steps
    
    ratios = run_experiment(n, min_d, max_d, steps)
    # plot_results(ratios)

    print("________ ADDITIONAL RESULTS FOR LOW DEGREE __________")

    # Run the additional experiments for low degree
    add_ratios = run_additionallowdeg_experiment()
    save_results_to_csv(add_ratios)