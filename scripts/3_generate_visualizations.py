import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def generate_network_graph():
    print("Loading association rules...")
    rules_path = "workspace/outputs/association_rules.csv"
    
    try:
        rules_df = pd.read_csv(rules_path)
    except FileNotFoundError:
        print(f"Error: {rules_path} not found. Run association mining first.")
        return
        
    # We will use the top rules for better visualization
    # Filtering by highest lift
    top_rules = rules_df.nlargest(20, 'lift')
    
    # Create directed graph
    G = nx.DiGraph()
    
    for _, row in top_rules.iterrows():
        # antecedents and consequents are stored as strings of sets/frozensets in the csv
        # let's parse them safely. They are like "frozenset({'Diabetes'})" or just comma separated if we formatted them?
        # Oh, in the previous script we didn't override the antecedents/consequents, they are saved as strings like "frozenset({'Diabetes'})"
        # Wait, mlxtend outputs frozensets. Let's clean it up.
        
        antecedent_str = str(row['antecedents']).replace('frozenset({', '').replace('})', '').replace("'", "")
        consequent_str = str(row['consequents']).replace('frozenset({', '').replace('})', '').replace("'", "")
        
        # Split by comma if there are multiple items
        ants = [a.strip() for a in antecedent_str.split(',')]
        cons = [c.strip() for c in consequent_str.split(',')]
        
        weight = row['lift']
        
        for a in ants:
            for c in cons:
                # Add nodes and edges
                G.add_edge(a, c, weight=weight)
                
    # Plotting
    plt.figure(figsize=(12, 10))
    
    # Use spring layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue', alpha=0.8)
    
    # Draw edges with varying thickness based on weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    # Normalize weights for thickness
    max_weight = max(weights) if weights else 1
    thickness = [(w/max_weight)*5 for w in weights]
    
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=thickness, arrowsize=20, arrowstyle='->')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_weight='bold')
    
    plt.title("Clinical Comorbidity & Treatment Network (Weighted by Lift)", fontsize=16)
    plt.axis('off')
    
    output_path = "workspace/visualizations/network_graph.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network graph saved to {output_path}")

if __name__ == "__main__":
    generate_network_graph()
