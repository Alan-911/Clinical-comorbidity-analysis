import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import time

st.set_page_config(
    page_title="Clinical Comorbidity Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for High-End UI ---
st.markdown("""
    <style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border: 1px solid #30363d;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
    }
    .metric-title {
        font-size: 14px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #58a6ff;
    }
    
    /* Headers */
    h1, h2, h3 {
        background: -webkit-linear-gradient(#58a6ff, #1f6feb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    
    /* Divider */
    hr {
        border-color: #30363d;
        margin: 30px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        rules_df = pd.read_csv(os.path.join(base_path, "outputs", "association_rules.csv"))
        itemsets_df = pd.read_csv(os.path.join(base_path, "outputs", "frequent_itemsets.csv"))
        return rules_df, itemsets_df
    except FileNotFoundError:
        return None, None

rules_df, itemsets_df = load_data()

# --- Sidebar UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
    st.markdown("## 🏥 System Controls")
    st.markdown("Analyze comorbidity and treatment associations from synthetic EHR data.")
    
    st.markdown("---")
    
    if rules_df is not None:
        # Extract unique diseases/treatments from rules to populate dropdown
        all_items = set()
        for antecedents in rules_df['antecedents']:
            items = str(antecedents).replace("frozenset({", "").replace("})", "").replace("'", "").split(", ")
            all_items.update([i.strip() for i in items])
        
        all_items = sorted(list(all_items))
        
        selected_primary = st.selectbox(
            "Select Primary Condition / Treatment:",
            ["All"] + all_items
        )
        
        min_lift_filter = st.slider("Minimum Lift Threshold:", 1.0, 10.0, 1.5, 0.5)
        min_conf_filter = st.slider("Minimum Confidence:", 0.5, 1.0, 0.6, 0.05)
    else:
        st.error("Data not found. Please run the data mining script first.")
        st.stop()
        
    st.markdown("---")
    st.markdown("**Algorithm:** FP-Growth & Apriori")
    st.markdown("**Metric Focus:** Lift & Confidence")

# --- Main UI ---
st.title("Discovery of Comorbidity and Treatment Patterns")
st.markdown("### Real-time Association Rule Mining Dashboard")

# Filter logic
if selected_primary == "All":
    filtered_rules = rules_df[
        (rules_df['lift'] >= min_lift_filter) & 
        (rules_df['confidence'] >= min_conf_filter)
    ]
else:
    # Check if selected item is in antecedents
    filtered_rules = rules_df[
        (rules_df['antecedents'].str.contains(selected_primary)) &
        (rules_df['lift'] >= min_lift_filter) & 
        (rules_df['confidence'] >= min_conf_filter)
    ]

# Metrics Row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Rules Discovered</div>
            <div class="metric-value">{len(filtered_rules)}</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    avg_conf = filtered_rules['confidence'].mean() if not filtered_rules.empty else 0
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Confidence</div>
            <div class="metric-value">{avg_conf:.2f}</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    max_lift = filtered_rules['lift'].max() if not filtered_rules.empty else 0
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Max Lift</div>
            <div class="metric-value">{max_lift:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Visuals and Tables
tab1, tab2, tab3 = st.tabs(["📊 Network Visualization", "📋 Top Association Rules", "📦 Frequent Itemsets"])

with tab1:
    st.markdown("### Comorbidity Network")
    st.markdown("This interactive graph shows relationships where **Node A** strongly implies **Node B**.")
    
    if len(filtered_rules) > 0:
        # Generate dynamic network graph for filtered data
        G = nx.DiGraph()
        top_plot_rules = filtered_rules.nlargest(15, 'lift')
        
        for _, row in top_plot_rules.iterrows():
            ant_str = str(row['antecedents']).replace("frozenset({", "").replace("})", "").replace("'", "")
            con_str = str(row['consequents']).replace("frozenset({", "").replace("})", "").replace("'", "")
            
            ants = [a.strip() for a in ant_str.split(',')]
            cons = [c.strip() for c in con_str.split(',')]
            
            for a in ants:
                for c in cons:
                    G.add_edge(a, c, weight=row['lift'])
                    
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        
        pos = nx.spring_layout(G, k=0.8, seed=42)
        
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='#58a6ff', alpha=0.9, edgecolors='#1f6feb', ax=ax)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u,v in edges]
        max_w = max(weights) if weights else 1
        thickness = [(w/max_w)*4 for w in weights]
        
        nx.draw_networkx_edges(G, pos, edge_color='#8b949e', width=thickness, arrowsize=20, arrowstyle='->', ax=ax)
        
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', font_color='white', ax=ax)
        
        plt.axis('off')
        st.pyplot(fig)
    else:
        st.warning("No rules match the current filters. Please adjust the sliders or select a different condition.")

with tab2:
    st.markdown("### Top Association Rules")
    if not filtered_rules.empty:
        display_cols = ['rule_string', 'support', 'confidence', 'lift']
        display_df = filtered_rules[display_cols].copy()
        display_df.columns = ['Rule (Antecedent → Consequent)', 'Support', 'Confidence', 'Lift']
        display_df['Support'] = display_df['Support'].apply(lambda x: f"{x:.4f}")
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.4f}")
        display_df['Lift'] = display_df['Lift'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.info("Adjust filters to see rules.")

with tab3:
    st.markdown("### Frequent Pattern Itemsets")
    if selected_primary == "All":
        filtered_itemsets = itemsets_df
    else:
        filtered_itemsets = itemsets_df[itemsets_df['itemsets_string'].str.contains(selected_primary)]
        
    display_item_df = filtered_itemsets[['itemsets_string', 'support']].sort_values(by='support', ascending=False)
    display_item_df.columns = ['Itemset', 'Support']
    display_item_df['Support'] = display_item_df['Support'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_item_df, use_container_width=True, height=400)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("🏥 **Data Source**: Synthetically generated clinical records mimicking real-world comorbidity dynamics. | Built with **Streamlit** & **FP-Growth**.")
