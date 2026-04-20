import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import base64

st.set_page_config(
    page_title="Carelink • Comorbidity Dashboard",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Helper to encode local image for background ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(base_path, "visualizations", "anatomical_model.png")
try:
    bg_img_b64 = get_base64_of_bin_file(img_path)
    bg_style = f"""
    @keyframes spin3D {{
        from {{ transform: translate(-50%, -50%) rotateY(0deg); }}
        to {{ transform: translate(-50%, -50%) rotateY(360deg); }}
    }}
    .bg-image {{
        position: fixed;
        top: 50%;
        left: 50%;
        height: 90vh;
        z-index: 0;
        opacity: 0.9;
        pointer-events: none;
        animation: spin3D 20s linear infinite;
        transform-style: preserve-3d;
    }}
    """
    bg_html = f'<img src="data:image/png;base64,{bg_img_b64}" class="bg-image">'
except Exception:
    bg_style = ""
    bg_html = ""

# --- CSS Styling (Cloud Design) ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Hide Streamlit default UI */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
        position: relative;
        z-index: 1;
    }}
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: #1f2937;
    }}
    
    /* Light Theme Cloud Background with Grid */
    .stApp {{
        background-color: #f8fafc;
        background-image: 
            linear-gradient(to right, #e2e8f0 1px, transparent 1px),
            linear-gradient(to bottom, #e2e8f0 1px, transparent 1px);
        background-size: 40px 40px;
    }}
    
    {bg_style}

    /* Top Navbar */
    .navbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 40px;
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 100px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        z-index: 10;
        position: relative;
    }}
    .navbar-brand {{
        font-size: 20px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .navbar-links {{
        display: flex;
        gap: 15px;
    }}
    .nav-btn {{
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s ease;
        background: transparent;
        color: #475569;
        border: none;
    }}
    .nav-btn.active {{
        background: #0f172a;
        color: white;
    }}

    /* Glassmorphism Cards */
    .glass-card {{
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        position: relative;
        z-index: 2;
    }}
    
    h3 {{
        font-size: 18px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 15px;
        margin-top: 0;
    }}
    
    /* Vitals */
    .vitals-row {{ display: flex; gap: 15px; margin-bottom: 20px; }}
    .vital-card {{ flex: 1; padding: 15px; border-radius: 15px; background: #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
    .vital-label {{ font-size: 12px; color: #64748b; font-weight: 600; }}
    .vital-value {{ font-size: 24px; font-weight: 700; color: #0f172a; }}
    
    /* Timeline */
    .timeline-item {{ margin-bottom: 15px; padding-left: 15px; border-left: 2px solid #e2e8f0; position: relative; }}
    .timeline-item::before {{
        content: ''; position: absolute; left: -6px; top: 0; width: 10px; height: 10px;
        border-radius: 50%; background: #3b82f6;
    }}
    .timeline-time {{ font-size: 12px; color: #94a3b8; font-weight: 600; }}
    .timeline-title {{ font-size: 14px; font-weight: 600; color: #0f172a; margin: 2px 0; }}
    .timeline-desc {{ font-size: 12px; color: #64748b; }}
    
    /* Metric blocks */
    .metric-block {{ text-align: center; }}
    .metric-block .val {{ font-size: 28px; font-weight: 700; color: #0f172a; }}
    .metric-block .lbl {{ font-size: 12px; color: #64748b; font-weight: 600; }}
    
    /* Selectbox styling override */
    div[data-baseweb="select"] {{
        border-radius: 10px !important;
        background: white !important;
        border: 1px solid #cbd5e1 !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Inject Background Image
if bg_html:
    st.markdown(bg_html, unsafe_allow_html=True)

# --- Top Navigation ---
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <span style="font-size: 24px;">⚕️</span> Carelink <span style="color:#94a3b8; font-weight:400;">• Comorbidity & Treatment</span>
    </div>
    <div class="navbar-links">
        <button class="nav-btn active">Dashboard</button>
        <button class="nav-btn">Appointments</button>
        <button class="nav-btn">Schedule</button>
        <button class="nav-btn">Labs Results</button>
    </div>
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;">
        <div style="width:35px; height:35px; border-radius:50%; background:#e2e8f0; display:flex; align-items:center; justify-content:center; margin-bottom:2px;">👤</div>
        <span style="font-size: 10px; font-weight: 700; color: #475569;">Patient's Profile</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        rules_df = pd.read_csv(os.path.join(base_path, "outputs", "association_rules.csv"))
        return rules_df
    except FileNotFoundError:
        return None

rules_df = load_data()

# Process unique items for the dropdown
all_items = []
if rules_df is not None:
    items_set = set()
    for antecedents in rules_df['antecedents']:
        items = str(antecedents).replace("frozenset({", "").replace("})", "").replace("'", "").split(", ")
        items_set.update([i.strip() for i in items])
    all_items = sorted(list(items_set))

# --- 3 Column Layout ---
col1, col2, col3 = st.columns([1, 1.2, 1.2], gap="large")

# === LEFT COLUMN: Context ===
with col1:
    # Care Schedule Widget
    st.markdown("""
    <div class="glass-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3>Care Schedule</h3>
            <span style="font-size:12px; color:#64748b; background:#f1f5f9; padding:5px 10px; border-radius:20px;">September 2025 ⌄</span>
        </div>
        <div style="margin-top:20px;">
            <div class="timeline-item">
                <div class="timeline-time">13:00</div>
                <div class="timeline-title">Blood Pressure Check</div>
                <div class="timeline-desc">Measure BP at rest if > 140/90 mmHg.</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-time">15:30</div>
                <div class="timeline-title">Dr. John Smith Consultation</div>
                <div class="timeline-desc">Video Call (Prepare last 3 BP readings)</div>
            </div>
            <div class="timeline-item">
                <div class="timeline-time">19:00</div>
                <div class="timeline-title">Symptom Log</div>
                <div class="timeline-desc">Pain in the right hypochondrium.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Data Exploration Widget
    st.markdown("""
    <div class="glass-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 10px;">
            <h3 style="margin-bottom:0;">Data Visualization</h3>
            <div style="width: 40%; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden;">
                <div style="width: 75%; height: 100%; background: #3b82f6;"></div>
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top: 15px;">
            <h3>Doctors On Call</h3>
            <span style="font-size:12px; color:#0f172a; font-weight:600; cursor:pointer;">View All</span>
        </div>
        <div style="display:flex; gap:10px; margin-top:10px; overflow-x:auto;">
            <div style="flex:1; text-align:center; background:#fff; padding:15px 5px; border-radius:15px; box-shadow:0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size:28px; margin-bottom:5px;">👨‍⚕️</div>
                <div style="font-size:11px; font-weight:700;">Doctor 1</div>
                <div style="font-size:9px; color:#64748b;">Oncologist</div>
            </div>
            <div style="flex:1; text-align:center; background:#fff; padding:15px 5px; border-radius:15px; box-shadow:0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size:28px; margin-bottom:5px;">👩‍⚕️</div>
                <div style="font-size:11px; font-weight:700;">Doctor 2</div>
                <div style="font-size:9px; color:#64748b;">Cardiologist</div>
            </div>
            <div style="flex:1; text-align:center; background:#fff; padding:15px 5px; border-radius:15px; box-shadow:0 2px 10px rgba(0,0,0,0.05);">
                <div style="font-size:28px; margin-bottom:5px;">👨‍⚕️</div>
                <div style="font-size:11px; font-weight:700;">Doctor 3</div>
                <div style="font-size:9px; color:#64748b;">Neurologist</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# === MIDDLE COLUMN: Nothing (Leave empty so 3D model shows through) ===
with col2:
    st.markdown("<div style='height: 80vh;'></div>", unsafe_allow_html=True)

# === RIGHT COLUMN: AI Analytics ===
with col3:
    # Vitals Row
    st.markdown("""
    <div class="vitals-row">
        <div class="vital-card">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <div class="vital-label">Heart rate</div>
                    <div class="vital-value">80-90 <span style="font-size:14px; color:#64748b;">bpm</span></div>
                </div>
                <div style="color:#ef4444;">❤️</div>
            </div>
            <div style="height:30px; width:100%; border-bottom:2px solid #ef4444; margin-top:10px;"></div>
        </div>
        <div class="vital-card">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <div class="vital-label">Brain activity</div>
                    <div class="vital-value">90-150 <span style="font-size:14px; color:#64748b;">Hz</span></div>
                </div>
                <div style="color:#eab308;">🧠</div>
            </div>
            <div style="height:30px; width:100%; border-bottom:2px solid #eab308; margin-top:10px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Bottom Row: Pattern Selection & Algorithm Comparison
    # We place Pattern Selection ABOVE the graph in the execution flow so it actually filters the graph!
    pat_col1, pat_col2 = st.columns(2)
    
    with pat_col1:
        st.markdown('<div class="glass-card" style="height:210px; padding-bottom:10px;">', unsafe_allow_html=True)
        st.markdown("<h3>Pattern Selection</h3>", unsafe_allow_html=True)
        selected_primary = st.selectbox("Primary Diagnosis", ["All"] + all_items, label_visibility="collapsed")
        
        # Apply filter to rules_df
        filtered_rules = rules_df.copy() if rules_df is not None else None
        if filtered_rules is not None and selected_primary != "All":
            filtered_rules = filtered_rules[filtered_rules['antecedents'].str.contains(selected_primary)]
            
        st.markdown("""
            <div style="margin-top:10px;">
            <button style="width:100%; padding:10px; background:#0f172a; color:white; border:none; border-radius:10px; font-weight:600; cursor:pointer;">
                Done
            </button>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with pat_col2:
        st.markdown('<div class="glass-card" style="height:210px;">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h3>Algorithm Comparison</h3>
        </div>
        <div style="display:flex; justify-content:space-between; margin-top:0px;">
            <div class="metric-block">
                <div class="lbl">Apriori Time</div>
                <div class="val">0.008s</div>
            </div>
            <div class="metric-block">
                <div class="lbl">FP-Growth</div>
                <div class="val">0.010s</div>
            </div>
        </div>
        <div style="margin-top:5px; border-top:1px solid #e2e8f0; padding-top:5px; text-align:center;">
            <div class="lbl">Patterns Discovered</div>
            <div class="val" style="color:#3b82f6;">144</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Interactive Network Graph via PyVis (Now uses filtered_rules)
    st.markdown('<div class="glass-card" style="padding: 10px;">', unsafe_allow_html=True)
    st.markdown('<h3 style="padding-left:10px; padding-top:10px;">Diagnosis & Treatment Rule Network</h3>', unsafe_allow_html=True)
    
    if filtered_rules is not None and len(filtered_rules) > 0:
        net = Network(height="300px", width="100%", bgcolor="white", font_color="#1f2937", directed=True)
        net.force_atlas_2based()
        
        top_rules = filtered_rules.nlargest(15, 'lift')
        
        for _, row in top_rules.iterrows():
            ant_str = str(row['antecedents']).replace("frozenset({", "").replace("})", "").replace("'", "")
            con_str = str(row['consequents']).replace("frozenset({", "").replace("})", "").replace("'", "")
            ants = [a.strip() for a in ant_str.split(',')]
            cons = [c.strip() for c in con_str.split(',')]
            
            for a in ants:
                a_color = "#0f172a" if "Disease" in a or a in ["Diabetes", "Hypertension", "Asthma"] else "#3b82f6"
                net.add_node(a, a, title=a, color=a_color, shape="dot", size=15)
                for c in cons:
                    c_color = "#0f172a" if "Disease" in c or c in ["Diabetes", "Hypertension", "Asthma"] else "#3b82f6"
                    net.add_node(c, c, title=c, color=c_color, shape="dot", size=15)
                    net.add_edge(a, c, title=f"Support: {row['support']:.2f}\\nConf: {row['confidence']:.2f}\\nLift: {row['lift']:.2f}", value=row['lift'], color="#94a3b8")
        
        net_path = os.path.join(base_path, "visualizations", "pyvis_graph.html")
        net.save_graph(net_path)
        with open(net_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=310)
    else:
        st.info("No rule data available for this selection.")
        
    st.markdown('</div>', unsafe_allow_html=True)

