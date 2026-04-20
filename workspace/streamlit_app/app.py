import streamlit as st
import pandas as pd
import numpy as np
import time
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import base64
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

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

# --- CSS Styling (Cloud Design & Animations) ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{ padding: 1rem; max-width: 100%; position: relative; z-index: 1; }}
    
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: #1f2937; }}
    
    .stApp {{
        background-color: #f8fafc;
        background-image: linear-gradient(to right, #e2e8f0 1px, transparent 1px), linear-gradient(to bottom, #e2e8f0 1px, transparent 1px);
        background-size: 40px 40px;
    }}
    
    {bg_style}

    .navbar {{
        display: flex; align-items: center; justify-content: space-between; padding: 10px 40px;
        background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(15px); border-radius: 100px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05); margin-bottom: 20px; position: relative; z-index: 10;
    }}
    .navbar-brand {{ font-size: 20px; font-weight: 700; display: flex; align-items: center; gap: 10px; }}
    .navbar-links {{ display: flex; gap: 15px; }}
    .nav-btn {{ padding: 10px 20px; border-radius: 50px; font-weight: 600; font-size: 14px; background: transparent; color: #475569; border: none; }}
    .nav-btn.active {{ background: #0f172a; color: white; }}

    .glass-card {{
        background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 20px; padding: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08); margin-bottom: 20px; position: relative; z-index: 2;
    }}
    
    h3 {{ font-size: 18px; font-weight: 600; color: #0f172a; margin-bottom: 15px; margin-top: 0; }}
    
    /* Vitals & Animations */
    .vitals-row {{ display: flex; gap: 10px; margin-bottom: 20px; }}
    .vital-card {{ flex: 1; padding: 15px; border-radius: 15px; background: #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.05); position: relative; overflow: hidden; }}
    .vital-label {{ font-size: 12px; color: #64748b; font-weight: 600; }}
    .vital-value {{ font-size: 24px; font-weight: 700; color: #0f172a; }}
    
    @keyframes heartbeat {{ 0% {{ transform: scale(1); }} 20% {{ transform: scale(1.25); }} 40% {{ transform: scale(1); }} 60% {{ transform: scale(1.15); }} 80% {{ transform: scale(1); }} 100% {{ transform: scale(1); }} }}
    .heart-icon {{ animation: heartbeat 1.5s infinite; display: inline-block; color: #ef4444; }}
    
    @keyframes brainwave {{ 0% {{ opacity: 0.5; }} 50% {{ opacity: 1; text-shadow: 0 0 10px #eab308; }} 100% {{ opacity: 0.5; }} }}
    .brain-icon {{ animation: brainwave 2s infinite; display: inline-block; color: #eab308; }}
    
    /* EKG Line Animation */
    @keyframes moveEKG {{ 0% {{ background-position: 0 0; }} 100% {{ background-position: -100px 0; }} }}
    .ekg-line {{
        height: 30px; width: 100%; margin-top: 10px;
        background: linear-gradient(90deg, transparent 0%, #ef4444 50%, transparent 100%);
        background-size: 100px 100%;
        animation: moveEKG 1s linear infinite;
        opacity: 0.5;
    }}
    .brain-line {{
        height: 30px; width: 100%; margin-top: 10px;
        background: repeating-linear-gradient(45deg, transparent, transparent 5px, #eab308 5px, #eab308 10px);
        animation: moveEKG 2s linear infinite;
        opacity: 0.5;
    }}
    
    .timeline-item {{ margin-bottom: 15px; padding-left: 15px; border-left: 2px solid #e2e8f0; position: relative; }}
    .timeline-item::before {{ content: ''; position: absolute; left: -6px; top: 0; width: 10px; height: 10px; border-radius: 50%; background: #3b82f6; }}
    
    /* Dark Theme Metrics Card */
    .dark-metric-card {{
        background: #0f172a; color: white; border-radius: 15px; padding: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }}
    .dark-metric-block {{ text-align: center; }}
    .dark-metric-block .val {{ font-size: 28px; font-weight: 700; color: #38bdf8; }}
    .dark-metric-block .lbl {{ font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }}
    
    div[data-baseweb="select"] {{ border-radius: 10px !important; background: white !important; border: 1px solid #cbd5e1 !important; }}
    </style>
""", unsafe_allow_html=True)

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

# --- Live Data Mining & Timers ---
def apriori_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        return res, time.time() - start
    return wrapper

def fpgrowth_timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        return res, time.time() - start
    return wrapper

@st.cache_data
def load_and_encode_data():
    trans_path = os.path.join(base_path, "transactions", "transactions.csv")
    try:
        df_trans = pd.read_csv(trans_path)
        dataset = [str(items).split(",") for items in df_trans["Items"].tolist()]
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        return pd.DataFrame(te_ary, columns=te.columns_)
    except Exception:
        return None

df_encoded = load_and_encode_data()

@st.cache_data
@apriori_timer
def run_apriori(df):
    return apriori(df, min_support=0.05, use_colnames=True)

@st.cache_data
@fpgrowth_timer
def run_fpgrowth(df):
    return fpgrowth(df, min_support=0.05, use_colnames=True)

if df_encoded is not None:
    freq_items_apriori, time_apriori = run_apriori(df_encoded)
    freq_items_fp, time_fp = run_fpgrowth(df_encoded)
    
    rules_df = association_rules(freq_items_fp, metric="confidence", min_threshold=0.5)
    
    all_items = set()
    for antecedents in rules_df['antecedents']:
        items = str(antecedents).replace("frozenset({", "").replace("})", "").replace("'", "").split(", ")
        all_items.update([i.strip() for i in items])
    all_items = sorted(list(all_items))
    runtime_diff = time_apriori - time_fp
else:
    rules_df = None
    all_items = []
    time_apriori, time_fp, runtime_diff = 0, 0, 0

# --- 3 Column Layout ---
col1, col2, col3 = st.columns([1, 1.2, 1.5], gap="large")

# === LEFT COLUMN: Context ===
with col1:
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
    
    # Legend & Metadata Card
    st.markdown("""
    <div class="glass-card" style="background: rgba(255, 255, 255, 0.95); border-left: 4px solid #3b82f6;">
        <h3>System Legend</h3>
        <div style="font-size: 12px; color: #475569; margin-bottom: 8px;">
            <strong style="color: #0f172a;">Support:</strong> Frequency of the pattern in the total dataset.
        </div>
        <div style="font-size: 12px; color: #475569; margin-bottom: 8px;">
            <strong style="color: #0f172a;">Confidence:</strong> Probability that the consequent event occurs given the antecedent.
        </div>
        <div style="font-size: 12px; color: #475569;">
            <strong style="color: #0f172a;">Lift:</strong> Strength of association. Lift > 1 implies a strong, non-random clinical correlation.
        </div>
    </div>
    """, unsafe_allow_html=True)

# === MIDDLE COLUMN: Transparent buffer ===
with col2:
    st.markdown("<div style='height: 80vh;'></div>", unsafe_allow_html=True)

# === RIGHT COLUMN: AI Analytics ===
with col3:
    # 1. Animated Vitals Component & Temperature Logic
    # Generate random temp or static for demo. Let's make it 38.5 to show Red, or user can toggle. We'll use 38.5.
    temp_val = 38.5
    temp_color = "#ef4444" if temp_val > 38 else "#22c55e" # Red if > 38, else Green
    temp_bg = "rgba(239, 68, 68, 0.1)" if temp_val > 38 else "rgba(34, 197, 94, 0.1)"
    
    st.markdown(f"""
    <div class="vitals-row">
        <div class="vital-card">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <div class="vital-label">Heart rate</div>
                    <div class="vital-value">80-90 <span style="font-size:14px; color:#64748b;">bpm</span></div>
                </div>
                <div class="heart-icon">❤️</div>
            </div>
            <div class="ekg-line"></div>
        </div>
        
        <div class="vital-card">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <div class="vital-label">Brain activity</div>
                    <div class="vital-value">90-150 <span style="font-size:14px; color:#64748b;">Hz</span></div>
                </div>
                <div class="brain-icon">🧠</div>
            </div>
            <div class="brain-line"></div>
        </div>
        
        <div class="vital-card" style="border-left: 4px solid {temp_color}; background: {temp_bg};">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <div class="vital-label" style="color: {temp_color};">Temperature (T)</div>
                    <div class="vital-value" style="color: {temp_color};">{temp_val}° <span style="font-size:14px; color: {temp_color};">C</span></div>
                </div>
                <div style="color: {temp_color}; font-size:20px;">🌡️</div>
            </div>
            <div style="height:30px; width:100%; border-bottom:2px solid {temp_color}; margin-top:10px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Association Rule Visualization
    st.markdown('<div class="glass-card" style="padding: 10px;">', unsafe_allow_html=True)
    st.markdown('<h3 style="padding-left:10px; padding-top:10px;">Diagnosis & Treatment Rule Network</h3>', unsafe_allow_html=True)
    
    # Render Pattern Selection inputs directly in Streamlit before rendering graph
    st.markdown("<div style='padding: 0 10px;'><h4 style='font-size:14px; color:#64748b; margin-bottom:5px;'>Pattern Selection</h4></div>", unsafe_allow_html=True)
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        primary_diagnosis = st.selectbox("Primary Diagnosis", ["All"] + all_items, key="p_diag")
    with sel_col2:
        target_medication = st.selectbox("Target Medication", ["All"] + all_items, key="t_med")

    # Filter Rules based on selection
    filtered_rules = rules_df.copy() if rules_df is not None else None
    if filtered_rules is not None:
        if primary_diagnosis != "All":
            filtered_rules = filtered_rules[filtered_rules['antecedents'].str.contains(primary_diagnosis)]
        if target_medication != "All":
            filtered_rules = filtered_rules[filtered_rules['consequents'].str.contains(target_medication)]

    # Draw Network Graph
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
                    # Edge representing Medical Events -> Lift
                    net.add_edge(a, c, title=f"Support: {row['support']:.2f}\\nConf: {row['confidence']:.2f}\\nLift: {row['lift']:.2f}", value=row['lift'], color="#94a3b8")
        
        net_path = os.path.join(base_path, "visualizations", "pyvis_graph.html")
        net.save_graph(net_path)
        with open(net_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=310)
    else:
        st.info("No matching medical events found for this selection.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3. Benchmarking Module (Dark Theme Metric Cards)
    st.markdown('<div class="dark-metric-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:white;">Live Algorithm Benchmarking</h3>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="display:flex; justify-content:space-around; margin-top:20px;">
        <div class="dark-metric-block">
            <div class="lbl">Apriori Runtime</div>
            <div class="val">{time_apriori:.4f}s</div>
        </div>
        <div class="dark-metric-block">
            <div class="lbl">FP-Growth Runtime</div>
            <div class="val">{time_fp:.4f}s</div>
        </div>
        <div class="dark-metric-block" style="border-left: 2px solid #334155; padding-left: 20px;">
            <div class="lbl" style="color: #22c55e;">Runtime Diff ($T_{{Apriori}} - T_{{FP}}$)</div>
            <div class="val" style="color: #22c55e;">{runtime_diff:.4f}s</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

