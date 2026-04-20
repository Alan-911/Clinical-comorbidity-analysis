import streamlit as st
import pandas as pd
import numpy as np
import time
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import base64
import re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

st.set_page_config(
    page_title="Carelink • Comorbidity Dashboard",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def st_html(html_str):
    """Flattens HTML to a single line to prevent Streamlit from parsing indented lines as Markdown code blocks."""
    flat_html = re.sub(r'\n\s*', ' ', html_str)
    st.markdown(flat_html, unsafe_allow_html=True)

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

# --- CSS Styling ---
st_html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{ padding: 1rem; max-width: 100%; position: relative; z-index: 1; }}
    
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: #0f172a; }}
    
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
        background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 15px; padding: 20px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08); margin-bottom: 20px; position: relative; z-index: 2;
    }}
    
    h3 {{ font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 15px; margin-top: 0; }}
    
    /* Vitals & Animations */
    .vitals-row {{ display: flex; gap: 10px; margin-bottom: 20px; }}
    .vital-card {{ flex: 1; padding: 15px; border-radius: 15px; background: #fff; box-shadow: 0 4px 15px rgba(0,0,0,0.05); position: relative; overflow: hidden; }}
    .vital-label {{ font-size: 12px; color: #64748b; font-weight: 600; margin-bottom: 2px; }}
    .vital-value {{ font-size: 24px; font-weight: 700; color: #0f172a; }}
    
    @keyframes heartbeat {{ 0% {{ transform: scale(1); }} 20% {{ transform: scale(1.25); }} 40% {{ transform: scale(1); }} 60% {{ transform: scale(1.15); }} 80% {{ transform: scale(1); }} 100% {{ transform: scale(1); }} }}
    .heart-icon {{ animation: heartbeat 1.5s infinite; display: inline-block; color: #ef4444; }}
    
    @keyframes brainwave {{ 0% {{ opacity: 0.5; }} 50% {{ opacity: 1; text-shadow: 0 0 10px #eab308; }} 100% {{ opacity: 0.5; }} }}
    .brain-icon {{ animation: brainwave 2s infinite; display: inline-block; color: #eab308; }}
    
    .ekg-line {{
        height: 30px; width: 100%; margin-top: 10px;
        background: linear-gradient(90deg, transparent 0%, #ef4444 50%, transparent 100%);
        background-size: 100px 100%; animation: moveEKG 1s linear infinite; opacity: 0.5;
    }}
    @keyframes moveEKG {{ 0% {{ background-position: 0 0; }} 100% {{ background-position: -100px 0; }} }}
    
    /* Timeline */
    .timeline-item {{ margin-bottom: 15px; padding-left: 15px; border-left: 2px solid #e2e8f0; position: relative; }}
    .timeline-item::before {{ content: ''; position: absolute; left: -6px; top: 0; width: 10px; height: 10px; border-radius: 50%; background: #3b82f6; }}
    .timeline-time {{ font-size: 12px; color: #94a3b8; font-weight: 600; }}
    .timeline-title {{ font-size: 14px; font-weight: 600; color: #0f172a; margin: 2px 0; }}
    .timeline-desc {{ font-size: 12px; color: #64748b; }}
    
    /* Form overrides for Pattern Selection */
    div[data-testid="stForm"] {{ border: none; padding: 0; background: transparent; }}
    div[data-baseweb="select"] {{ border-radius: 8px !important; background: white !important; border: 1px solid #cbd5e1 !important; }}
    button[data-testid="baseButton-primary"] {{ background-color: #0f172a !important; color: white !important; width: 100% !important; border-radius: 8px !important; padding: 10px !important; font-weight: 600 !important; border: none !important; margin-top: 10px !important; }}
    
    /* Custom spacing adjustments */
    .block-container {{ gap: 0rem; }}
    </style>
""")

if bg_html:
    st_html(bg_html)

# --- Top Navigation ---
st_html("""
<div class="navbar">
    <div class="navbar-brand">
        <span style="font-size: 24px;">⚕️</span> Carelink <span style="color:#94a3b8; font-weight:400; font-size:16px; margin-left:5px;">• Comorbidity & Treatment Patterns</span>
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
""")

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
    
    apriori_patterns = len(association_rules(freq_items_apriori, metric="confidence", min_threshold=0.5))
    fp_patterns = len(rules_df)
else:
    rules_df = None
    all_items = []
    time_apriori, time_fp = 0, 0
    apriori_patterns, fp_patterns = 0, 0

# --- 3 Column Layout ---
col1, col2, col3 = st.columns([1, 1.2, 1.5], gap="large")

# === LEFT COLUMN: Context ===
with col1:
    st_html("""
    <div class="glass-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 20px;">
            <h3 style="margin:0;">Care Schedule</h3>
            <span style="font-size:12px; color:#0f172a; font-weight:600; background:#f1f5f9; padding:6px 12px; border-radius:20px; border:1px solid #e2e8f0;">September 2025 ⌄</span>
        </div>
        <div>
            <div class="timeline-item">
                <div class="timeline-time">13:00</div>
                <div class="timeline-title">Blood Pressure Check</div>
                <div class="timeline-desc">Measure BP at rest if > 140/90 mmHg.<br><span style="color:#ef4444;">❤ 💊</span></div>
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
    """)
    
    st_html("""
    <div class="glass-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 15px;">
            <h3 style="margin:0;">Data Exploration & Visualization</h3>
            <span style="font-size:12px; color:#0f172a; font-weight:600; background:#f8fafc; padding:4px 10px; border-radius:15px; border:1px solid #e2e8f0; cursor:pointer;">View All</span>
        </div>
        <div style="display:flex; gap:10px; overflow-x:auto;">
            <div style="flex:1; text-align:center; background:#fff; padding:15px 5px; border-radius:15px; box-shadow:0 2px 10px rgba(0,0,0,0.05); border:1px solid #f1f5f9;">
                <div style="font-size:32px; margin-bottom:5px;">👨‍⚕️</div>
                <div style="font-size:12px; font-weight:700; color:#0f172a;">Dr. Daniel Lewis</div>
                <div style="font-size:10px; color:#64748b;">Oncologist</div>
                <div style="font-size:9px; font-weight:600; color:#0f172a; margin-top:5px; background:#f1f5f9; border-radius:10px; display:inline-block; padding:2px 6px;">Exp. 25 yrs</div>
            </div>
            <div style="flex:1; text-align:center; background:#fff; padding:15px 5px; border-radius:15px; box-shadow:0 2px 10px rgba(0,0,0,0.05); border:1px solid #f1f5f9;">
                <div style="font-size:32px; margin-bottom:5px;">👩‍⚕️</div>
                <div style="font-size:12px; font-weight:700; color:#0f172a;">Dr. Grace Walker</div>
                <div style="font-size:10px; color:#64748b;">Cardiologist</div>
                <div style="font-size:9px; font-weight:600; color:#0f172a; margin-top:5px; background:#f1f5f9; border-radius:10px; display:inline-block; padding:2px 6px;">Exp. 5 yrs</div>
            </div>
        </div>
    </div>
    """)

# === MIDDLE COLUMN: Transparent buffer ===
with col2:
    st_html("<div style='height: 80vh;'></div>")

# === RIGHT COLUMN: AI Analytics ===
with col3:
    # 1. Animated Vitals Component & Temperature Logic
    temp_val = 38.5
    temp_color = "#ef4444" if temp_val > 38 else "#22c55e" 
    temp_bg = "rgba(239, 68, 68, 0.05)" if temp_val > 38 else "rgba(34, 197, 94, 0.05)"
    
    st_html(f"""
    <div class="vitals-row">
        <div class="vital-card">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <div class="vital-label">Heart rate</div>
                    <div class="vital-value">80-90 <span style="font-size:12px; font-weight:600; color:#64748b;">bpm</span></div>
                </div>
                <div class="heart-icon" style="font-size:18px;">❤</div>
            </div>
            <div class="ekg-line"></div>
        </div>
        
        <div class="vital-card">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <div class="vital-label">Brain activity</div>
                    <div class="vital-value">90-150 <span style="font-size:12px; font-weight:600; color:#64748b;">Hz</span></div>
                </div>
                <div class="brain-icon" style="font-size:18px;">🧠</div>
            </div>
            <div style="height:30px; width:100%; margin-top:10px; background: repeating-linear-gradient(45deg, transparent, transparent 5px, rgba(234, 179, 8, 0.2) 5px, rgba(234, 179, 8, 0.2) 10px);"></div>
        </div>
        
        <div class="vital-card" style="background: {temp_bg}; border: 1px solid rgba(239, 68, 68, 0.1);">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <div class="vital-label" style="color: {temp_color};">Temperature</div>
                    <div class="vital-value" style="color: {temp_color};">{temp_val}°<span style="font-size:12px; font-weight:600; color: {temp_color};">C</span></div>
                </div>
                <div style="color: {temp_color}; font-size:18px;">🌡</div>
            </div>
            <div style="height:30px; width:100%; margin-top:10px; display:flex; flex-direction:column; justify-content:center;">
                 <div style="width:100%; height:8px; background:rgba(239, 68, 68, 0.2); border-radius:4px; overflow:hidden;">
                     <div style="width:85%; height:100%; background:{temp_color};"></div>
                 </div>
            </div>
        </div>
    </div>
    """)
    
    # Placeholder for the Network Graph so it renders structurally ABOVE the filters, 
    # but functionally uses the states defined BELOW it.
    graph_placeholder = st.empty()
    
    # 3. Bottom Row: Pattern Selection & Algorithm Comparison
    bottom_col1, bottom_col2 = st.columns([1, 1.4], gap="small")
    
    with bottom_col1:
        st_html("""
        <div class="glass-card" style="height: 100%; display:flex; flex-direction:column; padding: 20px;">
            <h3 style="margin-bottom: 15px; font-size: 16px;">Pattern Selection</h3>
        """)
        
        with st.form("pattern_form"):
            st_html("<div style='font-size:12px; font-weight:600; color:#0f172a; margin-bottom:5px;'>Primary Diagnosis</div>")
            primary_diag = st.selectbox("Primary Diagnosis", ["All"] + all_items, label_visibility="collapsed")
            
            st_html("<div style='font-size:12px; font-weight:600; color:#0f172a; margin-top:10px; margin-bottom:5px;'>Secondary Condition</div>")
            secondary_diag = st.selectbox("Secondary Condition", ["All"] + all_items, label_visibility="collapsed")
            
            submitted = st.form_submit_button("Done", type="primary")
            
        st_html('</div>')

    with bottom_col2:
        # Recreate the exact UI from the right side of the image for Algorithm Comparison
        st_html(f"""
        <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; padding: 20px;">
            <h3 style="margin-bottom: 15px; font-size: 16px; display: flex; align-items: center; gap: 8px;">
                <span style="font-size:16px;">≑</span> Algorithm Comparison
            </h3>
            <div style="display: flex; justify-content: space-between; flex: 1;">
                <!-- Left Side (Apriori) -->
                <div style="flex: 1; padding-right: 15px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 12px; font-weight: 600; color: #0f172a; margin-bottom: 2px;">Apriori Runtime</div>
                        <div style="font-size: 26px; font-weight: 700; color: #0f172a; margin-bottom: 10px;">{time_apriori:.1f}s</div>
                        <div style="font-size: 11px; color: #64748b; font-weight: 600; line-height: 1.3;">Number of Patterns<br>Discovered</div>
                    </div>
                    <button style="width: 100%; padding: 8px; background: white; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 8px; font-weight: 600; font-size: 12px; cursor: pointer; margin-top: 15px;">
                        View Details
                    </button>
                </div>
                <!-- Right Side (FP-Growth) -->
                <div style="flex: 1; padding-left: 15px; display: flex; flex-direction: column; justify-content: space-between; border-left: 1px solid #e2e8f0;">
                    <div>
                        <div style="font-size: 12px; font-weight: 600; color: #0f172a; margin-bottom: 2px;">FP-Growth Runtime</div>
                        <div style="font-size: 26px; font-weight: 700; color: #0f172a; margin-bottom: 10px;">{time_fp:.1f}s</div>
                        
                        <!-- CSS Bar Chart -->
                        <div style="display: flex; align-items: flex-end; justify-content: space-around; height: 50px; margin-top: 5px;">
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
                                <span style="font-size: 10px; font-weight: 600; color: #0f172a;">{apriori_patterns}</span>
                                <div style="width: 20px; height: 15px; background: #93c5fd; border-radius: 4px 4px 0 0;"></div>
                                <span style="font-size: 9px; color: #64748b; font-weight: 600;">Apriori</span>
                            </div>
                            <div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
                                <span style="font-size: 10px; font-weight: 600; color: #0f172a;">{fp_patterns}</span>
                                <div style="width: 20px; height: 40px; background: #3b82f6; border-radius: 4px 4px 0 0;"></div>
                                <span style="font-size: 9px; color: #64748b; font-weight: 600;">FP-Growth</span>
                            </div>
                        </div>
                    </div>
                    <button style="width: 100%; padding: 8px; background: #0f172a; color: white; border: none; border-radius: 8px; font-weight: 600; font-size: 12px; cursor: pointer; margin-top: 15px;">
                        View Details
                    </button>
                </div>
            </div>
        </div>
        """)

    # Now that we have the pattern selections, render the graph into the placeholder ABOVE
    with graph_placeholder.container():
        st_html("""
        <div class="glass-card" style="padding: 15px; margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="flex:1;">
                    <h3 style="margin-bottom:5px;">Diagnosis & Treatment Rule Network</h3>
                </div>
            </div>
            
            <div style="display:flex; justify-content:space-between; align-items:center; border-top: 1px solid #e2e8f0; padding-top:10px; margin-top:5px;">
                <div style="font-size: 10px; color: #0f172a; font-weight: 600;">Edge Color → Strength (Lift) <br><span style="color:#64748b; font-weight:400;">Thickness</span></div>
                <div style="font-size: 10px; color: #0f172a; font-weight: 600; text-align:right;">Node Color → Event Type <br><span style="color:#64748b; font-weight:400;">(Diagnosis, Medication)</span></div>
            </div>
        """)

        filtered_rules = rules_df.copy() if rules_df is not None else None
        if filtered_rules is not None:
            if primary_diag != "All":
                filtered_rules = filtered_rules[filtered_rules['antecedents'].str.contains(primary_diag)]
            if secondary_diag != "All":
                filtered_rules = filtered_rules[filtered_rules['consequents'].str.contains(secondary_diag)]

        if filtered_rules is not None and len(filtered_rules) > 0:
            net = Network(height="260px", width="100%", bgcolor="transparent", font_color="#0f172a", directed=True)
            net.force_atlas_2based(spring_length=80, spring_strength=0.08, overlap=0)
            
            top_rules = filtered_rules.nlargest(15, 'lift')
            
            for _, row in top_rules.iterrows():
                ant_str = str(row['antecedents']).replace("frozenset({", "").replace("})", "").replace("'", "")
                con_str = str(row['consequents']).replace("frozenset({", "").replace("})", "").replace("'", "")
                ants = [a.strip() for a in ant_str.split(',')]
                cons = [c.strip() for c in con_str.split(',')]
                
                for a in ants:
                    a_color = "#0f172a" if "Disease" in a or a in ["Diabetes", "Hypertension", "Asthma"] else "#3b82f6"
                    net.add_node(a, a, title=a, color=a_color, shape="ellipse", size=20, font={"color": "white", "size":12, "face":"Inter", "strokeWidth":0})
                    for c in cons:
                        c_color = "#0f172a" if "Disease" in c or c in ["Diabetes", "Hypertension", "Asthma"] else "#3b82f6"
                        net.add_node(c, c, title=c, color=c_color, shape="ellipse", size=20, font={"color": "white", "size":12, "face":"Inter", "strokeWidth":0})
                        
                        edge_label = f"Support: {row['support']:.2f}\nConfidence: {row['confidence']:.2f}\nLift: {row['lift']:.2f}"
                        net.add_edge(a, c, title=edge_label, value=row['lift'], color="#64748b", font={"size":8, "align":"middle"})
            
            net_path = os.path.join(base_path, "visualizations", "pyvis_graph.html")
            net.save_graph(net_path)
            with open(net_path, 'r', encoding='utf-8') as f:
                html_data = f.read()
            html_data = html_data.replace('background-color: transparent;', 'background-color: transparent;')
            components.html(html_data, height=270)
        else:
            st.info("No matching rules found. Please adjust the Pattern Selection.")
            
        st_html('</div>')
