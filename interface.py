import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="DDoS Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
    <style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Layout */
    .main {
        padding: 2rem 3rem;
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1 {
        color: #1a1a1a;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #34495e;
        font-weight: 600;
        font-size: 1.4rem;
    }
    
    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stMetric label {
        color: white !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    .success-card {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .danger-card {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* File Uploader */
    .uploadedFile {
        background: white;
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Tables */
    .dataframe {
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: 600;
        padding: 1rem;
        text-align: left;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 1.5rem;
        color: #666;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 8px;
    }
    
    /* Result Cards */
    .result-benign {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .result-attack {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        imputer = joblib.load('imputer.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        df = pd.read_csv('train_data.csv')
        feature_names = [col for col in df.columns if col != 'Label']
        return model, scaler, imputer, label_encoder, feature_names, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, False

model, scaler, imputer, label_encoder, feature_names, models_loaded = load_models()

# Header
st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>Network Security Analysis System</h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0.5rem 0 0 0;'>
            Advanced DDoS Attack Detection using Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### System Information")
    
    if models_loaded:
        st.markdown("""
        <div class='success-card'>
            <strong>Status:</strong> System Operational<br>
            <strong>Model:</strong> XGBoost Classifier<br>
            <strong>Accuracy:</strong> 99.57%<br>
            <strong>Features:</strong> 77 parameters<br>
            <strong>Classes:</strong> 9 categories
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Detectable Threats")
        if label_encoder:
            for idx, label in enumerate(label_encoder.classes_, 1):
                icon = "✓" if label == 'Benign' else "⚠" if label == 'Unknown' else "✗"
                st.markdown(f"{idx}. {icon} **{label}**")
    else:
        st.markdown("""
        <div class='danger-card'>
            <strong>Error:</strong> Models not loaded
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("Cybersecurity Research System v1.0")

# Main Content
if not models_loaded:
    st.error("**System Error:** Required model files not found. Please ensure all .pkl files are present.")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Flow Analysis", 
    "Batch Processing", 
    "Model Performance", 
    "Documentation"
])

# TAB 1: Single Analysis
with tab1:
    st.markdown("## Single Traffic Flow Analysis")
    st.markdown("Upload a CSV file containing network flow data for classification.")
    
    upload_file = st.file_uploader(
        "Select CSV File", 
        type=['csv'],
        help="File should contain 77 network flow features"
    )
    
    if upload_file:
        try:
            df_upload = pd.read_csv(upload_file)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.success(f"**File Loaded:** {df_upload.shape[0]} flows, {df_upload.shape[1]} features")
            with col2:
                st.info(f"**Size:** {upload_file.size / 1024:.1f} KB")
            
            with st.expander("Preview Data (First 10 Rows)"):
                st.dataframe(df_upload.head(10), use_container_width=True, height=300)
            
            if st.button("Analyze Traffic Flow", type="primary", use_container_width=True):
                with st.spinner("Processing traffic data..."):
                    sample = df_upload.iloc[0:1].copy()
                    
                    actual_label = None
                    if 'Label' in sample.columns:
                        actual_label = sample['Label'].values[0]
                        sample = sample.drop('Label', axis=1)
                    
                    sample = sample[feature_names]
                    sample_imputed = imputer.transform(sample)
                    sample_scaled = scaler.transform(sample_imputed)
                    
                    prediction = model.predict(sample_scaled)[0]
                    probabilities = model.predict_proba(sample_scaled)[0]
                    
                    predicted_label = label_encoder.classes_[prediction]
                    confidence = probabilities[prediction] * 100
                
                st.markdown("---")
                st.markdown("### Classification Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if predicted_label == 'Benign':
                        st.markdown("""
                        <div class='result-benign'>
                            <h2 style='color: #155724; margin: 0;'>BENIGN</h2>
                            <p style='color: #155724; margin: 0.5rem 0 0 0;'>Normal Network Traffic</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-attack'>
                            <h2 style='color: #721c24; margin: 0;'>{predicted_label}</h2>
                            <p style='color: #721c24; margin: 0.5rem 0 0 0;'>Attack Detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence Level", f"{confidence:.2f}%")
                
                with col3:
                    if actual_label:
                        if actual_label == predicted_label:
                            st.markdown("""
                            <div class='success-card'>
                                <strong>Verification:</strong> Correct Classification
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='warning-card'>
                                <strong>Verification:</strong> Mismatch<br>
                                <strong>Actual:</strong> {actual_label}
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("### Probability Distribution")
                
                prob_df = pd.DataFrame({
                    'Class': label_encoder.classes_,
                    'Probability': probabilities * 100
                }).sort_values('Probability', ascending=False)
                
                fig = go.Figure(go.Bar(
                    x=prob_df['Probability'],
                    y=prob_df['Class'],
                    orientation='h',
                    marker=dict(
                        color=prob_df['Probability'],
                        colorscale='RdYlGn_r',
                        showscale=False
                    ),
                    text=prob_df['Probability'].apply(lambda x: f'{x:.2f}%'),
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title='Classification Confidence by Attack Type',
                    xaxis_title='Probability (%)',
                    yaxis_title='Attack Type',
                    height=400,
                    template='plotly_white',
                    font=dict(family='Inter, sans-serif')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"**Processing Error:** {str(e)}")

# TAB 2: Batch Analysis
with tab2:
    st.markdown("## Batch Traffic Analysis")
    st.markdown("Process multiple network flows simultaneously for comprehensive threat assessment.")
    
    batch_file = st.file_uploader(
        "Select CSV File (Multiple Flows)", 
        type=['csv'], 
        key='batch',
        help="Upload file containing multiple network flows"
    )
    
    if batch_file:
        try:
            df_batch = pd.read_csv(batch_file)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Flows", f"{df_batch.shape[0]:,}")
            with col2:
                st.metric("Features", f"{df_batch.shape[1]}")
            with col3:
                st.metric("File Size", f"{batch_file.size / (1024*1024):.2f} MB")
            
            with st.expander("Data Preview"):
                st.dataframe(df_batch.head(20), use_container_width=True, height=400)
            
            if st.button("Process All Flows", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Analyzing network flows..."):
                    actual_labels = None
                    if 'Label' in df_batch.columns:
                        actual_labels = df_batch['Label'].copy()
                        df_batch_clean = df_batch.drop('Label', axis=1)
                    else:
                        df_batch_clean = df_batch.copy()
                    
                    status_text.text("Step 1/4: Preparing data...")
                    progress_bar.progress(25)
                    
                    df_batch_clean = df_batch_clean[feature_names]
                    batch_imputed = imputer.transform(df_batch_clean)
                    
                    status_text.text("Step 2/4: Normalizing features...")
                    progress_bar.progress(50)
                    
                    batch_scaled = scaler.transform(batch_imputed)
                    
                    status_text.text("Step 3/4: Running classification...")
                    progress_bar.progress(75)
                    
                    predictions = model.predict(batch_scaled)
                    probabilities = model.predict_proba(batch_scaled)
                    
                    status_text.text("Step 4/4: Generating report...")
                    progress_bar.progress(90)
                    
                    predicted_labels = [label_encoder.classes_[p] for p in predictions]
                    confidences = [probabilities[i][predictions[i]] * 100 for i in range(len(predictions))]
                    
                    results_df = pd.DataFrame({
                        'Flow_ID': range(1, len(predictions) + 1),
                        'Classification': predicted_labels,
                        'Confidence_%': confidences
                    })
                    
                    if actual_labels is not None:
                        results_df['Actual_Label'] = actual_labels.values
                        results_df['Status'] = results_df['Classification'] == results_df['Actual_Label']
                        results_df['Status'] = results_df['Status'].map({True: 'Correct', False: 'Incorrect'})
                    
                    progress_bar.progress(100)
                    time.sleep(0.3)
                    progress_bar.empty()
                    status_text.empty()
                
                st.markdown("---")
                st.markdown("### Analysis Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                total = len(predictions)
                benign_count = sum([1 for p in predicted_labels if p == 'Benign'])
                attack_count = total - benign_count
                avg_conf = np.mean(confidences)
                
                with col1:
                    st.metric("Total Analyzed", f"{total:,}")
                with col2:
                    st.metric("Benign Traffic", f"{benign_count:,}", f"{benign_count/total*100:.1f}%")
                with col3:
                    st.metric("Threats Detected", f"{attack_count:,}", f"{attack_count/total*100:.1f}%")
                with col4:
                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")
                
                st.markdown("### Threat Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    attack_dist = pd.Series(predicted_labels).value_counts()
                    
                    fig_pie = go.Figure(go.Pie(
                        labels=attack_dist.index,
                        values=attack_dist.values,
                        hole=0.4,
                        marker=dict(line=dict(color='white', width=2))
                    ))
                    
                    fig_pie.update_layout(
                        title='Traffic Classification Distribution',
                        template='plotly_white',
                        height=400,
                        font=dict(family='Inter, sans-serif')
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_bar = go.Figure(go.Bar(
                        x=attack_dist.values,
                        y=attack_dist.index,
                        orientation='h',
                        marker=dict(
                            color=attack_dist.values,
                            colorscale='Viridis',
                            showscale=False
                        ),
                        text=attack_dist.values,
                        textposition='auto'
                    ))
                    
                    fig_bar.update_layout(
                        title='Count by Classification Type',
                        xaxis_title='Number of Flows',
                        yaxis_title='Classification',
                        template='plotly_white',
                        height=400,
                        font=dict(family='Inter, sans-serif')
                    )
                    
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.markdown("### Detailed Results")
                
                st.dataframe(
                    results_df.style.format({'Confidence_%': '{:.2f}'}),
                    use_container_width=True,
                    height=400
                )
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Analysis Report (CSV)",
                    data=csv,
                    file_name=f"threat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                if actual_labels is not None:
                    st.markdown("---")
                    st.markdown("### Model Performance Verification")
                    
                    accuracy = accuracy_score(actual_labels, predicted_labels)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Accuracy", f"{accuracy*100:.2f}%")
                    with col2:
                        correct = sum(results_df['Status'] == 'Correct')
                        st.metric("Correct Classifications", f"{correct}/{total}")
                    with col3:
                        incorrect = total - correct
                        st.metric("Misclassifications", f"{incorrect}/{total}")
                    
                    with st.expander("View Classification Report"):
                        report = classification_report(actual_labels, predicted_labels, output_dict=True, zero_division=0)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(
                            report_df.style.format("{:.4f}"),
                            use_container_width=True
                        )
                        
        except Exception as e:
            st.error(f"**Processing Error:** {str(e)}")

# TAB 3: Performance
with tab3:
    st.markdown("## Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Accuracy", "99.57%", "+0.08% vs baseline")
    with col2:
        st.metric("Benign Detection Rate", "99.91%", "51,399/51,404")
    with col3:
        st.metric("Attack Detection Rate", "66.60%", "355/533 Syn attacks")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Technical Specifications")
        st.markdown("""
        <div class='info-card'>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr><td style='padding: 0.5rem 0; font-weight: 600;'>Algorithm</td><td>XGBoost Classifier</td></tr>
                <tr><td style='padding: 0.5rem 0; font-weight: 600;'>Estimators</td><td>300 trees</td></tr>
                <tr><td style='padding: 0.5rem 0; font-weight: 600;'>Max Depth</td><td>15 levels</td></tr>
                <tr><td style='padding: 0.5rem 0; font-weight: 600;'>Learning Rate</td><td>0.05</td></tr>
                <tr><td style='padding: 0.5rem 0; font-weight: 600;'>Training Time</td><td>~110 seconds</td></tr>
                <tr><td style='padding: 0.5rem 0; font-weight: 600;'>Training Samples</td><td>125,170 flows</td></tr>
                <tr><td style='padding: 0.5rem 0; font-weight: 600;'>Test Samples</td><td>306,201 flows</td></tr>
                <tr><td style='padding: 0.5rem 0; font-weight: 600;'>Features</td><td>77 parameters</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Per-Class Performance")
        
        performance_data = pd.DataFrame({
            'Class': ['Benign', 'Syn'],
            'Precision': [99.65, 90.10],
            'Recall': [99.91, 66.60],
            'F1-Score': [99.78, 76.59],
            'Support': [51404, 533]
        })
        
        st.dataframe(
            performance_data.style.format({
                'Precision': '{:.2f}%',
                'Recall': '{:.2f}%',
                'F1-Score': '{:.2f}%',
                'Support': '{:,}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    st.markdown("### Known Limitations")
    
    st.markdown("""
    <div class='warning-card'>
        <strong>Syn Attack Detection Challenge</strong><br><br>
        The model achieves 66.60% detection rate for Syn attacks due to significant feature similarity 
        with Benign traffic. This represents a known challenge in network security research where certain 
        attack types exhibit characteristics nearly identical to normal traffic patterns.
        <br><br>
        <strong>Mitigation Attempts:</strong>
        <ul>
            <li>Class weighting optimization</li>
            <li>SMOTE synthetic sample generation</li>
            <li>Threshold tuning techniques</li>
            <li>Hyperparameter optimization</li>
        </ul>
        Current performance represents state-of-the-art results for this specific dataset configuration.
    </div>
    """, unsafe_allow_html=True)

# TAB 4: Documentation
with tab4:
    st.markdown("## System Documentation")
    
    st.markdown("### Overview")
    st.markdown("""
    This system employs machine learning techniques to identify and classify Distributed Denial of 
    Service (DDoS) attacks in real-time network traffic. The system analyzes 77 distinct network flow 
    characteristics to determine whether traffic is benign or represents a security threat.
    """)
    
    st.markdown("### Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <strong>Data Pipeline</strong>
            <ol>
                <li>Raw network flow capture</li>
                <li>Feature extraction (77 parameters)</li>
                <li>Missing value imputation</li>
                <li>Feature standardization</li>
                <li>XGBoost classification</li>
                <li>Confidence scoring</li>
                <li>Result reporting</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <strong>Technology Stack</strong>
            <ul>
                <li><strong>Core Algorithm:</strong> XGBoost</li>
                <li><strong>Data Processing:</strong> Pandas, NumPy</li>
                <li><strong>ML Framework:</strong> Scikit-learn</li>
                <li><strong>Visualization:</strong> Plotly</li>
                <li><strong>Interface:</strong> Streamlit</li>
                <li><strong>Deployment:</strong> Python 3.x</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Dataset Information")
    st.markdown("""
    <div class='info-card'>
        <strong>Training Dataset:</strong> 125,170 network flows<br>
        <strong>Testing Dataset:</strong> 306,201 network flows<br>
        <strong>Feature Set:</strong> 77 network flow characteristics including:<br>
        <ul>
            <li>Protocol information</li>
            <li>Packet statistics (count, length, timing)</li>
            <li>Flow duration metrics</li>
            <li>Bidirectional traffic patterns</li>
            <li>Inter-arrival time features</li>
            <li>Flag distribution analysis</li>
        </ul>
        <strong>Classification Categories:</strong><br>
        Benign, LDAP, MSSQL, NetBIOS, Portmap, Syn, UDP, UDPLag, Unknown
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Performance Metrics Explanation")
    
    metrics_explanation = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Definition': [
            'Percentage of correct predictions overall',
            'Of predicted attacks, how many were actual attacks',
            'Of actual attacks, how many were detected',
            'Harmonic mean of precision and recall'
        ],
        'Formula': [
            '(TP + TN) / (TP + TN + FP + FN)',
            'TP / (TP + FP)',
            'TP / (TP + FN)',
            '2 × (Precision × Recall) / (Precision + Recall)'
        ]
    })
    
    st.dataframe(metrics_explanation, use_container_width=True, hide_index=True)
    
    st.markdown("### Usage Guidelines")
    st.markdown("""
    <div class='info-card'>
        <strong>Data Requirements:</strong>
        <ul>
            <li>CSV format with 77 network flow features</li>
            <li>Column names must match training data schema</li>
            <li>No authentication or encrypted data processing</li>
        </ul>
        
        <strong>Best Practices:</strong>
        <ul>
            <li>Validate input data format before analysis</li>
            <li>Review confidence scores alongside classifications</li>
            <li>Investigate low-confidence predictions manually</li>
            <li>Regular model retraining with new threat data recommended</li>
        </ul>
        
        <strong>Limitations:</strong>
        <ul>
            <li>Educational/research system - not production-hardened</li>
            <li>Limited to trained attack types</li>
            <li>May misclassify novel attack variants
            </li>
            <li>Performance degradation on adversarial attacks</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Citation")
    st.code("""
    DDoS Detection System v1.0
    Algorithm: XGBoost Gradient Boosting
    Training Dataset: Custom DDoS Dataset (125K+ flows)
    Accuracy: 99.57%
    Development: 2024
    """)
    st.markdown("""
    <div class='footer'>
    <strong>Network Security Analysis System</strong> | Version 1.0<br>
    Powered by XGBoost Machine Learning<br>
    For  Educational Purposes
    </div>
    """, unsafe_allow_html=True)
    
    
    