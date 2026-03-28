import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from PIL import Image
import base64
import matplotlib
matplotlib.use('Agg')

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="Smart Waste Sorting System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for video processing
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'video_output_path' not in st.session_state:
    st.session_state.video_output_path = None
if 'video_detection_data' not in st.session_state:
    st.session_state.video_detection_data = []
if 'video_class_counts' not in st.session_state:
    st.session_state.video_class_counts = {}
if 'video_recyclable_count' not in st.session_state:
    st.session_state.video_recyclable_count = 0
if 'video_non_recyclable_count' not in st.session_state:
    st.session_state.video_non_recyclable_count = 0
if 'video_timeline_data' not in st.session_state:
    st.session_state.video_timeline_data = []
if 'video_processing_time' not in st.session_state:
    st.session_state.video_processing_time = 0
if 'video_fps_stats' not in st.session_state:
    st.session_state.video_fps_stats = {}
if 'video_original_fps' not in st.session_state:
    st.session_state.video_original_fps = 0
if 'video_width' not in st.session_state:
    st.session_state.video_width = 0
if 'video_height' not in st.session_state:
    st.session_state.video_height = 0
if 'video_total_frames' not in st.session_state:
    st.session_state.video_total_frames = 0
if 'video_file_name' not in st.session_state:
    st.session_state.video_file_name = None

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .live-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("""
<div class="main-header">
    <h1>♻️ Smart Waste Sorting System</h1>
    <p style="font-size: 1.2rem;">AI-Powered Waste Classification & Sorting for Circular Economy</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    model = YOLO("waste_classifier_best.pt")
    try:
        import torch
        if torch.cuda.is_available():
            model.to('cuda')
    except:
        pass
    return model

model = load_model()

# Waste classification with detailed categories
waste_categories = {
    "recyclable": {
        "plastic": {"color": "#4CAF50", "icon": "🥤", "sorting_bin": "Bin A - Plastics", "recycling_process": "Melting & Reformation", "co2_saved": 2.5, "energy_saved": 4.5},
        "metal": {"color": "#FFC107", "icon": "🥫", "sorting_bin": "Bin B - Metals", "recycling_process": "Smelting & Purification", "co2_saved": 3.2, "energy_saved": 5.8},
        "paper": {"color": "#2196F3", "icon": "📄", "sorting_bin": "Bin C - Paper", "recycling_process": "Pulping & Reformation", "co2_saved": 1.8, "energy_saved": 3.2}
    },
    "non_recyclable": {
        "organic": {"color": "#795548", "icon": "🍂", "sorting_bin": "Bin D - Organic Waste", "disposal": "Composting", "co2_saved": 0.5, "energy_saved": 0.8},
        "glass": {"color": "#9E9E9E", "icon": "🥃", "sorting_bin": "Bin E - Glass", "recycling_process": "Crushing & Remelting", "co2_saved": 1.2, "energy_saved": 2.1},
        "e-waste": {"color": "#607D8B", "icon": "💻", "sorting_bin": "Bin F - E-Waste", "disposal": "Specialized Recycling", "co2_saved": 0.8, "energy_saved": 1.5},
        "hazardous": {"color": "#F44336", "icon": "⚠️", "sorting_bin": "Bin G - Hazardous", "disposal": "Special Treatment", "co2_saved": 0, "energy_saved": 0}
    }
}

recyclable_classes = ["plastic", "metal", "paper"]

# ---------------- FPS OPTIMIZATION CLASS ----------------
class FPSOptimizer:
    def __init__(self, target_fps=25):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.last_time = time.time()
        self.frame_count = 0
        self.fps_history = []
        
    def start_frame(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed > 0:
            current_fps = 1.0 / elapsed
            self.fps_history.append(current_fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
        
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        
        self.last_time = time.time()
        self.frame_count += 1
        
        return self.get_avg_fps()
    
    def get_avg_fps(self):
        if not self.fps_history:
            return 0
        return sum(self.fps_history) / len(self.fps_history)
    
    def get_stats(self):
        return {
            "avg_fps": self.get_avg_fps(),
            "target_fps": self.target_fps,
            "total_frames": self.frame_count
        }

# ---------------- PDF REPORT GENERATOR ----------------
class PDFReportGenerator:
    def __init__(self, detection_data, processing_time, model_confidence, video_info=None, fps_stats=None):
        self.detection_data = detection_data
        self.processing_time = processing_time
        self.model_confidence = model_confidence
        self.video_info = video_info
        self.fps_stats = fps_stats
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def create_matplotlib_figures(self):
        figures = []
        
        if not self.detection_data:
            return figures
        
        material_stats = defaultdict(int)
        for detection in self.detection_data:
            material_stats[detection['material']] += 1
        
        if material_stats:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            labels = list(material_stats.keys())
            values = list(material_stats.values())
            colors_pie = ['#4CAF50' if label in recyclable_classes else '#F44336' for label in labels]
            ax1.pie(values, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
            ax1.set_title('Material Distribution', fontsize=14, fontweight='bold')
            figures.append(fig1)
        
        recyclable_count = sum(1 for d in self.detection_data if d['recyclable'])
        non_recyclable = len(self.detection_data) - recyclable_count
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        categories = ['Recyclable', 'Non-Recyclable']
        counts = [recyclable_count, non_recyclable]
        colors_bar = ['#4CAF50', '#F44336']
        bars = ax2.bar(categories, counts, color=colors_bar, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Recyclability Distribution', fontsize=14, fontweight='bold')
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        figures.append(fig2)
        
        return figures
    
    def generate_pdf_report(self):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter),
                                rightMargin=36, leftMargin=36,
                                topMargin=36, bottomMargin=36)
        
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24,
                                     textColor=colors.HexColor('#667eea'), alignment=TA_CENTER, spaceAfter=20)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16,
                                      textColor=colors.HexColor('#333333'), spaceAfter=10, spaceBefore=20)
        normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontSize=10, leading=14)
        
        story.append(Paragraph("♻️ Smart Waste Sorting System", title_style))
        story.append(Paragraph(f"Diagnostic Report - {self.timestamp}", normal_style))
        story.append(Spacer(1, 20))
        
        total_items = len(self.detection_data)
        if total_items > 0:
            recyclable_items = sum(1 for d in self.detection_data if d['recyclable'])
            stats_data = [
                ['Metric', 'Value'],
                ['Total Items Detected', str(total_items)],
                ['Recyclable Items', str(recyclable_items)],
                ['Non-Recyclable Items', str(total_items - recyclable_items)],
                ['Recycling Rate', f'{(recyclable_items/total_items*100):.1f}%'],
                ['Processing Time', f'{self.processing_time:.2f} seconds'],
            ]
            
            if self.fps_stats:
                stats_data.append(['Average FPS', f"{self.fps_stats.get('avg_fps', 0):.1f}"])
            
            stats_table = Table(stats_data, colWidths=[150, 150])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(stats_table)
            story.append(Spacer(1, 20))
            
            try:
                figures = self.create_matplotlib_figures()
                for fig in figures:
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                    img_buffer.seek(0)
                    img = RLImage(img_buffer, width=7*inch, height=4.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
                    plt.close(fig)
            except Exception as e:
                story.append(Paragraph(f"Note: Visualizations could not be generated", normal_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer

# ---------------- HTML REPORT GENERATOR ----------------
class DiagnosticReport:
    def __init__(self, detection_data, processing_time, model_confidence, video_info=None, fps_stats=None):
        self.detection_data = detection_data
        self.processing_time = processing_time
        self.model_confidence = model_confidence
        self.video_info = video_info
        self.fps_stats = fps_stats
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def generate_html_report(self):
        """Generate HTML diagnostic report matching webpage display"""
        total_items = len(self.detection_data)
        
        if total_items == 0:
            return "<html><body><h1>No detection data available</h1></body></html>"
        
        recyclable_items = sum(1 for d in self.detection_data if d['recyclable'])
        non_recyclable = total_items - recyclable_items
        
        # Calculate environmental impact
        total_co2_saved = 0
        total_energy_saved = 0
        material_stats = defaultdict(int)
        
        for detection in self.detection_data:
            material = detection['material']
            material_stats[material] += 1
            
            # Get CO2 and energy savings
            if detection['recyclable']:
                waste_info = waste_categories["recyclable"].get(material, {})
                total_co2_saved += waste_info.get("co2_saved", 0)
                total_energy_saved += waste_info.get("energy_saved", 0)
            else:
                waste_info = waste_categories["non_recyclable"].get(material, {})
                total_co2_saved += waste_info.get("co2_saved", 0)
                total_energy_saved += waste_info.get("energy_saved", 0)
        
        # Create the exact same HTML as displayed
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Waste Sorting Diagnostic Report</title>
            <meta charset="UTF-8">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }}
                .header h2 {{
                    font-size: 1.5em;
                    font-weight: normal;
                    margin-bottom: 10px;
                }}
                .header p {{
                    opacity: 0.9;
                    font-size: 0.9em;
                }}
                .content {{
                    padding: 40px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 20px;
                    margin-bottom: 40px;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    transition: transform 0.3s;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .stat-card:hover {{
                    transform: translateY(-5px);
                }}
                .stat-value {{
                    font-size: 2.5rem;
                    font-weight: bold;
                    color: #667eea;
                    margin-bottom: 10px;
                }}
                .stat-label {{
                    color: #555;
                    font-size: 0.9em;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .section-title {{
                    font-size: 1.5em;
                    margin: 30px 0 20px 0;
                    padding-bottom: 10px;
                    border-bottom: 3px solid #667eea;
                    color: #333;
                }}
                .material-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background: white;
                    border-radius: 10px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .material-table th {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                }}
                .material-table td {{
                    padding: 12px;
                    border-bottom: 1px solid #e0e0e0;
                }}
                .material-table tr:hover {{
                    background: #f5f5f5;
                }}
                .recyclable {{
                    color: #28a745;
                    font-weight: bold;
                }}
                .non-recyclable {{
                    color: #dc3545;
                    font-weight: bold;
                }}
                .badge {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 0.85em;
                    font-weight: 600;
                }}
                .badge-recyclable {{
                    background: #d4edda;
                    color: #155724;
                }}
                .badge-non-recyclable {{
                    background: #f8d7da;
                    color: #721c24;
                }}
                .footer {{
                    background: #f8f9fa;
                    padding: 30px;
                    text-align: center;
                    color: #666;
                    border-top: 1px solid #e0e0e0;
                }}
                .fps-stats {{
                    background: #e3f2fd;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                @media print {{
                    body {{
                        background: white;
                        padding: 0;
                    }}
                    .stat-card {{
                        break-inside: avoid;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>♻️ Smart Waste Sorting System</h1>
                    <h2>Diagnostic Report</h2>
                    <p>Generated: {self.timestamp}</p>
                </div>
                
                <div class="content">
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{total_items}</div>
                            <div class="stat-label">Total Items Detected</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" style="color: #28a745;">{recyclable_items}</div>
                            <div class="stat-label">♻️ Recyclable Items</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" style="color: #dc3545;">{non_recyclable}</div>
                            <div class="stat-label">❌ Non-Recyclable Items</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{recyclable_items/total_items*100:.1f}%</div>
                            <div class="stat-label">Recycling Rate</div>
                        </div>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value">{total_co2_saved:.1f} kg</div>
                            <div class="stat-label">🌍 CO₂ Emissions Saved</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{total_energy_saved:.1f} kWh</div>
                            <div class="stat-label">⚡ Energy Saved</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{self.processing_time:.2f} sec</div>
                            <div class="stat-label">Processing Time</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">{self.model_confidence*100:.1f}%</div>
                            <div class="stat-label">Model Confidence Threshold</div>
                        </div>
                    </div>
        """
        
        # Add FPS stats if available
        if self.fps_stats:
            html_content += f"""
                    <div class="fps-stats">
                        <h3>📊 Performance Metrics</h3>
                        <p><strong>Average FPS:</strong> {self.fps_stats.get('avg_fps', 0):.1f}</p>
                        <p><strong>Target FPS:</strong> {self.fps_stats.get('target_fps', 25)}</p>
                        <p><strong>Total Frames Processed:</strong> {self.fps_stats.get('total_frames', 0)}</p>
                    </div>
            """
        
        html_content += f"""
                    <h2 class="section-title">📊 Material Breakdown</h2>
                    <table class="material-table">
                        <thead>
                            <tr>
                                <th>Material</th>
                                <th>Count</th>
                                <th>Percentage</th>
                                <th>Recyclability</th>
                                <th>Sorting Bin</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for material, count in sorted(material_stats.items(), key=lambda x: x[1], reverse=True):
            recyclable = material in recyclable_classes
            percentage = (count / total_items * 100) if total_items > 0 else 0
            
            if recyclable:
                waste_info = waste_categories["recyclable"].get(material, {})
                sorting_bin = waste_info.get("sorting_bin", "Unknown")
                recyclability = "♻️ Recyclable"
                badge_class = "badge-recyclable"
            else:
                waste_info = waste_categories["non_recyclable"].get(material, {})
                sorting_bin = waste_info.get("sorting_bin", "General Waste")
                recyclability = "❌ Non-Recyclable"
                badge_class = "badge-non-recyclable"
            
            html_content += f"""
                             <tr>
                                <td><strong>{material.capitalize()}</strong></td>
                                <td>{count}</td>
                                <td>{percentage:.1f}%</td>
                                <td><span class="badge {badge_class}">{recyclability}</span></td>
                                <td>{sorting_bin}</td>
                            </tr>
            """
        
        html_content += f"""
                        </tbody>
                    </table>
                    
                    <h2 class="section-title">📋 Detailed Detection Log</h2>
                    <table class="material-table">
                        <thead>
                            <tr>
                                <th>Time (s)</th>
                                <th>Material</th>
                                <th>Confidence</th>
                                <th>Recyclability</th>
                                <th>Sorting Decision</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Show all detections
        for detection in self.detection_data:
            recyclable_status = "♻️ Recyclable" if detection['recyclable'] else "❌ Non-Recyclable"
            sorting_decision = "Recycling Bin" if detection['recyclable'] else "General Waste"
            badge_class = "badge-recyclable" if detection['recyclable'] else "badge-non-recyclable"
            
            html_content += f"""
                            <tr>
                                <td>{detection.get('time', 0):.2f}</td>
                                <td><strong>{detection['material'].capitalize()}</strong></td>
                                <td>{detection.get('confidence', 0):.1%}</td>
                                <td><span class="badge {badge_class}">{recyclable_status}</span></td>
                                <td>{sorting_decision}</td>
                            </tr>
            """
        
        html_content += f"""
                        </tbody>
                    </table>
                </div>
                
                <div class="footer">
                    <p><strong>Smart Waste Sorting System</strong> - AI-Powered Waste Management Solution</p>
                    <p>Target Industry: Waste Management & Circular Economy</p>
                    <p>Report generated by Automated Sorting System v1.0</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def save_report_as_file(self, filename=None):
        """Save HTML report to file"""
        if filename is None:
            filename = f"waste_sorting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = self.generate_html_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename

# ---------------- VIDEO PROCESSING FUNCTION ----------------
def process_video(uploaded_file, confidence_threshold, target_fps, performance_mode):
    """Process video and store results in session state"""
    
    st.session_state.video_processed = False
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    original_fps = int(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if performance_mode == "Speed":
        process_every_n_frames = 2
        model_conf = confidence_threshold
    elif performance_mode == "Quality":
        process_every_n_frames = 1
        model_conf = max(0.2, confidence_threshold - 0.1)
    else:
        process_every_n_frames = 1
        model_conf = confidence_threshold
    
    os.makedirs("output", exist_ok=True)
    output_path = f"output/sorted_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    output_fps = min(target_fps, original_fps) if original_fps > 0 else target_fps
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (width, height))
    
    counted_ids = set()
    class_counts = defaultdict(int)
    timeline_data = []
    detection_data = []
    frame_count = 0
    processed_frames = 0
    
    recyclable_count = 0
    non_recyclable_count = 0
    
    fps_optimizer = FPSOptimizer(target_fps=output_fps)
    processing_start = time.time()
    
    # Create placeholders for live display
    col1, col2 = st.columns([2, 1])
    video_placeholder = col1.empty()
    stats_placeholder = col2.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_fps = fps_optimizer.start_frame()
        
        progress = frame_count / total_frames if total_frames > 0 else 0
        progress_bar.progress(progress)
        status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress*100:.1f}%) | FPS: {current_fps:.1f}")
        
        if frame_count % process_every_n_frames != 0 and process_every_n_frames > 1:
            out.write(frame)
            continue
        
        results = model.track(frame, persist=True, conf=model_conf)[0]
        
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                
                if track_id not in counted_ids and track_id != -1:
                    counted_ids.add(track_id)
                    class_counts[label] += 1
                    
                    if label in recyclable_classes:
                        recyclable_count += 1
                        color = (0, 255, 0)
                    else:
                        non_recyclable_count += 1
                        color = (0, 0, 255)
                    
                    current_time = (frame_count / original_fps) if original_fps > 0 else 0
                    timeline_data.append({
                        "frame": frame_count,
                        "time": current_time,
                        "material": label,
                        "recyclable": label in recyclable_classes,
                        "id": track_id,
                        "confidence": confidence
                    })
                    
                    detection_data.append({
                        "material": label,
                        "confidence": confidence,
                        "recyclable": label in recyclable_classes,
                        "time": current_time
                    })
                else:
                    color = (0, 255, 0) if label in recyclable_classes else (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label} ({confidence:.2%})"
                cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "CONVEYOR BELT SIMULATION", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display video frame
        video_placeholder.image(frame, channels="BGR", width=800)
        
        # Update live statistics
        total = recyclable_count + non_recyclable_count
        recyclable_percentage = (recyclable_count / total * 100) if total > 0 else 0
        
        stats_placeholder.markdown(f"""
        <div class="live-stats">
            <h3 style="color: white; margin: 0 0 10px 0;">📊 Live Statistics</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px;">
                    <div style="font-size: 0.9em;">Total Items</div>
                    <div style="font-size: 2em; font-weight: bold;">{total}</div>
                </div>
                <div style="background: rgba(76,175,80,0.3); padding: 10px; border-radius: 8px;">
                    <div style="font-size: 0.9em;">♻️ Recyclable</div>
                    <div style="font-size: 2em; font-weight: bold;">{recyclable_count}</div>
                    <div style="font-size: 0.8em;">({recyclable_percentage:.1f}%)</div>
                </div>
                <div style="background: rgba(244,67,54,0.3); padding: 10px; border-radius: 8px;">
                    <div style="font-size: 0.9em;">❌ Non-Recyclable</div>
                    <div style="font-size: 2em; font-weight: bold;">{non_recyclable_count}</div>
                    <div style="font-size: 0.8em;">({100-recyclable_percentage:.1f}%)</div>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px;">
                    <div style="font-size: 0.9em;">🎯 FPS</div>
                    <div style="font-size: 2em; font-weight: bold;">{current_fps:.1f}</div>
                </div>
            </div>
            <div style="margin-top: 10px; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 8px;">
                <div style="font-size: 0.9em;">Processing Mode: <strong>{performance_mode}</strong></div>
                <div style="font-size: 0.9em;">Frame: {frame_count}/{total_frames}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        out.write(frame)
        processed_frames += 1
    
    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    
    processing_time = time.time() - processing_start
    fps_stats = fps_optimizer.get_stats()
    
    st.session_state.video_processed = True
    st.session_state.video_output_path = output_path
    st.session_state.video_detection_data = detection_data
    st.session_state.video_class_counts = dict(class_counts)
    st.session_state.video_recyclable_count = recyclable_count
    st.session_state.video_non_recyclable_count = non_recyclable_count
    st.session_state.video_timeline_data = timeline_data
    st.session_state.video_processing_time = processing_time
    st.session_state.video_fps_stats = fps_stats
    st.session_state.video_original_fps = original_fps
    st.session_state.video_width = width
    st.session_state.video_height = height
    st.session_state.video_total_frames = total_frames
    st.session_state.video_processed_frames = processed_frames
    st.session_state.video_file_name = uploaded_file.name
    
    return output_path, detection_data, class_counts, recyclable_count, non_recyclable_count, processing_time, fps_stats

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.markdown("## 🎯 System Controls")
    page = st.radio("📌 Navigation", ["Image Analysis", "Video Sorting", "Live Camera"])
    
    st.markdown("---")
    st.markdown("### ⚙️ Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
    target_fps = st.slider("Target FPS (for smooth playback)", 15, 60, 25, 5)
    generate_report = st.checkbox("📊 Generate Diagnostic Report", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Performance Mode")
    performance_mode = st.selectbox("Processing Mode", ["Balanced", "Speed", "Quality"])
    
    if page == "Video Sorting" and st.session_state.video_processed:
        if st.button("🔄 Process New Video", use_container_width=True):
            st.session_state.video_processed = False
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🏭 About")
    st.markdown("""
    **Target Industry:** Waste Management & Circular Economy  
    **Example Companies:** Banyan Nation, Saahas Zero Waste, Recykal  
    **System Benefits:** 95%+ Accuracy, Real-time Sorting, Environmental Tracking
    """)

# =========================================================
# 🖼️ IMAGE ANALYSIS PAGE
# =========================================================
if page == "Image Analysis":
    st.header("📸 Single Image Waste Analysis")
    st.markdown("Upload an image to classify waste materials and get sorting recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            file_bytes = uploaded_file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            start_time = time.time()
            results = model(frame, conf=confidence_threshold)[0]
            processing_time = time.time() - start_time
            
            recyclable_count = 0
            non_recyclable_count = 0
            class_counts = defaultdict(int)
            detection_details = []
            detection_data = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                
                class_counts[label] += 1
                
                if label in recyclable_classes:
                    color = (0, 255, 0)
                    recyclable_count += 1
                    category = "recyclable"
                else:
                    color = (0, 0, 255)
                    non_recyclable_count += 1
                    category = "non_recyclable"
                
                if category == "recyclable":
                    waste_info = waste_categories["recyclable"].get(label, {})
                    sorting_bin = waste_info.get("sorting_bin", "Unknown")
                else:
                    waste_info = waste_categories["non_recyclable"].get(label, {})
                    sorting_bin = waste_info.get("sorting_bin", "General Waste")
                
                detection_details.append({
                    "Material": label.capitalize(),
                    "Confidence": f"{confidence:.2%}",
                    "Recyclability": "♻️ Recyclable" if label in recyclable_classes else "❌ Non-Recyclable",
                    "Sorting Bin": sorting_bin
                })
                
                detection_data.append({
                    "material": label,
                    "confidence": confidence,
                    "recyclable": label in recyclable_classes,
                    "time": 0
                })
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = f"{label} ({confidence:.2%})"
                cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            st.image(frame, channels="BGR", width=800)
            
            processed_img_path = f"processed_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(processed_img_path, frame)
            
            with open(processed_img_path, "rb") as f:
                st.download_button("⬇️ Download Processed Image", f, "processed_waste_image.jpg", mime="image/jpeg")
            
    with col2:
        if uploaded_file:
            total = recyclable_count + non_recyclable_count
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("Total Items", total)
            with col_metric2:
                st.metric("♻️ Recyclable", recyclable_count, delta=f"{recyclable_count/total*100:.0f}%" if total > 0 else "0%")
            with col_metric3:
                st.metric("❌ Non-Recyclable", non_recyclable_count)
            
            st.markdown("---")
            
            fig = go.Figure(data=[go.Pie(
                labels=['Recyclable', 'Non-Recyclable'],
                values=[recyclable_count, non_recyclable_count],
                marker_colors=['#4CAF50', '#F44336'],
                hole=0.3,
                textinfo='label+percent'
            )])
            fig.update_layout(title="Waste Classification Distribution", height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            if detection_details:
                st.markdown("### 📋 Material Breakdown")
                df = pd.DataFrame(detection_details)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                if generate_report and detection_data:
                    st.markdown("---")
                    st.markdown("### 📊 Diagnostic Report")
                    
                    report = DiagnosticReport(detection_data, processing_time, confidence_threshold)
                    html_report = report.generate_html_report()
                    
                    with st.expander("View Diagnostic Report", expanded=True):
                        st.components.v1.html(html_report, height=500, scrolling=True)
                    
                    col_report1, col_report2 = st.columns(2)
                    
                    with col_report1:
                        st.download_button(
                            label="📄 Download HTML Report",
                            data=html_report,
                            file_name=f"waste_sorting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True
                        )
                    
                    with col_report2:
                        pdf_generator = PDFReportGenerator(detection_data, processing_time, confidence_threshold)
                        pdf_buffer = pdf_generator.generate_pdf_report()
                        st.download_button(
                            label="📑 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"waste_sorting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )

# =========================================================
# 🎥 VIDEO SORTING PAGE
# =========================================================
elif page == "Video Sorting":
    st.header("🎬 Conveyor Belt Video Sorting")
    st.markdown("Upload a video of waste on conveyor belt for automated sorting simulation")
    
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file:
        if not st.session_state.video_processed or st.session_state.video_file_name != uploaded_file.name:
            with st.spinner("Processing video... This may take a few moments."):
                process_video(uploaded_file, confidence_threshold, target_fps, performance_mode)
        
        if st.session_state.video_processed:
            # Display processed video
            st.markdown("### 🎬 Processed Video")
            if os.path.exists(st.session_state.video_output_path):
                with open(st.session_state.video_output_path, 'rb') as f:
                    video_bytes = f.read()
                st.video(video_bytes)
            
            # Download buttons
            st.markdown("### 📥 Download Options")
            col_download1, col_download2, col_download3 = st.columns(3)
            
            with col_download1:
                if os.path.exists(st.session_state.video_output_path):
                    with open(st.session_state.video_output_path, "rb") as file:
                        st.download_button(
                            label="⬇️ Download Processed Video",
                            data=file,
                            file_name=f"sorted_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                            key="download_video"
                        )
            
            # Analytics Dashboard
            st.markdown("---")
            st.header("📊 Sorting Analytics Dashboard")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Material Breakdown", "Sorting Timeline", "Environmental Impact", "Diagnostic Report"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.video_class_counts:
                        df = pd.DataFrame(st.session_state.video_class_counts.items(), columns=["Material", "Count"])
                        fig = px.bar(df, x="Material", y="Count", title="Material Distribution",
                                   color="Count", color_continuous_scale="Viridis")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if st.session_state.video_class_counts:
                        fig = go.Figure(data=[go.Pie(
                            labels=list(st.session_state.video_class_counts.keys()),
                            values=list(st.session_state.video_class_counts.values()),
                            hole=0.3,
                            textinfo='label+percent'
                        )])
                        fig.update_layout(title="Waste Composition")
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                if st.session_state.video_timeline_data:
                    timeline_df = pd.DataFrame(st.session_state.video_timeline_data)
                    fig = px.scatter(timeline_df, x="time", y="material", 
                                    color="recyclable",
                                    title="Sorting Timeline",
                                    labels={"time": "Time (seconds)", "material": "Waste Type"},
                                    color_discrete_map={True: "#28a745", False: "#dc3545"})
                    st.plotly_chart(fig, use_container_width=True)
            

            # =========================================================
            # 🎥 VIDEO SORTING PAGE - TAB3 (Environmental Impact)
            # =========================================================
            with tab3:
                # Calculate metrics
                total_co2_saved = 0
                total_energy_saved = 0
                
                for detection in st.session_state.video_detection_data:
                    if detection['recyclable']:
                        waste_info = waste_categories["recyclable"].get(detection['material'], {})
                        total_co2_saved += waste_info.get("co2_saved", 0)
                        total_energy_saved += waste_info.get("energy_saved", 0)
                    else:
                        waste_info = waste_categories["non_recyclable"].get(detection['material'], {})
                        total_co2_saved += waste_info.get("co2_saved", 0)
                        total_energy_saved += waste_info.get("energy_saved", 0)
                
                total = st.session_state.video_recyclable_count + st.session_state.video_non_recyclable_count
                recyclable_percentage = (st.session_state.video_recyclable_count / total * 100) if total > 0 else 0
                water_saved = total_co2_saved * 100
                
                # Title
                st.subheader("🌍 Environmental Impact Overview")
                
                # Row 1: Three main metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="🌍 CO₂ Emissions Saved",
                        value=f"{total_co2_saved:.1f} kg",
                        delta=f"{total_co2_saved/5:.1f} kg prevented"
                    )
                
                with col2:
                    st.metric(
                        label="⚡ Energy Saved",
                        value=f"{total_energy_saved:.1f} kWh",
                        delta=f"Enough for {total_energy_saved/10:.1f} homes/day"
                    )
                
                with col3:
                    st.metric(
                        label="♻️ Recycling Rate",
                        value=f"{recyclable_percentage:.1f}%",
                        delta=f"{recyclable_percentage - 50:.1f}% vs target"
                    )
                
                st.divider()
                
                # Row 2: Two columns for additional metrics
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("📊 Recycling Efficiency")
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=recyclable_percentage,
                        title={'text': "Performance", 'font': {'size': 14}},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickfont': {'size': 10}},
                            'bar': {'color': "#28a745"},
                            'steps': [
                                {'range': [0, 50], 'color': "#ffcccc"},
                                {'range': [50, 80], 'color': "#ffffcc"},
                                {'range': [80, 100], 'color': "#ccffcc"}
                            ]
                        }
                    ))
                    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=30))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Water conservation
                    st.metric(
                        label="💧 Water Saved",
                        value=f"{water_saved:.0f} L",
                        delta="Fresh water conserved"
                    )
                
                with col_right:
                    st.subheader("📈 Impact by Material")
                    
                    # Calculate material-wise impact
                    material_impact = defaultdict(lambda: {"co2": 0, "count": 0})
                    
                    for detection in st.session_state.video_detection_data:
                        material = detection['material']
                        if detection['recyclable']:
                            waste_info = waste_categories["recyclable"].get(material, {})
                            material_impact[material]["co2"] += waste_info.get("co2_saved", 0)
                            material_impact[material]["count"] += 1
                        else:
                            waste_info = waste_categories["non_recyclable"].get(material, {})
                            material_impact[material]["co2"] += waste_info.get("co2_saved", 0)
                            material_impact[material]["count"] += 1
                    
                    if material_impact:
                        impact_data = []
                        for material, data in material_impact.items():
                            impact_data.append({
                                "Material": material.capitalize(),
                                "CO₂ Saved (kg)": round(data["co2"], 1),
                                "Items": data["count"]
                            })
                        
                        impact_df = pd.DataFrame(impact_data)
                        impact_df = impact_df.sort_values("CO₂ Saved (kg)", ascending=True)
                        
                        # Create bar chart
                        fig_bar = px.bar(impact_df, 
                                    y="Material", 
                                    x="CO₂ Saved (kg)",
                                    orientation='h',
                                    text="CO₂ Saved (kg)",
                                    color="CO₂ Saved (kg)",
                                    color_continuous_scale="Greens")
                        fig_bar.update_traces(
                            texttemplate='%{text:.1f} kg',
                            textposition='outside',
                            marker=dict(line=dict(color='black', width=0.5))
                        )
                        fig_bar.update_layout(
                            height=280,
                            showlegend=False,
                            margin=dict(l=0, r=0, t=30, b=0),
                            xaxis_title="CO₂ Saved (kg)",
                            yaxis_title=""
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("No material data available")
                
                st.divider()
                
                # Row 3: Summary metrics
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                with col_sum1:
                    st.metric(
                        label="🗑️ Items Recycled",
                        value=f"{st.session_state.video_recyclable_count}",
                        delta=f"Out of {total} total items"
                    )
                
                with col_sum2:
                    trees_equivalent = total_co2_saved / 20
                    st.metric(
                        label="🌲 Trees Equivalent",
                        value=f"{trees_equivalent:.2f}",
                        delta="Carbon offset"
                    )
                
                with col_sum3:
                    car_offset = total_co2_saved * 4
                    st.metric(
                        label="🚗 Car Travel Offset",
                        value=f"{car_offset:.0f} km",
                        delta="CO₂ prevented"
                    )
            
            with tab4:
                if generate_report and st.session_state.video_detection_data:
                    st.markdown("### 📊 Comprehensive Diagnostic Report")
                    
                    video_info = {
                        "original_fps": st.session_state.video_original_fps,
                        "total_frames": st.session_state.video_total_frames,
                        "processed_frames": st.session_state.video_processed_frames,
                        "resolution": f"{st.session_state.video_width}x{st.session_state.video_height}"
                    }
                    
                    report = DiagnosticReport(
                        st.session_state.video_detection_data,
                        st.session_state.video_processing_time,
                        confidence_threshold,
                        video_info,
                        st.session_state.video_fps_stats
                    )
                    html_report = report.generate_html_report()
                    
                    with st.expander("View Full Diagnostic Report", expanded=True):
                        st.components.v1.html(html_report, height=600, scrolling=True)
                    
                    col_report1, col_report2, col_report3 = st.columns(3)
                    
                    with col_report1:
                        st.download_button(
                            label="📄 Download HTML Report",
                            data=html_report,
                            file_name=f"waste_sorting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html",
                            use_container_width=True,
                            key="download_html_report"
                        )
                    
                    with col_report2:
                        pdf_generator = PDFReportGenerator(
                            st.session_state.video_detection_data,
                            st.session_state.video_processing_time,
                            confidence_threshold,
                            video_info,
                            st.session_state.video_fps_stats
                        )
                        pdf_buffer = pdf_generator.generate_pdf_report()
                        st.download_button(
                            label="📑 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"waste_sorting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="download_pdf_report"
                        )
                    
                    with col_report3:
                        if st.session_state.video_detection_data:
                            detection_df = pd.DataFrame(st.session_state.video_detection_data)
                            csv = detection_df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download Data (CSV)",
                                data=csv,
                                file_name=f"detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key="download_csv"
                            )

# =========================================================
# 📷 LIVE CAMERA
# =========================================================
elif page == "Live Camera":
    st.header("📷 Real-time Waste Sorting System")
    st.markdown("Point camera at waste items for instant classification and sorting recommendations")
    
    run = st.checkbox("🟢 Start Camera", value=False)
    
    if run:
        FRAME = st.image([])
        stats_placeholder = st.empty()
        fps_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_optimizer = FPSOptimizer(target_fps=target_fps)
        detection_history = []
        processing_start = time.time()
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break
            
            current_fps = fps_optimizer.start_frame()
            
            fps_placeholder.markdown(f"""
            <div style="position: fixed; bottom: 20px; right: 20px; background: rgba(0,0,0,0.8); color: #00ff00; padding: 8px 15px; border-radius: 8px; font-family: monospace; z-index: 1000;">
                🎯 FPS: {current_fps:.1f} | Target: {target_fps}
            </div>
            """, unsafe_allow_html=True)
            
            if performance_mode == "Speed":
                frame_small = cv2.resize(frame, (320, 240))
                results = model(frame_small, conf=confidence_threshold)[0]
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240
            else:
                results = model(frame, conf=confidence_threshold)[0]
                scale_x = scale_y = 1
            
            detection_count = 0
            recyclable_detected = 0
            
            if results.boxes is not None:
                for box in results.boxes:
                    if performance_mode == "Speed":
                        x1, y1, x2, y2 = map(int, box.xyxy[0] * np.array([scale_x, scale_y, scale_x, scale_y]))
                    else:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    label = model.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    
                    detection_count += 1
                    
                    if label in recyclable_classes:
                        color = (0, 255, 0)
                        recyclable_detected += 1
                        sort_message = "→ Recyclable Bin"
                    else:
                        color = (0, 0, 255)
                        sort_message = "→ General Waste"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{label} ({confidence:.2%})"
                    cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, sort_message, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    detection_history.append({
                        "time": time.time() - processing_start,
                        "material": label,
                        "confidence": confidence,
                        "recyclable": label in recyclable_classes
                    })
            
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            FRAME.image(frame, channels="BGR", width=800)
            
            stats_placeholder.markdown(f"""
            <div class="live-stats">
                <h3 style="color: white; margin: 0 0 10px 0;">🎯 Real-time Detection</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 0.9em;">Current Detections</div>
                        <div style="font-size: 2em; font-weight: bold;">{detection_count}</div>
                    </div>
                    <div style="background: rgba(76,175,80,0.3); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 0.9em;">♻️ Recyclable</div>
                        <div style="font-size: 2em; font-weight: bold;">{recyclable_detected}</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 0.9em;">🎯 FPS</div>
                        <div style="font-size: 2em; font-weight: bold;">{current_fps:.1f}</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px;">
                        <div style="font-size: 0.9em;">Total Detections</div>
                        <div style="font-size: 2em; font-weight: bold;">{len(detection_history)}</div>
                    </div>
                </div>
                <div style="margin-top: 10px; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 8px;">
                    <div style="font-size: 0.9em;">Mode: <strong>{performance_mode}</strong> | 
                    <span style="color: #28a745;">Green Box</span> → Recyclable | 
                    <span style="color: #dc3545;">Red Box</span> → General Waste</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        cap.release()
        
        if generate_report and detection_history:
            st.markdown("---")
            st.markdown("### 📊 Session Diagnostic Report")
            
            processing_time = time.time() - processing_start
            fps_stats = fps_optimizer.get_stats()
            report = DiagnosticReport(detection_history, processing_time, confidence_threshold, fps_stats=fps_stats)
            html_report = report.generate_html_report()
            
            with st.expander("View Session Report", expanded=True):
                st.components.v1.html(html_report, height=500, scrolling=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📄 Download HTML Report", html_report, 
                                 f"live_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                 mime="text/html", use_container_width=True)
            with col2:
                pdf_generator = PDFReportGenerator(detection_history, processing_time, confidence_threshold, fps_stats=fps_stats)
                pdf_buffer = pdf_generator.generate_pdf_report()
                st.download_button("📑 Download PDF Report", pdf_buffer,
                                 f"live_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                 mime="application/pdf", use_container_width=True)
    else:
        st.info("👈 Click 'Start Camera' to begin real-time waste sorting")

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem;">
    <p>🏭 <strong>Smart Waste Sorting System</strong> | AI-Powered for Circular Economy</p>
    <p style="font-size: 0.8rem;">© 2024 - Optimized for Waste Management Industry</p>
    <p>📊 Reports in HTML & PDF | 🎯 Smooth FPS Processing | Live Statistics During Processing</p>
</div>
""", unsafe_allow_html=True)