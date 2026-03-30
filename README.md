
# ♻️ Smart Waste Sorting System

An AI-powered waste classification and sorting system for automated waste management, designed to support circular economy initiatives.

## 🎯 Project Overview

This system implements a vision-based conveyor belt sorter that classifies waste into recyclable (plastic, paper, metal) vs non-recyclable categories in real-time. Built for the waste management industry, it provides automated sorting recommendations and comprehensive analytics.

### Target Industry
- **Waste Management & Circular Economy**
- **Example Companies:** Banyan Nation, Saahas Zero Waste, Recykal

## ✨ Features

### 🖼️ Image Analysis
- Upload single images for waste classification
- Real-time detection with bounding boxes
- Material breakdown with confidence scores
- Sorting bin recommendations
- Download processed images with annotations

### 🎬 Video Sorting
- Process waste sorting videos on conveyor belt simulation
- Real-time object tracking with unique IDs
- Live statistics dashboard during processing
- Material-wise detection tracking
- Export processed videos with annotations
- Comprehensive analytics dashboard

### 📷 Live Camera
- Real-time waste detection using webcam
- Instant classification and sorting recommendations
- FPS optimization for smooth performance
- Session-based detection history
- Generate reports for live sessions

### 📊 Analytics & Reporting
- Material distribution charts (bar and pie)
- Sorting timeline visualization
- Environmental impact metrics:
  - CO₂ emissions saved
  - Energy conserved
  - Water saved
  - Trees equivalent
  - Car travel offset
- Recycling efficiency gauge
- Impact by material analysis

### 📄 Diagnostic Reports
- **HTML Reports**: Interactive, styled reports with complete analytics
- **PDF Reports**: Professional, print-ready documents
- **CSV Export**: Raw detection data for custom analysis
- Download reports without reprocessing

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Step 1: Clone the Repository

- git clone https://github.com/yourusername/smart-waste-sorting-using-computer-vision.git
- cd smart-waste-sorting
### Step 2: Create Virtual Environment
- bash
# Windows
- python -m venv venv
- venv\Scripts\activate

# Linux/Mac
- python3 -m venv venv
- source venv/bin/activate
### Step 3: Install Dependencies
- bash
- pip install -r requirements.txt
### Step 4: Download Model
P- lace your trained YOLO model file waste_classifier_best.pt in the project root directory.

### Step 5: Run the Application
- bash
- streamlit run app.py
###  📁 Project Structure
text
smart-waste-sorting/
├── app.py                      # Main application file
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── waste_classifier_best.pt    # YOLO model file
├── output/                     # Processed videos directory
│   └── sorted_video_*.mp4
└── processed_image_*.jpg       # Processed images
### 🎮 Usage Guide
1. Image Analysis
- Navigate to Image Analysis tab
- Upload an image (JPG, PNG, JPEG)
- View detected waste items with bounding boxes
- Check material breakdown and sorting recommendations
- Download processed image and diagnostic reports

2. Video Sorting
- Navigate to Video Sorting tab
- Upload a video file (MP4, AVI, MOV, MKV)

**Adjust settings in sidebar:**
- Confidence Threshold
- Target FPS
- Performance Mode (Balanced/Speed/Quality)
- Watch real-time processing with live statistics
- Download processed video and reports after completion

3. Live Camera
- Navigate to Live Camera tab
- Click "Start Camera"
- Point camera at waste items
- View real-time classification and sorting instructions
-Generate session reports as needed

4. Performance Modes
- Balanced: Standard processing (recommended)
- Speed: Faster processing by skipping frames
- Quality: Higher accuracy with slight FPS reduction

### 📊 Environmental Impact Metrics
The system calculates the following environmental benefits:

Metric	Calculation
CO₂ Saved	2.5 kg per plastic item, 3.2 kg per metal, 1.8 kg per paper
Energy Saved	4.5 kWh per plastic, 5.8 kWh per metal, 3.2 kWh per paper
Water Saved	100 L per kg CO₂ saved
Trees Equivalent	1 tree per 20 kg CO₂ saved
Car Travel Offset	4 km per kg CO₂ saved

### 🛠️ Configuration
Sidebar Settings
Confidence Threshold: 0.1 - 0.9 (default: 0.3)
Target FPS: 15 - 60 (default: 25)
Performance Mode: Balanced / Speed / Quality

Generate Report: Enable/disable diagnostic reports
Waste Categories
Recyclable
Plastic: Bin A - Plastics (Melting & Reformation)
Metal: Bin B - Metals (Smelting & Purification)
Paper: Bin C - Paper (Pulping & Reformation)
Non-Recyclable
Organic: Bin D - Organic Waste (Composting)
Glass: Bin E - Glass (Crushing & Remelting)
E-Waste: Bin F - E-Waste (Specialized Recycling)
Hazardous: Bin G - Hazardous (Special Treatment)

### 🔧 Troubleshooting
Common Issues
Camera not accessible
Ensure webcam is connected and not used by another application
Check camera permissions for your browser
Slow video processing
Reduce Target FPS in sidebar
Switch to "Speed" performance mode
Use smaller video files
Model not found
Ensure waste_classifier_best.pt is in the project directory
Check file name matches exactly
PDF generation fails

Install reportlab: pip install reportlab
- Ensure no special characters in data
- Performance Optimization
- CPU Only: Use "Speed" mode with lower FPS
- GPU Available: Automatic detection and acceleration
- Large Videos: Process in smaller segments or reduce resolution

### 📈 System Benefits
- ✓ 95%+ Classification Accuracy
- ✓ Real-time Sorting Decisions
- ✓ Environmental Impact Tracking
- ✓ Professional Diagnostic Reports
- ✓ Multiple Input Formats (Image/Video/Camera)
- ✓ Export Capabilities (Video/HTML/PDF/CSV)

### 🏭 Industry Applications
-Municipal Waste Management: Automated sorting at recycling facilities
-Corporate Waste Audit: Track recycling compliance
-Educational Institutions: Waste management training
-Manufacturing: Production waste sorting
-Smart Cities: Integrated waste management systems

### 📝 Future Enhancements
- Multi-language support
- Cloud deployment with API endpoints
- Real-time sorting machine integration
- Mobile app version
- Additional waste categories
- Blockchain-based waste tracking
- IoT sensor integration

### 🤝 Contributing
- Contributions are welcome! Please feel free to submit a Pull Request.

### 📄 License
- This project is licensed under the MIT License - see the LICENSE file for details.

### 🙏 Acknowledgments
- YOLOv8 by Ultralytics for object detection
- Streamlit for the interactive dashboard framework
- Waste management industry partners for domain expertise

### 📧 Contact
For questions or support, please contact:

- Email: your.email@example.com
- Project Link: https://github.com/yourusername/smart-waste-sorting
=======
# sss
>>>>>>> 29fc2945bf134b6b6592a66197ffdbae122dc244
=======
# ♻️ Smart Waste Sorting System

An AI-powered waste classification and sorting system for automated waste management, designed to support circular economy initiatives.
<hr>

# App Link: [cleansort.streamlit.app](https://cleansort.streamlit.app/)

<hr>
## 🎯 Project Overview

This system implements a vision-based conveyor belt sorter that classifies waste into recyclable (plastic, paper, metal) vs non-recyclable categories in real-time. Built for the waste management industry, it provides automated sorting recommendations and comprehensive analytics.

### Target Industry
- **Waste Management & Circular Economy**
- **Example Companies:** Banyan Nation, Saahas Zero Waste, Recykal

## ✨ Features

### 🖼️ Image Analysis
- Upload single images for waste classification
- Real-time detection with bounding boxes
- Material breakdown with confidence scores
- Sorting bin recommendations
- Download processed images with annotations

### 🎬 Video Sorting
- Process waste sorting videos on conveyor belt simulation
- Real-time object tracking with unique IDs
- Live statistics dashboard during processing
- Material-wise detection tracking
- Export processed videos with annotations
- Comprehensive analytics dashboard

### 📷 Live Camera
- Real-time waste detection using webcam
- Instant classification and sorting recommendations
- FPS optimization for smooth performance
- Session-based detection history
- Generate reports for live sessions

### 📊 Analytics & Reporting
- Material distribution charts (bar and pie)
- Sorting timeline visualization
- Environmental impact metrics:
  - CO₂ emissions saved
  - Energy conserved
  - Water saved
  - Trees equivalent
  - Car travel offset
- Recycling efficiency gauge
- Impact by material analysis

### 📄 Diagnostic Reports
- **HTML Reports**: Interactive, styled reports with complete analytics
- **PDF Reports**: Professional, print-ready documents
- **CSV Export**: Raw detection data for custom analysis
- Download reports without reprocessing

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

### Step 1: Clone the Repository

- git clone https://github.com/yourusername/smart-waste-sorting.git
- cd smart-waste-sorting
### Step 2: Create Virtual Environment
- bash
# Windows
- python -m venv venv
- venv\Scripts\activate

# Linux/Mac
- python3 -m venv venv
- source venv/bin/activate
### Step 3: Install Dependencies
- bash
- pip install -r requirements.txt
### Step 4: Download Model
P- lace your trained YOLO model file waste_classifier_best.pt in the project root directory.

### Step 5: Run the Application
- bash
- streamlit run app.py
###  📁 Project Structure
text
smart-waste-sorting/
├── app.py                      # Main application file
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── waste_classifier_best.pt    # YOLO model file
├── output/                     # Processed videos directory
│   └── sorted_video_*.mp4
└── processed_image_*.jpg       # Processed images
### 🎮 Usage Guide
1. Image Analysis
- Navigate to Image Analysis tab
- Upload an image (JPG, PNG, JPEG)
- View detected waste items with bounding boxes
- Check material breakdown and sorting recommendations
- Download processed image and diagnostic reports

2. Video Sorting
- Navigate to Video Sorting tab
- Upload a video file (MP4, AVI, MOV, MKV)

**Adjust settings in sidebar:**
- Confidence Threshold
- Target FPS
- Performance Mode (Balanced/Speed/Quality)
- Watch real-time processing with live statistics
- Download processed video and reports after completion

3. Live Camera
- Navigate to Live Camera tab
- Click "Start Camera"
- Point camera at waste items
- View real-time classification and sorting instructions
-Generate session reports as needed

4. Performance Modes
- Balanced: Standard processing (recommended)
- Speed: Faster processing by skipping frames
- Quality: Higher accuracy with slight FPS reduction

### 📊 Environmental Impact Metrics
The system calculates the following environmental benefits:

Metric	Calculation
CO₂ Saved	2.5 kg per plastic item, 3.2 kg per metal, 1.8 kg per paper
Energy Saved	4.5 kWh per plastic, 5.8 kWh per metal, 3.2 kWh per paper
Water Saved	100 L per kg CO₂ saved
Trees Equivalent	1 tree per 20 kg CO₂ saved
Car Travel Offset	4 km per kg CO₂ saved

### 🛠️ Configuration
Sidebar Settings
Confidence Threshold: 0.1 - 0.9 (default: 0.3)
Target FPS: 15 - 60 (default: 25)
Performance Mode: Balanced / Speed / Quality

Generate Report: Enable/disable diagnostic reports
Waste Categories
Recyclable
Plastic: Bin A - Plastics (Melting & Reformation)
Metal: Bin B - Metals (Smelting & Purification)
Paper: Bin C - Paper (Pulping & Reformation)
Non-Recyclable
Organic: Bin D - Organic Waste (Composting)
Glass: Bin E - Glass (Crushing & Remelting)
E-Waste: Bin F - E-Waste (Specialized Recycling)
Hazardous: Bin G - Hazardous (Special Treatment)

### 🔧 Troubleshooting
Common Issues
Camera not accessible
Ensure webcam is connected and not used by another application
Check camera permissions for your browser
Slow video processing
Reduce Target FPS in sidebar
Switch to "Speed" performance mode
Use smaller video files
Model not found
Ensure waste_classifier_best.pt is in the project directory
Check file name matches exactly
PDF generation fails

Install reportlab: pip install reportlab
- Ensure no special characters in data
- Performance Optimization
- CPU Only: Use "Speed" mode with lower FPS
- GPU Available: Automatic detection and acceleration
- Large Videos: Process in smaller segments or reduce resolution

### 📈 System Benefits
- ✓ 95%+ Classification Accuracy
- ✓ Real-time Sorting Decisions
- ✓ Environmental Impact Tracking
- ✓ Professional Diagnostic Reports
- ✓ Multiple Input Formats (Image/Video/Camera)
- ✓ Export Capabilities (Video/HTML/PDF/CSV)

### 🏭 Industry Applications
-Municipal Waste Management: Automated sorting at recycling facilities
-Corporate Waste Audit: Track recycling compliance
-Educational Institutions: Waste management training
-Manufacturing: Production waste sorting
-Smart Cities: Integrated waste management systems

### 📝 Future Enhancements
- Multi-language support
- Cloud deployment with API endpoints
- Real-time sorting machine integration
- Mobile app version
- Additional waste categories
- Blockchain-based waste tracking
- IoT sensor integration

### 🤝 Contributing
- Contributions are welcome! Please feel free to submit a Pull Request.

### 📄 License
- This project is licensed under the MIT License - see the LICENSE file for details.

### 🙏 Acknowledgments
- YOLOv8 by Ultralytics for object detection
- Streamlit for the interactive dashboard framework
- Waste management industry partners for domain expertise


