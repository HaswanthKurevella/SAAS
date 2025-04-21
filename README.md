# Student Attention Analysis System

## Overview
The **Student Attention Analysis System** is an AI-driven tool designed to analyze and improve student engagement in real-time. Using **deep learning, computer vision, and gaze-tracking algorithms**, the system detects facial expressions and eye movements to generate an "attention score." This score helps optimize teaching strategies and enhances learning outcomes.

## Features
- **Real-time engagement analysis** through facial expression recognition and eye-tracking.
- **Attention score generation** to measure student focus.
- **Scalable architecture** using compact cameras and real-time processing.
- **Interactive dashboard & mobile app** for personalized feedback.
- **Robust AI models** built on Convolutional Neural Networks (CNNs).
- **Handles real-world challenges** like lighting variations and occlusions.
- **Privacy and scalability focused** design.
- **Applicable in multiple domains:** classrooms, e-learning, corporate training, and healthcare.

## Implementation Details
1. **Data Collection**: Captures real-time video feed of students.
2. **Preprocessing**: Extracts facial landmarks and eye regions.
3. **Model Processing**: Uses CNNs and gaze-tracking algorithms to analyze attention.
4. **Score Calculation**:
   - **Gaze Ratio Calculation**: `Gaze Ratio = W_left / W_right`
   - **Attention Score**: `Score = w1 × Gaze + w2 × Emotion`
5. **Output & Visualization**:
   - Displays attention metrics on dashboards.
   - Provides insights via a mobile app.

## Installation
### Prerequisites
- Python 3.x
- OpenCV
- TensorFlow/Keras
- dlib
- Flask (for the web app/dashboard)
- NumPy, Pandas, Matplotlib (for data analysis)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Student-Attention-Analysis.git
cd Student-Attention-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the system
python main.py
```

## Usage
- **Step 1**: Connect a webcam or use a video input.
- **Step 2**: Run the program to analyze attention in real-time.
- **Step 3**: View the results on the dashboard.
- **Step 4**: Use insights to adjust teaching strategies.

## Future Enhancements
- **Audio analysis** for detecting distractions.
- **Advanced ML models** for improved accuracy.
- **Support for multiple students in a classroom.**

## Contributing
Feel free to fork the repository and contribute! Submit pull requests with enhancements or bug fixes.

## License
This project is licensed under the MIT License.

## Contact
For questions or contributions, reach out at [haswanthkurevella1@gmail.com ].
