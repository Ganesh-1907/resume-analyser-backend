# 🐍 Python Backend: AI Interview Logic

This directory contains the core AI engine and media processing modules for the Resume Analyzer.

## 📦 Core Modules

### 1. `resume_parser.py`
- **Role**: Extracts text and skills from resumes.
- **Technology**: `pdfplumber`, `python-docx`, and custom Regex.
- **Logic**: Uses a priority-based skill detection system with fuzzy matching to normalize skill names (e.g., "Node.js" and "node.js" are mapped to the same ID).

### 2. `similarity_matcher.py`
- **Role**: Evaluates the correctness of candidate answers.
- **Models**: `SentenceTransformer` (`all-MiniLM-L6-v2`).
- **Algorithm**:
    - **Semantic Similarity**: Encodes answers into vector space and calculates cosine similarity.
    - **Keyword Coverage**: Checks for specific technical terms required for a "correct" answer.
    - **Length Adequacy**: Ensures responses are substantial enough to be meaningful.

### 3. `audio_video_processor.py`
- **Role**: Handles media files and extracts features.
- **Transcription**: `OpenAI Whisper` (base model).
- **Video Analysis**: `OpenCV` with Haar Cascades for face/eye detection.
- **Audio Analysis**: `Librosa` for SNR (Signal-to-Noise Ratio), energy levels, and volume consistency.

### 4. `interview_engine.py`
- **Role**: The orchestrator of the interview session.
- **Logic**: 
    - **Diverse Skill Coverage**: Randomly selects questions from identified skill categories to ensure a broad assessment.
    - **Scoring Orchestrator**: Aggregates scores from content, audio, and video analysis into a final weighted result.

### 5. `report_generator.py`
- **Role**: Finalizes the data for the user.
- **Output**: Generates both `final_report.json` and a styled `final_report.html`.

## ⚙️ Requirements & Setup

The backend requires several system-level dependencies:
- **FFmpeg**: For audio extraction from video files.
- **Python 3.8+**: Recommended for best compatibility with ML libraries.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Evaluation Weights
- **Content Accuracy**: 60%
- **Audio Clarity**: 20%
- **Video Confidence**: 20%
