# 🤖 AI Resume Analyzer & Video Interviewer

An intelligent web application that analyzes resumes, extracts technical skills, and conducts AI-powered video interviews with real-time transcription.

## ✨ Features

- 📄 **Resume Parsing**: Upload PDF/DOCX resumes and automatically extract technical skills
- 🎯 **Smart Skill Detection**: AI-powered skill extraction covering 100+ technologies
- 📹 **Video Interviews**: Record video answers with camera and microphone
- 🎤 **Real-Time Transcription**: Speech-to-text conversion using OpenAI Whisper
- 🧠 **Semantic Evaluation**: Answer evaluation using sentence transformers
- 📊 **Comprehensive Reports**: Detailed performance analysis with scores and recommendations
- 🎨 **Modern UI**: Clean, responsive interface with real-time feedback

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **AI/ML**: 
  - Sentence Transformers (semantic similarity)
  - OpenAI Whisper (speech transcription)
  - OpenCV (video processing)
  - Librosa (audio analysis)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Resume Parsing**: pdfplumber, python-docx
- **Deployment**: Render / Railway (cloud platforms)

## 📋 Prerequisites

- Python 3.10+
- FFmpeg (for audio/video processing)
- Webcam and microphone (for video interviews)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-resume-analyzer.git
cd ai-resume-analyzer
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

**Windows:**
- Download from: https://ffmpeg.org/download.html
- Add to PATH

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

## 🎮 Usage

### 1. Run the application
```bash
python main.py
```

### 2. Open browser
Navigate to: `http://localhost:5000`

### 3. Upload resume
- Enter your name
- Upload PDF or DOCX resume
- Skills are automatically extracted

### 4. Complete interview
- Answer 10 technical questions
- Choose video or text mode
- Get instant feedback

### 5. View results
- Download JSON/HTML reports
- Review detailed scores
- See recommendations

## 📁 Project Structure

```
ai-resume-analyzer/
├── config/
│   ├── qa_database.json      # Interview questions database
│   └── skills.json            # Technical skills database
├── models/
│   ├── interview_session.py   # Session management
│   └── user_profile.py        # User data models
├── modules/
│   ├── audio_processor.py     # Audio transcription
│   ├── interview_engine.py    # Interview logic
│   ├── report_generator.py    # PDF/HTML reports
│   ├── resume_parser.py       # Resume text extraction
│   └── similarity_matcher.py  # AI answer evaluation
├── templates/
│   └── index.html             # Frontend UI
├── main.py                    # Flask application
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎯 How It Works

1. **Resume Upload**: User uploads PDF/DOCX resume
2. **Skill Extraction**: AI extracts technical skills using pattern matching
3. **Question Generation**: System selects 10 relevant questions based on skills
4. **Video Interview**: User records video answers (or types text)
5. **Transcription**: Whisper AI transcribes spoken answers to text
6. **Evaluation**: Semantic similarity scoring using sentence transformers
7. **Report Generation**: Comprehensive HTML/JSON reports with scores

## 📊 Scoring System

- **Content Score (60%)**: Semantic similarity + keyword coverage
- **Audio Quality (20%)**: Duration and clarity
- **Video Presence (20%)**: Face detection and engagement

**Final Rating:**
- 90-100%: Excellent
- 80-89%: Very Good
- 70-79%: Good
- 60-69%: Fair
- <60%: Needs Improvement

## 🔧 Configuration

Edit `config/skills.json` to add new skills:
```json
{
  "programming_languages": ["Python", "Java", "JavaScript"],
  "web_frameworks": ["Django", "Flask", "React"]
}
```

Edit `config/qa_database.json` to add questions:
```json
{
  "python": [
    {
      "id": "py001",
      "question": "Explain list comprehension in Python",
      "difficulty": "medium",
      "expected_answer": "...",
      "keywords": ["list", "comprehension", "syntax"]
    }
  ]
}
```

## 🌐 Deployment

### Deploy to Render (Free)

1. Push to GitHub
2. Sign up at [render.com](https://render.com)
3. Create new Web Service
4. Connect GitHub repo
5. Settings:
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn main:app --bind 0.0.0.0:$PORT --timeout 120`
6. Deploy!

### Deploy to Railway (Free)

1. Push to GitHub
2. Sign up at [railway.app](https://railway.app)
3. New Project → Deploy from GitHub
4. Select repository
5. Done! Auto-configured

## 🐛 Troubleshooting

**Camera not working:**
- Ensure HTTPS is enabled (required for camera access)
- Check browser permissions
- Try different browser (Chrome recommended)

**Transcription timeout:**
- Reduce video length to <2 minutes
- Increase timeout in deployment settings
- Use text mode as alternative

**Skill extraction issues:**
- Ensure resume has clear formatting
- Add custom skills to `config/skills.json`
- Use PDF format for best results

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- Sentence Transformers for semantic analysis
- Flask community for excellent documentation
- All open-source contributors

## 📧 Contact

For questions or feedback, reach out at: your.email@example.com

---

⭐ **Star this repo if you found it helpful!** ⭐