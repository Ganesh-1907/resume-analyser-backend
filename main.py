import sys
import os
import json
import hashlib
import uuid
import time
import numpy as np

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from modules.resume_parser import ResumeParser
from modules.audio_video_processor import AudioVideoProcessor
from modules.similarity_matcher import SimilarityMatcher
from modules.interview_engine import InterviewEngine
from modules.report_generator import ReportGenerator

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'mp4', 'webm', 'avi', 'mov', 'wav', 'mp3', 'ogg'}

# Global session storage (in production, use database or Redis)
current_user = None
interview_engine = None
skill_report = None

# ── DATA STORE (JSON file based) ─────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.json')

def load_data():
    if not os.path.exists(DATA_FILE):
        return {"users": {}}
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ── PRELOAD MODELS ON STARTUP ─────────────────────────────────────────────────
print("\n" + "="*70)
print("🚀 PRELOADING AI MODELS (ONE-TIME STARTUP)")
print("="*70)

audio_video_processor = AudioVideoProcessor()
similarity_matcher = SimilarityMatcher()

print("✅ All models loaded and ready!")
print("="*70 + "\n")

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('config', exist_ok=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def allowed_file(filename, extensions=None):
    if extensions is None:
        extensions = app.config['ALLOWED_EXTENSIONS']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

# ── AUTH ENDPOINTS ────────────────────────────────────────────────────────────
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not name or not email or not password:
        return jsonify({"success": False, "error": "Name, email and password are required"}), 400

    db = load_data()
    if email in db['users']:
        return jsonify({"success": False, "error": "Email already registered"}), 409

    user_id = str(uuid.uuid4())
    token = str(uuid.uuid4())

    db['users'][email] = {
        "id": user_id,
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
        "token": token,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reports": []
    }
    save_data(db)

    return jsonify({
        "success": True,
        "token": token,
        "user": {"id": user_id, "name": name, "email": email}
    })

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password required"}), 400

    db = load_data()
    user = db['users'].get(email)

    if not user or user['password_hash'] != hash_password(password):
        return jsonify({"success": False, "error": "Invalid email or password"}), 401

    # Refresh token
    token = str(uuid.uuid4())
    db['users'][email]['token'] = token
    save_data(db)

    return jsonify({
        "success": True,
        "token": token,
        "user": {"id": user['id'], "name": user['name'], "email": user['email']}
    })

def get_user_from_token(token):
    """Look up user by their auth token"""
    db = load_data()
    for user in db['users'].values():
        if user.get('token') == token:
            return user
    return None

@app.route('/api/profile', methods=['GET'])
def get_profile():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = get_user_from_token(token)
    if not user:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    return jsonify({
        "success": True,
        "user": {
            "id": user['id'],
            "name": user['name'],
            "email": user['email'],
            "created_at": user.get('created_at', ''),
            "reports": user.get('reports', [])
        }
    })

@app.route('/api/profile', methods=['PUT'])
def update_profile():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = get_user_from_token(token)
    if not user:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    data = request.get_json()
    db = load_data()
    email = user['email']

    if data.get('name'):
        db['users'][email]['name'] = data['name'].strip()

    save_data(db)
    return jsonify({"success": True, "message": "Profile updated"})

@app.route('/api/save-report', methods=['POST'])
def save_report():
    """Save interview report to user's profile"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user = get_user_from_token(token)
    if not user:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400

    db = load_data()
    email = user['email']

    report_entry = {
        "id": str(uuid.uuid4()),
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "interview_type": data.get('interview_type', 'audio'),
        "overall_score": data.get('overall_score', 0),
        "rating": data.get('rating', ''),
        "total_questions": data.get('total_questions', 0),
        "resume_name": data.get('resume_name', ''),
        "skills": data.get('skills', []),
        "recommendations": data.get('recommendations', []),
        "detailed_analysis": data.get('detailed_analysis', [])
    }

    if 'reports' not in db['users'][email]:
        db['users'][email]['reports'] = []

    db['users'][email]['reports'].insert(0, report_entry)
    # Keep only latest 20 reports
    db['users'][email]['reports'] = db['users'][email]['reports'][:20]

    save_data(db)
    return jsonify({"success": True, "report_id": report_entry['id']})

# ── EXISTING ENDPOINTS ────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    global current_user, skill_report

    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not allowed_file(file.filename, {'pdf', 'docx'}):
        return jsonify({"success": False, "error": "Only PDF and DOCX files are supported"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        parser = ResumeParser()
        resume_text = parser.extract_text(filepath)

        if not resume_text:
            return jsonify({"success": False, "error": "Could not extract text from resume. Please check file format."}), 400

        skills = parser.extract_skills(resume_text)
        contact_info = parser.extract_contact_info(resume_text)
        skill_report = parser.generate_skill_report()

        current_user = {
            "name": request.form.get('name', 'Candidate'),
            "email": contact_info.get('email'),
            "phone": contact_info.get('phone'),
            "resume_path": filepath,
            "resume_filename": filename
        }

        print(f"\n✅ Successfully parsed resume for {current_user['name']}")
        print(f"📊 Found {skill_report['total_skills']} skills")

        return jsonify({
            "success": True,
            "user_profile": current_user,
            "skills_extracted": skills,
            "top_skills": skill_report["top_skills"],
            "total_skills": skill_report["total_skills"]
        })

    except Exception as e:
        print(f"❌ Error in upload_resume: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/start-interview', methods=['POST'])
def start_interview():
    global interview_engine, current_user, skill_report

    if not current_user or not skill_report:
        return jsonify({"success": False, "error": "Resume not uploaded yet"}), 400

    try:
        all_skills = skill_report["top_skills"]
        top_skills = all_skills[:15] if len(all_skills) > 15 else all_skills

        print(f"\n📋 Selected {len(top_skills)} skills for interview")
        
        interview_type = request.json.get('interview_type', 'audio') if request.is_json else 'audio'
        interview_engine = InterviewEngine(top_skills, similarity_matcher=similarity_matcher, interview_type=interview_type)

        return jsonify({
            "success": True,
            "message": "Interview started successfully",
            "total_questions": interview_engine.total_questions,
            "skills_for_interview": top_skills
        })
    except Exception as e:
        print(f"❌ Error starting interview: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/get-next-question', methods=['GET'])
def get_next_question():
    global interview_engine

    if not interview_engine:
        return jsonify({"success": False, "error": "Interview not started"}), 400

    try:
        question = interview_engine.get_next_question()

        if not question:
            return jsonify({
                "success": False,
                "message": "All questions completed",
                "interview_complete": True
            })

        current_num = len(interview_engine.asked_questions)
        total = interview_engine.total_questions

        return jsonify({
            "success": True,
            "question_id": question["id"],
            "question": question["question"],
            "category": question.get("category", "General"),
            "difficulty": question.get("difficulty", "medium"),
            "question_number": current_num,
            "total_questions": total
        })
    except Exception as e:
        print(f"❌ Error getting question: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/submit-answer', methods=['POST'])
def submit_answer():
    global interview_engine

    if not interview_engine:
        return jsonify({"success": False, "error": "Interview not started"}), 400

    try:
        question_id = request.form.get('question_id')
        if not question_id:
            return jsonify({"success": False, "error": "Question ID required"}), 400

        user_answer = request.form.get('answer', '').strip()

        transcription = ""
        transcription_display = ""
        audio_analysis = {}
        video_analysis = {}

        if 'video_file' in request.files:
            try:
                print("\n🎬 FAST VIDEO PROCESSING (preloaded models)")
                video_file = request.files['video_file']
                video_filename = secure_filename(f"response_{question_id}_{int(time.time())}.webm")
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
                video_file.save(video_path)

                results = audio_video_processor.process_interview_response(video_path)
                transcription = results.get("transcription", "")
                audio_analysis = results.get("audio_analysis", {})
                video_analysis = results.get("video_analysis", {})

                if transcription:
                    user_answer = transcription
                    transcription_display = transcription

                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                except:
                    pass

            except Exception as e:
                print(f"\n⚠️  Video processing error: {e}")
                import traceback
                traceback.print_exc()

        elif 'audio_file' in request.files:
            try:
                print("\n🎤 FAST AUDIO PROCESSING")
                audio_file = request.files['audio_file']
                audio_filename = secure_filename(f"audio_{question_id}_{int(time.time())}.wav")
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
                audio_file.save(audio_path)

                transcription = audio_video_processor.transcribe_audio(audio_path)
                if transcription:
                    user_answer = transcription
                    transcription_display = transcription

                audio_analysis = audio_video_processor.analyze_audio_quality(audio_path)

                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                except:
                    pass

            except Exception as e:
                print(f"⚠️  Audio processing error: {e}")
                import traceback
                traceback.print_exc()

        if not user_answer or len(user_answer.strip()) == 0:
            # If no answer provided (e.g. timeout or processing failure), use a placeholder
            # so the interview can continue to the next question.
            if 'video_file' in request.files or 'audio_file' in request.files:
                user_answer = "[Transcription unavailable - please ensure FFmpeg is installed on server]"
            else:
                user_answer = "[No answer provided before timeout]"
            
            transcription_display = "No answer recorded"

        evaluation = interview_engine.submit_answer(
            question_id,
            user_answer,
            audio_analysis=audio_analysis,
            video_analysis=video_analysis,
            transcription=transcription
        )

        evaluation = convert_to_serializable(evaluation)

        return jsonify({
            "success": True,
            "evaluation": evaluation,
            "transcription": transcription_display,
            "message": "Answer evaluated successfully",
            "answer_used": user_answer[:100] + "..." if len(user_answer) > 100 else user_answer
        })

    except Exception as e:
        print(f"\n❌ Error submitting answer: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/get-interview-summary', methods=['GET'])
def get_interview_summary():
    global interview_engine, current_user, skill_report

    if not interview_engine:
        return jsonify({"success": False, "error": "Interview not started"}), 400

    try:
        interview_summary = interview_engine.get_interview_summary()
        interview_summary = convert_to_serializable(interview_summary)

        report_generator = ReportGenerator()
        final_report = report_generator.generate_full_report(
            current_user,
            interview_summary,
            skill_report
        )

        final_report = convert_to_serializable(final_report)

        report_generator.export_to_json('final_report.json')
        report_generator.export_to_html('final_report.html')

        return jsonify({
            "success": True,
            "interview_summary": interview_summary,
            "final_report": final_report,
            "resume_filename": current_user.get('resume_filename', ''),
            "interview_type": interview_summary.get('session_info', {}).get('interview_type', 'audio')
        })
    except Exception as e:
        print(f"❌ Error getting summary: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/download-report', methods=['GET'])
def download_report():
    try:
        return send_file('final_report.json',
                        as_attachment=True,
                        download_name='interview_report.json',
                        mimetype='application/json')
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/final_report.html', methods=['GET'])
def get_html_report():
    try:
        return send_file('final_report.html', mimetype='text/html')
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Smart AI Resume Analyzer & Video Interviewer")
    print("="*70)
    print("📱 Server starting at http://localhost:5000")
    print("="*70 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5000)