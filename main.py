import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from modules.resume_parser import ResumeParser
from modules.audio_video_processor import AudioVideoProcessor
from modules.similarity_matcher import SimilarityMatcher
from modules.interview_engine import InterviewEngine
from modules.report_generator import ReportGenerator
import time
import json
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'mp4', 'webm', 'avi', 'mov', 'wav', 'mp3', 'ogg'}

# Global session storage (in production, use database or Redis)
current_user = None
interview_engine = None
skill_report = None

# PRELOAD MODELS ON STARTUP - NO DELAYS DURING INTERVIEW
print("\n" + "="*70)
print("🚀 PRELOADING AI MODELS (ONE-TIME STARTUP)")
print("="*70)

# Initialize global processors with preloaded models
audio_video_processor = AudioVideoProcessor()
similarity_matcher = SimilarityMatcher()

print("✅ All models loaded and ready!")
print("="*70 + "\n")

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('config', exist_ok=True)

def allowed_file(filename, extensions=None):
    """Check if file has allowed extension"""
    if extensions is None:
        extensions = app.config['ALLOWED_EXTENSIONS']
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
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

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    """Handle resume upload and skill extraction"""
    global current_user, skill_report
    
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    if not allowed_file(file.filename, {'pdf', 'docx'}):
        return jsonify({"success": False, "error": "Only PDF and DOCX files are supported"}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Parse resume
        parser = ResumeParser()
        resume_text = parser.extract_text(filepath)
        
        if not resume_text:
            return jsonify({"success": False, "error": "Could not extract text from resume. Please check file format."}), 400
        
        # Extract skills and contact info
        skills = parser.extract_skills(resume_text)
        contact_info = parser.extract_contact_info(resume_text)
        skill_report = parser.generate_skill_report()
        
        # Store user profile
        current_user = {
            "name": request.form.get('name', 'Candidate'),
            "email": contact_info.get('email'),
            "phone": contact_info.get('phone'),
            "resume_path": filepath
        }
        
        print(f"\n✅ Successfully parsed resume for {current_user['name']}")
        print(f"📊 Found {skill_report['total_skills']} skills across {len(skill_report['skill_categories'])} categories")
        
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
    """Initialize interview session with exactly 10 questions - INSTANT (models preloaded)"""
    global interview_engine, current_user, skill_report
    
    if not current_user or not skill_report:
        return jsonify({"success": False, "error": "Resume not uploaded yet"}), 400
    
    try:
        # FIXED: Increase from 5 to 15 skills for better coverage
        all_skills = skill_report["top_skills"]
        top_skills = all_skills[:15] if len(all_skills) > 15 else all_skills
        
        print(f"\n📋 Selected {len(top_skills)} skills from {len(all_skills)} total skills")
        print(f"🎯 Interview will focus on: {', '.join(top_skills)}")
        
        # Use preloaded similarity_matcher - NO LOADING TIME
        interview_engine = InterviewEngine(top_skills, similarity_matcher=similarity_matcher)
        
        print(f"\n🎯 Starting interview for skills: {', '.join(top_skills[:10])}...")
        print(f"📝 Total questions planned: {interview_engine.total_questions}")
        
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
    """Get the next interview question"""
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
        
        print(f"\n❓ Question {current_num}/{total}: {question['question'][:50]}...")
        
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
    """
    OPTIMIZED: Fast video/audio processing with preloaded models
    """
    global interview_engine
    
    if not interview_engine:
        return jsonify({"success": False, "error": "Interview not started"}), 400
    
    try:
        question_id = request.form.get('question_id')
        if not question_id:
            return jsonify({"success": False, "error": "Question ID required"}), 400
        
        # Get text answer if provided
        user_answer = request.form.get('answer', '').strip()
        
        # Initialize variables
        transcription = ""
        transcription_display = ""
        audio_analysis = {}
        video_analysis = {}
        
        # Handle video file if present
        if 'video_file' in request.files:
            try:
                print("\n🎬 FAST VIDEO PROCESSING (preloaded models)")
                
                video_file = request.files['video_file']
                video_filename = secure_filename(f"response_{question_id}_{int(time.time())}.webm")
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
                video_file.save(video_path)
                
                print(f"✅ Video saved: {video_path}")
                
                # FAST PROCESSING with preloaded models
                results = audio_video_processor.process_interview_response(video_path)
                
                transcription = results.get("transcription", "")
                audio_analysis = results.get("audio_analysis", {})
                video_analysis = results.get("video_analysis", {})
                
                if transcription:
                    user_answer = transcription
                    transcription_display = transcription
                    print(f"✅ TRANSCRIBED: {transcription[:100]}...")
                else:
                    print("⚠️  Transcription empty or failed")
                
                # Cleanup
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                        print(f"🗑️  Cleaned up video file")
                except:
                    pass
                    
            except Exception as e:
                print(f"\n⚠️  Video processing error: {e}")
                import traceback
                traceback.print_exc()
        
        # Handle audio-only file if present
        elif 'audio_file' in request.files:
            try:
                print("\n🎤 FAST AUDIO PROCESSING")
                
                audio_file = request.files['audio_file']
                audio_filename = secure_filename(f"audio_{question_id}_{int(time.time())}.wav")
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
                audio_file.save(audio_path)
                
                # Fast transcription with preloaded model
                transcription = audio_video_processor.transcribe_audio(audio_path)
                if transcription:
                    user_answer = transcription
                    transcription_display = transcription
                    print(f"✅ TRANSCRIBED: {transcription[:200]}...")
                
                # Fast audio analysis
                audio_analysis = audio_video_processor.analyze_audio_quality(audio_path)
                
                # Cleanup
                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                except:
                    pass
                    
            except Exception as e:
                print(f"⚠️  Audio processing error: {e}")
                import traceback
                traceback.print_exc()
        
        # Validate we have an answer
        if not user_answer or len(user_answer.strip()) == 0:
            return jsonify({
                "success": False, 
                "error": "No answer provided. Please record your answer or type a response."
            }), 400
        
        print(f"\n📊 Evaluating answer (length: {len(user_answer)} chars)...")
        print(f"Answer text: {user_answer[:150]}...")
        
        # Evaluate answer - uses preloaded similarity model
        evaluation = interview_engine.submit_answer(
            question_id, 
            user_answer,
            audio_analysis=audio_analysis,
            video_analysis=video_analysis,
            transcription=transcription
        )
        
        # CRITICAL FIX: Convert numpy types to native Python types
        evaluation = convert_to_serializable(evaluation)
        
        print(f"✅ Evaluation complete!")
        print(f"   Content Score: {evaluation.get('content_evaluation', {}).get('total_score', 0):.1%}")
        print(f"   Composite Score: {evaluation.get('composite_score', 0):.1%}")
        
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
    """Get final interview results and generate reports"""
    global interview_engine, current_user, skill_report
    
    if not interview_engine:
        return jsonify({"success": False, "error": "Interview not started"}), 400
    
    try:
        print("\n" + "="*70)
        print("📈 GENERATING FINAL REPORT")
        print("="*70)
        
        # Get interview summary
        interview_summary = interview_engine.get_interview_summary()
        
        # Convert to serializable format
        interview_summary = convert_to_serializable(interview_summary)
        
        # Generate reports
        report_generator = ReportGenerator()
        final_report = report_generator.generate_full_report(
            current_user,
            interview_summary,
            skill_report
        )
        
        # Convert final report
        final_report = convert_to_serializable(final_report)
        
        # Export to files
        report_generator.export_to_json('final_report.json')
        report_generator.export_to_html('final_report.html')
        
        print(f"✅ Final Score: {final_report['interview_performance']['overall_score']:.1%}")
        print(f"⭐ Rating: {final_report['interview_performance']['rating']}")
        print("="*70 + "\n")
        
        return jsonify({
            "success": True,
            "interview_summary": interview_summary,
            "final_report": final_report
        })
    except Exception as e:
        print(f"❌ Error getting summary: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/download-report', methods=['GET'])
def download_report():
    """Download JSON report"""
    try:
        return send_file('final_report.json', 
                        as_attachment=True, 
                        download_name='interview_report.json',
                        mimetype='application/json')
    except Exception as e:
        print(f"❌ Error downloading report: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/final_report.html', methods=['GET'])
def get_html_report():
    """Serve HTML report"""
    try:
        return send_file('final_report.html', mimetype='text/html')
    except Exception as e:
        print(f"❌ Error serving HTML report: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Smart AI Resume Analyzer & Video Interviewer")
    print("="*70)
    print("📱 Server starting at http://localhost:5000")
    print("\n⚡ Key Features:")
    print("   ✅ Resume parsing and skill extraction")
    print("   ✅ Video/Audio recording with FAST transcription")
    print("   ✅ Exactly 10 questions covering all your skills")
    print("   ✅ AI-powered answer evaluation")
    print("   ✅ Models preloaded - NO DELAYS during interview")
    print("   ✅ Comprehensive performance reports")
    print("\n🔥 OPTIMIZATIONS:")
    print("   ⚡ Models loaded at startup (one-time)")
    print("   ⚡ No loading delays between questions")
    print("   ⚡ Fast video processing (5-10 seconds)")
    print("   ⚡ Auto camera permissions on interview start")
    print("   ⚡ Improved skill detection (15 skills vs 5)")
    print("   ⚡ Smart question distribution (1 per skill)")
    print("="*70 + "\n")
    
    # Run the application
    app.run(debug=False)
    