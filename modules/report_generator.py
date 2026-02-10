import json
from datetime import datetime
from typing import Dict

class ReportGenerator:
    def __init__(self):
        self.report = {}
    
    def generate_full_report(self, user_profile: Dict, interview_summary: Dict, 
                            skill_report: Dict) -> Dict:
        """Generate comprehensive interview report"""
        
        self.report = {
            "report_generated_at": datetime.now().isoformat(),
            "candidate_profile": {
                "name": user_profile.get("name", "Unknown"),
                "email": user_profile.get("email"),
                "identified_skills": skill_report.get("top_skills", []),
                "skill_categories_found": skill_report.get("skill_categories", {})
            },
            "interview_performance": {
                "overall_score": interview_summary["performance"]["overall_score"],
                "rating": interview_summary["performance"]["rating"],
                "content_average": interview_summary["performance"].get("content_average", 0),
                "audio_average": interview_summary["performance"].get("audio_average", 0),
                "video_average": interview_summary["performance"].get("video_average", 0),
                "questions_asked": interview_summary["session_info"]["questions_asked"],
                "interview_duration_minutes": round(interview_summary["session_info"]["duration_seconds"] / 60, 2)
            },
            "detailed_analysis": interview_summary["performance"]["detailed_scores"],
            "recommendations": interview_summary["recommendations"]
        }
        
        return self.report
    
    def export_to_json(self, file_path: str = "final_report.json"):
        """Export report to JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.report, f, indent=2)
            print(f"✓ Report saved to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving JSON report: {e}")
            return None
    
    def export_to_html(self, file_path: str = "final_report.html"):
        """Export report to beautiful HTML file"""
        try:
            perf = self.report.get("interview_performance", {})
            profile = self.report.get("candidate_profile", {})
            
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interview Report - {profile.get('name', 'Candidate')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
        }}
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 14px;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #333;
            font-size: 24px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .score-display {{
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%);
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .score-number {{
            font-size: 64px;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .rating-badge {{
            display: inline-block;
            padding: 10px 30px;
            background: #667eea;
            color: white;
            border-radius: 25px;
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        .meta-info {{
            color: #666;
            font-size: 16px;
        }}
        .score-breakdown {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .score-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .score-card-title {{
            font-size: 14px;
            color: #666;
            margin-bottom: 10px;
        }}
        .score-card-value {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }}
        .info-row:last-child {{
            border-bottom: none;
        }}
        .info-label {{
            font-weight: 600;
            color: #333;
        }}
        .info-value {{
            color: #666;
        }}
        .skills {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }}
        .skill-badge {{
            background: #e7f3ff;
            color: #667eea;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th {{
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #dee2e6;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .recommendations {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            border-radius: 4px;
        }}
        .recommendations h3 {{
            color: #856404;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        .recommendations ul {{
            list-style-position: inside;
            color: #856404;
        }}
        .recommendations li {{
            margin-bottom: 10px;
            line-height: 1.6;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 12px;
            background: #f8f9fa;
        }}
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Interview Report</h1>
            <p>Generated on {self.report.get('report_generated_at', 'N/A')}</p>
        </div>
        
        <div class="content">
            <!-- Score Section -->
            <div class="score-display">
                <div class="score-number">{perf.get('overall_score', 0):.1%}</div>
                <div class="rating-badge">{perf.get('rating', 'N/A')}</div>
                <div class="meta-info">
                    {perf.get('questions_asked', 0)} Questions • {perf.get('interview_duration_minutes', 0)} Minutes
                </div>
                
                <div class="score-breakdown">
                    <div class="score-card">
                        <div class="score-card-title">📝 Content Score</div>
                        <div class="score-card-value">{perf.get('content_average', 0):.0%}</div>
                    </div>
                    <div class="score-card">
                        <div class="score-card-title">🎤 Audio Quality</div>
                        <div class="score-card-value">{perf.get('audio_average', 0):.0%}</div>
                    </div>
                    <div class="score-card">
                        <div class="score-card-title">👤 Video Presence</div>
                        <div class="score-card-value">{perf.get('video_average', 0):.0%}</div>
                    </div>
                </div>
            </div>
            
            <!-- Candidate Profile -->
            <div class="section">
                <h2>👤 Candidate Profile</h2>
                <div class="info-row">
                    <span class="info-label">Name:</span>
                    <span class="info-value">{profile.get('name', 'N/A')}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Email:</span>
                    <span class="info-value">{profile.get('email', 'N/A')}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Skills Identified:</span>
                    <span class="info-value">{len(profile.get('identified_skills', []))}</span>
                </div>
                <div class="skills">
                    {''.join([f'<span class="skill-badge">{skill}</span>' for skill in profile.get('identified_skills', [])[:15]])}
                </div>
            </div>
            
            <!-- Performance Summary -->
            <div class="section">
                <h2>📊 Performance Summary</h2>
                <div class="info-row">
                    <span class="info-label">Overall Score:</span>
                    <span class="info-value">{perf.get('overall_score', 0):.1%}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Rating:</span>
                    <span class="info-value">{perf.get('rating', 'N/A')}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Content Accuracy:</span>
                    <span class="info-value">{perf.get('content_average', 0):.1%}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Presentation Quality:</span>
                    <span class="info-value">{((perf.get('audio_average', 0) + perf.get('video_average', 0)) / 2):.1%}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Questions Answered:</span>
                    <span class="info-value">{perf.get('questions_asked', 0)}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Time Taken:</span>
                    <span class="info-value">{perf.get('interview_duration_minutes', 0)} minutes</span>
                </div>
            </div>
            
            <!-- Detailed Scores -->
            <div class="section">
                <h2>🔍 Detailed Question Analysis</h2>
                <table>
                    <thead>
                        <tr>
                            <th style="width: 40%">Question</th>
                            <th>Category</th>
                            <th>Content</th>
                            <th>Audio</th>
                            <th>Video</th>
                            <th>Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f'''
                        <tr>
                            <td>{item['question'][:60]}{'...' if len(item['question']) > 60 else ''}</td>
                            <td>{item['category']}</td>
                            <td><strong>{item['content_score']['total_score']:.0%}</strong></td>
                            <td><strong>{item['audio_score']:.0%}</strong></td>
                            <td><strong>{item['video_score']:.0%}</strong></td>
                            <td><strong>{item['composite_score']:.0%}</strong></td>
                        </tr>
                        ''' for item in self.report.get('detailed_analysis', [])])}
                    </tbody>
                </table>
            </div>
            
            <!-- Recommendations -->
            <div class="section">
                <div class="recommendations">
                    <h3>💡 Recommendations for Improvement</h3>
                    <ul>
                        {''.join([f'<li>{rec}</li>' for rec in self.report.get('recommendations', [])])}
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by AI Resume Analyzer • Powered by Sentence Transformers, OpenCV & Flask</p>
        </div>
    </div>
</body>
</html>"""
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"✓ HTML report saved to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving HTML report: {e}")
            return None