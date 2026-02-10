import json
import random
from typing import Dict, List, Set
from datetime import datetime
from modules.similarity_matcher import SimilarityMatcher

class InterviewEngine:
    def __init__(self, user_skills: List[str], similarity_matcher: SimilarityMatcher = None):
        self.user_skills = [skill.lower() for skill in user_skills]
        self.matcher = similarity_matcher if similarity_matcher else SimilarityMatcher()
        self.qa_database = self._load_qa_database()
        self.interview_session = {
            "start_time": datetime.now().isoformat(),
            "questions": [],
            "answers": [],
            "transcriptions": [],
            "audio_scores": [],
            "video_scores": [],
            "content_scores": [],
            "total_score": 0
        }
        self.asked_questions = set()
        self.total_questions = 10  # Fixed limit
        self.questions_per_skill = {}
        self.skill_coverage = {}  # Track which skills have been asked
    
    def _load_qa_database(self) -> Dict:
        """Load QA database"""
        try:
            with open("config/qa_database.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading QA database: {e}")
            return {}
    
    def _normalize_skill(self, skill: str) -> str:
        """Normalize skill name for matching"""
        normalized = skill.lower().strip()
        # Handle common variations
        normalized = normalized.replace("node.js", "nodejs")
        normalized = normalized.replace("next.js", "nextjs")
        normalized = normalized.replace("vue.js", "vue")
        normalized = normalized.replace("express.js", "express")
        normalized = normalized.replace("spring boot", "springboot")
        normalized = normalized.replace(".net", "dotnet")
        normalized = normalized.replace("asp.net", "aspnet")
        normalized = normalized.replace("c#", "csharp")
        normalized = normalized.replace("c++", "cpp")
        normalized = normalized.replace("scikit-learn", "scikitlearn")
        normalized = normalized.replace("scikit learn", "scikitlearn")
        return normalized
    
    def _find_matching_db_skill(self, user_skill: str) -> str:
        """Find matching skill in QA database with fuzzy matching"""
        user_skill_norm = self._normalize_skill(user_skill)
        
        # Direct match
        if user_skill_norm in self.qa_database:
            return user_skill_norm
        
        # Fuzzy matching - check if user skill is substring of DB skill or vice versa
        for db_skill in self.qa_database.keys():
            db_skill_norm = self._normalize_skill(db_skill)
            
            # Check if user skill is part of DB skill (e.g., "react" in "react.js")
            if user_skill_norm in db_skill_norm or db_skill_norm in user_skill_norm:
                return db_skill
            
            # Check without special characters
            user_clean = user_skill_norm.replace('.', '').replace('-', '').replace('_', '').replace(' ', '')
            db_clean = db_skill_norm.replace('.', '').replace('-', '').replace('_', '').replace(' ', '')
            
            if user_clean == db_clean:
                return db_skill
        
        return None
    
    def _distribute_questions_across_skills(self) -> Dict[str, int]:
        """
        SMART DISTRIBUTION: Ensure 1 question per important skill, then distribute remaining.
        Goal: If we have 12 important skills and 10 questions, ask 10 different skills (1 each).
        """
        if not self.user_skills:
            return {}
        
        # Find which user skills have questions available in DB
        available_skill_mappings = {}  # user_skill -> db_skill
        
        for user_skill in self.user_skills:
            db_skill = self._find_matching_db_skill(user_skill)
            if db_skill and len(self.qa_database[db_skill]) > 0:
                available_skill_mappings[user_skill] = db_skill
        
        print(f"\n📊 SKILL-TO-QUESTION MAPPING:")
        print(f"   User skills provided: {len(self.user_skills)}")
        print(f"   Skills with questions: {len(available_skill_mappings)}")
        
        if not available_skill_mappings:
            print("⚠️  No matching skills found. Using all available questions.")
            # Fallback: use all available skills in database
            available_skills = [skill for skill, questions in self.qa_database.items() 
                              if len(questions) > 0]
            distribution = {}
            for i, skill in enumerate(available_skills[:self.total_questions]):
                distribution[skill] = 1
            return distribution
        
        # STRATEGY: 1 question per skill, prioritize diversity
        distribution = {}
        available_skills = list(available_skill_mappings.values())
        
        # If we have MORE skills than questions (e.g., 15 skills, 10 questions)
        # Select top 10 skills
        if len(available_skills) >= self.total_questions:
            selected_skills = available_skills[:self.total_questions]
            for skill in selected_skills:
                distribution[skill] = 1
            print(f"   📌 Strategy: 1 question each from {self.total_questions} different skills")
        
        # If we have FEWER skills than questions (e.g., 6 skills, 10 questions)
        # Give 1 question to each skill, then distribute remaining
        else:
            # First, give 1 question to each skill
            for skill in available_skills:
                distribution[skill] = 1
            
            remaining = self.total_questions - len(available_skills)
            
            # Distribute remaining questions evenly
            for i in range(remaining):
                skill_to_add = available_skills[i % len(available_skills)]
                distribution[skill_to_add] += 1
            
            print(f"   📌 Strategy: Each skill gets ≥1 question, {remaining} extra distributed")
        
        print(f"   📋 Distribution: {distribution}")
        
        # Initialize skill coverage tracker
        for skill in distribution.keys():
            self.skill_coverage[skill] = 0
        
        return distribution
    
    def get_next_question(self) -> Dict:
        """
        Get next interview question ensuring DIVERSE SKILL COVERAGE.
        Priority: Ask from skills that haven't been covered yet.
        """
        # Check if we've reached the limit
        if len(self.asked_questions) >= self.total_questions:
            return None
        
        # First call - plan the distribution
        if not self.questions_per_skill:
            self.questions_per_skill = self._distribute_questions_across_skills()
        
        selected_question = None
        
        # PRIORITY 1: Get questions from skills that haven't been asked yet
        uncovered_skills = [skill for skill, count in self.skill_coverage.items() if count == 0]
        
        if uncovered_skills:
            # Randomly select from uncovered skills
            skill = random.choice(uncovered_skills)
            skill_questions = self.qa_database.get(skill, [])
            available_questions = [q for q in skill_questions if q["id"] not in self.asked_questions]
            
            if available_questions:
                selected_question = random.choice(available_questions)
                self.skill_coverage[skill] += 1
                print(f"   ✓ Selected from UNCOVERED skill: {skill}")
        
        # PRIORITY 2: Get from skills that still need more questions
        if not selected_question:
            for skill, count_needed in self.questions_per_skill.items():
                current_count = self.skill_coverage.get(skill, 0)
                if current_count < count_needed:
                    skill_questions = self.qa_database.get(skill, [])
                    available_questions = [q for q in skill_questions if q["id"] not in self.asked_questions]
                    
                    if available_questions:
                        selected_question = random.choice(available_questions)
                        self.skill_coverage[skill] = current_count + 1
                        print(f"   ✓ Selected from skill needing more: {skill}")
                        break
        
        # PRIORITY 3: Get from any user skill
        if not selected_question:
            for user_skill in self.user_skills:
                db_skill = self._find_matching_db_skill(user_skill)
                if db_skill:
                    skill_questions = self.qa_database.get(db_skill, [])
                    available_questions = [q for q in skill_questions if q["id"] not in self.asked_questions]
                    
                    if available_questions:
                        selected_question = random.choice(available_questions)
                        print(f"   ✓ Selected from user skill: {db_skill}")
                        break
        
        # PRIORITY 4: Get from any available question
        if not selected_question:
            all_questions = []
            for skill_questions in self.qa_database.values():
                all_questions.extend(skill_questions)
            available_questions = [q for q in all_questions if q["id"] not in self.asked_questions]
            
            if available_questions:
                selected_question = random.choice(available_questions)
                print(f"   ✓ Selected from fallback pool")
        
        if selected_question:
            self.asked_questions.add(selected_question["id"])
            return selected_question
        
        return None
    
    def submit_answer(self, question_id: str, user_answer: str, 
                     audio_analysis: Dict = None, video_analysis: Dict = None,
                     transcription: str = None) -> Dict:
        """Process and evaluate user's answer with audio/video analysis"""
        # Find the question in database
        question = None
        for skill_questions in self.qa_database.values():
            for q in skill_questions:
                if q["id"] == question_id:
                    question = q
                    break
        
        if not question:
            return {"error": "Question not found"}
        
        # Use transcription if available, otherwise use text answer
        answer_to_evaluate = transcription if transcription else user_answer
        
        # Get expected answer and keywords
        expected_answer = question.get("expected_answer", "")
        expected_keywords = question.get("expected_keywords", [])
        
        # Evaluate the content
        content_evaluation = self.matcher.evaluate_answer(
            answer_to_evaluate,
            expected_answer,
            expected_keywords
        )
        
        # Calculate presentation scores
        audio_score = 0
        video_score = 0
        
        if audio_analysis:
            # Audio contributes 20% of presentation score
            audio_score = audio_analysis.get("quality_score", 0) / 100
        
        if video_analysis:
            # Video contributes 20% of presentation score
            video_score = video_analysis.get("confidence_score", 0) / 100
        
        # Calculate composite score
        # Content: 60%, Audio: 20%, Video: 20%
        content_score = content_evaluation["total_score"]
        composite_score = (content_score * 0.6) + (audio_score * 0.2) + (video_score * 0.2)
        
        evaluation = {
            "content_evaluation": content_evaluation,
            "audio_score": round(audio_score, 3),
            "video_score": round(video_score, 3),
            "composite_score": round(composite_score, 3),
            "presentation_quality": self._rate_presentation(audio_score, video_score)
        }
        
        # Store in session
        self.interview_session["questions"].append(question)
        self.interview_session["answers"].append(user_answer)
        self.interview_session["transcriptions"].append(transcription or "")
        self.interview_session["audio_scores"].append(audio_score)
        self.interview_session["video_scores"].append(video_score)
        self.interview_session["content_scores"].append(content_evaluation)
        
        # Update total score
        self._update_total_score()
        
        return evaluation
    
    def _rate_presentation(self, audio_score: float, video_score: float) -> str:
        """Rate presentation quality"""
        avg_presentation = (audio_score + video_score) / 2
        
        if avg_presentation >= 0.85:
            return "Excellent - Clear and confident"
        elif avg_presentation >= 0.70:
            return "Very Good - Professional delivery"
        elif avg_presentation >= 0.55:
            return "Good - Adequate presentation"
        elif avg_presentation >= 0.40:
            return "Fair - Room for improvement"
        else:
            return "Needs Improvement - Practice recommended"
    
    def _update_total_score(self):
        """Calculate and update total composite score"""
        if self.interview_session["content_scores"]:
            # Calculate average content score
            content_avg = sum(s["total_score"] for s in self.interview_session["content_scores"]) / len(self.interview_session["content_scores"])
            
            # Calculate average presentation scores
            audio_avg = sum(self.interview_session["audio_scores"]) / len(self.interview_session["audio_scores"])
            video_avg = sum(self.interview_session["video_scores"]) / len(self.interview_session["video_scores"])
            
            # Composite: 60% content, 20% audio, 20% video
            total = (content_avg * 0.6) + (audio_avg * 0.2) + (video_avg * 0.2)
            self.interview_session["total_score"] = round(total, 3)
    
    def get_interview_summary(self) -> Dict:
        """Get complete interview summary with performance analysis"""
        return {
            "session_info": {
                "start_time": self.interview_session["start_time"],
                "questions_asked": len(self.interview_session["questions"]),
                "duration_seconds": (datetime.now() - datetime.fromisoformat(self.interview_session["start_time"])).total_seconds()
            },
            "performance": {
                "overall_score": self.interview_session["total_score"],
                "rating": self._calculate_rating(self.interview_session["total_score"]),
                "content_average": round(sum(s["total_score"] for s in self.interview_session["content_scores"]) / len(self.interview_session["content_scores"]), 3) if self.interview_session["content_scores"] else 0,
                "audio_average": round(sum(self.interview_session["audio_scores"]) / len(self.interview_session["audio_scores"]), 3) if self.interview_session["audio_scores"] else 0,
                "video_average": round(sum(self.interview_session["video_scores"]) / len(self.interview_session["video_scores"]), 3) if self.interview_session["video_scores"] else 0,
                "detailed_scores": [
                    {
                        "question": q["question"],
                        "category": q.get("category", "general"),
                        "difficulty": q.get("difficulty", "medium"),
                        "content_score": self.interview_session["content_scores"][i],
                        "audio_score": self.interview_session["audio_scores"][i],
                        "video_score": self.interview_session["video_scores"][i],
                        "composite_score": (self.interview_session["content_scores"][i]["total_score"] * 0.6 + 
                                          self.interview_session["audio_scores"][i] * 0.2 + 
                                          self.interview_session["video_scores"][i] * 0.2),
                        "user_answer": self.interview_session["answers"][i],
                        "transcription": self.interview_session["transcriptions"][i]
                    }
                    for i, q in enumerate(self.interview_session["questions"])
                ]
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _calculate_rating(self, score: float) -> str:
        """Convert numerical score to rating"""
        if score >= 0.85:
            return "Excellent"
        elif score >= 0.70:
            return "Very Good"
        elif score >= 0.55:
            return "Good"
        elif score >= 0.40:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate personalized recommendations based on performance"""
        recommendations = []
        
        # Content recommendations
        content_avg = sum(s["total_score"] for s in self.interview_session["content_scores"]) / len(self.interview_session["content_scores"]) if self.interview_session["content_scores"] else 0
        
        if content_avg < 0.5:
            recommendations.append("💡 Focus on reviewing fundamental concepts in your skill areas")
            recommendations.append("📚 Practice explaining technical concepts with more detail and keywords")
        elif content_avg < 0.7:
            recommendations.append("💡 Work on depth of knowledge and include more technical details")
            recommendations.append("📚 Use industry-standard terminology in your explanations")
        else:
            recommendations.append("🌟 Excellent technical knowledge! Keep building on this strong foundation")
        
        # Audio/Presentation recommendations
        audio_avg = sum(self.interview_session["audio_scores"]) / len(self.interview_session["audio_scores"]) if self.interview_session["audio_scores"] else 0
        
        if audio_avg < 0.5:
            recommendations.append("🎤 Improve audio quality: speak clearly and at moderate pace")
            recommendations.append("🔊 Practice answering in a quiet environment with good microphone")
        elif audio_avg < 0.7:
            recommendations.append("🎤 Good audio quality, work on speaking pace and clarity")
        
        # Video/Confidence recommendations
        video_avg = sum(self.interview_session["video_scores"]) / len(self.interview_session["video_scores"]) if self.interview_session["video_scores"] else 0
        
        if video_avg < 0.5:
            recommendations.append("👤 Improve eye contact and maintain good posture during interviews")
            recommendations.append("📹 Practice on camera to build confidence and professional presence")
        elif video_avg < 0.7:
            recommendations.append("👤 Good presence! Work on maintaining consistent eye contact")
        
        # Analyze weak categories
        if self.interview_session["content_scores"]:
            category_scores = {}
            for i, q in enumerate(self.interview_session["questions"]):
                category = q.get("category", "general")
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(self.interview_session["content_scores"][i]["total_score"])
            
            for category, scores in category_scores.items():
                avg_cat_score = sum(scores) / len(scores)
                if avg_cat_score < 0.6:
                    recommendations.append(f"📖 Review {category} concepts to strengthen your understanding")
        
        return recommendations[:6]  # Limit to 6 recommendations