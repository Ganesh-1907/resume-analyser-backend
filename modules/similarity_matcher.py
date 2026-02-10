import json
from typing import Dict, List
import re

class SimilarityMatcher:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.qa_database = self._load_qa_database()
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer, util
            self.model = SentenceTransformer(self.model_name)
            self.util = util
            print("✓ AI model loaded successfully")
        except ImportError:
            print("⚠ sentence-transformers not installed. Using fallback similarity.")
            self.model = None
        except Exception as e:
            print(f"⚠ Error loading model: {e}. Using fallback similarity.")
            self.model = None
    
    def _load_qa_database(self) -> Dict:
        """Load QA database"""
        try:
            with open("config/qa_database.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading QA database: {e}")
            return {}
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts (0-1)"""
        if not self.model:
            # Fallback to simple word overlap similarity
            return self._simple_similarity(text1, text2)
        
        try:
            embeddings1 = self.model.encode(text1, convert_to_tensor=True)
            embeddings2 = self.model.encode(text2, convert_to_tensor=True)
            similarity = self.util.pytorch_cos_sim(embeddings1, embeddings2).item()
            return max(0, min(1, similarity))
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return self._simple_similarity(text1, text2)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Fallback simple similarity based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    def keyword_coverage(self, user_answer: str, expected_keywords: List[str]) -> float:
        """Calculate keyword coverage score"""
        if not expected_keywords:
            return 1.0
            
        user_answer_lower = user_answer.lower()
        matched_keywords = 0
        
        for keyword in expected_keywords:
            keyword_lower = keyword.lower()
            # Replace underscores with spaces for matching
            pattern = r'\b' + re.escape(keyword_lower.replace('_', ' ')) + r'\b'
            if re.search(pattern, user_answer_lower):
                matched_keywords += 1
        
        return matched_keywords / len(expected_keywords)
    
    def evaluate_answer(self, user_answer: str, expected_answer: str, 
                       expected_keywords: List[str]) -> Dict:
        """Evaluate user answer using semantic similarity + keywords + length"""
        
        # Semantic similarity (60% weight)
        semantic_score = self.semantic_similarity(user_answer, expected_answer)
        
        # Keyword coverage (30% weight)
        keyword_score = self.keyword_coverage(user_answer, expected_keywords)
        
        # Length adequacy (10% weight)
        answer_length = len(user_answer.split())
        length_score = min(1.0, answer_length / 50)  # Optimal around 50 words
        
        # Calculate total score
        total_score = (semantic_score * 0.6 + 
                      keyword_score * 0.3 + 
                      length_score * 0.1)
        
        return {
            "semantic_score": round(semantic_score, 3),
            "keyword_score": round(keyword_score, 3),
            "length_score": round(length_score, 3),
            "total_score": round(total_score, 3),
            "keywords_found": self._extract_matched_keywords(user_answer, expected_keywords)
        }
    
    def _extract_matched_keywords(self, user_answer: str, expected_keywords: List[str]) -> List[str]:
        """Extract matched keywords from user answer"""
        user_answer_lower = user_answer.lower()
        matched = []
        
        for keyword in expected_keywords:
            keyword_lower = keyword.lower()
            pattern = r'\b' + re.escape(keyword_lower.replace('_', ' ')) + r'\b'
            if re.search(pattern, user_answer_lower):
                matched.append(keyword)
        
        return matched
    
    def get_relevant_questions(self, skills: List[str], difficulty: str = "medium") -> List[Dict]:
        """Get questions relevant to user skills"""
        relevant_questions = []
        
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower in self.qa_database:
                questions = self.qa_database[skill_lower]
                # Filter by difficulty if specified
                filtered = [q for q in questions if q.get("difficulty") == difficulty]
                relevant_questions.extend(filtered)
        
        # Remove duplicates based on question ID
        unique_questions = {q["id"]: q for q in relevant_questions}
        return list(unique_questions.values())