import re
import json
from typing import Dict, List, Set
import os


class ResumeParser:
    def __init__(self, skills_file: str = "config/skills.json"):
        self.skills_database = self._load_skills(skills_file)
        self.extracted_skills = {}

        # CRITICAL: Define priority skills that MUST be detected and asked about
        self.priority_skills = {
            # Programming Languages (HIGH PRIORITY)
            "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
            "php", "ruby", "swift", "kotlin", "scala", "c", "cpp", "c sharp",

            # Web Frontend (HIGH PRIORITY)
            "react", "angular", "vue.js", "vue", "next.js", "nextjs", "html", "css",
            "bootstrap", "tailwind", "tailwind css", "jquery", "html5", "css3",
            "sass", "scss", "webpack", "vite",

            # Web Backend (HIGH PRIORITY)
            "node.js", "nodejs", "express.js", "express", "django", "flask", "fastapi",
            "spring", "spring boot", "springboot", ".net", "dotnet", "asp.net", "laravel",
            "ruby on rails", "rails", "symfony", "nest.js", "nestjs",

            # Databases (HIGH PRIORITY)
            "sql", "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
            "dynamodb", "cassandra", "oracle", "sql server", "mariadb", "sqlite",
            "firebase", "couchdb", "neo4j", "influxdb",

            # Cloud Platforms (HIGH PRIORITY)
            "aws", "azure", "google cloud", "gcp", "docker", "kubernetes", "k8s",
            "heroku", "digitalocean", "netlify", "vercel", "ibm cloud",

            # DevOps & Tools (HIGH PRIORITY)
            "git", "github", "gitlab", "jenkins", "gitlab ci", "github actions",
            "circleci", "travis ci", "terraform", "ansible", "cloudformation",
            "ci/cd", "cicd", "linux", "unix", "bash", "shell", "powershell",
            "prometheus", "grafana",

            # AI/ML (HIGH PRIORITY)
            "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn",
            "scikit learn", "sklearn", "keras", "xgboost", "lightgbm",
            "llm", "rag", "prompt engineering", "nlp", "computer vision", "openai",
            "hugging face", "langchain", "vector database", "embeddings", "pandas",
            "numpy", "matplotlib", "seaborn",

            # APIs & Architecture
            "rest api", "restful", "graphql", "microservices", "api", "grpc",
            "rabbitmq", "kafka", "message queue", "websocket",

            # Testing
            "jest", "pytest", "mocha", "jasmine", "junit", "selenium", "cypress",
            "playwright", "unittest", "testng"
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_skills(self, file_path: str) -> Dict:
        """Load skills database from JSON"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading skills: {e}")
            return {}

    def _normalize_skill_name(self, skill: str) -> str:
        """Normalize skill name for better matching"""
        normalized = skill.lower().strip()
        normalized = normalized.replace("node.js",      "nodejs")
        normalized = normalized.replace("next.js",      "nextjs")
        normalized = normalized.replace("vue.js",       "vue")
        normalized = normalized.replace("express.js",   "express")
        normalized = normalized.replace("spring boot",  "springboot")
        normalized = normalized.replace("scikit-learn", "scikitlearn")
        normalized = normalized.replace("scikit learn", "scikitlearn")
        normalized = normalized.replace(".net",         "dotnet")
        normalized = normalized.replace("asp.net",      "aspnet")
        normalized = normalized.replace("c#",           "csharp")
        normalized = normalized.replace("c++",          "cpp")
        return normalized

    def _build_fuzzy_pattern(self, skill: str) -> str:
        """
        Build a flexible regex pattern that tolerates different separator
        styles between words (dots, hyphens, underscores, spaces, or none).

        ROOT-CAUSE FIX
        --------------
        The original code passed a raw string literal as the *repl* argument
        of re.sub().  Python's re engine processes backslashes in repl strings,
        so any sequence like backslash-s raises:
            re.error: bad escape at position 5

        The fix is to pass a **lambda** instead.  A callable's return value is
        used verbatim and is never scanned for escape sequences, so the
        character-class string arrives in the output unchanged.

        Examples (approximate):
            'machine learning'  becomes  'machine[sep]*learning'
            'scikit-learn'      becomes  'scikit[sep]*learn'
            'node.js'           becomes  'node[sep]*js'
        where [sep]* matches zero-or-more dots, hyphens, underscores, or spaces.
        """
        escaped = re.escape(skill)
        # Replace every escaped separator (space, dot, hyphen, underscore)
        # with a group that allows zero-or-more separators.
        fuzzy = re.sub(
            r'(\\\\[.\-_ ]|\ )',            # matches escaped dot/hyphen/underscore OR literal space
            lambda _: r'[.\-_ ]*',           # lambda keeps the replacement verbatim
            escaped
        )
        return fuzzy

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF resume"""
        text = ""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except ImportError:
            print("pdfplumber not installed. Install with: pip install pdfplumber")
            return ""
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX resume"""
        try:
            from docx import Document
            doc = Document(docx_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except ImportError:
            print("python-docx not installed. Install with: pip install python-docx")
            return ""
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""

    def extract_text(self, file_path: str) -> str:
        """Extract text from resume (PDF or DOCX)"""
        if file_path.endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self.extract_text_from_docx(file_path)
        else:
            raise ValueError("Unsupported file format. Use PDF or DOCX.")

    # ------------------------------------------------------------------
    # Skill extraction
    # ------------------------------------------------------------------

    def extract_skills(self, resume_text: str) -> Dict[str, List[str]]:
        """
        Extract technical skills from resume text with PRIORITY-BASED detection.
        Ensures important skills are detected with flexible matching.
        """
        resume_text_lower = resume_text.lower()
        found_skills = {}
        priority_skills_found = set()

        # ── First pass: exact / near-exact matching against the DB ──────────
        for category, skills in self.skills_database.items():
            found_skills[category] = []
            for skill in skills:
                skill_lower = skill.lower()

                # Pattern 1: Exact word boundary match
                pattern1 = r'\b' + re.escape(skill_lower) + r'\b'

                # Pattern 2: With dots/dashes escaped (e.g., Node\.js)
                pattern2 = re.escape(skill_lower.replace('.', r'\.'))

                # Pattern 3: Without special chars (e.g., nodejs for node.js)
                skill_no_special = re.sub(r'[.\-_\s]', '', skill_lower)
                pattern3 = r'\b' + re.escape(skill_no_special) + r'\b'

                if (re.search(pattern1, resume_text_lower) or
                        re.search(pattern2, resume_text_lower) or
                        re.search(pattern3, resume_text_lower)):
                    found_skills[category].append(skill)

                    if (skill_lower in self.priority_skills or
                            skill_no_special in self.priority_skills):
                        priority_skills_found.add(skill)

        # ── Second pass: fuzzy matching for priority skills ──────────────────
        # Catches variants not stored verbatim in the database
        # (e.g. "ReactJS", "scikit learn", "Node.js" vs "nodejs").
        for priority_skill in self.priority_skills:
            # Skip if we already found this skill
            if any(
                self._normalize_skill_name(s) == self._normalize_skill_name(priority_skill)
                for s in priority_skills_found
            ):
                continue

            # *** FIX: use _build_fuzzy_pattern() instead of inline re.sub ***
            fuzzy_pattern = self._build_fuzzy_pattern(priority_skill)

            if re.search(fuzzy_pattern, resume_text_lower, re.IGNORECASE):
                skill_added = False

                # Try to match back to an existing DB entry
                for category, skills in self.skills_database.items():
                    for db_skill in skills:
                        if (self._normalize_skill_name(db_skill) ==
                                self._normalize_skill_name(priority_skill)):
                            if db_skill not in found_skills[category]:
                                found_skills[category].append(db_skill)
                                priority_skills_found.add(db_skill)
                                skill_added = True
                                break
                    if skill_added:
                        break

                # If no DB match, create an entry in "other_tools"
                if not skill_added:
                    found_skills.setdefault("other_tools", [])
                    skill_title = priority_skill.title()
                    if skill_title not in found_skills["other_tools"]:
                        found_skills["other_tools"].append(skill_title)
                        priority_skills_found.add(skill_title)

        self.extracted_skills = found_skills

        print(f"\n📊 SKILL EXTRACTION RESULTS:")
        print(f"   Total skills found: {sum(len(v) for v in found_skills.values())}")
        print(f"   Priority skills found: {len(priority_skills_found)}")
        if priority_skills_found:
            print(f"   Priority skills: {', '.join(list(priority_skills_found)[:10])}...")

        return found_skills

    def get_top_skills(self, top_n: int = 20) -> List[str]:
        """
        Get top N skills with PRIORITY-BASED ORDERING.
        Priority skills come first, then others.
        """
        priority_found = []
        other_found = []

        for category, skills in self.extracted_skills.items():
            for skill in skills:
                skill_normalized = self._normalize_skill_name(skill)
                if (skill_normalized in self.priority_skills or
                        skill.lower() in self.priority_skills):
                    if skill not in priority_found:
                        priority_found.append(skill)
                else:
                    if skill not in other_found:
                        other_found.append(skill)

        all_skills = priority_found + other_found

        print(f"\n🎯 TOP SKILLS (Priority-based):")
        print(f"   Priority skills: {len(priority_found)}")
        print(f"   Other skills:    {len(other_found)}")
        if all_skills:
            print(f"   Top {min(top_n, len(all_skills))}: {', '.join(all_skills[:top_n])}")

        return all_skills[:top_n] if all_skills else []

    def extract_contact_info(self, resume_text: str) -> Dict:
        """Extract email and phone from resume"""
        contact_info = {}

        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, resume_text)
        contact_info['email'] = emails[0] if emails else None

        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
        phones = re.findall(phone_pattern, resume_text)
        contact_info['phone'] = phones[0] if phones else None

        return contact_info

    def generate_skill_report(self) -> Dict:
        """Generate detailed skill extraction report"""
        return {
            "extracted_skills": self.extracted_skills,
            "total_skills":     sum(len(v) for v in self.extracted_skills.values()),
            "top_skills":       self.get_top_skills(20),
            "skill_categories": {k: len(v) for k, v in self.extracted_skills.items()}
        }