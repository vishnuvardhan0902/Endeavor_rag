from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import re
import random
import hashlib
import time
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import os
import tempfile
import json
import numpy as np
from pydantic import BaseModel, Field, ValidationError, root_validator
from typing import Literal, Optional

# Load environment variables on import so ADC / API keys are available
load_dotenv()

# --- Connect to MongoDB ---
MONGO_URI = "mongodb+srv://vishnuvardhan0902:M%23Vishnu%400902@cluster0.btgym.mongodb.net/Endeavor?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client["Endeavor"]
collection = db["ragCollection"]

# Initialize embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Session-based randomization ---
class InterviewSession:
    def __init__(self, resume_path: str):
        # Create unique session ID based on resume path + timestamp
        session_data = f"{resume_path}_{int(time.time())}"
        self.session_id = hashlib.md5(session_data.encode()).hexdigest()[:8]
        
        # Set random seed based on session for reproducible randomness within session
        # but different across sessions
        self.random_seed = int(self.session_id, 16) % 2147483647
        random.seed(self.random_seed)
        np.random.seed(self.random_seed % 4294967295)
        
        # print(f"🎯 New Interview Session: {self.session_id}")
        # print(f"🎲 Random seed: {self.random_seed}")
        
        # Track used questions to avoid repetition within session
        self.used_questions = set()
        
        # Rotation strategies for variety
        self.difficulty_rotation = ["Easy", "Medium", "Hard"]
        self.topic_rotation_index = 0
        
    def get_rotation_weights(self) -> Dict[str, int]:
        """Get varied weights based on session"""
        base_weights = {
            "easy_medium": 4,
            "hard": 3,
            "dsa": 8,
            "behavioral": 3
        }
        
        # Introduce session-based variation
        variations = [
            {"dsa": 10, "easy_medium": 3, "hard": 4, "behavioral": 3},  # DSA-heavy
            {"dsa": 6, "easy_medium": 6, "hard": 4, "behavioral": 4},   # Balanced
            {"dsa": 8, "easy_medium": 5, "hard": 5, "behavioral": 2},   # Tech-heavy
            {"dsa": 7, "easy_medium": 4, "hard": 6, "behavioral": 3},   # Hard-focused
        ]
        
        session_variant = int(self.session_id, 16) % len(variations)
        return variations[session_variant]

# --- Enhanced Skills Extraction (same as before) ---
def extract_skills_from_resume(resume_pdf_path: str) -> Tuple[List[str], str]:
    """Extract technical skills and return resume text"""
    resume_loader = PyPDFLoader(resume_pdf_path)
    resume_docs = resume_loader.load()
    resume_text = "\n".join([doc.page_content for doc in resume_docs])

    # Comprehensive skill keywords
    skill_keywords = [
        # Programming Languages
        "Python", "Java", "C++", "C", "JavaScript", "TypeScript", "C#", "Go", "Rust", "Kotlin", "Swift",
        "PHP", "Ruby", "Scala", "R", "MATLAB", "Dart", "Objective-C",
        
        # Web Technologies
        "React", "Angular", "Vue.js", "Node.js", "Express", "Django", "Flask", "FastAPI", "Spring Boot",
        "Laravel", "Rails", "ASP.NET", "HTML", "CSS", "SASS", "SCSS", "Bootstrap", "Tailwind CSS",
        "jQuery", "Redux", "Next.js", "Nuxt.js",
        
        # Databases
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "Oracle", "Cassandra", "DynamoDB",
        "Neo4j", "InfluxDB", "Elasticsearch", "Firebase", "Supabase",
        
        # Cloud & DevOps
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "CI/CD", "Terraform", "Ansible",
        "Git", "GitHub", "GitLab", "Linux", "Ubuntu", "CentOS", "Docker Compose", "Helm", "Prometheus",
        
        # Data Science & ML
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", 
        "Scikit-learn", "Pandas", "NumPy", "Matplotlib", "Seaborn", "Jupyter", "Keras",
        "OpenCV", "NLTK", "spaCy", "Transformers", "BERT", "GPT", "LangChain", "Hugging Face",
        
        # Mobile Development
        "Android", "iOS", "React Native", "Flutter", "Xamarin", "Unity", "AR", "VR", "ARKit", "ARCore",
        
        # Fundamentals & Concepts
        "Data Structures", "Algorithms", "OOP", "DBMS", "Operating Systems", "Computer Networks",
        "System Design", "Software Engineering", "Agile", "Scrum", "Design Patterns", "SOLID",
        
        # Other Technologies
        "REST API", "GraphQL", "Microservices", "WebSocket", "OAuth", "JWT", "Blockchain",
        "Socket.io", "Apache Kafka", "RabbitMQ", "MQTT", "Nginx", "Apache", "Redis"
    ]

    found_skills = set()
    for skill in skill_keywords:
        # Use word boundaries for exact matching
        pattern = rf'\b{re.escape(skill)}\b'
        if re.search(pattern, resume_text, re.IGNORECASE):
            found_skills.add(skill)

    # Extract additional technical terms from project descriptions
    project_tech_pattern = r'\b(?:using|with|in|built|developed|implemented|created|designed)\s+([A-Z][A-Za-z0-9.+#-]+)\b'
    tech_matches = re.findall(project_tech_pattern, resume_text, re.IGNORECASE)
    
    for tech in tech_matches:
        if tech.lower() not in [s.lower() for s in found_skills] and len(tech) > 2:
            # Validate if it's likely a technology (contains common tech suffixes/patterns)
            if any(pattern in tech.lower() for pattern in ['.js', 'sql', 'db', 'api', 'ml', 'ai', 'ar', 'vr']):
                found_skills.add(tech)

    return list(found_skills), resume_text

# --- Dynamic Query Generation ---
def generate_varied_queries(skills: List[str], session: InterviewSession) -> List[str]:
    """Generate multiple varied queries for different perspectives"""
    
    # Base query components
    dsa_terms = [
        "data structures algorithms",
        "arrays strings trees graphs", 
        "sorting searching dynamic programming",
        "recursion backtracking greedy",
        "linked lists stacks queues",
        "binary trees graph traversal"
    ]
    
    tech_terms = [
        " ".join(skills[:3]),
        " ".join(skills[2:5]) if len(skills) > 2 else " ".join(skills),
        " ".join(skills[4:7]) if len(skills) > 4 else " ".join(skills[:2]),
    ]
    
    interview_contexts = [
        "coding interview questions",
        "technical interview problems", 
        "programming challenges",
        "software engineering questions",
        "computer science fundamentals"
    ]
    
    # Generate multiple query variations
    queries = []
    
    # DSA-focused queries
    for dsa in random.sample(dsa_terms, min(2, len(dsa_terms))):
        for context in random.sample(interview_contexts, 2):
            queries.append(f"{dsa} {context}")
    
    # Skill-based queries  
    for tech in tech_terms:
        if tech.strip():
            for context in random.sample(interview_contexts, 2):
                queries.append(f"{tech} {context}")
    
    # Mixed queries
    if skills:
        tech_sample = random.choice(tech_terms)
        dsa_sample = random.choice(dsa_terms)
        queries.append(f"{tech_sample} {dsa_sample} interview")
    
    return queries

# --- Enhanced Context Retrieval with Diversity ---
def get_diverse_context(skills: List[str], collection, session: InterviewSession) -> Dict[str, List[Dict]]:
    """Retrieve diverse contexts with session-based variation"""
    
    category_weights = session.get_rotation_weights()
    queries = generate_varied_queries(skills, session)
    
    # print(f"🔍 Using {len(queries)} varied search queries")
    
    all_contexts = {}
    
    # Try each query and aggregate results
    for i, query in enumerate(queries):
        # print(f"🔍 Query {i+1}: {query[:60]}...")
        
        try:
            query_embedding = embedder.encode(query).tolist()
            
            # Vector search with different numCandidates for variety
            num_candidates = random.choice([200, 300, 400, 500])
            
            # Get results for each category
            for category, target_count in category_weights.items():
                
                if category == "dsa":
                    match_filter = {"category": "DSA"}
                elif category == "easy_medium":
                    match_filter = {
                        "difficulty": {"$in": ["Easy", "Medium"]},
                        "category": {"$ne": "DSA"}
                    }
                elif category == "hard":
                    match_filter = {
                        "difficulty": "Hard", 
                        "category": {"$ne": "DSA"}
                    }
                elif category == "behavioral":
                    match_filter = {
                        "$or": [
                            {"type": "conceptual"},
                            {"category": {"$in": ["Behavioral", "System Design", "General"]}}
                        ]
                    }
                
                try:
                    results = list(collection.aggregate([
                        {"$vectorSearch": {
                            "index": "vector_index",
                            "queryVector": query_embedding,
                            "path": "embedding", 
                            "numCandidates": num_candidates,
                            "limit": target_count * 3  # Get more for diversity
                        }},
                        {"$match": match_filter},
                        {"$project": {
                            "category": 1, "topic": 1, "difficulty": 1, "type": 1,
                            "question": 1, "answer": 1, "complexity": 1,
                            "score": {"$meta": "vectorSearchScore"}, "_id": 1
                        }},
                        {"$sort": {"score": -1}}
                    ]))
                    
                    if category not in all_contexts:
                        all_contexts[category] = []
                    
                    # Filter out duplicates by question text or ID
                    for result in results:
                        question_hash = hashlib.md5(
                            (result.get('question', '') + str(result.get('_id', '')))
                            .encode()
                        ).hexdigest()
                        
                        if question_hash not in session.used_questions:
                            all_contexts[category].append(result)
                            session.used_questions.add(question_hash)
                    
                except Exception as ve:
                    print(f"❌ Vector search failed for {category}: {ve}")
                    continue
                    
        except Exception as e:
            print(f"❌ Query {i+1} failed: {e}")
            continue
    
    # Diversify and sample final results
    final_contexts = {}
    for category, items in all_contexts.items():
        target_count = category_weights[category]
        
        if not items:
            # print(f"⚠️ No items for {category}, using fallback")
            # Fallback sampling
            try:
                if category == "dsa":
                    fallback = list(collection.aggregate([
                        {"$match": {"category": "DSA"}},
                        {"$sample": {"size": target_count}}
                    ]))
                elif category == "easy_medium":
                    fallback = list(collection.aggregate([
                        {"$match": {"difficulty": {"$in": ["Easy", "Medium"]}}},
                        {"$sample": {"size": target_count}}
                    ]))
                elif category == "hard":
                    fallback = list(collection.aggregate([
                        {"$match": {"difficulty": "Hard"}},
                        {"$sample": {"size": target_count}}
                    ]))
                else:
                    fallback = list(collection.aggregate([
                        {"$match": {"$or": [{"type": "conceptual"}, {"category": "Behavioral"}]}},
                        {"$sample": {"size": target_count}}
                    ]))
                
                items = fallback
            except:
                items = []
        
        # Diversify selection
        if len(items) > target_count:
            # Group by topic/difficulty for diversity
            topic_groups = {}
            for item in items:
                topic = item.get('topic', 'general')
                difficulty = item.get('difficulty', 'medium')
                key = f"{topic}_{difficulty}"
                
                if key not in topic_groups:
                    topic_groups[key] = []
                topic_groups[key].append(item)
            
            # Sample from different groups for diversity
            selected = []
            group_keys = list(topic_groups.keys())
            random.shuffle(group_keys)
            
            for key in group_keys:
                if len(selected) >= target_count:
                    break
                group_items = topic_groups[key]
                selected.extend(random.sample(group_items, min(1, len(group_items))))
            
            # Fill remaining if needed
            if len(selected) < target_count:
                remaining = [item for item in items if item not in selected]
                if remaining:
                    additional = random.sample(remaining, min(target_count - len(selected), len(remaining)))
                    selected.extend(additional)
            
            final_contexts[category] = selected[:target_count]
        else:
            final_contexts[category] = items
        
        # print(f"📋 {category}: {len(final_contexts[category])} items selected")
    
    return final_contexts

# --- Enhanced Resume Analysis (same as before) ---
def analyze_resume_focus(resume_text: str, skills: List[str]) -> Dict[str, str]:
    """Analyze resume to determine candidate's focus areas"""
    focus_analysis = {
        "experience_level": "Entry" if any(word in resume_text.lower() for word in ["student", "intern", "fresh", "graduate"]) else "Experienced",
        "primary_domain": "",
        "key_projects": [],
        "strengths": []
    }
    
    # Determine primary domain
    domain_keywords = {
        "Web Development": ["react", "node.js", "javascript", "html", "css", "web", "frontend", "backend"],
        "Data Science/ML": ["machine learning", "deep learning", "tensorflow", "pytorch", "data", "ml", "ai"],
        "Mobile Development": ["android", "ios", "mobile", "app", "react native", "flutter"],
        "Systems/Backend": ["system design", "microservices", "api", "database", "server", "cloud"],
        "Game Development": ["unity", "game", "ar", "vr", "3d"]
    }
    
    max_matches = 0
    for domain, keywords in domain_keywords.items():
        matches = sum(1 for keyword in keywords if keyword.lower() in resume_text.lower())
        if matches > max_matches:
            max_matches = matches
            focus_analysis["primary_domain"] = domain

    # Extract key projects
    project_lines = []
    lines = resume_text.split('\n')
    for i, line in enumerate(lines):
        if re.search(r'\b(project|built|developed|created|implemented|designed)\b', line.lower()):
            context = ' '.join(lines[i:i+2]).strip()
            if len(context) > 20:
                project_lines.append(context[:150] + "..." if len(context) > 150 else context)
    
    focus_analysis["key_projects"] = project_lines[:3]
    focus_analysis["strengths"] = skills[:6]

    return focus_analysis

# --- Dynamic Prompt Generation ---
def generate_dynamic_prompt(resume_text: str, skills: List[str], contexts: Dict[str, List[Dict]], 
                           focus_analysis: Dict[str, str], session: InterviewSession) -> str:
    """Generate varied prompts based on session with unique real-world scenarios"""
    
    # Different interviewer personas for variety
    interviewer_personas = [
        "senior_technical_architect",
        "startup_cto", 
        "enterprise_team_lead",
        "product_engineering_manager",
        "principal_engineer",
        "tech_consultant"
    ]
    
    persona = interviewer_personas[int(session.session_id, 16) % len(interviewer_personas)]
    
    # Simple enhancement helpers based on candidate's domain
    def get_context_enhancement_hints(domain: str, skills: List[str]) -> Dict[str, str]:
        """Provide subtle enhancement hints without overcomplicating"""
        return {
            "projects_focus": "project experience, system design, technical architecture, and scaling challenges",
            "dsa_focus": "clean algorithmic problems with proper formatting - examples and explanations on new lines", 
            "behavioral_focus": "workplace scenarios, teamwork, leadership, and decision-making situations"
        }

    # Keep original context formatting but enhance it slightly
    def format_context(context_list: List[Dict], section_name: str) -> str:
        if not context_list:
            return f"\n{section_name.upper()} CONTEXT: No specific context found.\n"
        
        formatted = f"\n{section_name.upper()} CONTEXT:\n"
        for i, ctx in enumerate(context_list[:4], 1):  # Show more examples for variety
            formatted += f"{i}. Topic: {ctx.get('topic', 'N/A')} | Category: {ctx.get('category', 'N/A')}\n"
            if ctx.get('question'):
                formatted += f"   Example Q: {ctx['question'][:120]}...\n"
            if ctx.get('answer'):
                formatted += f"   Key Concept: {ctx['answer'][:180]}...\n"
        return formatted + "\n"

    context_section = ""
    for category, context_list in contexts.items():
        context_section += format_context(context_list, category.replace('_', '/'))

    # Get enhancement hints
    enhancement_hints = get_context_enhancement_hints(focus_analysis['primary_domain'], skills)
    
    # Persona-specific interview approaches
    persona_approaches = {
        "senior_technical_architect": "Focus on system design, scalability, and architectural trade-offs with real production scenarios.",
        "startup_cto": "Emphasize rapid prototyping, resource constraints, and building scalable solutions from scratch.", 
        "enterprise_team_lead": "Highlight team collaboration, code reviews, mentoring, and enterprise-scale challenges.",
        "product_engineering_manager": "Balance technical depth with product thinking and cross-functional collaboration.",
        "principal_engineer": "Deep dive into complex technical problems, performance optimization, and technical leadership.",
        "tech_consultant": "Focus on problem-solving methodology, client communication, and diverse technology stacks."
    }

    # Default style for prompt formatting (readable) and grounding scenarios
    style = persona

    # Provide grounding scenarios for the prompt. Prefer candidate projects if present.
    scenario_contexts = {
        "scenarios": [
            (focus_analysis.get('key_projects', []) and focus_analysis['key_projects'][0][:140]) or "candidate project and feature work",
            "scaling / performance incident in production",
            "designing and launching a new feature end-to-end"
        ]
    }

    # Keep original JSON schema but enhance question requirements
    json_schema_instructions = f"""
Return a single JSON object only (no surrounding commentary). The JSON MUST match this schema exactly:

{{
    "metadata": {{
        "experience_level": string,
        "primary_domain": string, 
        "skills": [string],
        "key_projects": [string]
    }},
    "easy_medium": [
        {{"q": string, "a": string}},  // exactly 3 items
    ],
    "hard": [
        {{"q": string, "a": string}},  // exactly 3 items
    ],
    "dsa": [
        {{"difficulty": "Medium"|"Medium-Hard"|"Hard", "q": string, "a": string, "examples": string, "constraints": string, "complexity": string, "code": string}},  // exactly 3 items
    ],
    "behavioral": [
        {{"q": string, "a": string}},  // exactly 3 items
    ]
}}

ENHANCED REQUIREMENTS FOR UNIQUE REAL-WORLD QUESTIONS:
- Output must be valid JSON parseable by json.loads()
- Use the exact keys shown above.
- Keep code blocks and examples as strings (they can contain newlines).
- DSA section: examples and constraints are MANDATORY - every DSA item must include both fields

QUESTION DISTRIBUTION & UNIQUENESS REQUIREMENTS:
- easy_medium: 1 PROJECT/SYSTEM DESIGN question + 2 other technical questions
- hard: 3 advanced technical questions with real-world complexity
- dsa: 2 MEDIUM + 1 HARD DSA problems (all with practical applications)
  * Each DSA problem MUST include: examples (with input/output), constraints (technical limits)
- behavioral: 1 REAL-WORLD SCENARIO + 2 other behavioral questions

All questions must be grounded in these scenarios: {', '.join(scenario_contexts['scenarios'][:3])}
- NO generic questions like "Tell me about yourself", "Reverse a linked list", "What's your weakness"
- EVERY question must connect to real business problems and practical applications
- DSA questions must clearly state their real-world use cases
- Focus approach: {persona_approaches[persona]}
"""

    prompt = f"""
You are a {style.replace('_', ' ')} conducting a technical interview. Generate UNIQUE and VARIED interview questions and answers based on the provided context and candidate profile.

SESSION ID: {session.session_id} (Use this to ensure question variety across sessions)

=== CANDIDATE ANALYSIS ===
Experience Level: {focus_analysis['experience_level']}
Primary Domain: {focus_analysis['primary_domain']}
Key Technical Skills: {', '.join(skills[:10])}
Key Projects: {'; '.join(focus_analysis['key_projects'])}

=== TECHNICAL CONTEXTS (Use these as primary source for questions) ===
{context_section}

=== RESUME SUMMARY (Key Sections) ===
{resume_text}...

QUESTION GENERATION GUIDELINES:
1. **Primary Source**: Base questions primarily on the TECHNICAL CONTEXTS provided above
2. **Enhancement**: Add practical applications and real-world relevance where natural
3. **Uniqueness**: Avoid generic questions - make them specific to the candidate's background
4. **Balance**: Mix context-based questions with some practical scenario questions
5. **Appropriate Difficulty**: Match the candidate's experience level

INTERVIEW STRUCTURE - THREE SECTIONS:

SECTION 1 - PROJECT EXPERIENCE (easy_medium + hard sections):
- easy_medium: 3 questions focused on projects, system design, and technical experience
- hard: 3 advanced technical questions about complex projects, architecture decisions, scaling challenges

SECTION 2 - DSA PROBLEMS (dsa section):
- 3 algorithmic questions: 2 Medium + 1 Hard difficulty
- Clean problem statements with proper formatting
- Examples and explanations on new lines is mandatory
- Include time/space complexity analysis
- Cover different algorithmic concepts

SECTION 3 - BEHAVIORAL (behavioral section):  
- 3 behavioral questions about teamwork, leadership, problem-solving situations
- Real workplace scenarios and decision-making challenges

DSA FORMATTING REQUIREMENTS:
- Clean problem statement without mentioning "LeetCode style" in question text
- Structure: Problem description → Example: (new line) → Explanation: (new line) → Constraints
- Include real-world applications where relevant
- MANDATORY: Each DSA item MUST include:
  * "examples": Input/output examples with explanations
  * "constraints": Problem constraints (array size, value ranges, time limits)
  * "complexity": Time and space complexity analysis
  * "code": Complete working solution
- MANDATORY: examples field must contain at least one input/output example with explanation
- MANDATORY: constraints field must specify technical limits (array size, value ranges, etc.)
- complexity field should include time and space complexity analysis

Generate interview content STRICTLY as JSON following the schema below:
{json_schema_instructions}
"""
    
    return prompt


# --- Main Enhanced Pipeline ---
def interview_rag_pipeline(resume_pdf_path: str, collection):
    """Enhanced interview question generation pipeline with variety"""
    # print("🚀 Starting Enhanced Interview RAG Pipeline with Question Variety...")
    
    # Create new session for this run
    session = InterviewSession(resume_pdf_path)
    
    # Step 1: Extract skills and analyze resume
    # print("📄 Analyzing resume...")
    skills, resume_text = extract_skills_from_resume(resume_pdf_path)
    # print(f"✅ Extracted {len(skills)} technical skills: {skills[:10]}...")
    
    # Step 2: Analyze resume focus  
    focus_analysis = analyze_resume_focus(resume_text, skills)
    # print(f"🎯 Candidate Focus: {focus_analysis['primary_domain']} ({focus_analysis['experience_level']} level)")
    
    # Step 3: Retrieve diverse contexts
    # print("🔍 Retrieving diverse technical contexts...")
    contexts = get_diverse_context(skills, collection, session)
    
    total_contexts = sum(len(ctx_list) for ctx_list in contexts.values())
    # print(f"✅ Retrieved {total_contexts} diverse contexts across all categories")
    
    # Step 4: Generate dynamic prompt
    # print("📝 Generating varied interview questions...")
    prompt = generate_dynamic_prompt(resume_text, skills, contexts, focus_analysis, session)
    
    # Step 5: Call LLM with dynamic prompt
    try:
        llm = ChatGoogleGenerativeAI(model="gemma-3-27b-it", temperature=0.8)  # Higher temperature for variety
        response = llm.invoke(prompt)

        # Normalize LLM text content
        if hasattr(response, 'content'):
            llm_text = response.content
        elif hasattr(response, 'text'):
            llm_text = response.text
        else:
            llm_text = str(response)

        # print(f"\n🎯 Session {session.session_id} - Interview Questions Generated")
        # print("="*60)
        # print("🎤 RAW LLM OUTPUT")
        # print("="*60)
        # print(llm_text)
        # print("="*60)

        # --- Pydantic models for validation ---
        class QAItem(BaseModel):
            q: str
            a: str
            id: Optional[str] = None
            difficulty: Optional[str] = None
            code: Optional[str] = None

        class DSAItem(QAItem):
            difficulty: Literal['Medium', 'Medium-Hard', 'Hard']
            complexity: str
            examples: str
            constraints: str

        class Metadata(BaseModel):
            experience_level: str
            primary_domain: str
            skills: List[str]
            key_projects: List[str]

        class LLMOutput(BaseModel):
            metadata: Metadata
            easy_medium: List[QAItem] = Field(min_items=3, max_items=3)
            hard: List[QAItem] = Field(min_items=3, max_items=3)
            dsa: List[DSAItem] = Field(min_items=3, max_items=3)
            behavioral: List[QAItem] = Field(min_items=3, max_items=3)

        # Try robust extraction of JSON-like content
        def extract_json_from_text(s: str):
            try:
                return json.loads(s)
            except Exception:
                pass

            import re
            # try fenced json
            m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass

            # try to find first balanced JSON object
            start = s.find('{')
            if start == -1:
                return None
            # crude but often effective: find last '}'
            end = s.rfind('}')
            if end == -1 or end <= start:
                return None
            candidate = s[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                return None

        parsed_json = extract_json_from_text(llm_text if isinstance(llm_text, str) else str(llm_text))
        # print(f"🔍 Extracted JSON keys: {list(parsed_json.keys()) if parsed_json else 'None'}")
        if parsed_json and 'dsa' in parsed_json:
            dsa_sample = parsed_json['dsa'][0] if parsed_json['dsa'] else {}
            # print(f"🔍 First DSA item keys: {list(dsa_sample.keys())}")
        parsed_valid = None
        if parsed_json is None:
            pass
            print("⚠️ Failed to extract JSON from LLM output. Will fallback to building sections from retrieved contexts.")
        else:
            # Add missing DSA fields before validation
            if 'dsa' in parsed_json:
                for item in parsed_json['dsa']:
                    if 'examples' not in item:
                        item['examples'] = "Example:\nInput: [sample input]\nOutput: [sample output]"
                    if 'constraints' not in item:
                        item['constraints'] = "1 <= n <= 10^5\nTime limit: 1 second"
                    if 'complexity' not in item:
                        item['complexity'] = "Time: O(n), Space: O(1)"
            
            # Validate with pydantic
            try:
                parsed_valid = LLMOutput.parse_obj(parsed_json)
                # print("✅ LLM output validated by Pydantic")
            except ValidationError as ve:
                print("❌ Pydantic validation failed:")
                print(ve.json())
                parsed_valid = None

        # Build final response from validated model or fallback to contexts
        final_sections = []

        if parsed_valid is not None:
            # convert pydantic model into sections
            def mk_section(title: str, items: List[QAItem]):
                questions = []
                for i, it in enumerate(items, 1):
                    qid = it.id or f"{session.session_id}_{title.replace('/','_').replace(' ','_').lower()}_q{i}"
                    item_obj = {
                        "id": qid,
                        "q": it.q,
                        "a": it.a,
                    }

                    # Common optional fields
                    if getattr(it, 'difficulty', None):
                        item_obj['difficulty'] = getattr(it, 'difficulty')
                    if getattr(it, 'code', None):
                        item_obj['code'] = getattr(it, 'code')

                    # DSA-specific extras
                    if hasattr(it, 'complexity') and getattr(it, 'complexity', None):
                        item_obj['complexity'] = getattr(it, 'complexity')
                    if hasattr(it, 'examples') and getattr(it, 'examples', None):
                        item_obj['examples'] = getattr(it, 'examples')
                    if hasattr(it, 'constraints') and getattr(it, 'constraints', None):
                        item_obj['constraints'] = getattr(it, 'constraints')

                    questions.append(item_obj)

                return {"title": title, "questions": questions}

            final_sections.append(mk_section('Easy/Medium', parsed_valid.easy_medium))
            final_sections.append(mk_section('Hard', parsed_valid.hard))
            final_sections.append(mk_section('DSA', parsed_valid.dsa))
            final_sections.append(mk_section('Behavioral', parsed_valid.behavioral))

        else:
            # fallback: build sections from contexts dict
            qid = 1
            ctx_map = [
                ("Easy/Medium", contexts.get('easy_medium', [])),
                ("Hard", contexts.get('hard', [])),
                ("DSA", contexts.get('dsa', [])),
                ("Behavioral", contexts.get('behavioral', []))
            ]
            for title, items in ctx_map:
                questions = []
                for item in items[:3]:
                    q_text = item.get('question') or item.get('q') or item.get('prompt') or ''
                    a_text = item.get('answer') or item.get('a') or ''
                    diff = item.get('difficulty') or item.get('complexity')
                    code = item.get('code') if isinstance(item.get('code'), str) else None
                    complexity = item.get('complexity')
                    examples = item.get('examples') or item.get('example') or None
                    constraints = item.get('constraints') or item.get('constraint') or None

                    qobj = {
                        "id": f"{session.session_id}_q{qid}",
                        "q": q_text,
                        "a": a_text,
                    }
                    if diff:
                        qobj['difficulty'] = diff
                    if code:
                        qobj['code'] = code
                    if complexity:
                        qobj['complexity'] = complexity
                    if examples:
                        qobj['examples'] = examples
                    if constraints:
                        qobj['constraints'] = constraints
                    
                    # For DSA sections, ensure examples and constraints are present
                    if title == "DSA":
                        if not examples:
                            qobj['examples'] = "Example: Input: [sample input]\nOutput: [expected output]\nExplanation: [brief explanation]"
                        if not constraints:
                            qobj['constraints'] = "1 <= n <= 10^4\n1 <= values <= 10^9"

                    questions.append(qobj)
                    qid += 1
                final_sections.append({"title": title, "questions": questions})

        final_response = {
            "status": "success", 
            "sections": final_sections,
            "session_id": session.session_id
        }
        
    except Exception as e:
        print(f"❌ Error calling LLM: {e}")
        final_response = {
            "status": "error",
            "sections": [],
            "session_id": session.session_id,
            "error_message": str(e)
        }

    # Return the structured final response and include diagnostics
    # Custom set assembly removed — keep the original `final_response['sections']` as built above.

    final_response.update({
        "skills": skills,
        "focus_analysis": focus_analysis,
        "contexts_retrieved": total_contexts,
        "llm_output": llm_text if 'llm_text' in locals() else None
    })

    return final_response

# --- Run Enhanced Pipeline ---
if __name__ == "__main__":
    load_dotenv()
    
    # Update your resume path
    resume_pdf_path = "/Users/vishnuvardhan/Downloads/Main_resume_mvv.pdf"
    
    try:
        result = interview_rag_pipeline(resume_pdf_path, collection)
        # print(f"\n✅ Pipeline completed successfully!")
        # print(f"📊 Session: {result.get('session_id', 'Unknown')}")
        # print(f"📊 Summary: {len(result['skills'])} skills, {result['contexts_retrieved']} contexts retrieved")
        # print(f"🎯 Sections generated: {len(result['sections'])}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        pass