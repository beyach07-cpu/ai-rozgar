import streamlit as st
import urllib.parse
import re
from PIL import Image, ImageOps, ImageEnhance
import pypdf
import pytesseract
from sentence_transformers import SentenceTransformer, util

# Page Config
st.set_page_config(
    page_title="AI Rozgar — Career Diagnostic",
    page_icon="💼",
    layout="wide"
)

# -------------------------------------------------
# 1. LOAD MODEL (Cached taake baar baar load na ho)
# -------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------------------------------
# 2. 50 JOBS & REQUIRED CORE SKILLS DATABASE
# -------------------------------------------------
JOB_ROLES = {
    "Data Scientist": ["Python", "Machine Learning", "SQL", "Pandas", "Scikit-Learn", "Statistics"],
    "Frontend Developer": ["JavaScript", "React", "HTML", "CSS", "Tailwind CSS", "Git"],
    "Backend Developer": ["Python", "Django", "Node.js", "PostgreSQL", "REST APIs", "Docker"],
    "Full Stack Developer": ["React", "Node.js", "MongoDB", "Express", "TypeScript", "CI/CD"],
    "DevOps Engineer": ["Docker", "Kubernetes", "Linux", "AWS", "Terraform", "CI/CD"],
    "Cloud Architect": ["AWS", "Azure", "Cloud Security", "System Design", "Networking"],
    "AI/ML Engineer": ["Deep Learning", "PyTorch", "TensorFlow", "NLP", "Python", "Transformers"],
    "Data Analyst": ["Excel", "SQL", "PowerBI", "Tableau", "Data Cleaning", "Python"],
    "Business Analyst": ["Requirements Gathering", "Agile", "Jira", "SQL", "Process Modeling"],
    "Cybersecurity Analyst": ["Network Security", "Ethical Hacking", "SIEM", "Linux", "Firewalls"],
    "Mobile Developer (Flutter)": ["Flutter", "Dart", "Firebase", "State Management", "REST APIs"],
    "Android Developer": ["Kotlin", "Android Studio", "Jetpack Compose", "Coroutines", "MVVM"],
    "iOS Developer": ["Swift", "SwiftUI", "Xcode", "CocoaPods", "CoreData"],
    "Product Manager": ["Roadmapping", "User Research", "Agile", "Scrum", "A/B Testing"],
    "UI/UX Designer": ["Figma", "Wireframing", "User Research", "Prototyping", "Design Systems"],
    "QA Automation Tester": ["Selenium", "Python", "Test Automation", "Jira", "Postman"],
    "Database Administrator": ["SQL", "MySQL", "PostgreSQL", "Performance Tuning", "Backup Recovery"],
    "Blockchain Developer": ["Solidity", "Smart Contracts", "Ethereum", "Web3.js", "Cryptography"],
    "Embedded Systems Engineer": ["C", "C++", "Microcontrollers", "RTOS", "I2C/SPI"],
    "Game Developer": ["Unity", "C#", "3D Math", "Game Physics", "Shader Programming"],
    "Digital Marketing Specialist": ["SEO", "Google Ads", "Content Marketing", "Google Analytics"],
    "SEO Specialist": ["Keyword Research", "On-Page SEO", "Backlinks", "Technical SEO", "Ahrefs"],
    "Content Strategist": ["Copywriting", "SEO Writing", "Content Strategy", "Proofreading"],
    "HR Recruiter": ["Talent Acquisition", "Technical Screening", "Interviewing", "HRIS"],
    "Financial Analyst": ["Financial Modeling", "Advanced Excel", "Valuation", "Forecasting"],
    "Network Engineer": ["Cisco CCNA", "Routing & Switching", "TCP/IP", "DNS/DHCP", "VPN"],
    "Systems Administrator": ["Linux Administration", "Windows Server", "Active Directory", "Bash"],
    "Graphic Designer": ["Adobe Photoshop", "Adobe Illustrator", "Branding", "Typography"],
    "Video Editor": ["Premiere Pro", "After Effects", "Color Grading", "Sound Design"],
    "E-Commerce Specialist": ["Shopify", "Amazon FBA", "Inventory Control", "PPC Advertising"],
    "Solutions Architect": ["System Architecture", "Microservices", "Cloud Native", "Scalability"],
    "Data Engineer": ["Apache Spark", "Kafka", "SQL", "ETL Pipelines", "Airflow"],
    "NLP Engineer": ["Transformers", "Hugging Face", "spaCy", "Tokenization", "LLMs"],
    "Computer Vision Engineer": ["OpenCV", "YOLO", "Image Processing", "PyTorch", "Object Detection"],
    "Robotics Engineer": ["ROS", "Python", "C++", "Control Systems", "Sensor Fusion"],
    "Site Reliability Engineer (SRE)": ["Prometheus", "Grafana", "Linux", "SLO/SLI", "Incident Response"],
    "Technical Writer": ["API Documentation", "Markdown", "Git", "Developer Documentation"],
    "Operations Manager": ["Supply Chain", "Budgeting", "Vendor Management", "Process Optimization"],
    "Prompt Engineer": ["Prompt Design", "LangChain", "Vector Databases", "Model Evaluation"],
    "ERP Consultant (SAP)": ["SAP S/4HANA", "Business Processes", "ABAP", "ERP Configuration"],
    "Supply Chain Analyst": ["Inventory Planning", "Data Analysis", "ERP Systems", "Forecasting"],
    "Customer Success Manager": ["Onboarding", "Churn Reduction", "Zendesk", "Account Management"],
    "Legal Tech / Compliance Analyst": ["GDPR", "Data Privacy", "Regulatory Compliance", "Contract Audit"],
    "Public Relations Specialist": ["Media Relations", "Press Releases", "Crisis Communication", "Branding"],
    "AR/VR Developer": ["Unity", "Unreal Engine", "C#", "OpenXR", "Spatial Audio"],
    "Biomedical Data Analyst": ["Bioinformatics", "Python", "R", "Clinical Data Analysis", "Biostatistics"],
    "Penetration Tester": ["Metasploit", "Burp Suite", "Kali Linux", "Vulnerability Assessment"],
    "Scrum Master": ["Scrum", "Agile Coaching", "Sprint Planning", "Jira", "Conflict Resolution"],
    "Technical Support Engineer": ["Customer Support", "Troubleshooting", "Ticketing Tools", "Basic Networking"],
    "Sales / Business Development": ["Lead Generation", "Cold Calling", "CRM", "Negotiation", "B2B Sales"]
}

# Pre-compute job embeddings once
job_titles = list(JOB_ROLES.keys())
job_descriptions = [f"Job role: {title}. Core competencies: {', '.join(skills)}" for title, skills in JOB_ROLES.items()]
job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)

# -------------------------------------------------
# 3. ROBUST TEXT EXTRACTION (MOBILE + LAPTOP SYNC)
# -------------------------------------------------
def preprocess_image_for_ocr(image):
    """Mobile photo orientation aur contrast fix karta hai taake text clear read ho"""
    try:
        # EXIF auto-rotation (Mobile camera angle fix)
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    
    # Grayscale conversion
    image = image.convert('L')
    
    # Contrast boost (Blurry text ko dark banata hai)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.8)
    return image

def extract_text(file_obj, raw_text):
    if raw_text and len(raw_text.strip()) > 30:
        return raw_text.strip()
    
    if not file_obj:
        return ""
    
    text = ""
    file_name = file_obj.name.lower()
    
    # PDF Processing
    if file_name.endswith('.pdf'):
        try:
            reader = pypdf.PdfReader(file_obj)
            for page in reader.pages:
                page_content = page.extract_text()
                if page_content:
                    text += page_content + "\n"
        except Exception:
            pass
            
    # Image / Photo Processing (PNG, JPG, JPEG)
    else:
        try:
            img = Image.open(file_obj)
            processed_img = preprocess_image_for_ocr(img)
            # Custom Tesseract configuration for consistent line reads
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed_img, config=custom_config)
        except Exception:
            try:
                # Fallback basic scan
                text = pytesseract.image_to_string(Image.open(file_obj))
            except Exception:
                pass
                
    return text.strip()

def check_skill_in_text(skill, text):
    """Regex boundary matching: case mismatch ya word formatting errors ko door karta hai"""
    escaped = re.escape(skill.lower())
    pattern = rf'\b{escaped}\b'
    return bool(re.search(pattern, text.lower()))

# -------------------------------------------------
# 4. STREAMLIT USER INTERFACE
# -------------------------------------------------
st.title("💼 AI Rozgar — Intelligent Career Diagnostic")
st.write("Apna CV upload karein aur 50 industry jobs ke hisaab se **Readiness %**, **Missing Skills Roadmap**, aur **1-Click Cover Letter** hasil karein.")

col1, col2 = st.columns([1, 1.25], gap="large")

with col1:
    st.subheader("📄 Upload CV")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF, PNG, JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Best results ke liye PDF ya saaf photo upload karein."
    )
    pasted_text = st.text_area(
        "Ya Direct Text Paste Karein",
        height=140,
        placeholder="Agar photo blur ho to yahan direct CV text copy paste kar sakte hain..."
    )
    analyze_btn = st.button("🚀 Run AI Analysis", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Career Diagnostic Report")
    
    if analyze_btn:
        with st.spinner("Analyzing resume against 50 industry job profiles..."):
            cv_text = extract_text(uploaded_file, pasted_text)
            
            if not cv_text or len(cv_text) < 30:
                st.error("⚠️ CV se text read nahi ho saka. Barah-e-karam saaf PDF upload karein ya direct text paste karein.")
            else:
                # Semantic Similarity Calculation
                cv_emb = model.encode(cv_text, convert_to_tensor=True)
                cosine_scores = util.cos_sim(cv_emb, job_embeddings)[0]
                
                scores = [(job_titles[i], float(cosine_scores[i])) for i in range(len(job_titles))]
                scores.sort(key=lambda x: x[1], reverse=True)
                
                top_role, top_val = scores[0]
                # Normalized percentage formula
                top_pct = min(max(int((top_val + 0.25) * 100), 20), 98)
                
                # Hero Match Card
                st.success(f"### 🎯 Primary Job Match: **{top_role}** ({top_pct}% Ready)")
                
                tab1, tab2, tab3 = st.tabs(["💡 Missing Skills & YouTube Links", "📈 All Top Matches", "📝 Tailored Cover Letter"])
                
                with tab1:
                    target_skills = JOB_ROLES[top_role]
                    # Upgraded regex matching
                    found = [s for s in target_skills if check_skill_in_text(s, cv_text)]
                    missing = [s for s in target_skills if not check_skill_in_text(s, cv_text)]
                    
                    if found:
                        st.markdown("**Existing Strengths Detected:**")
                        st.markdown(" ".join([f"`✓ {s}`" for s in found]))
                    
                    st.divider()
                    
                    if missing:
                        st.markdown(f"#### Ye **{len(missing)} skills** seekh kar aapka match **100%** ho sakta hai:")
                        for s in missing:
                            yt_query = urllib.parse.quote(f"{s} complete crash course tutorial for beginners")
                            yt_link = f"https://www.youtube.com/results?search_query={yt_query}"
                            st.markdown(f"- 🔴 **{s}**: [Free YouTube Lectures Yahan Dekhein ↗]({yt_link})")
                    else:
                        st.info("🎉 Shabash! Aap is job profile ke tamam zaroori core skills meet kar rahe hain.")
                
                with tab2:
                    st.write("**Top 6 Relevant Job Matches:**")
                    for role, score in scores[:6]:
                        pct = min(max(int((score + 0.25) * 100), 15), 98)
                        st.write(f"**{role}** — `{pct}%`")
                        st.progress(pct / 100)
                
                with tab3:
                    cover_letter = f"""Dear Hiring Team,

I am writing to express my strong interest in the {top_role} role. Based on my technical background and practical experience, my skills align closely with your core requirements, particularly in {', '.join(found[:3]) if found else 'core domain practices'}.

I am proactive about continuous technical growth and am currently refining advanced workflows in {', '.join(missing[:2]) if missing else 'emerging industry technologies'} to deliver high-impact results for your team.

Thank you for your time and consideration. I look forward to the opportunity to discuss my application further.

Sincerely,
Applicant"""
                    st.text_area("Ready-to-use Tailored Cover Letter", value=cover_letter, height=220)
    else:
        st.info("Resume upload karein aur 'Run AI Analysis' par click karein.")