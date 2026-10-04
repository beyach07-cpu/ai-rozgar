import streamlit as st
import urllib.parse
import re
from PIL import Image, ImageOps, ImageEnhance
import pypdf
import pytesseract
from sentence_transformers import SentenceTransformer, util

# Page Configuration
st.set_page_config(
    page_title="AI Rozgar — Career Diagnostic",
    page_icon="💼",
    layout="wide"
)

# -------------------------------------------------
# 1. LOAD MODEL (Cached)
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

job_titles = sorted(list(JOB_ROLES.keys()))
job_descriptions = [f"Job role: {title}. Core competencies: {', '.join(skills)}" for title, skills in sorted(JOB_ROLES.items())]
job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)

# -------------------------------------------------
# 3. TEXT & SKILL EXTRACTION
# -------------------------------------------------
def preprocess_image_for_ocr(image):
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
    image = image.convert('L')
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.8)

def extract_text(file_obj, raw_text):
    if raw_text and len(raw_text.strip()) > 30:
        return raw_text.strip()
    if not file_obj:
        return ""
    text = ""
    file_name = file_obj.name.lower()
    if file_name.endswith('.pdf'):
        try:
            reader = pypdf.PdfReader(file_obj)
            for page in reader.pages:
                c = page.extract_text()
                if c: text += c + "\n"
        except Exception:
            pass
    else:
        try:
            img = Image.open(file_obj)
            processed = preprocess_image_for_ocr(img)
            text = pytesseract.image_to_string(processed, config=r'--oem 3 --psm 6')
        except Exception:
            try:
                text = pytesseract.image_to_string(Image.open(file_obj))
            except Exception:
                pass
    return text.strip()

def check_skill_in_text(skill, text):
    escaped = re.escape(skill.lower())
    pattern = rf'\b{escaped}\b'
    return bool(re.search(pattern, text.lower()))

# -------------------------------------------------
# 4. APP UI (ENGLISH)
# -------------------------------------------------
st.title("💼 AI Rozgar — Intelligent Career Diagnostic")
st.write("Upload your resume to evaluate job alignment across 50 top industry roles, identify skill gaps, and access free learning resources.")

col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.subheader("📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF, PNG, JPG)",
        type=["pdf", "png", "jpg", "jpeg"]
    )
    pasted_text = st.text_area(
        "Or Paste Resume Text Directly",
        height=140,
        placeholder="Paste your resume text here if you prefer not to upload a file..."
    )
    analyze_btn = st.button("🚀 Run AI Analysis", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Career Diagnostic Report")
    
    if analyze_btn:
        with st.spinner("Analyzing resume against 50 industry job profiles..."):
            cv_text = extract_text(uploaded_file, pasted_text)
            
            if not cv_text or len(cv_text) < 30:
                st.error("⚠️ Unable to extract text. Please upload a clear PDF file or paste the plain text directly.")
            else:
                st.session_state["cv_text"] = cv_text

    if "cv_text" in st.session_state:
        cv_text = st.session_state["cv_text"]
        cv_emb = model.encode(cv_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(cv_emb, job_embeddings)[0]
        
        scores_dict = {}
        scores_list = []
        for i, title in enumerate(job_titles):
            val = float(cosine_scores[i])
            pct = min(max(int((val + 0.25) * 100), 20), 98)
            scores_dict[title] = pct
            scores_list.append((title, pct))
            
        scores_list.sort(key=lambda x: x[1], reverse=True)
        top_auto_role, top_auto_pct = scores_list[0]
        
        # Primary Recommendation Banner
        st.success(f"🎯 Primary Recommended Role: **{top_auto_role}** ({top_auto_pct}% Match)")
        
        # Job Selection
        st.markdown("### 🔍 Select a Specific Role to Inspect:")
        selected_job = st.selectbox(
            "Choose a target job profile:",
            options=job_titles,
            index=job_titles.index(top_auto_role)
        )
        
        current_job_pct = scores_dict[selected_job]
        req_skills = JOB_ROLES[selected_job]
        
        found_skills = [s for s in req_skills if check_skill_in_text(s, cv_text)]
        missing_skills = [s for s in req_skills if not check_skill_in_text(s, cv_text)]
        
        st.info(f"Readiness Score for **{selected_job}**: **{current_job_pct}%**")

        tab1, tab2, tab3 = st.tabs([
            f"💡 {selected_job} Missing Skills", 
            "📈 Top Role Match Scores", 
            f"📝 {selected_job} Cover Letter"
        ])
        
        # TAB 1: Skill Roadmap
        with tab1:
            if found_skills:
                st.markdown("**Identified Strengths:**")
                st.markdown(" ".join([f"`✓ {s}`" for s in found_skills]))
            
            st.divider()
            
            if missing_skills:
                st.markdown(f"#### Skills to acquire for a 100% match in **{selected_job}**:")
                for s in missing_skills:
                    yt_query = urllib.parse.quote(f"{s} complete course tutorial for beginners")
                    yt_link = f"https://www.youtube.com/results?search_query={yt_query}"
                    st.markdown(f"- 🔴 **{s}**: [Watch Free Tutorials on YouTube ↗]({yt_link})")
            else:
                st.balloons()
                st.success(f"🎉 Excellent! You have covered all essential core skills for **{selected_job}**.")
        
        # TAB 2: Match Overview
        with tab2:
            st.write("**Top Relevant Role Matches:**")
            for role, pct in scores_list[:10]:
                st.write(f"**{role}** — `{pct}%`")
                st.progress(pct / 100)
                
        # TAB 3: Cover Letter
        with tab3:
            cover_letter = f"""Dear Hiring Team,

I am writing to express my enthusiastic interest in the {selected_job} position. After reviewing your core requirements, I am confident that my technical skills and professional background align strongly with your expectations, particularly in {', '.join(found_skills[:3]) if found_skills else 'core industry workflows'}.

I place a strong emphasis on continuous growth and am currently expanding my capabilities in {', '.join(missing_skills[:2]) if missing_skills else 'advanced architecture and best practices'} to ensure modern, high-impact contributions to your team.

Thank you for your time and consideration. I look forward to the opportunity to discuss how my qualifications align with your objectives.

Sincerely,
Applicant"""
            st.text_area("Tailored Cover Letter (1-Click Ready)", value=cover_letter, height=220)
    else:
        st.info("Upload your resume and click 'Run AI Analysis' to view results.")