import streamlit as st
import urllib.parse
from PIL import Image
import pypdf
import pytesseract
from sentence_transformers import SentenceTransformer, util

# Page Config
st.set_page_config(page_title="AI Rozgar - Career Matcher", page_icon="💼", layout="wide")

# Cache model loading for fast performance
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Jobs Database
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
    "Game Developer": ["Unity", "C#", "3D Math", "Game Physics", "Shader Programming"],
    "Digital Marketing Specialist": ["SEO", "Google Ads", "Content Marketing", "Google Analytics"],
    "SEO Specialist": ["Keyword Research", "On-Page SEO", "Backlinks", "Technical SEO", "Ahrefs"],
    "HR Recruiter": ["Talent Acquisition", "Technical Screening", "Interviewing", "HRIS"],
    "Financial Analyst": ["Financial Modeling", "Advanced Excel", "Valuation", "Forecasting"],
    "Network Engineer": ["Cisco CCNA", "Routing & Switching", "TCP/IP", "DNS/DHCP", "VPN"],
    "Data Engineer": ["Apache Spark", "Kafka", "SQL", "ETL Pipelines", "Airflow"],
    "Prompt Engineer": ["Prompt Design", "LangChain", "Vector Databases", "Model Evaluation"]
}

job_titles = list(JOB_ROLES.keys())
job_descriptions = [f"Job role: {title}. Core competencies: {', '.join(skills)}" for title, skills in JOB_ROLES.items()]
job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)

def extract_text(file_obj, raw_text):
    if raw_text and len(raw_text.strip()) > 30:
        return raw_text.strip()
    if not file_obj:
        return ""
    text = ""
    if file_obj.name.lower().endswith('.pdf'):
        try:
            reader = pypdf.PdfReader(file_obj)
            for page in reader.pages:
                t = page.extract_text()
                if t: text += t + "\n"
        except Exception:
            pass
    else:
        try:
            image = Image.open(file_obj)
            text = pytesseract.image_to_string(image)
        except Exception:
            pass
    return text.strip()

# UI Layout
st.title("💼 AI Rozgar — Intelligent Career Diagnostic")
st.write("Apna CV upload karein aur 50 jobs ke sath match, missing skills ke YouTube lectures aur 1-click cover letter payein.")

col1, col2 = st.columns([1, 1.3], gap="medium")

with col1:
    st.subheader("📄 Upload CV")
    uploaded_file = st.file_uploader("Upload Resume (PDF ya Photo)", type=["pdf", "png", "jpg", "jpeg"])
    pasted_text = st.text_area("Ya Direct Text Paste Karein", height=150, placeholder="Paste CV text here...")
    analyze_btn = st.button("🚀 Run AI Analysis", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Career Diagnostic Report")
    if analyze_btn:
        with st.spinner("Analyzing resume against industry roles..."):
            cv_text = extract_text(uploaded_file, pasted_text)
            
            if not cv_text or len(cv_text) < 30:
                st.error("⚠️ Text read nahi ho saka. Barah-e-karam clear PDF/Image upload karein ya direct text paste karein.")
            else:
                cv_emb = model.encode(cv_text, convert_to_tensor=True)
                cosine_scores = util.cos_sim(cv_emb, job_embeddings)[0]
                
                scores = [(job_titles[i], float(cosine_scores[i])) for i in range(len(job_titles))]
                scores.sort(key=lambda x: x[1], reverse=True)
                
                top_role, top_val = scores[0]
                top_pct = min(int((top_val + 0.28) * 100), 98)
                
                # Hero Card
                st.success(f"### Top Match: **{top_role}** ({top_pct}% Ready)")
                
                tab1, tab2, tab3 = st.tabs(["💡 Missing Skills & Lectures", "📊 Match % (Top Roles)", "📝 1-Click Cover Letter"])
                
                with tab1:
                    target_skills = JOB_ROLES[top_role]
                    missing = [skill for skill in target_skills if skill.lower() not in cv_text.lower()]
                    found = [skill for skill in target_skills if skill.lower() in cv_text.lower()]
                    
                    if found:
                        st.markdown("**Existing Strengths Detected:** " + ", ".join([f"`{s}`" for s in found]))
                    
                    if missing:
                        st.markdown("#### Ye skills seekhein 100% match ke liye:")
                        for skill in missing:
                            yt_link = f"https://www.youtube.com/results?search_query={urllib.parse.quote(skill + ' crash course tutorial')}"
                            st.markdown(f"- 🔴 **{skill}**: [YouTube Free Lectures Yahan Dekhein ↗]({yt_link})")
                    else:
                        st.info("🎉 Shabash! Aap is role ke tamam core technical skills meet kar rahe hain.")
                
                with tab2:
                    for role, score in scores[:6]:
                        pct = min(int((score + 0.28) * 100), 98)
                        st.write(f"**{role}** ({pct}%)")
                        st.progress(pct / 100)
                
                with tab3:
                    cover_letter = f"""Dear Hiring Team,

I am writing to express my strong enthusiasm for the {top_role} role. My practical skills align closely with your core requirements, specifically in {', '.join(found[:3]) if found else 'key domain workflows'}. 

I am also actively sharpening my expertise in {', '.join(missing[:2]) if missing else 'emerging standards'} to ensure high-impact results for your engineering team.

Sincerely,
Applicant"""
                    st.text_area("Ready-to-use Cover Letter", value=cover_letter, height=200)
    else:
        st.info("CV upload karein aur 'Run AI Analysis' dabayein.")