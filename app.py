%%writefile app.py
import urllib.parse
from PIL import Image
import pypdf
import pytesseract
import gradio as gr
from sentence_transformers import SentenceTransformer, util

# Model load
model = SentenceTransformer('all-MiniLM-L6-v2')

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

def extract_text(file_obj, raw_text_input):
    if raw_text_input and len(raw_text_input.strip()) > 30:
        return raw_text_input.strip()
    if not file_obj:
        return ""
    path = file_obj.name if hasattr(file_obj, 'name') else file_obj
    text = ""
    if path.lower().endswith('.pdf'):
        try:
            reader = pypdf.PdfReader(path)
            for page in reader.pages:
                t = page.extract_text()
                if t: text += t + "\n"
        except Exception:
            pass
    else:
        try:
            text = pytesseract.image_to_string(Image.open(path))
        except Exception:
            pass
    return text.strip()

def analyze_resume(file_obj, raw_text_input):
    cv_text = extract_text(file_obj, raw_text_input)
    if not cv_text or len(cv_text) < 40:
        msg = "<div style='background: #fef2f2; border: 1px solid #f87171; border-radius: 8px; padding: 16px; color: #991b1b;'>⚠️ CV text read nahi ho saka. Clear PDF ya Image upload karein.</div>"
        return msg, {}, "", "", ""

    cv_embedding = model.encode(cv_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(cv_embedding, job_embeddings)[0]

    scores = [(job_titles[i], float(cosine_scores[i])) for i in range(len(job_titles))]
    scores.sort(key=lambda x: x[1], reverse=True)

    top_role, top_val = scores[0]
    top_score_pct = min(int((top_val + 0.28) * 100), 98)

    chart_data = {role: min(int((score + 0.28) * 100), 98) for role, score in scores[:8]}

    target_skills = JOB_ROLES[top_role]
    missing = [skill for skill in target_skills if skill.lower() not in cv_text.lower()]
    found = [skill for skill in target_skills if skill.lower() in cv_text.lower()]

    badge_html = f"""
    <div style='background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px; padding: 20px; color: #ffffff;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='background: #3b82f6; font-size: 11px; padding: 4px 8px; border-radius: 4px; font-weight: 600;'>Primary Recommendation</span>
                <h2 style='margin: 8px 0 0 0; font-size: 22px;'>{top_role}</h2>
            </div>
            <div style='text-align: right;'>
                <span style='font-size: 32px; font-weight: 800; color: #38bdf8;'>{top_score_pct}%</span>
                <div style='font-size: 12px; color: #94a3b8;'>Match Score</div>
            </div>
        </div>
    </div>
    """

    roadmap_html = "<div style='display: flex; flex-direction: column; gap: 8px; margin-top: 10px;'>"
    if missing:
        for skill in missing:
            yt_query = urllib.parse.quote(f"{skill} crash course tutorial")
            yt_link = f"https://www.youtube.com/results?search_query={yt_query}"
            roadmap_html += f"""
            <div style='display: flex; justify-content: space-between; align-items: center; background: #ffffff; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px;'>
                <span style='color: #e11d48; font-weight: 600;'>• Missing: {skill}</span>
                <a href='{yt_link}' target='_blank' style='background: #ef4444; color: white; padding: 4px 10px; border-radius: 4px; font-size: 12px; text-decoration: none;'>Watch on YouTube ↗</a>
            </div>
            """
    else:
        roadmap_html += "<div style='color: #059669;'>Aap is role ke liye mukammal ready hain!</div>"
    roadmap_html += "</div>"

    cover_letter = f"""Dear Hiring Team,

I am writing to express my enthusiasm for the {top_role} role. My practical skills align closely with your core requirements, specifically in {', '.join(found[:3]) if found else 'key domain workflows'}. I am also proactively leveling up in {', '.join(missing[:2]) if missing else 'emerging standards'} to ensure high-impact results.

Sincerely,
Applicant
"""
    return badge_html, chart_data, roadmap_html, cover_letter, cv_text[:500]

custom_css = ".gradio-container { max-width: 1000px !important; margin: auto !important; }"

with gr.Blocks(title="AI Rozgar", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💼 AI Rozgar — Career Matcher")
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.TabItem("📄 Upload File"):
                    cv_file = gr.File(label="Upload Resume", file_types=[".pdf", ".png", ".jpg", ".jpeg"])
                with gr.TabItem("✍️ Text Paste"):
                    cv_raw_text = gr.Textbox(label="Paste CV Text", lines=6)
            analyze_btn = gr.Button("🚀 Run AI Analysis", variant="primary")
            with gr.Accordion("Preview Extracted Text", open=False):
                raw_preview = gr.Textbox(lines=4, interactive=False)

        with gr.Column(scale=6):
            hero_output = gr.HTML(value="<div>Upload resume to view analysis.</div>")
            with gr.Tabs():
                with gr.TabItem("📊 Match %"):
                    chart_output = gr.Label(label="Top Job Match", num_top_classes=6)
                with gr.TabItem("💡 Missing Skills"):
                    roadmap_output = gr.HTML()
                with gr.TabItem("📝 Cover Letter"):
                    cover_letter_box = gr.Textbox(label="Cover Letter", lines=6)

    analyze_btn.click(
        fn=analyze_resume,
        inputs=[cv_file, cv_raw_text],
        outputs=[hero_output, chart_output, roadmap_output, cover_letter_box, raw_preview]
    )

if __name__ == "__main__":
    demo.launch()