import re
import pdfplumber
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

st.markdown("""
<style>
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
.stApp { background: radial-gradient(circle at 20% 0%, #1e1b4b 0%, #0b1020 50%); }
.block-container { padding-top: 1rem; max-width: 1100px; }

.hero { text-align: center; padding: 30px 10px 22px 10px; }
.hero h1 {
  font-size: 3.2rem; font-weight: 800; margin: 0;
  background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #9ca3af; font-size: 1.1rem; margin: 8px 0 14px 0; }
.badge {
  display: inline-block; padding: 6px 14px; margin: 4px; border-radius: 999px;
  background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.4);
  color: #c4b5fd; font-size: 0.85rem; font-weight: 600;
}

.stTextArea textarea {
  background: rgba(255,255,255,0.06) !important; border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.12) !important; color: #e5e7eb !important;
}
[data-testid="stFileUploader"] section {
  background: rgba(255,255,255,0.05); border: 2px dashed #6d5bd0; border-radius: 16px;
}
div.stButton > button {
  background: linear-gradient(90deg, #8b5cf6, #3b82f6); color: white; border: 0;
  border-radius: 14px; padding: 0.75rem 1rem; font-weight: 700; font-size: 1.05rem;
  box-shadow: 0 0 22px rgba(139,92,246,0.45); transition: all 0.2s ease;
}
div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 0 32px rgba(139,92,246,0.7); color: white; }

.stat {
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px; padding: 18px; text-align: center; margin: 10px 0 18px 0;
}
.stat .num { font-size: 2rem; font-weight: 800; color: white; }
.stat .cap { color: #9ca3af; font-size: 0.9rem; }

.glass {
  display: flex; gap: 22px; align-items: flex-start;
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
  border-radius: 20px; padding: 22px; margin-bottom: 16px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}
.ring {
  width: 100px; height: 100px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
}
.ring-in {
  width: 78px; height: 78px; border-radius: 50%; background: #0f1530;
  display: flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 1.3rem; color: white;
}
.info { flex: 1; }
.fname { font-size: 1.2rem; font-weight: 700; color: white; }
.meta { color: #9ca3af; font-size: 0.9rem; margin-bottom: 8px; }
.bar-label { color: #cbd5e1; font-size: 0.82rem; margin-top: 6px; }
.bar-bg { background: rgba(255,255,255,0.10); border-radius: 8px; height: 8px; }
.bar-fill { height: 8px; border-radius: 8px; }
.label { font-weight: 700; color: #e5e7eb; margin-top: 12px; font-size: 0.9rem; }
.chip {
  display: inline-block; padding: 4px 12px; margin: 3px 4px 3px 0;
  border-radius: 999px; font-size: 0.82rem; font-weight: 600;
}
.chip-green { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }
.chip-red { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
</style>
""", unsafe_allow_html=True)

skills_list = [
    "python", "java", "c++", "sql", "html", "css", "javascript",
    "react", "machine learning", "data analysis", "excel",
    "git", "django", "flask", "communication", "teamwork"
]

tips = {
    "python": "Python basics aur ek chhota project banao.",
    "java": "Java OOP concepts aur DSA practice karo.",
    "c++": "C++ mein DSA problems solve karo.",
    "sql": "SQL queries (JOIN, GROUP BY) practice karo.",
    "html": "HTML se ek simple webpage banao.",
    "css": "CSS se webpage ko style karna seekho.",
    "javascript": "JavaScript basics aur DOM seekho.",
    "react": "React se ek chhota app banao.",
    "machine learning": "Scikit-learn se ek ML project banao.",
    "data analysis": "Pandas aur Excel se data analysis seekho.",
    "excel": "Excel formulas aur pivot tables seekho.",
    "git": "Git/GitHub pe apna project upload karo.",
    "django": "Django se ek simple website banao.",
    "flask": "Flask se ek chhoti API banao.",
    "communication": "Resume mein presentations/team projects ka zikr karo.",
    "teamwork": "Group projects ya hackathon ka experience likho.",
}


def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def analyze(text, job_description):
    text_lower = text.lower()
    jd_lower = job_description.lower()

    resume_skills = [s for s in skills_list if s in text_lower]
    jd_skills = [s for s in skills_list if s in jd_lower]

    matched = [s for s in jd_skills if s in resume_skills]
    missing = [s for s in jd_skills if s not in resume_skills]
    skill_score = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0

    email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return {
        "email": email.group(0) if email else "Nahi mila",
        "matched": matched,
        "missing": missing,
        "skill_score": skill_score,
    }


def score_color(score):
    if score >= 70:
        return "#22c55e"
    if score >= 40:
        return "#f59e0b"
    return "#ef4444"


def chips(items, css_class):
    if not items:
        return "<span style='color:#9ca3af'>Koi nahi</span>"
    return "".join(f"<span class='chip {css_class}'>{s}</span>" for s in items)


def ring(score, color):
    return (
        "<div class='ring' style='background:conic-gradient(" + color + " "
        + str(round(min(score, 100))) + "%, rgba(255,255,255,0.10) 0);'>"
        + "<div class='ring-in'>" + f"{score:.0f}%" + "</div></div>"
    )


def mini_bar(label, value, color):
    return (
        "<div class='bar-label'>" + label + f" &nbsp;{value:.0f}%</div>"
        + "<div class='bar-bg'><div class='bar-fill' style='width:"
        + str(round(min(value, 100))) + "%;background:" + color + ";'></div></div>"
    )


def stat(num, cap):
    return "<div class='stat'><div class='num'>" + str(num) + "</div><div class='cap'>" + cap + "</div></div>"


# ---------- Header ----------
st.markdown(
    "<div class='hero'><h1>📄 Resume Analyzer</h1>"
    "<p>Job description se resumes match karo aur best candidate dhundho</p>"
    "<span class='badge'>⚡ TF-IDF Matching</span>"
    "<span class='badge'>🎯 Skill Detection</span>"
    "<span class='badge'>📊 Smart Ranking</span></div>",
    unsafe_allow_html=True,
)

# ---------- Input ----------
left, right = st.columns(2)
with left:
    job_description = st.text_area("📝 Job Description", height=190,
                                   placeholder="Yahan job description paste karo...")
with right:
    files = st.file_uploader("📎 Resume PDFs upload karo", type="pdf",
                             accept_multiple_files=True)

analyze_clicked = st.button("🚀 Analyze Resumes", use_container_width=True)

# ---------- Analysis ----------
if analyze_clicked:
    if not files or not job_description.strip():
        st.warning("Job description aur kam se kam ek resume zaroor daalo.")
    else:
        texts = [read_pdf(f) for f in files]

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([job_description] + texts)
        similarities = cosine_similarity(matrix[0:1], matrix[1:])[0]

        results = []
        for f, text, sim in zip(files, texts, similarities):
            data = analyze(text, job_description)
            data["file"] = f.name
            data["text_score"] = sim * 100
            data["score"] = 0.7 * data["skill_score"] + 0.3 * data["text_score"]
            results.append(data)

        results.sort(key=lambda x: x["score"], reverse=True)
        st.session_state["results"] = results

# ---------- Results ----------
results = st.session_state.get("results")

if results:
    avg = sum(r["score"] for r in results) / len(results)
    c1, c2, c3 = st.columns(3)
    c1.markdown(stat(len(results), "Total Resumes"), unsafe_allow_html=True)
    c2.markdown(stat(f"{results[0]['score']:.1f}%", "Best Score"), unsafe_allow_html=True)
    c3.markdown(stat(f"{avg:.1f}%", "Average Score"), unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🏆 Ranking", "📊 Chart & Download"])

    with tab1:
        medals = ["🥇", "🥈", "🥉"]
        for i, r in enumerate(results, start=1):
            medal = medals[i - 1] if i <= 3 else f"#{i}"
            color = score_color(r["score"])
            card = (
                "<div class='glass'>"
                + ring(r["score"], color)
                + "<div class='info'>"
                + f"<div class='fname'>{medal} {r['file']}</div>"
                + f"<div class='meta'>📧 {r['email']}</div>"
                + mini_bar("Skills match", r["skill_score"], "#8b5cf6")
                + mini_bar("Text similarity", r["text_score"], "#3b82f6")
                + "<div class='label'>✅ Match hui skills</div>" + chips(r["matched"], "chip-green")
                + "<div class='label'>❌ Missing skills</div>" + chips(r["missing"], "chip-red")
                + "</div></div>"
            )
            st.markdown(card, unsafe_allow_html=True)

            if r["missing"]:
                with st.expander(f"💡 {r['file']} ke liye suggestions"):
                    for s in r["missing"]:
                        st.write(f"- **{s}**: {tips.get(s, 'Is skill ko seekho.')}")

    with tab2:
        chart_data = pd.DataFrame(
            {"Score": [r["score"] for r in results]},
            index=[r["file"] for r in results],
        )
        st.bar_chart(chart_data)

        df = pd.DataFrame([
            {
                "Rank": i,
                "File": r["file"],
                "Email": r["email"],
                "Final Score": round(r["score"], 1),
                "Skills Score": round(r["skill_score"], 1),
                "Text Similarity": round(r["text_score"], 1),
                "Matched": ", ".join(r["matched"]),
                "Missing": ", ".join(r["missing"]),
            }
            for i, r in enumerate(results, start=1)
        ])
        st.download_button(
            "⬇️ Result CSV download karo",
            df.to_csv(index=False).encode("utf-8"),
            "resume_ranking.csv",
            "text/csv",
        )