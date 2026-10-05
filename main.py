import os
import re
import pdfplumber

folder = "resumes"

skills_list = [
    "python", "java", "c++", "sql", "html", "css", "javascript",
    "react", "machine learning", "data analysis", "excel",
    "git", "django", "flask", "communication", "teamwork"
]

job_description = """
We are looking for a Python developer with knowledge of SQL, Git,
machine learning and good communication skills.
"""


def read_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def analyze(text):
    text_lower = text.lower()
    jd_lower = job_description.lower()

    resume_skills = [s for s in skills_list if s in text_lower]
    jd_skills = [s for s in skills_list if s in jd_lower]

    matched = [s for s in jd_skills if s in resume_skills]
    missing = [s for s in jd_skills if s not in resume_skills]
    score = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0

    email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return {
        "email": email.group(0) if email else "Nahi mila",
        "matched": matched,
        "missing": missing,
        "score": score,
    }


results = []
for file in os.listdir(folder):
    if file.lower().endswith(".pdf"):
        text = read_pdf(os.path.join(folder, file))
        data = analyze(text)
        data["file"] = file
        results.append(data)

results.sort(key=lambda x: x["score"], reverse=True)

print("===== RANKING =====")
for i, r in enumerate(results, start=1):
    print(f"{i}. {r['file']} - {r['score']:.1f}%")
    print("   Email:", r["email"])
    print("   Match hui:", r["matched"])
    print("   Missing:", r["missing"])