import os
from dotenv import load_dotenv
from openai import OpenAI
from weasyprint import HTML
import tempfile

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_questions_from_openai(skills):
    """
    Generate 3 technical interview questions per skill using OpenAI's Chat API.
    """
    prompt = f"Generate 3 technical interview questions for each of these skills: {', '.join(skills)}."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in generating technical interview questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        print("OpenAI error:", e)
        return None


def render_questions_to_pdf(questions_dict, username):
    """
    Converts interview questions to a styled PDF using WeasyPrint.
    """
    html = f"<h1>Interview Preparation for {username}</h1><hr>"
    for skill, questions in questions_dict.items():
        html += f"<h2>{skill}</h2><pre>{questions}</pre><br>"

    # Save to a temporary PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    HTML(string=html).write_pdf(temp_file.name)
    return temp_file.name


def generate_portfolio_html_with_ai(name, user_image_url, email, phone, skills, education, experience, projects):
    skill_str = ', '.join(map(str, skills))
    education_str = ', '.join(map(str, education))
    experience_str = ', '.join(map(str, experience))
    projects_str = ', '.join(map(str, projects))
   
    prompt = f"""
Create a clean, responsive personal portfolio website in a single HTML file with embedded CSS.

Details:
Name: {name}
Email: {email}
Phone: {phone}
Image URL: {user_image_url}
Skills: {', '.join(skill_str)}
Education: {', '.join(education_str)}
Experience: {', '.join(experience_str)}
Projects: {', '.join(projects_str)}

Design Requirements:
- Use modern, responsive HTML and inline CSS.
- Include a section for image, skills, education, experience, and projects.
- Use best UI/UX practices.
- Do NOT include any JavaScript.
- Return only valid HTML.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional web designer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=3000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI generation failed: {str(e)}")
