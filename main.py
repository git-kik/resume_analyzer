from flask import Flask, render_template, request,redirect, url_for, flash, Response
import os
from werkzeug.utils import secure_filename
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager,login_user, login_required, logout_user, current_user
from models import db, User, Resume, ResumeMatch
from dotenv import load_dotenv
import csv
import io
from flask import render_template, redirect, url_for, flash
from flask_login import login_user
from forms import RegistrationForm, LoginForm
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from flask_login import current_user
from flask_admin import expose
from flask_admin import helpers as admin_helpers
from flask_admin.menu import MenuLink
from get_text import *
import requests


load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///resume_analyzer.db'
app.config['UPLOAD_FOLDER'] = 'upload_resume/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit


db.init_app(app)

try:
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully.")
except Exception as e:
    print(f"❌ Error creating DB tables: {e}")


admin = Admin(app, name='Admin Dashboard', template_mode='bootstrap3')


class AdminModelView(ModelView):
    def is_accessible(self):
        return current_user.is_authenticated and current_user.role == 'admin'

    def inaccessible_callback(self, name, **kwargs):
        # Redirect to login page if user doesn't have access
        return redirect(url_for('login'))

admin.add_view(AdminModelView(User, db.session, name='Admin', endpoint='user_admin'))
admin.add_link(MenuLink(name='Logout', category='', url='/logout'))



login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))



## Loading models ##################
svc_classifier_categorization = pickle.load(open('model/svc_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('model/tfidf_vectorizer_categorization.pkl', 'rb'))
knn_classifier_job_recommendation = pickle.load(open('model/knn_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('model/tfidf_vectorizer_job_recommendation.pkl', 'rb'))



# Clean resume==========================================================================================================
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText



# Prediction and Category Name
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = svc_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

# Prediction and Category Name
def job_recommendation(resume_text):
    resume_text= cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = knn_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job



os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('front_page.html')

@app.route('/home')
def home():
    return render_template('home.html')




@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data, role=form.role.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please login.')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for('home') if user.role == 'jobseeker' else url_for('dashboard'))
        else:
            flash('Invalid email or password.')
    return render_template('login.html', form=form)


@app.route('/dashboard')
@login_required
def dashboard():
    user_role = current_user.role 
    return render_template('dashboard.html', user_role=user_role)



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))



def fetch_learn_catalog():
    url = "https://learn.microsoft.com/api/catalog/"
    params = {"locale": "en-us"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch catalog data. Status code: {response.status_code}")
        return None


def recommend_courses_for_skills(missing_skills, catalog_data):
    recommended_items = []
    courses = catalog_data.get('courses', [])
    certifications = catalog_data.get('certifications', [])

    # Check for courses matching missing skills
    for skill in missing_skills:
        # for course in courses:
        #     title = course.get('title', '').lower()
        #     if skill.lower() in title:
        #         recommended_items.append({
        #             'name': course.get('title', 'No Title'),
        #             'url': course.get('url', '#')
        #         })
        #         break  

        # Check for certifications matching missing skills
        for certification in certifications:
            title = certification.get('title', '').lower()
            if skill.lower() in title:
                recommended_items.append({
                    'name': certification.get('title', 'No Title'),
                    'url': certification.get('url', '#')
                })
                break  

    return recommended_items


@app.route('/result', methods=['POST'])
@login_required
def result():
    if current_user.role != 'jobseeker':
        return "Access denied", 403
    
    resume = request.files['resume']
    job_desc = request.form['job_description']

    filename = secure_filename(resume.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    resume.save(path)

    resume_text = extract_text(path)

    category = predict_category(resume_text)
    job = job_recommendation(resume_text)

    extracted_name = extract_name_from_resume(resume_text)
    extracted_education = extract_education_from_resume(resume_text)
    extracted_email = extract_email_from_resume(resume_text)
    extracted_phone = extract_contact_number_from_resume(resume_text)
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc)
    matched = set(resume_skills) & set(job_skills)
    missing = set(job_skills) - set(resume_skills)

    tfidf = TfidfVectorizer().fit_transform([job_desc, resume_text])
    cosine_score = round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)

    total_skills = len(job_skills) if len(job_skills) > 0 else 1  # Avoid division by zero
    skill_data = [{"skill": skill, "percentage": round(100 / total_skills, 2)} for skill in matched]

    # Fetch  data from  learn microsoft catalog api
    catalog_data = fetch_learn_catalog()

    recommended_courses = recommend_courses_for_skills(missing, catalog_data)

    try:
        resume = Resume(
            user_id=current_user.id,
            resume_name = filename,
            name=extracted_name,
            email=extracted_email,
            phone=extracted_phone,
            skills=",".join(resume_skills),
            source='jobseeker',
            education=", ".join(extracted_education) if isinstance(extracted_education, list) else extracted_education,
            job_catg_pred = category,
            job_recommendation = job,
            similarity_score = cosine_score
        )
        db.session.add(resume)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error: {e}")


    return render_template("result.html",
                           category=category,
                           job=job,
                           name=extracted_name,
                           education=", ".join(extracted_education) if isinstance(extracted_education, list) else extracted_education,
                           email=extracted_email,
                           phone=extracted_phone,
                           matched_skills=matched,
                           missing_skills=missing,
                           cosine_score=cosine_score,
                           resume_filename=filename,
                           skill_data=skill_data,
                           recommended_courses = recommended_courses)

        
@app.route('/view_resumes')
@login_required
def view_resumes():
    if current_user.role == 'hr' or 'admin':
        resumes = Resume.query.all()
        return render_template('resumes.html', resumes=resumes,user_role=current_user.role)
    else:
        return "Unauthorized", 403




@app.route('/download_resumes_csv')
@login_required
def download_resumes_csv():
    if current_user.role != 'hr':
        return "Unauthorized", 403

    # Query all resumes
    resumes = Resume.query.all()

    # Create an in-memory output file for the CSV data
    output = io.StringIO()
    writer = csv.writer(output)

    # Write the header row
    writer.writerow(['Resume Name', 'Name', 'Email', 'Phone', 'Skills', 'Education', 'Similarity Score'])

    # Write data rows
    for resume in resumes:
        writer.writerow([
            resume.resume_name,
            resume.name,
            resume.email,
            resume.phone,
            resume.skills,
            resume.education,
            resume.similarity_score
        ])

    # Prepare the response with appropriate headers
    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=resumes.csv'
        }
    )



@app.route('/matcher', methods=['GET', 'POST'])
@login_required
def matcher():
    if current_user.role != 'hr':
        return "Access denied", 403

    if request.method == 'GET':
        return render_template("matchresume.html", show_form=True)

    job_desc = request.form['job_description']
    resumes = request.files.getlist('resumes')

    if len(resumes) == 0:
        return render_template("matchresume.html", error="Please upload at least 2 resumes.")

    resume_texts = []
    resume_names = []
    resume_records = []

    # Step 1: Save resumes and extract data
    for resume in resumes:
        filename = secure_filename(resume.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume.save(path)
        text = extract_text(path)

        resume_texts.append(text)
        resume_names.append(filename)

        category = predict_category(text)
        job = job_recommendation(text)

        extracted_name = extract_name_from_resume(text)
        extracted_education = extract_education_from_resume(text)
        extracted_email = extract_email_from_resume(text)
        extracted_phone = extract_contact_number_from_resume(text)
        resume_skills = extract_skills(text)

        resume_record = Resume(
            user_id=current_user.id,
            resume_name=filename,
            name=extracted_name,
            email=extracted_email,
            phone=extracted_phone,
            skills=",".join(resume_skills),
            education=", ".join(extracted_education) if isinstance(extracted_education, list) else extracted_education,
            source='hr/admin',
            job_catg_pred=category,
            job_recommendation=job
        )
        db.session.add(resume_record)
        resume_records.append(resume_record)

    db.session.commit()  # Commit all resumes first

    # Step 2: Compute similarity scores
    all_docs = [job_desc] + resume_texts
    vectorizer = TfidfVectorizer().fit_transform(all_docs)
    vectors = vectorizer.toarray()

    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    similarities = cosine_similarity([job_vector], resume_vectors)[0]

    top_results = []

    # Step 3: Save similarity scores and prepare results
    for i, score in enumerate(similarities):
        match_record = ResumeMatch(
            resume_id=resume_records[i].id,
            job_description=job_desc,
            similarity_score=round(score * 100, 2)
        )
        db.session.add(match_record)
        top_results.append((resume_names[i], round(score * 100, 2)))

    db.session.commit()

    # Step 4: Sort results by similarity score
    top_results.sort(key=lambda x: x[1], reverse=True)

    return render_template("matchresume.html", top_results=top_results, show_form=False)




if __name__ == '__main__':
    app.run(debug=True)




