# Import flask and other modules
from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename # Import this module for file handling
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, EqualTo
import os

import numpy as np
from keras.models import load_model
import librosa
import requests
import time
import wave
from tempfile import NamedTemporaryFile
from datetime import datetime

from flask_socketio import SocketIO, emit
from threading import Thread, Event




# Create app instance
app = Flask(__name__)

# Configure app secret key and database URI
app.config['SECRET_KEY'] = 'some-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads' # Create an upload folder
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

socketio = SocketIO(app)


# Create database instance
db = SQLAlchemy(app)

# Create login manager instance and initialize it with app
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create User model that inherits from UserMixin
class User(UserMixin, db.Model):
    # Define table name and columns
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), unique=True, nullable=False)
    profile_pic = db.Column(db.String(200), default='default.jpg')

    # Define a method to return a user object by id
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

# Creating Question model
class Question(db.Model):
    __tablename__ = 'questions'
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    author_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    author = db.relationship('User', backref='questions')

# Creating the Replies Model
class Reply(db.Model):
    __tablename__ = 'replies'
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    author_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)

    author = db.relationship('User', backref='replies')
    question = db.relationship('Question', backref='replies')

class Link(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    esp32_api = db.Column(db.String(255), nullable=False)


# create the database tables if they do not exist
with app.app_context():
    db.create_all()

# Create a form class for registration
class RegisterForm(FlaskForm):
    # Define the form fields and validators
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    email = StringField('Email', validators=[DataRequired()])
    phone = StringField('Phone', validators=[DataRequired()])

# Create a form class for login
class LoginForm(FlaskForm):
    # Define the form fields and validators
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])


# Define a route for the home page
@app.route('/')
def home():
    return render_template('home.html', user=current_user)

# Define a route for the register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Create an instance of the register form
    form = RegisterForm()

    # If the form is validated on submission, get the form data
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        email = form.email.data
        phone = form.phone.data

        # Check if the username or email or phone already exists in the database
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already taken.')
            return redirect(url_for('register'))
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already registered.')
            return redirect(url_for('register'))
        
        user = User.query.filter_by(phone=phone).first()
        if user:
            flash('Phone number already registered.')
            return redirect(url_for('register'))

        # If not, create a new user object with hashed password and add it to the database 
        # Use default picture for new users 
        user = User(username=username, password=generate_password_hash(password), email=email, phone=phone)
        db.session.add(user)
        db.session.commit()

        # Log in the user and redirect to the account page 
        login_user(user)
        flash('Registration successful.')
        return redirect(url_for('account'))
    
    # If the request method is GET or the form is not validated, render the register template with the form object 
    return render_template('register.html', form=form)

# Define a route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    # Create an instance of the login form
    form = LoginForm()

    # If the form is validated on submission, get the form data
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        # Check if the username exists in the database and the password matches
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            # Log in the user and redirect to the account page
            login_user(user)
            # Set the "username" key in the session object
            session["username"] = username
            flash('Login successful.')
            return redirect(url_for('account'))

        # If not, flash an error message and redirect to the login page
        flash('Invalid username or password.')
        return redirect(url_for('login'))

    # If the request method is GET or the form is not validated, render the login template with the form object
    return render_template('login.html', form=form)


# Define a route for the update page
@app.route('/update', methods=['GET', 'POST'])
@login_required  # Require the user to be logged in to access this route
def update():
    # Create an instance of the register form
    form = RegisterForm()
    # If the request method is GET, render the update template with current user and form data
    if request.method == 'GET':
        return render_template('update.html', user=current_user, form=form)
    # Pass both the user and form objects
    # If request method is POST get data from forms
    username = request.form.get("username")
    password = request.form.get("password")
    email = request.form.get("email")
    phone = request.form.get("phone")
    profile_pic = request.files.get("profile_pic")
    # Get profile picture file from request.files dictionary
    # Check if any of fields are changed and update them in database accordingly
    if username != current_user.username:
        user = User.query.filter_by(username=username).first()
        if user:
            flash("Username already taken.")
            return redirect(url_for("update"))
        else:
            current_user.username = username
    if password != "":
        current_user.password = generate_password_hash(password)
    if email != current_user.email:
        user = User.query.filter_by(email=email).first()
        if user:
            flash("Email already registered.")
            return redirect(url_for("update"))
        else:
            current_user.email = email
    if phone != current_user.phone:
        user = User.query.filter_by(phone=phone).first()
        if user:
            flash("Phone number already registered.")
            return redirect(url_for("update"))
        else:
            current_user.phone = phone
    if profile_pic:
        # Check if there is a profile picture file uploaded
        old_profile_pic_path = os.path.join(app.config['UPLOAD_FOLDER'], current_user.profile_pic)
        # Check if the old profile picture file exists before trying to remove it
        if os.path.exists(old_profile_pic_path):
            os.remove(old_profile_pic_path)
        current_user.profile_pic = secure_filename(profile_pic.filename)
        profile_pic.save(os.path.join(app.config['UPLOAD_FOLDER'], current_user.profile_pic))
    # Commit changes to database
    db.session.commit()
    flash("Update successful.")
    return redirect(url_for("account"))


# Define a route for account page 
@app.route('/account')
@login_required # Require user to be logged in to access this route 
def account():
    link = Link.query.order_by(Link.id.desc()).first()
    if link:
        esp32_api = link.esp32_api
    else:
        esp32_api = ''
    return render_template("account.html",user=current_user, esp32_api=esp32_api)

@app.route('/link', methods=['GET', 'POST'])
@login_required
def link():
    if request.method == 'POST':
        esp32_api = request.form['esp32_api']
        session['esp32_api'] = esp32_api
        print(esp32_api)

    return render_template("account.html", user=current_user)

# Websocket socketio functions
thread = None
stop_event = Event()

def send_hello(esp32_api):
    while not stop_event.is_set():
        
            # Folder to save audio files
        AUDIO_FOLDER = "E:/DENNIS/Machine learning/Machine Learning Model Deployment/FYP Model deployment/received audio"

        ESP32_API = esp32_api

        # Send a GET request to ESP32 API endpoint
        response = requests.get(ESP32_API)
        # Check if the response is OK
        if response.status_code == 200:
            # Get the binary content of the response
            audio_data = response.content
            # Get the suggested filename from the content disposition header
            filename = response.headers.get("content-disposition").split("=")[-1]
            # Create the full path for saving the file
            filepath = AUDIO_FOLDER + "/" + filename
            # Append a timestamp to the filename to make it unique so as to save a new file whenever the server runs
            filename = filename[:-4] + "_" + str(int(time.time())) + ".wav"
            # Create the full path for saving the file
            filepath = AUDIO_FOLDER + "/" + filename

            print("Receiving audio file from ESP32")

            # Open the wav file in write binary mode
            with wave.open(filepath, "wb") as out_f:
                # Set the parameters of the wav file
                out_f.setnchannels(2) # Stereo channel
                out_f.setsampwidth(2) # 16-bit sample width
                out_f.setframerate(4500) # sample rate
                # Write the data to the wav file
                out_f.writeframesraw(audio_data)

                # Print a success message
                print("Conversion done!")
                print("Now classifying the audio file.....")


                    #loading the saved ANN model
            model = load_model('Trained ANN model/FYP_Beehivemodel.hdf5')

            audio, sample_rate = librosa.load(filepath, res_type='kaiser_fast')
            mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
                #Reshape MFCC feature to 2-D array since its one file
            mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
                #Getting the predicted label for each input sample.
            x_predict=model.predict(mfccs_scaled_features)
            predicted_label=np.argmax(x_predict,axis=1)

            if predicted_label == 0:
                classification = 'ABSENT'
                print("Queen Bee is ABSENT")

                ESP32_PhoneCall = 'http://192.168.4.1/phonecall'
                response = requests.get(ESP32_PhoneCall)

            else:
                classification = 'PRESENT'
                print("Queen Bee is PRESENT")

            socketio.emit('my response', {'data': ''})
            time.sleep(2)
            socketio.emit('my response', {'data': classification})
            time.sleep(2)

        else:
            socketio.emit('my response', {'data': ''})
            time.sleep(2)
            socketio.emit('my response', {'data': "Failed to get audio from ESP32"})
            time.sleep(2)
            # Return an error message
            return f"Failed to get audio from ESP32: {response.status_code}"
            
        print("\nRecording the audio with ESP32")


@socketio.on('connect')
def handle_connect():
    global thread
    if thread is not None:
        stop_event.set()
        thread.join()
        stop_event.clear()
    print("Handling connect")
    esp32_api = session.get('esp32_api')
    thread = Thread(target=send_hello, args=(esp32_api,))
    thread.start()



@app.route('/community')
@login_required
def community():
    # Fetch all questions from the database
    questions = Question.query.order_by(Question.timestamp.desc()).all()
    return render_template('community.html', user=current_user, questions=questions)

@app.route('/post_question', methods=['GET', 'POST'])
@login_required
def post_question():
    if request.method == 'POST':
        question = request.form['question']
        # Create a new Question object and add it to the database
        new_question = Question(content=question, author=current_user)
        db.session.add(new_question)
        db.session.commit()
        flash('Question posted successfully.')
        return redirect(url_for('community'))
    return render_template('post_question.html', user=current_user)

# Route to edit questions
@app.route('/edit_question/<int:question_id>', methods=['GET', 'POST'])
@login_required
def edit_question(question_id):
    # Fetch the question from the database
    question = Question.query.get_or_404(question_id)
    # Check if the current user is the author of the question
    if current_user != question.author:
        flash('You are not authorized to edit this question.')
        return redirect(url_for('community'))
    if request.method == 'POST':
        # Update the question content and save changes to the database
        question.content = request.form['question']
        db.session.commit()
        flash('Question updated successfully.')
        return redirect(url_for('community'))
    return render_template('edit_question.html', question=question, user=current_user)

# Route to display a specific question and its replies:
@app.route('/question/<int:question_id>')
@login_required
def question(question_id):
    # Fetch the question from the database
    question = Question.query.get_or_404(question_id)
    # Fetch the replies to the question from the database indescending order
    replies = Reply.query.filter_by(question_id=question_id).order_by(Reply.timestamp.desc()).all()
    return render_template('question.html', question=question, replies=replies, user=current_user)

# Route for adding reply
@app.route('/add_reply/<int:question_id>', methods=['POST'])
@login_required
def add_reply(question_id):
    # Get the reply content from the form
    content = request.form['reply']
    # Create a new Reply object
    reply = Reply(content=content, author=current_user, question_id=question_id)
    # Add the new reply to the database
    db.session.add(reply)
    db.session.commit()
    # Redirect back to the question page
    return redirect(url_for('question', question_id=question_id))


@app.route("/about")
@login_required
def about():
    return render_template("about.html", user=current_user)


@app.route("/own_classification", methods=['GET', 'POST'])
@login_required
def own_classification():
    if request.method == 'POST':
        nchannels = int(request.form['nchannels'])
        sampwidth = int(request.form['samplewidth'])
        framerate = int(request.form['samplerate'])
        audiosample = request.files['audiosample']
        
        # Save the uploaded file to a temporary location
        with NamedTemporaryFile(delete=False) as temp:
            audiosample.save(temp.name)
        
        # Open the temporary file in read binary mode
        with open(temp.name, "rb") as inp_f:
            data = inp_f.read()
        
        # Delete the temporary file
        os.unlink(temp.name)
        
        with wave.open("converted/output.wav", "wb") as out_f:
            out_f.setnchannels(nchannels)
            out_f.setsampwidth(sampwidth)
            out_f.setframerate(framerate)
            out_f.writeframesraw(data)        

                #loading the saved ANN model
        model = load_model('Trained ANN model/FYP_Beehivemodel.hdf5')
        filepath = "converted/output.wav" 

        audio, sample_rate = librosa.load(filepath, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
            #Reshape MFCC feature to 2-D array since its one file
        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
            #Getting the predicted label for each input sample.
        x_predict=model.predict(mfccs_scaled_features)
        predicted_label=np.argmax(x_predict,axis=1)

    if predicted_label == 0:
        classification = 'ABSENT'
        print("Queen Bee is ABSENT")

    else:
        classification = 'PRESENT'
        print("Queen Bee is PRESENT")

    return render_template('own_classification.html', prediction=classification,  user=current_user)
    sys.exit()



# The user will be asked to confirm if they want log out after clicking on logout  
@app.route("/logout")
def logout():
   # Use a script tag to embed JavaScript code  
   return """
   <script>
   // Ask them to confirm if they want to log out  
   var answer=confirm ("Are you sure you want to log out?");
   // If they click OK , remove their name from their session and redirect them to home page  
   if (answer) {
     window.location.href="/logout_confirm";
   }
   // If they click Cancel , stay on current page  
   else {
     window.history.back();
   }
   </script>
   """

@app.route("/logout_confirm")
def logout_confirm():
   # Remove their name from their session  
   session.pop ("username" , None)
   flash ("You have logged out")
   return redirect (url_for ("home"))

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=3000, debug=True)
