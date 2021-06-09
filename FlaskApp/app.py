
import os
from flask import Flask,render_template, flash, request, redirect, url_for
from flask.helpers import send_file, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = './files/users'
ALLOWED_EXTENSIONS = {'zip', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    print(filename,filename.rsplit('.', 1)[1].lower())
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
@app.route("/home")
def index():
    return render_template('sample/index.html', number="8")

@app.route('/send', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
            f.save(os.path.join('./files/users', secure_filename(f.filename)))
            return send_file(os.path.join('./files/users', secure_filename(f.filename)), as_attachment=True)
    return 'Something went wrong'

    # print("enter")
    # print(request.method)
    # print(request)
    # if request.method == 'POST':
    #     # check if the post request has the file part
    #     if 'file' not in request.files:
    #         print("enter")
    #         flash('No file part')
    #         return redirect(request.url)
    #     file = request.files['file']
    #     # If the user does not select a file, the browser submits an
    #     # empty file without a filename.
    #     if file.filename == '':
    #         flash('No selected file')
    #         return redirect(request.url)
    #     if file and allowed_file(file.filename):
    #         print("enter")
    #         flash("entered")
    #         filename = secure_filename(file.filename)
    #         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #         return redirect(url_for('download_file', name=filename))
    # return '''
    # <!doctype html>
    # <title>Upload new File</title>
    # <h1>Upload new File</h1>
    # <form method=post enctype=multipart/form-data>
    #   <input type=file name=file>
    #   <input type=submit value=Upload>
    # </form>
    # '''