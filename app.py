from flask import Flask

from flask_cors import CORS, cross_origin
from flask_cors import CORS
import logging



app = Flask(__name__)



from controller.api_v1 import api_v1

cors = CORS(app, resources={r"/v1.0/*": {"origins": "*"}})
logging.getLogger('flask_cors').level = logging.DEBUG
app.register_blueprint(api_v1, url_prefix='/v1.0')


# @app.before_first_request
# def create_tables():
#     db.create_all()

@app.route('/')
def hello():
    # r.set('foo1', 'bar')
    # print(r.get('foo123'))
    return 'NotBG api server v1.0 up and running'


# @app.teardown_appcontext
# def shutdown_session(exception=None):
#     session.remove()


if __name__ == '__main__':
    # app.run(threaded=True)
    app.run(debug=True)




# from flask import Flask, request, jsonify, redirect, url_for
# import os
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# UPLOAD_FOLDER = 'static/uploads/'
# app.secret_key = "secret key"
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/', methods=['GET'])
# def home():
#     return "Image Upload Service"

# @app.route('/upload_image', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No image selected for uploading"}), 400
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         full_size_image_output = do_mask()
#         file_url = url_for('static', filename='uploads/' + filename, _external=True)
#         return jsonify({"message": "Image successfully uploaded", "file_url": file_url}), 200
#     else:
#         return jsonify({"error": "Allowed image types are - png, jpg, jpeg, gif"}), 400

# if __name__ == "__main__":
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run(debug=True)
    