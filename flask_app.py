import base64
import cv2
import os
import numpy as np
from flask import Flask, render_template, send_from_directory, url_for, request
from flask_uploads import UploadSet, DOCUMENTS, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from LayoutXlm_extraction import initialize_layout_model, initialize_layout_processor, initialize_easyocr_detector, whole_pipeline, initialize_yolo_text_detector, initialize_mcrnn_recognition
from yolo_obb_pipeline import initialize_yolo_obb


# yolo text detection download fix >:)
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


app = Flask(__name__)
app.config['UPLOADED_DOCUMENTS_DEST'] = 'docs'
app.config['SECRET_KEY'] = 'ac90825480812ea4e02bc39aceaa80e1'

documents = UploadSet('documents', DOCUMENTS+IMAGES)
configure_uploads(app, documents)

# crutch
layout_model, yolo_model, layout_processor = None, None, None


class UploadForm(FlaskForm):
    doc = FileField(
        validators=[
            FileAllowed(documents, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')


@app.route('/docs/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_DOCUMENTS_DEST'], filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if request.method == "GET":
        return render_template('index.html', form=form)

    if request.method == "POST":
        if form.validate_on_submit():
            filename = documents.save(form.doc.data)
            file_url = url_for('get_file', filename=filename)

            # receipt data extraction
            results = whole_pipeline(layout_model, layout_processor, text_detection_model, text_recognition_model, yolo_obb_model, filename)
            # parsing annotated images to base64 for html
            for res in results:
                res['annotated_image'] = convert_pil_to_base64(res['annotated_image'])

            return render_template('result.html', form=form, filename=filename, file_url=file_url,
                                   results_list=results)
        return 400


def convert_pil_to_base64(pil_image):
    img_data = cv2.imencode('.jpg', np.asarray(pil_image))[1].tostring()
    img_data = base64.b64encode(img_data)
    img_data = img_data.decode()
    return img_data


if __name__ == '__main__':
    layout_model = initialize_layout_model()
    yolo_obb_model = initialize_yolo_obb('yolov8-obb_checkpoint/best_nano.pt')
    layout_processor = initialize_layout_processor()
    # ocr_engine = initialize_easyocr_detector()
    text_detection_model = initialize_yolo_text_detector()
    text_recognition_model = initialize_mcrnn_recognition()
    print('Model and Processor are initialized')

    #app.run(debug=True)
    # test image for debugging
    for image_name in os.listdir('docs'):
        whole_pipeline(layout_model, layout_processor, text_detection_model, text_recognition_model, yolo_obb_model,
                       image_name)
        break


