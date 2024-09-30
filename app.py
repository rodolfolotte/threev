import io
import numpy as np
import base64

from PIL import Image
from flask import Flask
from flask import request
from flask import jsonify
from flask import send_file
from ultralytics import YOLO
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
cors = CORS(app)

UPLOAD_FOLDER = '/usr/src/app/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'gif'}

app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = YOLO('/usr/src/app/yolov8m-seg.pt')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return ("Hello World! ThreeV exam! 3 This solution was organized and coded "
            "by Rodolfo G. Lotte in agreement to the ThreeV candidate process.")

@app.route('/test', methods=['POST','PUT'])
def test():
    file = request.files['file']
    filename = secure_filename(file.filename)
    return filename

@app.route('/img2grayscale', methods=['POST'])
def img2grayscale():
    try:
        if 'file' not in request.files:
            return 'No image found', 400

        file = request.files['file']
        print(file.content_type)
        print(file.filename)

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #     return redirect(url_for('uploaded_file', filename=filename))

        img = Image.open(file)
        processed_img = img.convert('L')

        # b = base64.b64decode(processed_img.encode('utf-8'))
        img_byte_arr = io.BytesIO()
        img_byte_arr.seek(0)
        # img_byte_arr = img_byte_arr.getvalue()
        processed_img.save(img_byte_arr, format='PNG')
        return send_file(img_byte_arr,
                         mimetype='image/png',
                         as_attachment=True)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

        img_np = np.array(img)

        results = model.predict(source=img_np, save=False)
        if not hasattr(results[0], 'masks'):
            return jsonify({'error': 'No masks found in the prediction'}), 400

        masks = results[0].masks.data.cpu().numpy()

        for mask in masks:
            mask_resized = np.stack([mask, mask, mask], axis=-1) * 255
            mask_resized = mask_resized.astype(np.uint8)

            img_np[mask_resized[:, :, 0] > 0] = [255, 0, 0]

        img_with_masks = Image.fromarray(img_np)
        img_io = io.BytesIO()
        img_with_masks.save(img_io, 'PNG', quality=70)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='segmented_image.png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)