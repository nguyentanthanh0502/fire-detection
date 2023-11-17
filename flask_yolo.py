from flask import Flask, request, render_template, jsonify
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"
# Load mô hình YOLO từ file .pt
yolo_model = YOLO("./best.pt")

@app.route('/', methods=['POST', 'GET'])
def detect_objects():
    if request.method == 'POST':
        # Xử lý yêu cầu POST
        image = request.files['file']
        if image:
            input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(input_image_path)
            image = cv2.imread(input_image_path)
            results = yolo_model.predict(source=image)
            
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                im.save(os.path.join(app.config['UPLOAD_FOLDER'], 'results.jpg'))# Probs object for classification outputs
                output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.jpg')  
                
                
            
            return render_template('index.html', input_image=input_image_path, output_image=output_image_path)

        return 'Không được'
    else:
        # Trả về trang HTML cho yêu cầu GET
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
