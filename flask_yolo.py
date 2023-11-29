from flask import Flask, request, render_template
import os, cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import base64
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"
yolo_model = YOLO("./best.pt")
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            results = yolo_model.predict(source=frame, conf=0.5, iou=0.5, stream=True)
            for r in results:
                im_array = r.plot()  
                im = Image.fromarray(im_array[..., ::-1])
                im.save(output_path)
                socketio.emit('update_image', {'image_path': output_path})
            socketio.emit('processing_done')
        else:
            break
      
@app.route('/', methods=['POST', 'GET'])
def detect_objects():
    if request.method == 'POST':
        f = request.files['file']
        # Kiểm tra định dạng của file trước khi lưu
        if f.filename.endswith('.jpg'):
            input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
            f.save(input_file_path)
            
            # Thực hiện object detection
            image = cv2.imread(input_file_path)
            results = yolo_model.predict(source=image, conf=0.5, iou=0.5)
            
            for r in results:
                im_array = r.plot()  
                im = Image.fromarray(im_array[..., ::-1])  
                im.save(os.path.join(app.config['UPLOAD_FOLDER'], 'results.jpg'))
                output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.jpg')  
                
            return render_template('index.html', input_image=input_file_path, output_image=output_file_path)
        
        elif f.filename.endswith('.mp4'):
            input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.mp4')
            output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')  
            f.save(input_file_path)
            
            socketio.start_background_task(process_video, input_file_path, output_file_path)
            
            
            return render_template('index.html', output_video=output_file_path)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    
    socketio.run(app, debug=True)
