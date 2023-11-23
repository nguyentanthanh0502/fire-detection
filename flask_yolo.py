from flask import Flask, request, render_template
import os, cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"
yolo_model = YOLO("./best.pt")

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        exit()

    # Lấy thông số kích thước và tần suất khung hình của video gốc
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))

    # Tạo video writer cho video kết quả
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            results = yolo_model.predict(source=frame, conf=0.5, iou=0.5, stream=True)
            for r in results:
                im_array = r.plot()  
                im = Image.fromarray(im_array[..., ::-1]) 

                # Chuyển đổi từ PIL Image sang OpenCV image
                result_frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

                # Ghi frame vào video kết quả
                out.write(result_frame)

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    
                
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
            output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.mp4')  
            f.save(input_file_path)
            #process_video(input_file_path, output_file_path)
            

            return render_template('index.html', input_video=input_file_path, output_video=output_file_path)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
