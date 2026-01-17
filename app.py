from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                # Mirror the frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Analyze emotion on the flipped frame
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                
                for res in results:
                    x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
                    emotions = res['emotion']
                    
                    # BIAS REMOVAL: Force detection of secondary emotions if Neutral is low
                    if emotions['neutral'] < 80: 
                        emotions['neutral'] = 0   
                    
                    top_emotion = max(emotions, key=emotions.get)
                    
                    # Draw Cyan box and label on the mirrored view
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                    cv2.putText(frame, f"SENTIMENTAL GAMBIT: {top_emotion.upper()}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            except Exception:
                pass 

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
