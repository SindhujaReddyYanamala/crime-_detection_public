from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
import cv2
from core.yolo import detect_crime
from django.contrib.auth import authenticate, login as auth_login
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
import os
from django.conf import settings

def login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            auth_login(request, user)
            return redirect('dashboard')
        else:
            return render(request, 'login.html', {
                'error': 'Invalid Username or Password',
                'is_login': True
            })

    return render(request, 'login.html', {'is_login': True})


def register(request):
    return render(request, 'register.html')


def dashboard(request):
    if request.method == "POST" and request.FILES.get("video"):
        video = request.FILES["video"]

        fs = FileSystemStorage()
        filename = fs.save(video.name, video)

        # Absolute path
        full_path = os.path.join(settings.MEDIA_ROOT, filename)

        print("SAVED VIDEO PATH:", full_path)  # Debug

        request.session["video_path"] = full_path

        return redirect("stream")

    return render(request, "dashboard.html")



def live(request):
    return render(request, "live.html")


def stream(request):
    return render(request, 'stream.html')


def livestream(request):
    return render(request, 'livestream.html')





def video_feed(request):
    video_path = request.session.get("video_path")

    if not video_path:
        return HttpResponse("No video uploaded")

    def generate():
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("ERROR: Cannot open video file")
            return

        while True:
            success, frame = cap.read()

            if not success:
                print("Video ended")
                break

            # -------- RESIZE (IMPORTANT) --------
            frame = cv2.resize(frame, (800, 500))

            # -------- DETECTION --------
            frame, crime = detect_crime(frame)

            # -------- ENCODE --------
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return StreamingHttpResponse(
        generate(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )




def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Camera not opening")
        return

    while True:
        success, frame = cap.read()

        if not success:
            break

        # -------- RESIZE --------
        frame = cv2.resize(frame, (800, 500))

        # -------- DETECTION --------
        frame, crime = detect_crime(frame)

        _, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()



def live_crime_feed(request):
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )
