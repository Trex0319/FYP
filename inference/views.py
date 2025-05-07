# inference/views.py

from django.shortcuts import render
from django.http import JsonResponse
from .utils import extract_frames, run_roboflow_inference
import os
import zipfile
from django.conf import settings
from django.http import HttpResponse
from wsgiref.util import FileWrapper
from django.views.decorators.csrf import csrf_exempt
from .models import Prediction

def home(request):
    """
    Render the home page with the video upload form.
    """
    return render(request, 'index.html')

def analyze_video(request):
    """
    Handle video upload, frame extraction, and Roboflow inference.
    """
    if request.method == 'POST' and request.FILES.get('video_file'):
        # Save the uploaded video file
        video_file = request.FILES['video_file']
        video_path = os.path.join(settings.MEDIA_ROOT, video_file.name)
        
        # Ensure the MEDIA_ROOT directory exists
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

        with open(video_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)

        # Extract frames
        output_dir = os.path.join(settings.MEDIA_ROOT, 'frames')
        frame_paths = extract_frames(video_path, output_dir, frame_rate=10)

        # Run Roboflow inference
        api_url = "https://detect.roboflow.com/traffic-training-pn2uu/2"
        api_key = "Z7USQy9C3FHdMDukOSO2"
        predictions = run_roboflow_inference(frame_paths, api_url, api_key)

        # Prepare response
        results = []
        for frame_path, prediction in predictions.items():
            results.append({
                'frame': os.path.basename(frame_path),
                'result': prediction
            })

        return JsonResponse(results, safe=False)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def download_frames(request):
    # Path to the directory containing frame images
    frames_dir = os.path.join(settings.MEDIA_ROOT, 'frames')

    # Check if the directory exists
    if not os.path.exists(frames_dir):
        return HttpResponse("No frames found.", status=404)

    # Create a temporary ZIP file
    zip_filename = 'frames.zip'
    zip_path = os.path.join(settings.MEDIA_ROOT, zip_filename)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(frames_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, frames_dir))

    # Serve the ZIP file for download
    response = HttpResponse(FileWrapper(open(zip_path, 'rb')), content_type='application/zip')
    response['Content-Disposition'] = f'attachment; filename="{zip_filename}"'
    return response

@csrf_exempt
def get_predictions(request, frame_name):
    try:
        # Fetch predictions for the specific frame
        predictions = Prediction.objects.filter(frame=frame_name).values(
            "x", "y", "width", "height", "class_label", "confidence"
        )
        return JsonResponse(list(predictions), safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)