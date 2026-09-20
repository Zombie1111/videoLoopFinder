echo "Ensure FFmpeg is installed and .exe path in frameMatcher.config https://ffmpeg.org/download.html"
echo "Ensure Python is installed with ☑ Add Python to PATH https://www.python.org/downloads/"
ffmpeg -version
python --version
python -m pip install opencv-python pillow imagehash
python -c "import cv2, PIL, imagehash; print('Everything installed OK')"
pause