import os
import uvicorn
from fastapi import FastAPI
import cv2
import numpy as np
from subprocess import Popen, PIPE
from moviepy import editor

app = FastAPI()

WORKING_DIR = "/Users/liuyanjun/Desktop/tmp"


@app.post("/download_youtube")
def download_youtube(url: str):
    if not os.path.isdir(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    p = Popen(["pytube", f"{url}"], stdout=PIPE, stderr=PIPE)
    output, error = p.communicate()
    if p.returncode != 0:
        raise ValueError(f"pytube failed {p.returncode} {output} {error}")
    else:
        return {"message": "Success"}


@app.post("/cartoon")
def cartoonize(image: str, output_video: str, fps: int):
    def contour(img: object):
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(im_gray, (3, 3), 0)
        edges = cv2.Canny(gaussian, 1, 255)
        contours_draw, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours_draw

    def flutter(img: object, videoWrite):
        # get dimensions
        h, w = img.shape[:2]
        # set wavelength
        wave_x = 2 * w
        wave_y = h

        # set amount, number of frames and delay
        amount_x = 10
        amount_y = 5
        num_frames = 100
        delay = 50
        border_color = (128, 128, 128)
        # create X and Y ramps
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)

        # loop and change phase
        for j in range(0, 10):
            for i in range(0, num_frames):
                # compute phase to increment over 360 degree for number of frames specified so makes full cycle
                phase_x = i / num_frames
                phase_y = phase_x

                # create sinusoids in X and Y, add to ramps and tile out to fill to size of image
                x_sin = amount_x * np.sin(2 * np.pi * (x / wave_x + phase_x)) + x
                map_x = np.tile(x_sin, (h, 1))
                y_sin = amount_y * np.sin(2 * np.pi * (y / wave_y + phase_y)) + y
                map_y = np.tile(y_sin, (w, 1)).transpose()

                # do the warping using remap
                result = cv2.remap(img.copy(), map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=border_color)

                # show result
                cv2.waitKey(delay)

                # convert to PIL format and save frames
                result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                videoWrite.write(result)

    img = cv2.imread(image)
    _output_video = os.path.join(WORKING_DIR, f"{output_video}.mp4")
    img_height, img_width = img.shape[:2]
    size = (img_width, img_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWrite = cv2.VideoWriter(_output_video, fourcc, fps, size)

    # Apply some Gaussian blur on the image
    img_gb = cv2.GaussianBlur(img, (7, 7), 0)
    # Apply some Median blur on the image
    img_mb = cv2.medianBlur(img_gb, 5)
    # Apply a bilateral filer on the image
    img_bf = cv2.bilateralFilter(img_mb, 5, 80, 80)
    # Use the laplace filter to detect edges
    img_lp_al = cv2.Laplacian(img_bf, cv2.CV_8U, ksize=5)
    # Convert the image to greyscale (1D)
    img_lp_al_grey = cv2.cvtColor(img_lp_al, cv2.COLOR_BGR2GRAY)

    # Manual image thresholding
    _, EdgeImage = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #
    # # Remove some additional noise
    blur_al = cv2.GaussianBlur(img_lp_al_grey, (5, 5), 0)
    # # Apply a threshold (Otsu)
    _, tresh_al = cv2.threshold(blur_al, 245, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invert the black and the white
    inverted_Bilateral_1 = cv2.subtract(255, tresh_al)
    # Reshape the image
    img_reshaped = img.reshape((-1, 3))
    # convert to np.float32
    img_reshaped = np.float32(img_reshaped)
    # Set the Kmeans criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set the amount of K (colors)
    K = 8
    # Apply Kmeans
    _, label, center = cv2.kmeans(img_reshaped, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Covert it back to np.int8
    # Reduce the colors of the original image
    div = 64
    img_bins = img // div * div + div // 2

    inverted_Bilateral = cv2.cvtColor(inverted_Bilateral_1, cv2.COLOR_GRAY2RGB)
    # Combine the edge image and the binned image
    cartoon_Bilateral = cv2.bitwise_and(inverted_Bilateral, img_bins)
    contours_draw = contour(cartoon_Bilateral)
    # create canvas
    canvas = np.zeros(img.shape, np.uint8)
    canvas.fill(255)
    frames = len(contours_draw)
    for contour in range(frames - 1, 0, -1):
        step = cv2.drawContours(canvas, contours_draw, contour, (0, 0, 0), 3)
        videoWrite.write(step)
    for _ in range(3 * int(frames / 10)):
        videoWrite.write(cartoon_Bilateral)

    cartoon_Bilateral_gb = cv2.GaussianBlur(cartoon_Bilateral, (7, 7), 0)
    cartoon_Bilateral_mb = cv2.medianBlur(cartoon_Bilateral_gb, 5)
    cartoon_Bilateral_bf = cv2.bilateralFilter(cartoon_Bilateral_mb, 5, 80, 80)
    for _ in range(int(frames / 10)):
        videoWrite.write(cartoon_Bilateral_bf)
    for _ in range(int(frames / 10)):
        videoWrite.write(cartoon_Bilateral_mb)
    for _ in range(int(frames / 10)):
        videoWrite.write(cartoon_Bilateral_gb)
    for _ in range(int(frames / 10)):
        videoWrite.write(cartoon_Bilateral_mb)
    for _ in range(int(frames / 10)):
        videoWrite.write(cartoon_Bilateral_bf)
    for _ in range(3 * int(frames / 10)):
        videoWrite.write(cartoon_Bilateral)
    flutter(cartoon_Bilateral, videoWrite)
    videoWrite.release()

    return {"message": "Success"}


@app.post("/contour")
def contour(image: str, output_video: str, fps: int):
    _output_video = os.path.join(WORKING_DIR, f"{output_video}.mp4")
    img = cv2.imread(image)
    img_height, img_width = img.shape[:2]
    size = (img_width, img_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWrite = cv2.VideoWriter(_output_video, fourcc, fps, size)

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(im_gray, (3, 3), 0)
    edges = cv2.Canny(gaussian, 1, 255)

    # create canvas
    canvas = np.zeros(img.shape, np.uint8)
    canvas.fill(255)

    contours_draw, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(range(len(contours_draw) - 1, 0, -1))
    for contour in range(len(contours_draw) - 1, 0, -1):
        step = cv2.drawContours(canvas, contours_draw, contour, (0, 0, 0), 3)
        videoWrite.write(step)
    output_jpg = os.path.join(WORKING_DIR, f"{output_video}.png")
    cv2.imwrite(output_jpg, step)
    print(videoWrite.isOpened())
    videoWrite.release()
    print(videoWrite.isOpened())
    return {"message": f"{_output_video} Success"}


@app.post("/extract_music")
def extract_music(mp4: str, mp3: str):
    if not os.path.isdir(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    mp3 = os.path.join(WORKING_DIR, f"{mp3}.mp3")
    p = Popen(["ffmpeg", "-i", f"{mp4}", "-vn", "-acodec", "libmp3lame", "-q:a", "2", f"{mp3}"], stdout=PIPE,
              stderr=PIPE)
    output, error = p.communicate()
    if p.returncode != 0:
        raise ValueError(f"pytube failed {p.returncode} {output} {error}")
    else:
        return {"message": "Success"}


@app.post("/remove_sound")
def remove_sound(old_video: str, new_video: str):
    if not os.path.isdir(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    try:
        new_video = os.path.join(WORKING_DIR, new_video)
        _vedio = editor.VideoFileClip(old_video)
        _vedio = _vedio.without_audio()
        _vedio.write_videofile(new_video)
        return {"message": "Success"}
    except Exception as err:
        raise err


@app.post("/contour_video")
def contour_video(input_video: str, output_video: str, fps: int):
    cap = cv2.VideoCapture(input_video)
    _output_video = os.path.join(WORKING_DIR, f"{output_video}.mp4")
    count = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        count += 1

        if ret:
            if count == 1:
                img_width = int(cap.get(3))
                img_height = int(cap.get(4))
                size = (img_width, img_height)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                videoWrite = cv2.VideoWriter(_output_video, fourcc, fps, size)

                im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gaussian = cv2.GaussianBlur(im_gray, (3, 3), 0)
                edges = cv2.Canny(gaussian, 1, 255)

                # create canvas
                canvas = np.zeros(frame.shape, np.uint8)
                canvas.fill(255)

                contours_draw, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                for contour in range(len(contours_draw) - 1, 0, -1):
                    step = cv2.drawContours(canvas, contours_draw, contour, (0, 0, 0), 3)
                    # videoWrite.write(step)
            # Break the loop
            else:
                im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gaussian = cv2.GaussianBlur(im_gray, (3, 3), 0)
                edges = cv2.Canny(gaussian, 1, 255)

                # create canvas
                canvas = np.zeros(frame.shape, np.uint8)
                canvas.fill(255)
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                step = cv2.drawContours(canvas, contours, -1, (0, 0, 0), 3)
                videoWrite.write(step)
        else:
            break

    videoWrite.release()
    return {"message": f"{_output_video} Success"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
