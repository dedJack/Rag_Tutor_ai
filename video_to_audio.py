import os
import subprocess

files = os.listdir("videos")

for file in files :
    file_number = file.split("_ ")[0].split("#")[1]
    file_name = file.split("_ ")[1].replace("720P.mp4","").replace("360P.mp4","")
    print(file_number, file_name)
    subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{file_number}_{file_name}.mp3"])