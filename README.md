from video_processing import task_Worker
t = task_Worker()
url = 'https://storage.googleapis.com/picturize/videocolorize/1022e970dc72798c8b83ea8084b51fb2.mp4'
id = 1726
task = {"url": url, "id": id}
print(task)
t.run(task)
