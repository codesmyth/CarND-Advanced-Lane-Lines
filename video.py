from moviepy.editor import VideoFileClip
from image_generation import process_image

output = './output_images/lanes_detected.mp4'

clip1 = VideoFileClip("./project_video.mp4")
alf_clip = clip1.fl_image(process_image)
alf_clip.write_videofile(output, audio=False)