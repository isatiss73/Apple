from FileManager import FileManager

video_path = 'BadApple.mp4'
output_dir = 'var_2208'


manager = FileManager(video_path, output_dir)
manager.extract_to_video(blur_radius=5)
