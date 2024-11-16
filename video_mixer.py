import sys
import os
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QWidget, QFileDialog, QLabel, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

class VideoProcessor(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    
    def __init__(self, video_files):
        super().__init__()
        self.video_files = video_files
        
    def analyze_audio(self, audio_data, sr):
        # 使用librosa分析音频能量
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        return onset_times
        
    def split_video(self, video_path):
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # 提取音频数据
        audio_array = audio.to_soundarray()
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)  # 转换为单声道
            
        # 获取分割点
        split_points = self.analyze_audio(audio_array, audio.fps)
        
        # 生成视频片段
        clips = []
        for i in range(len(split_points) - 1):
            if split_points[i+1] - split_points[i] > 0.5:  # 只保留长度超过0.5秒的片段
                clip = video.subclip(split_points[i], split_points[i+1])
                clips.append(clip)
                
        video.close()
        return clips
        
    def run(self):
        all_clips = []
        for i, video_file in enumerate(self.video_files):
            clips = self.split_video(video_file)
            all_clips.extend(clips)
            self.progress.emit(int((i + 1) / len(self.video_files) * 50))
            
        # 随机打乱并选择片段
        random.shuffle(all_clips)
        total_duration = 0
        final_clips = []
        target_duration = random.uniform(30, 60)
        
        for clip in all_clips:
            if total_duration + clip.duration <= target_duration:
                final_clips.append(clip)
                total_duration += clip.duration
            if total_duration >= target_duration:
                break
                
        # 合成最终视频
        if final_clips:
            final_video = concatenate_videoclips(final_clips)
            output_path = os.path.join(os.path.dirname(self.video_files[0]), 'mixed_video.mp4')
            final_video.write_videofile(output_path, 
                                      codec='libx264', 
                                      audio_codec='aac')
            self.progress.emit(100)
            self.finished.emit(output_path)
        else:
            self.finished.emit("")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频混剪工具")
        self.setMinimumSize(400, 300)
        
        # 创建主界面
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # 添加控件
        self.select_btn = QPushButton("选择视频文件")
        self.select_btn.clicked.connect(self.select_videos)
        
        self.status_label = QLabel("请选择要混剪的视频文件")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.process_videos)
        self.process_btn.setEnabled(False)
        
        # 添加到布局
        layout.addWidget(self.select_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.process_btn)
        
        main_widget.setLayout(layout)
        self.video_files = []
        
    def select_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择视频文件",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if files:
            self.video_files = files
            self.status_label.setText(f"已选择 {len(files)} 个视频文件")
            self.process_btn.setEnabled(True)
            
    def process_videos(self):
        if not self.video_files:
            return
            
        self.select_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.processor = VideoProcessor(self.video_files)
        self.processor.progress.connect(self.update_progress)
        self.processor.finished.connect(self.processing_finished)
        self.processor.start()
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def processing_finished(self, output_path):
        self.select_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if output_path:
            self.status_label.setText(f"处理完成！输出文件：{output_path}")
        else:
            self.status_label.setText("处理失败，请重试")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
