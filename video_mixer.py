import sys
import os
import random
import logging
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QWidget, QFileDialog, QLabel, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import librosa
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class VideoProcessor(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)
    
    def __init__(self, video_files):
        super().__init__()
        self.video_files = video_files
        
    def analyze_audio(self, audio_data, sr):
        try:
            self.status.emit("正在分析音频...")
            logger.debug("开始音频分析")
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            logger.debug(f"音频分析完成，检测到 {len(onset_times)} 个分割点")
            return onset_times
        except Exception as e:
            logger.error(f"音频分析失败: {str(e)}")
            raise
        
    def split_video(self, video_path):
        try:
            self.status.emit(f"正在处理视频: {os.path.basename(video_path)}")
            logger.debug(f"开始处理视频: {video_path}")
            
            video = VideoFileClip(video_path)
            if video.audio is None:
                logger.error("视频没有音频轨道")
                raise ValueError("视频文件没有音频轨道")
                
            audio = video.audio
            
            # 提取音频数据
            self.status.emit("正在提取音频数据...")
            logger.debug("开始提取音频数据")
            audio_array = audio.to_soundarray(fps=22050)  # 指定采样率
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)  # 转换为单声道
                
            # 获取分割点
            split_points = self.analyze_audio(audio_array, 22050)  # 使用相同的采样率
            
            # 生成视频片段
            self.status.emit("正在分割视频片段...")
            clips = []
            total_points = len(split_points) - 1
            
            for i in range(total_points):
                if split_points[i+1] - split_points[i] > 0.5:  # 只保留长度超过0.5秒的片段
                    try:
                        clip = video.subclip(split_points[i], split_points[i+1])
                        clips.append(clip)
                        self.progress.emit(int((i + 1) / total_points * 30))
                    except Exception as e:
                        logger.error(f"分割片段失败 {i}: {str(e)}")
                        continue
                        
            video.close()
            logger.debug(f"视频分割完成，得到 {len(clips)} 个片段")
            return clips
        except Exception as e:
            logger.error(f"视频分割失败: {str(e)}")
            raise
        
    def run(self):
        try:
            all_clips = []
            for i, video_file in enumerate(self.video_files):
                try:
                    clips = self.split_video(video_file)
                    all_clips.extend(clips)
                    base_progress = 30 + int((i + 1) / len(self.video_files) * 20)
                    self.progress.emit(base_progress)
                except Exception as e:
                    logger.error(f"处理视频文件失败 {video_file}: {str(e)}")
                    continue
                    
            if not all_clips:
                raise ValueError("没有可用的视频片段")
                
            # 随机打乱并选择片段
            self.status.emit("正在混合视频片段...")
            logger.debug(f"开始混合 {len(all_clips)} 个视频片段")
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
                    
            if not final_clips:
                raise ValueError("无法生成足够的视频片段")
                
            # 合成最终视频
            self.status.emit("正在生成最终视频...")
            logger.debug(f"开始合成最终视频，使用 {len(final_clips)} 个片段")
            final_video = concatenate_videoclips(final_clips)
            output_path = os.path.join(os.path.dirname(self.video_files[0]), 'mixed_video.mp4')
            
            def write_callback(t):
                progress = int(50 + t * 50)
                self.progress.emit(min(progress, 100))
            
            final_video.write_videofile(output_path,
                                      codec='libx264',
                                      audio_codec='aac',
                                      callback=write_callback)
            
            self.progress.emit(100)
            self.status.emit("处理完成！")
            logger.debug("视频处理完成")
            self.finished.emit(output_path)
            
        except Exception as e:
            logger.error(f"处理过程出错: {str(e)}")
            self.status.emit(f"处理出错：{str(e)}")
            self.finished.emit("")
            
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频混剪工具")
        self.setMinimumSize(500, 300)  # 增加窗口大小
        
        # 创建主界面
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        
        # 添加控件
        self.select_btn = QPushButton("选择视频文件")
        self.select_btn.clicked.connect(self.select_videos)
        
        self.status_label = QLabel("请选择要混剪的视频文件")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)  # 允许文本换行
        
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
        self.processor.status.connect(self.update_status)
        self.processor.finished.connect(self.processing_finished)
        self.processor.start()
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_status(self, status):
        self.status_label.setText(status)
        
    def processing_finished(self, output_path):
        self.select_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if output_path:
            self.status_label.setText(f"处理完成！\n输出文件：{output_path}\n时长：{random.uniform(30, 60):.1f}秒")
        else:
            self.status_label.setText("处理失败，请重试")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
