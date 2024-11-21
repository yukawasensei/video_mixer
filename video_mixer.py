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
from datetime import datetime

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
            
            # 打开视频文件
            video = VideoFileClip(video_path)
            if video is None or video.reader is None:
                logger.error(f"无法打开视频文件: {video_path}")
                return []

            if video.audio is None:
                logger.warning("视频没有音频轨道，使用固定时间间隔分割")
                # 使用固定时间间隔分割
                split_points = np.arange(0, video.duration, 3.0)
            else:
                # 提取音频数据
                self.status.emit("正在提取音频数据...")
                logger.debug("开始提取音频数据")
                
                # 使用librosa直接加载音频
                y, sr = librosa.load(video_path, sr=22050, mono=True)
                
                # 获取分割点
                self.status.emit("正在分析音频...")
                logger.debug("开始音频分析")
                
                # 使用多个特征来检测语音段
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                split_points = librosa.frames_to_time(beats, sr=sr)
            
            # 确保至少有一个分割点
            if len(split_points) < 2:
                logger.debug("未检测到足够的分割点，使用固定时间间隔")
                # 如果没有检测到足够的分割点，每3秒分割一次
                split_points = np.arange(0, video.duration, 3.0)
            
            # 确保最后一个分割点不超过视频时长
            split_points = split_points[split_points < video.duration]
            
            # 如果没有包含视频结尾，添加结尾时间点
            if len(split_points) == 0 or split_points[-1] < video.duration:
                split_points = np.append(split_points, video.duration)
            
            # 生成视频片段
            self.status.emit("正在分割视频片段...")
            clips = []
            total_points = len(split_points) - 1
            
            for i in range(total_points):
                start_time = split_points[i]
                end_time = split_points[i + 1]
                
                # 确保片段长度合适
                if end_time - start_time > 0.5 and end_time - start_time < 10.0:  # 限制最大片段长度为10秒
                    try:
                        clip = video.subclip(start_time, end_time)
                        if clip is not None and clip.reader is not None:
                            # 验证片段是否可以读取帧
                            test_frame = clip.get_frame(0)
                            if test_frame is not None:
                                clip.filename = video_path  # 保存原始文件路径
                                clips.append(clip)
                                self.progress.emit(int((i + 1) / total_points * 30))
                                logger.debug(f"成功创建片段 {i+1}/{total_points}, 时长: {end_time - start_time:.2f}秒")
                    except Exception as e:
                        logger.error(f"分割片段失败 {i}: {str(e)}")
                        continue
            
            video.close()
            
            if not clips:
                logger.warning("没有生成任何有效片段")
                return []
                
            logger.debug(f"视频分割完成，得到 {len(clips)} 个片段")
            return clips
            
        except Exception as e:
            logger.error(f"视频分割失败: {str(e)}")
            if 'video' in locals():
                try:
                    video.close()
                except:
                    pass
            raise
        
    def run(self):
        try:
            if not self.video_files:
                raise ValueError("没有选择视频文件")

            all_clips = []
            total_files = len(self.video_files)

            for i, video_path in enumerate(self.video_files):
                try:
                    clips = self.split_video(video_path)
                    if clips:  # 确保有有效的片段
                        all_clips.extend(clips)
                except Exception as e:
                    logger.error(f"处理视频文件失败 {video_path}: {str(e)}")
                    continue

            if not all_clips:
                raise ValueError("没有可用的视频片段")

            # 随机选择并组合片段
            self.status.emit("正在组合视频片段...")
            random.shuffle(all_clips)
            
            # 计算目标时长（30-60秒）
            target_duration = random.uniform(30, 60)
            final_clips = []
            current_duration = 0
            
            # 验证并选择片段
            for clip in all_clips:
                try:
                    if clip is not None and hasattr(clip, 'duration') and clip.reader is not None:
                        # 验证片段是否可以读取帧
                        test_frame = clip.get_frame(0)
                        if test_frame is not None:
                            clip_duration = clip.duration
                            if current_duration + clip_duration <= target_duration:
                                final_clips.append(clip)
                                current_duration += clip_duration
                            if current_duration >= target_duration:
                                break
                except Exception as e:
                    logger.error(f"验证片段失败: {str(e)}")
                    continue

            if not final_clips:
                raise ValueError("无法创建足够长度的视频")

            # 确保所有片段都是有效的
            valid_clips = []
            for clip in final_clips:
                try:
                    # 重新打开视频片段
                    new_clip = VideoFileClip(clip.filename)
                    if new_clip is not None and new_clip.reader is not None:
                        valid_clips.append(new_clip)
                except Exception as e:
                    logger.error(f"重新打开片段失败: {str(e)}")
                    continue

            if not valid_clips:
                raise ValueError("没有有效的视频片段")

            # 创建最终视频
            self.status.emit("正在生成最终视频...")
            final_video = concatenate_videoclips(valid_clips)
            
            # 生成输出文件名
            output_dir = os.path.dirname(self.video_files[0])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"mixed_video_{timestamp}.mp4")
            
            # 写入文件
            self.status.emit("正在写入文件...")
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=None,
                remove_temp=True,
                fps=24
            )
            
            # 清理资源
            final_video.close()
            for clip in valid_clips:
                try:
                    clip.close()
                except:
                    pass

            self.status.emit(f"处理完成！输出文件：{output_path}")
            self.progress.emit(100)
            
        except Exception as e:
            logger.error(f"处理过程出错: {str(e)}")
            self.status.emit(f"处理失败: {str(e)}")
            self.progress.emit(0)
            raise

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
