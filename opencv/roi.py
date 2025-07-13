import cv2
import imageio
import numpy as np

# 逐帧读取GIF文件
gif_path = "1.gif"
reader = imageio.get_reader(gif_path)

# ROI设置
x, y, w, h = 0, 150, 720, 800  # 设定ROI区域 (x, y, width, height)

# 保存裁剪后的帧
roi_frames = []

for i,frame in enumerate(reader):
    # 将frame转换为OpenCV格式
    if i<3:
        continue
    else:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # 裁剪ROI
        roi = frame[y:y + h, x:x + w]

        # 转换回RGB格式
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # 添加到列表
        roi_frames.append(roi_rgb)

# 释放资源
reader.close()

# 保存新的GIF
output_path = "output.gif"
imageio.mimsave(output_path, roi_frames, format='GIF', duration=0.01)  # duration是每一帧的持续时间
