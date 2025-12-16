import cv2
import matplotlib.pyplot as plt
import numpy as np

# # 读取图像
# img=cv2.imread('piano&rose.jpg')    # 字符串参数为路径,默认为彩色图，可选参数设0为灰度图
#
# # 显示图像
# cv2.imshow('piano&rose',img)        # cv2.imshow()函数有两个参数，第一个是给窗口命名，第二个是要显示的图像
# cv2.waitKey(0)
# # waitKey函数是一个等待键盘事件的函数,最常见的便是与显示图像窗口imshow函数搭配使用.
# # 参数值delay<=0时等待时间无限长,delay为正整数n时至少等待n毫秒的时间才结束。
# # 在等待的期间按下任意按键时函数结束，返回按键的键值（ascii码），等待时间结束仍未按下按键则返回-1。
# cv2.destroyAllWindows()
# # 图片有shape属性，img.shape（h*w*3）。opencv是BGR的三通道顺序。
#
# # 保存图像
# cv2.imwrite('example.jpg',img)      # 传入路径和变量名。注意要乘以255，因为图片的数值本应在0-255，但是在显示时被设置在（0-1）之间了。
#
# # 读取视频
# vc = cv2.VideoCapture('Fireflies Fly.mov')      # 这里传入路径
# while vc.isOpened():                # 这里vc.isOpened()用于判断视频是否成功打开
#     ret,frame=vc.read()             # 用vc.read()读取视频中的每一帧.ret为一个布尔变量,成功读到为True,frame为读取的帧的数据.
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # 将读到的帧转换成灰度图
#     # cv2.imshow('frame', gray)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(10) & 0xFF == ord('q'):  # 按q键退出。这里的意思是如果键盘输入的按键转换成ASCII码和q的ASCII码相同，则break
#         break
# vc.release()
# cv2.destroyAllWindows()
#
# # 读取摄像头
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)         # cv2.CAP_DSHOW是在win系统中需要加的一个参数
# while cap.isOpened():
#     ret, frame = cap.read()
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # 这句将每一帧彩色图片转换成灰度图片
#     # cv2.imshow('frame', gray)
#     cv2.imshow('frame', frame)        # 显示彩色视频
#     if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
#         break
# cap.release()
# cv2.destroyAllWindows()
#
#
# ROI图像截取
# img=cv2.imread('piano&rose.jpg')
# piano=img[0:50,0:200]       # 对图片数组进行切片，左上角为0，0点，h方向从0到50，w方向从0到200
# cv2.imshow('piano&rose',piano)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 颜色通道提取
# b,g,r=cv2.split(img)        # 将bgr三通道的值分离出来
# img=cv2.merge((b,g,r))      # 合并通道
#
# # 将g通道（或任意通道）上的值归零
# img[:,:,1]=0
#
#
# # 边界填充
# img=cv2.imread('piano&rose.jpg')
# top,bottom,left,right=(10,10,10,10)
# replicate = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_REPLICATE)
# # 为图片进行边界填充需要用到cv2.copyMakeBorder()函数。
# # 需传入的参数为图片变量，上下左右的填充像素数，填充方式变量.
# #   cv2.BORDER_CONSTANT 添加有颜色的常数值边界，还需要加一个参数value。eg:value=(255,0,0)
# #   cv2.BORDER_REFLIECT 边界元素的镜像。例如：fedcba | abcdefgh | hgfedcb
# #   cv2.BORDER_REFLECT101 跟上面一样，但稍作改动，例如：gfedcb | abcdefgh | gfedcba
# #   cv2.BORDER_REPLICATE 复制最后一个像素值。例如: aaaaaa| abcdefgh|hhhhhhh
# #   cv2.BORDER_WRAP 像这样: cdefgh| abcdefgh|abcdefg
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)       # cv是bgr通道，所以当cv读取的图片需要放入plt中显示时，需要变换成rgb通道
# replicate = cv2.cvtColor(replicate,cv2.COLOR_BGR2RGB)
# plt.subplot(211),plt.imshow(img),plt.title('original')
# plt.subplot(212),plt.imshow(replicate),plt.title('replicate')
# plt.show()
#
#
# # 数值计算:图像融合
# img1=cv2.imread('piano&rose.jpg')
# img2=cv2.imread('program.jpg')
# a,b,c=(0.5,0.5,0)
# img2=cv2.resize(img2,(img1.shape[1],img1.shape[0]))
# # 要保证两张图片大小相同，所以resize。
# # 传入需要resize的图片变量和元组(w,h)。要注意是(w,h)。
# img=cv2.addWeighted(img1,a,img2,b,c)
# # 叠加函数。公式：img = a·img1+b·img2+c
# cv2.imshow('combine',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('example.jpg',img)
#
# # 关于resize同比例放缩的方法
# img1=cv2.imread('piano&rose.jpg')
# img1=cv2.resize(img1,(0,0),fx=0.5,fy=0.5)      # 元组传入(0,0),水平fx和垂直fy缩到0.5倍
# cv2.imshow('combine',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# # 图像阈值
# img=cv2.imread('piano&rose.jpg',0)
# ret1 , thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret2 , thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret3 , thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret4 , thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret5 , thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# # cv2.threshold()函数需要传入4个参数。图片变量（单通道图像，如灰度图），阈值，最大值及操作方式。返回一个阈值（otsu法会用这个阈值）和一个图像。
# #   cv2.THRESH_BINARY：二值法。超过阈值的像素点取最大值，小于阈值取0。
# #   cv2.THRESH_BINARY_INV：上一个参数的反转。
# #   cv2.THRESH_TRUNC：大于阈值的像素点设为阈值，否则不变。
# #   cv2.THRESH_TOZERO：大于阈值的像素点保持不变，小于阈值设为0。
# #   cv2.THRESH_TOZERO_INV：小于阈值的像素点保持不变，大于阈值设为0。
# titles = ['original image','Binary','binary-inv','trunc','tozero','tozero-inv']
# images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()
#
# # Otsu's二值化（对于函数返回的阈值的解释）：
# # 前面对于阈值的处理上，我们选择的阈值都是127。有的图像可能阈值不是127得到的效果更好。那么这里我们需要算法自己去寻找到一个阈值。
# # 而Otsu’s就可以自己找到一个认为最好的阈值。
# # Otsu’s方法会产生一个阈值，那么函数cv2.threshold的的第二个参数（设置阈值）就是0了，并且在cv2.threshold的方法参数中还得加上语句cv2.THRESH_OTSU。
# img=cv2.imread('piano&rose.jpg',0)
# ret1 , thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret2 , thresh2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# titles = ['original image','Binary','otsu-Binary']
# images = [img,thresh1,thresh2]
# for i in range(3):
#     plt.subplot(1,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()
#
#
# # 图像平滑处理（去噪）
# img=cv2.imread('piano&rose.jpg')
# blur1=cv2.blur(img,(3,3))                       # cv2.blur()函数，一个均值滤波函数。需要传入要滤波的图像，和卷积核的大小。
# blur2 = cv2.GaussianBlur(img,(5,5),sigmaX=0)    # 高斯滤波函数。需传入图像，核的大小及X方向的标准差。若不设定y方向标准差，则sigmaY=sigmaX。
# blur3 = cv2.medianBlur(img,5)                   # 中值滤波函数。
# blur4 = cv2.bilateralFilter(img,9,75,75)        # 双边滤波函数。双边滤波器可以去除无关噪声，同时保持较好的边缘信息。但是，其速度比绝大多数滤波器都慢。
#                                                 # 参数为图像本身，邻域直径为9，两个 75 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差。
#                                                 # 两个标准差太小（小于10）会导致没有效果，太大（大于150）会导致看起来像卡通。
# titles = ['blur','GaussianBlur','medianBlur','bilateralFilter']
# images = [blur1,blur2,blur3,blur4]
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()
#
#
# # 图像形态学操作
# img = cv2.imread('piano&rose.jpg',0)
# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations=1)                # cv2.erode():腐蚀函数去图像毛刺等，但同时会减少图像的信息。核越大，迭代次数越高，效果越明显。
# dilation = cv2.dilate(img,kernel,iterations=1)              # 膨胀操作，与腐蚀互为逆运算。
# opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)       # 开运算：先腐蚀再膨胀，参数传入cv2.MORPH_OPEN。
# closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)      # 闭运算：先膨胀再腐蚀，参数传入cv2.MORPH_CLOSE。
# gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)  # 形态学梯度:结果为膨胀减腐蚀。结果为原图像的边界。
# tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)      # 礼帽运算：用原图像减开运算结果。即被去除的毛刺信息。
# blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)  # 黑帽运算：用闭运算结果减原图像。
# cv2.imshow('gradient',gradient)
# k=cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# # 图像梯度的计算方法（图像边缘位置）
# # sobel算子：卷积核如下，水平梯度右减左，垂直梯度下减上。
# # Gx=[[-1,0,1             Gy=[[-1,-2,-1
# #      -2,0,2                   0,0,0
# #      -1,0,1]]                 1,2,1]]
# im=cv2.imread('piano&rose.jpg',0)
# x = cv2.Sobel(im, cv2.CV_16S, 1, 0)
# # cv2.Sobel()函数，传入图片，图像深度，梯度方向dx和dy。
# #   图像深度：因为求出来的梯度不一定在0-255之间，所以用cv2.CV_16S这个参数。为了不丢信息。后面再转。
# #   梯度方向，分别计算xy方向，要求的那个方向参数为1。
# y = cv2.Sobel(im, cv2.CV_16S, 0, 1)
# absX = cv2.convertScaleAbs(x)           # 这个函数将数据转回uint8的形式
# absY = cv2.convertScaleAbs(y)
# dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
# cv2.imshow("Result", dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # Scharr算子（卷积核对边缘更敏感），用法与sobel算子相同。
# # Gx=[[-3,0,3             Gy=[[-3,-10,-3
# #      -10,0,10                  0,0,0
# #      -3,0,3]]                 3,10,3]]
# cv2.Scharr()
# # Laplacian算子，与二阶导有关，适合和其他算法结合，不建议单独使用。
# # 与sobel算子的区别是不需要分xy方向
# # Gx=[[0,1,0
# #      1,-4,1
# #      0,1,0]]
# cv2.Laplacian()
#
#
# # 边缘检测
# img = cv2.imread('piano&rose.jpg',0)
# edges = cv2.Canny(img,80,250)
# # 设置两个阀值：minVal和maxVal。
# # 图像的灰度梯度>maxVal，判定为边界；灰度梯度<minVal，被抛弃。
# # minVal<灰度梯度<maxVal，就要看这个点是否被确定为真正边界点相连，如果是，就认为它也是边界点，如果不是就抛弃。
# # maxVal: 带来最明显的差异，增大maxVal无疑会导致原来的边界点可能会直接消失。但这种消失时是成片消失。
# # minVal: 增大minVal，会导致有些待定像素点被弃用，也就是靠近边界像素点的介于双阈值之间的被弃用。导致的现象就是边界出现破损，这种非成片消失。只是边界信息不完整。
# # 如果边界信息缺损，那么适当的减小minVal;如果有不想要的区域出现，那么适当的增加MaxVal。
# cv2.imshow("Canny", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# # 图像金字塔：高斯金字塔
# img = cv2.imread('piano&rose.jpg')
# img_d=cv2.pyrDown(img)              # 高斯金字塔下采样
# img_dd=cv2.pyrDown(img_d)
# img_ddu=cv2.pyrUp(img_dd)           # 高斯金字塔上采样
# cv2.imshow("down",img_d-img_ddu)    # 进行上下采样时，信息会丢失，所以可以看到丢失的信息有哪些（拉普拉斯金字塔）
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 图像金字塔：拉普拉斯金字塔
# # 拉普拉斯金字塔可以由高斯金字塔计算得来。公式为：L_i=G_i-cv2.pyrUp(G_i+1),G_i+1=cv2.pyrDown(G_i)
# # 即，原图像减去原图像下采样后上采样的结果为该层金字塔，其中下采样后的图像为下一层的原图像。拉普拉斯金字塔常被用在图像压缩，图像融合中。
#
#
# # 图像轮廓
# # 边缘和轮廓的区别：边缘是零散的，轮廓是整体的
# img = cv2.imread('piano&rose.jpg')
# img=cv2.pyrDown(img)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)      # 对图像进行二值处理
# contours,hierarchy = cv2.findContours(thresh,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
# # cv2.findContours()：寻找轮廓函数。传入3个参数。二值化后的图像，轮廓检索模式和轮廓逼近方法。返回轮廓和层次。
# # mode参数: cv2.RETR_EXTERNAL,只检索最外面的轮廓
# #           cv2.RETR_LIST,并保存到一条链表当中
# #           cv2.RETR_CCOMP,
# #           cv2.RETR_TREE,检索所有轮廓并重构嵌套轮廓的整个层次.(比较推荐)
# # method参数:cv2.CHAIN_APPROX_NONE,保存所有的边界点.
# #           cv2.CHAIN_APPROX_SIMPLE,两点确定一条线的方法,保存重要的点.
# draw_img1=img.copy()         # 绘制轮廓之前需要复制一下原图，因为绘制轮廓函数会直接修改原图。
# draw_img1= cv2.drawContours(draw_img1,contours,29,(0,255,0),2)
# # cv2.drawContour()：绘制轮廓函数。传入画出轮廓的图，轮廓信息，等高线，轮廓的线条颜色green，线条宽度3。
# # 等高线参数用来指定画第几个轮廓。传入-1时表示绘制所有轮廓。
# cv2.imshow("down",draw_img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 下面是对于获得的轮廓进行分析（不完全）
# cnt=contours[29]     # 提取出一个轮廓
# area=cv2.contourArea(cnt)       # 求面积
# M=cv2.moments(cnt)              # 求矩。图像的矩可以帮助我们计算图像的质心，面积等。
# perimeter = cv2.arcLength(cnt,True)     # 周长。True参数表示闭合
# # 轮廓近似
# # 将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定。（Douglas-Peucker算法）
# epsilon=0.01*cv2.arcLength(cnt,True)     # 计算准确度
# approx = cv2.approxPolyDP(cnt,epsilon,True)
# # cv2.approxPolyDP()：用多边形的曲线代替轮廓。输入一条轮廓，准确度和True（闭合）。
# # 准确度的计算一般来自于原本轮廓的长度。epsilon越小越像原本的轮廓。
# draw_img2=img.copy()
# draw_img2= cv2.drawContours(draw_img2,[approx],-1,(0,255,0),2)   # 这个位置要用中括号括一下。
# cv2.imshow("down",draw_img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 轮廓外接矩形
# x,y,w,h=cv2.boundingRect(cnt)       # 返回4个变量，点（x,y）是矩形的左上角，w,h分别为矩形的宽和高。
# img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)      # 用返回的4个变量画矩形。
# cv2.imshow("down",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print('轮廓面积与边界矩形比例为：{:.4f}'.format(area/(w*h)))
