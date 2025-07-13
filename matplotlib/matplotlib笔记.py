import matplotlib.pyplot as plt
import numpy as np


# 基本
x = np.linspace(-3, 3, 50)
y1= 2*x + 1
y2=x**2
plt.figure(num=1,figsize=(8,5)) # 同时显示多张图时用，num参数为标号顺序，figsize参数为图片大小。
plt.plot(x, y1,label='up')                 # plot函数有color参数，linewidth（线宽), linestyle（线型）,label(标签)可以选择，label用legend函数显示。
# plt.figure(num=2)
plt.plot(x, y2,label='down')
plt.xlim((-1,2))                # 设置显示的横坐标轴的范围（-1到2）。也可以只设置一边，xmin和xmax参数。
plt.ylim((-2,3))
plt.xlabel('I am x')            # 声明横轴的内容
plt.ylabel('I am y')
new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)           # 设置x轴的刻度
plt.yticks([-2, -1.8, -1, 1.22, 3],['really bad', 'bad', 'normal', 'good', 'really good'])  # 设置y轴上的标签，一一对应。
# 设置和移动坐标轴
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 使用plt.gca获取当前坐标轴信息。
# 使用spines设置边框：
#   右侧边框：使用.set_color设置边框颜色：默认白色，none为不显示；
#   上边框：使用.set_color设置边框颜色：默认白色；
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')             # 将底部的轴设为x轴，左边设为y轴
ax.spines['bottom'].set_position(('data', 0))   # 将x轴放在y轴0的位置。
ax.spines['left'].set_position(('data',0))      # 将y轴放在x轴0的位置。
plt.legend()                # legend函数的参数有loc，labels，handles。loc参数为设置位置。
plt.show()


# # 标出重要的信息
# x = np.linspace(-3, 3, 50)
# y = 2*x + 1
# plt.figure(num=1, figsize=(8, 5),)
# plt.plot(x, y)
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# x0 = 1
# y0 = 2*x0 + 1
# plt.scatter(x0, y0, s=50, color='b')        # scatter函数，本是用来画散点图的，也可以用来标出点的位置。参数s大小，color颜色。
# plt.plot([x0, x0], [0, y0], 'k--', linewidth=2.5)
# plt.annotate('2x+1=%s' % y0, xy=(x0, y0), xytext=(+30, -30),
#              textcoords='offset points', fontsize=16,
#              arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
# # annotate函数，对标出点进行文字描述。
# #   xy参数，被注释的点的位置；xytext参数，传入一个坐标。
# #   textcoords参数，offset points是指相对于被注释点，在xytext的坐标处注释。fontsize字体大小。
# plt.text(x=0.25,y=3,s='here!',fontsize=16)
# plt.show()


# # 当数据把坐标等挡住时
# x = np.linspace(-3, 3, 50)
# y = 0.1*x
# plt.figure()
# plt.plot(x, y, linewidth=10, zorder=1)
# plt.ylim(-2, 2)
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data', 0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data', 0))
# for label in ax.get_xticklabels() + ax.get_yticklabels():       # ax.get_xticklabels() + ax.get_yticklabels()为取出所有的坐标
#     label.set_fontsize(12)                                      # 设置坐标的字体大小为12。
#     label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.7))
#     # bbox函数为设置坐标的前景，facecolor为前景颜色，edgecolor设置边框，本处设置边框为无，alpha设置透明度。
# plt.show()


# # 画图种类
# #     散点图
# n=1024
# X=np.random.normal(0,1,n)   # 生成n个均值是0，标准差是1的随机数
# Y=np.random.normal(0,1,n)
# T = np.arctan2(Y,X)
# plt.scatter(X,Y,s=75, c=T, alpha=0.5)
# plt.xlim((-1.5,1.5))
# plt.ylim((-1.5,1.5))
# plt.show()

# # 条形图
# n = 12
# X = np.arange(n)
# Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)    # 生成随机数
# Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
# plt.bar(X, +Y1)             # 生成条形图
# plt.bar(X, -Y2)
# plt.xlim(-.5, n)
# plt.xticks(())
# plt.ylim(-1.25, 1.25)
# plt.yticks(())
# for x, y in zip(X, Y1):
#     # ha: 水平对齐方式
#     # va: 垂直对齐方式
#     plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')# 显示数据
# for x, y in zip(X, Y2):
#     plt.text(x, -y - 0.05, '%.2f' % y, ha='center', va='top')
# plt.show()

# # 等高线（可用于画隐函数图像）
# def f(x,y):
#     return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 -y**2)  # 画这个函数的等高线
# n = 256
# x = np.linspace(-3, 3, n)
# y = np.linspace(-3, 3, n)
# X,Y = np.meshgrid(x, y)         # 生成网格。等高线就这么画
# plt.contourf(X, Y, f(X, Y), levels=8, alpha=0.75, cmap=plt.cm.hot)         # 添加颜色，传入mesh后的X,Y,f(X,Y)
# C=plt.contour(X, Y, f(X, Y), levels=8, colors='black')      # 画等高线，画levels+1条等高线，自动选择高度。
# plt.clabel(C,inline=True, fontsize=10)                      # 添加高度，inline参数为true时。将高度写在等高线上
# plt.show()

# # 生成随机矩阵对应的图片
# a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
#               0.365348418405, 0.439599930621, 0.525083754405,
#               0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)
# plt.imshow(a, interpolation='nearest', cmap='jet', origin='upper')          # 这个函数可以将矩阵转化成图片
# plt.colorbar(shrink=0.92)      # 添加颜色和数字之间对应的标注。shrink参数可以修改colorbar的长度，变为原来的百分之92
# plt.show()

# # 三维图
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = Axes3D(fig)        # 先定义一个图像窗口，并在窗口上添加3D坐标轴，画3D图官方操作。
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
# R_1 = X ** 2 + 2*Y ** 2
# R_2=4*X+4*Y-6
# # Z = np.sin(R)
# ax.plot_surface(X, Y, R_1, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# # 传入三个坐标轴X,Y,Z。rstride和cstride参数分别为网格线的行跨度和列跨度。
# ax.contourf(X, Y, R_1, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
# ax.plot_surface(X, Y, R_2, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# # 传入三个坐标轴X,Y,Z。rstride和cstride参数分别为网格线的行跨度和列跨度。
# ax.contourf(X, Y, R_2, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
# # 画等高线。zdir参数为从哪个轴压下去。offset参数为压在指定轴的哪个坐标上。
# ax.set_zlim(-1,10)
# plt.show()
# # plt.savefig('fig.png', bbox_inches='tight')     # 这里用plt.show()会有警告，直接保存3D图比较好。


# # 多图合并显示
# # 均匀小图
# plt.figure()  # 创建一个图片
# plt.subplot(2, 2, 1)  # 将图片分成两行两列，在第一个位置上创建一个小图
# plt.plot([0, 1], [0, 1])
# plt.subplot(2, 2, 2)  # 将图片分成两行两列，在第二个位置上创建一个小图
# plt.plot([0, 1], [0, 2])
# plt.subplot(2, 2, 3)  # 将图片分成两行两列，在第三个位置上创建一个小图
# plt.plot([0, 1], [0, 1])
# plt.subplot(2, 2, 4)  # 将图片分成两行两列，在第四个位置上创建一个小图
# plt.plot([0, 1], [0, 2])
# plt.show()
#     # 不均匀小图
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot([0,1],[0,1])
# plt.subplot(2,3,4)
# plt.plot([0,1],[0,2])
# plt.subplot(2,3,5)
# plt.plot([0,1],[0,1])
# plt.subplot(2,3,6)
# plt.plot([0,1],[0,2])
# plt.suptitle('全局标题',fontsize=20,x=0.5,y=0.99)
# plt.tight_layout()    # 调整坐标与标题重叠
# plt.show()

# # 次坐标轴，共享x轴，y轴不同的数据
# x = np.arange(0, 10, 0.1)
# y1 = 0.05 * x**2
# y2 = -1 * y1
# fig, ax1 = plt.subplots()   # subplot()返回一个 fig图像对象，ax对象。
# ax2 = ax1.twinx()           # 生成另一个坐标轴
# ax1.plot(x, y1, 'g-')
# ax1.set_xlabel('X data')
# ax1.set_ylabel('Y1 data', color='g')
# ax2.plot(x, y2, 'b-')
# ax2.set_ylabel('Y2 data', color='b')
# plt.show()

# # 箱型小图
# y=np.random.rand(4,2,100)
# figure,axes=plt.subplots(2,2)
# for i in range(y.shape[0]):
#     div,mod=divmod(i,2)
#     axes[div][mod].boxplot((y[i][0],y[i][1]),patch_artist=True,showfliers=False)
#     axes[div][mod].set_xticklabels(['1', '2'])
#     axes[div][mod].set_title(str(i+1))
#     # axes[div][mod].set_ylim([0, 90])
# plt.show()

# 在水平/垂直的位置划线
# plt.axhline(0.9, color='k', linestyle='--', linewidth=1)
# plt.avhline()
