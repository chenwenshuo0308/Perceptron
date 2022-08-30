# 这是一个感知机示例



import pandas as pd  #数据分析库
import numpy as np #数值计算扩展，用来存储和处理大型矩阵
import matplotlib.pyplot as plt #导入画图工具包
from sklearn.datasets import load_iris
#从机器学习包中导入load_iris数据集
#ris数据集的中文名是安德森鸢尾花卉数据集，英文全称是Anderson’s Iris data set
#iris包含150个样本，对应数据集的每行数据
#每行数据包含每个样本的四个特征(花萼长度、花萼宽度、花瓣长度、花瓣宽度)和样本的类别信息(targe)
#iris数据集是一个150行5列的二维表
#iris数据集及简介链接:https://blog.csdn.net/java1573/article/details/78865495
iris=load_iris() #导入数据
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#将iris中的四个特征数据放入df矩阵中
#DataFrame介绍：https://www.jianshu.com/p/8024ceef4fe2
df['label']=iris.target
#在df矩阵的最后插入一列label即样本类别
df.columns=['sepal length','sepal width','petal length','petal width','label']
#输入print(df)即可查看
plt.figure(figsize=(15, 8))
#figsize:指定figure的宽和高
#plt.figure()的使用:https://blog.csdn.net/m0_37362454/article/details/81511427
plt.subplot(131)
#subplot绘制多个子图介绍:https://www.cnblogs.com/xiaoboge/p/9683056.html
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
#画散点图
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('original data')
plt.legend()

data = np.array(df.iloc[:100, [0, 1, -1]])
#读取df的前100行，第一、二、五列（'sepal length','sepal width','label')
#array：计算机为数组分配一段连续的内存，从而支持对数组随机访问
#要选取连续多列就该使用df.iloc
#iloc的用法示例：https://blog.csdn.net/qq_39697564/article/details/87855167
X, y = data[:,:-1], data[:,-1]
#令X等于'sepal length','sepal width'的100个二维数据，令y等于100个类别（只有0,1）
y = np.array([1 if i == 1 else -1 for i in y])
#对标记数据进行标准化为1或-1，当y为1时仍为1，其他的变为-1

class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype = np.float32)
        self.b = 0
        self.l_rate = 0.1
        #ones在这生成一个比data列数少1的数组，此处data[0]表示data的第1行，len表示长度
        #选取w,b和学习率l_rate的初值
        # ones介绍：https://blog.csdn.net/qq_28618765/article/details/7808545
        #_init_介绍：https://www.zhihu.com/question/46973549

    def sign(self,x,w,b):
        y = np.dot(x,w) + b
        return y

    def fit(self, X_train, y_train):
        is_wrong = False
        while not is_wrong:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True
        return 'Perceptron Model!'
#随机梯度下降法
# range创建一个整数列表，默认从0开始
# range() 函数用法: http://www.runoob.com/python/python-func-range.html
#python中的self介绍：https://blog.csdn.net/xrinosvip/article/details/89647884

perceptron = Model()
perceptron.fit(X, y)

x_points = np.linspace(4,7,10)
#从4到7之间创建10个等差序列，包括7
#linspace的用法:https://blog.csdn.net/weixin_40103118/article/details/78787454
y_ = -(perceptron.w[0]*x_points + perceptron.b)/perceptron.w[1]
#根据上面的结果得到了方程，画图时根据w1x1+w2x2+b=0进行画图
#由于画图时是二维的，这里的横轴x表示x1即第一个特征，纵轴y表示x2即第二个特征
plt.subplot(132)
plt.plot(x_points, y_)
plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('perceptron byhand')
plt.legend()

from sklearn.linear_model import Perceptron
clf = Perceptron(fit_intercept=True, max_iter=1000, shuffle=True)
#fit_intercept表示是否保留截距
clf.fit(X, y)

print(clf.coef_)
#输出系数w
print(clf.intercept_)
#输出截距b
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
#根据上面的结果得到了方程，画图时根据w1x1+w2x2+b=0进行画图
#由于画图时是二维的，这里的横轴x表示x1即第一个特征，纵轴y表示x2即第二个特征
plt.subplot(133)
plt.plot(x_ponits, y_)

plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('perceptron by sklearn')
plt.legend()

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
#调整子图间距
plt.savefig("demo.jpg")
plt.show()
