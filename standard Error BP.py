# -*- coding: UTF-8 -*-
#版权所有：广州大学 机械与电气工程学院 陈立隆
###########################################################################

#输入：D={(Xk,Yk)},k=0~11
      #学习率：Eta=0.1
#过程：
##1. 在(0，1)范围内随机初始化网络中所有连接权和阈值
##2. repeat
##3.    for all (Xk,Yk)∈D do
##4.        根据当前参数和式(5.3)计算当前样本的输出Yk；
##5.        根据式(5.10)计算输出层神经元的梯度项Gj；
##6.        根据式(5.15)计算隐层神经元的梯度项Eh；
##7.        根据式(5.11)-(5.14)更新连接权Whj，Vhj与阈值Theta，Gamma
##8.    end for
##9. until 达到停止条件
#输出：连接权与阈值确定的多层前馈神经网络

###########################################################################

#标准BP算法

import numpy as np

#设定原始参数
d=8;q=5;l=3;
Eta=0.1;
np.random.seed(20201231);
Theta=np.random.uniform(0.0,1.0,[l,1]);
Gamma=np.random.uniform(0.0,1.0,[q,1]);
#八个输入神经元到五个隐层神经元
Vih=np.random.uniform(0.0,1.0,[d,q]);
#五个隐层神经元到三个输出神经元
Whj=np.random.uniform(0.0,1.0,[q,l]);

#导入训练集 
#西瓜数据3.0中的前12个，其中，第二个和第四个样本又和测试集的第二个和第四个样本对调
tmp=np.loadtxt("D:/Machine Learning Homework/Train_Data.csv", delimiter=",")
Y_Out=np.loadtxt("D:/Machine Learning Homework/Y_out.CSV", delimiter=",")
Y_Out_C=Y_Out[:,1];
tmp1=np.array(tmp)
Train_data=tmp1[0:, 1:]  # 加载数据部分
Train_label=tmp1[0:, 0]  # 加载类别标签部分
Train_label=np.asarray(Train_label).astype('float32')  # 加载类别标签部分

#定义sigmoid函数
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

#训练样本数量
m=np.size(Train_label,0);
print("训练样本数量：");
print(m)

alpha=np.zeros(q);
beta=np.zeros(q);
vx=np.zeros(d);
wb=np.zeros(q);
Ek=np.zeros(m)
Bh=np.zeros(q);
Yk_Out=np.zeros([m,l]);
Gj=np.zeros([m,l]);
Eh=np.zeros(q);
wg=np.zeros(l);
deltaWhj=np.zeros([q,l]);
deltaTheta=np.zeros(l);
deltaVh=np.zeros([d,q]);
deltaGamma=np.zeros(q);
sub_Y=np.zeros(l);
Y_Out_CC=np.zeros(m);

#设定初始累计误差，以便开始训练
E=1;
#迭代次数
Generation=0;
M=0;

#神经网络计算 最终累计误差不超过0.01
while E>=0.01:
    Generation=Generation+1;
    for k in range(0,m,1):
        #计算Bh
        for h in range(0,q,1):
            for i in range(0,d,1):
                vx[i]=Vih[i][h]*Train_data[k][i];
            alpha[h]=np.sum(vx);
            Bh[h]=sigmoid(alpha[h]);
            
        #计算beta
        for j in range(0,l,1):
            for h in range(0,q,1):
                wb[h]=Whj[h][j]*Bh[h];
            beta[j]=np.sum(wb);
        
        #计算输出和梯度
        for j in range(0,l,1):
            Yk_Out[k][j]=sigmoid(beta[j]-Theta[j]);
            Gj[k][j]=Yk_Out[k][j]*(1-Yk_Out[k][j])*(Y_Out[k][j]-Yk_Out[k][j]);
            
        #计算Eh
        for h in range(0,q,1):
            for j in range(0,l,1):
                wg[j]=Whj[h][j]*Gj[k][j];
            Eh[h]=Bh[h]*(1-Bh[h])*np.sum(wg);

        #计算并更新阈值和连接权
        for h in range(0,q,1):
            for j in range(0,l,1):
                deltaWhj[h][j]=Eta*Gj[k][j]*Bh[h];
                Whj[h][j]=Whj[h][j]+deltaWhj[h][j];
            deltaGamma[h]=-Eta*Eh[h];
            Gamma[h]=Gamma[h]+deltaGamma[h];
        for j in range(0,l,1):
            deltaTheta[j]=-Eta*Gj[k][j];
            Theta[j]=deltaTheta[j]+Theta[j];
        for i in range(0,d,1):
            for h in range(0,q,1):
                deltaVh[i][h]=Eta*Eh[h]*Train_data[k][i];
                Vih[i][h]=Vih[i][h]+deltaVh[i][h];
        
        #计算均方误差
        for j in range(0,l,1):
            sub_Y[j]=(Yk_Out[k][j]-Y_Out[k][j])*(Yk_Out[k][j]-Y_Out[k][j]);   
        Ek[k]=0.5*np.sum(sub_Y);
    E=np.sum(Ek)/m;
    
#计算训练精度
for k in range(0,m,1):
    if (np.sum(Yk_Out[k][0]*Yk_Out[k][0]+Yk_Out[k][1]*Yk_Out[k][1]\
              +Yk_Out[k][2]*Yk_Out[k][2]))**0.5>0.5:
        Y_Out_CC[k]=1;
    else:
        Y_Out_CC[k]=0;
    if Y_Out_CC[k]==Y_Out_C[k]:
        M=M+1;
accurateRatio=M/m;

print("迭代次数：")
print(Generation)
print("Vih=")
print(Vih)
print("Whj=")
print(Whj)
print("Theta=")
print(Theta)
print("Gamma=")
print(Gamma)
print("均方误差为=")
print(E)
print("训练输出为：")
print(Yk_Out)
print("原始输出为")
print(Y_Out)
print("训练精度：")
print(accurateRatio)
