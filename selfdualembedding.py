import matplotlib.gridspec
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec 

import scipy.optimize

class FirstMethod:
    #実行可能内点初期解を持つ自己双対線形計画問題を生成
    def generate_initial_sol(self,M):
        M_dim1 = M.shape[0]
        e = np.ones(M_dim1).reshape(-1,1)
        r = e - M @ e 
        zero = np.zeros((1,1))
        M1 = np.concatenate((M, r),axis=1)
        M2 = np.concatenate((-r.T,zero),axis=1)
        M_processed = np.concatenate((M1,M2),axis=0) #実行可能内点を持つ歪行列
        x_processed = np.concatenate((e,np.ones((1,1))), axis=0) #実行可能内点初期解
        s_processed = np.concatenate((e,np.ones((1,1))),axis = 0) #実行可能内点初期解
        return (M_processed,x_processed,s_processed)
    
class newton:
    #newton法で解の更新方向を定める。以下は予測子・修正子法を採用
    def __init__(self,M,x,s,mu,step):
        self.M = M
        self.x = x
        self.s = s
        self.mu = mu
        self.step = step
        
    def newtonMethod(self):
        #各ステップごとに計算
        X = np.diag(self.x.flatten())
        S = np.diag(self.s.flatten())
        X_inv = np.linalg.inv(X)
        XinvS = X_inv @ S
        mat = np.linalg.inv(self.M + XinvS)
        
        if self.step:
            # predictor step, delta = 0
            delta_x =  mat @ (-self.s)
            delta_s = -self.s - XinvS @ delta_x
        else:
            # correltor step, delta = 1
            diagXinv = X_inv @ np.ones(len(X)).reshape(-1,1)
            delta_x = mat @ (self.mu * diagXinv - self.s)
            delta_s = -self.s - XinvS @ delta_x + self.mu * diagXinv
        return (delta_x,delta_s) 

class MainMethod:
    def __init__(self,M):
        self.M = M
    
    def interior_point_method(self,A,b,c):
        error_tol = 10 ** -10 #許容誤差
        m,n = A.shape
        #実行可能初期解を持つ自己双対問題を作成
        initials = FirstMethod()
        (skewMat,init_x,init_s) = initials.generate_initial_sol(self.M)
        #初期解
        x = init_x
        s = init_s
        
        #以下プロット用
        self.fig = plt.figure()
        self.gs = GridSpec(nrows=3,ncols=4,height_ratios=[2,2,1.5],width_ratios= [2.5,1,1,1],figure = self.fig)
        ax1 = self.fig.add_subplot(self.gs[:2,:2])
        ax2 = self.fig.add_subplot(self.gs[0,2:4])
        ax3 = self.fig.add_subplot(self.gs[1,2:4])
        
        x1 = np.array([x[m,0]/x[m+n,0]])
        x2 = np.array([x[m+1,0]/x[m+n,0]])
        mus = np.zeros(0)
        iter_num = 0
        obj_val = np.array([c[0,0]*x1[iter_num] + c[1,0]*x2[iter_num]])
        Xaxis, Yaxsis = np.mgrid[0:5:100j,0:5:100j]
        
        while True:
            if np.dot(x.T,s)[0,0] < error_tol:
                iter = np.arange(iter_num) #反復回数
                ax1.plot(x1,x2,marker= 'o',color = "black",markersize = 4)
                ax1.set_title("plot")
                ax1.set_xlabel("x1")
                ax1.set_ylabel("x2")
                contourf = ax1.contourf(Xaxis,Yaxsis, c[0,0]*Xaxis + c[1,0]*Yaxsis, levels = 10, cmap = "coolwarm") #目的関数の等高線
                self.fig.colorbar(contourf,ax = ax1,shrink = 0.5 )
                                
                for i in range(m): 
                    #制約の直線を作図
                    z = A[i,0] * Xaxis + A[i,1] * Yaxsis
                    ax1.contour(Xaxis, Yaxsis, z, levels = [b[i,0]],colors = "black",linestyles = "solid" )
                ax1.set_aspect("equal")
                
                ax2.plot(iter,mus,marker = ".",markersize = 4)
                ax2.set_ylabel("mu value")
                ax3.plot(np.append(iter,iter_num + 1),obj_val,marker = ".",markersize = 4)
                ax3.set_xlabel("iter")
                ax3.set_ylabel("object value")    
                return (x,s)
            else:                
                
                #予測子
                mu_pre = np.dot(x.T,s)[0,0] / len(x)
                predictor = newton(skewMat,x,s,mu_pre,True)
                (delta_x_pre,delta_s_pre) = predictor.newtonMethod()
                step_pre = 1  / ( 2*math.sqrt(len(x))) #ステップサイズは下限を使用
                #解の更新
                x = x + step_pre * delta_x_pre
                s = s + step_pre * delta_s_pre
                iter_num += 1
                #プロット用に格納
                x1 = np.append(x1,x[m,0]/x[m+n,0])
                x2 = np.append(x2,x[m+1,0]/x[m+n,0])
                mus = np.append(mus,mu_pre)
                obj_val = np.append(obj_val,c[0,0]*x1[iter_num] + c[1,0]*x2[iter_num])
                
                #修正子
                mu_cor = np.dot(x.T,s)[0,0] / len(x)
                corrector = newton(skewMat,x,s,mu_cor,False)
                (delta_x_cor,delta_s_cor) = corrector.newtonMethod()
                step_cor = 1 #修正子ではステップサイズ１
                #解の更新
                x = x + step_cor * delta_x_cor
                s = s + step_cor * delta_s_cor
                iter_num += 1
                #プロット用に格納
                x1 = np.append(x1,x[m,0]/x[m+n,0])
                x2 = np.append(x2,x[m+1,0]/x[m+n,0])
                mus = np.append(mus,mu_cor)
                obj_val = np.append(obj_val,c[0,0]*x1[iter_num] + c[1,0]*x2[iter_num])
                            
class solve:
    def solveLP(self,c,A,b):
        #自己双対問題を作成
        m,n = A.shape
        zero_m = np.zeros((m,m))
        zero_n = np.zeros((n,n))
        zero = np.zeros((1,1))
        M1 = np.concatenate((zero_m,A,-b),axis=1)
        M2 = np.concatenate((-A.T,zero_n,c),axis=1)
        M3 = np.concatenate((b.T,-c.T,zero),axis=1)
        M = np.concatenate((M1,M2,M3),axis=0)
        
        
        method = MainMethod(M)
        #実行可能解を持つ自己双対線形問題の解（次元が2つ上）
        x_sol_SD = method.interior_point_method(A,b,c)[0] 
        #解のスケールを調節
        dual_sol = x_sol_SD[:m,:] / x_sol_SD[m + n,:]
        main_sol = x_sol_SD[m:m+n,:] /x_sol_SD[m + n,:]
        dual_value = np.dot(b.T,dual_sol)[0,0]
        main_value = np.dot(c.T,main_sol)[0,0]
        
        #以下プロット用
        fig = method.fig
        gs = method.gs
        axtext = fig.add_subplot(gs[2,:])
        
        result1 = scipy.optimize.linprog(-c,A_ub=A,b_ub=b)
        axtext.text(0.5,0.5,f"solution = ( {main_sol[0,0]},{main_sol[1,0]}) \n optimal value = {main_value}\n dual optimal value = {dual_value} \n scipy solution = {result1.x} \n scipy optimal value = {result1.fun}",
                    fontsize = 11,
                    ha = "center",
                    va = "center"
                    )
        axtext.axis("off")
        fig.subplots_adjust(wspace=1,hspace=0.5)
        plt.show()
        # return dual_sol,main_sol,dual_value,main_value

# 目的関数の係数 (最大化なので符号を反転)
c =np.array( [[3], [5]])

# 制約条件
# A =np.array( [
#     [1, 2],   # x + 2y <= 8
#     [3, 1]    # 3x + y <= 9
# ])
# b = np.array([[8], [9]])    # 制約の右辺

# c = np.array([[5], [2]])  # 目的関数の係数

# # 行列 (m×n)
# A = np.array([[1, 2],     # 制約1
#               [2, 1]])    # 制約2

# # 列ベクトル (m×1)
# b = np.array([[10],       # 制約1の右辺
#               [8]])  

# c = np.array([[4], [3]])  # 目的関数の係数

# # 行列 (m×n)
# A = np.array([
#     [1, 2],  # 制約1
#     [2, 1],  # 制約2
#     [1, 1]   # 制約3
# ])

# # 列ベクトル (m×1)
# b = np.array([
#     [12],    # 制約1の右辺
#     [14],    # 制約2の右辺
#     [8]      # 制約3の右辺
# ])

c = np.array([[3],  # x の係数
              [2]])  # y の係数

# 制約条件の係数行列 (A @ [x, y] <= b の形)
A = np.array([
    [1, 1],    # x + y <= 3
    [-1, 1]    # -x + y <= -1  (x - y >= 1)
])

#制約条件の右辺ベクトル（2行1列の2次元配列）
b = np.array([[3],    # 制約1の右辺
              [-1]])  # 制約2の右辺

# c = np.array([[5],  # x の係数
#               [4]])  # y の係数

# # 制約条件の係数行列 (A @ [x, y] <= b の形)
# A = np.array([
#     [1, 1],    # x + y <= 4
#     [-1, 1],   # -x + y <= -2  (x - y >= 2)
#     [-2, -1],  # -2x - y <= -5  (2x + y >= 5)
#     [0, -1]    # y <= 2
# ])

# # 制約条件の右辺ベクトル（2行1列の形）
# b = np.array([[4],    # 制約1の右辺
#               [-2],   # 制約2の右辺
#               [-5],   # 制約3の右辺
#               [2]])   # 制約4の右辺

r = solve()
result = r.solveLP(c,A,b)


                
                