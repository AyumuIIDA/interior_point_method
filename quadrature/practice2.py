import numpy as np

class gauss:
    def __init__(self,a,b,f,n):
        self.a = a
        self.b = b
        self.f = f
        self.n = n
        
    def coefLgendre(self):
        """
        ルジャンドル多項式のmonic三項漸化式の係数を計算する関数
        """
        a = np.zeros(self.n)
        b = np.array([])
        b = np.append(b,2)
        for i in range(1,self.n):
            bi = 1/(4-1/i**2)
            b = np.append(b,bi)
        return a,b

    def zpLegendre(self):
        """
        Golub-Welsch Algorismによってnode,weightを求める関数
        """
        a,b = self.coefLgendre()
        # Jacobi行列の生成
        JacM = np.zeros((self.n,self.n))
        for i in range(self.n):
            JacM[i,i] = a[i]
        for i in range(self.n-1):
            # b[0]はp_-1の係数であり、Jacobi行列に関与しない
            beta = np.sqrt(b[i+1]) 
            JacM[i,i+1] = beta
            JacM[i+1,i] = beta
        # 固有値分解
        eigvals,eigvecs = np.linalg.eigh(JacM)
        x = eigvals # node
        w = 2*(eigvecs[0,:]**2) # weight
        ind = np.argsort(x)
        x_sorted = x[ind]
        w_sorted = w[ind]
        
        return x_sorted,w_sorted

    def gaussian_quadratures(self):
        """
        積分経路(a,b)についてGaussian Quadrature実行
        """
        x,w = self.zpLegendre()
        # (-1,1)→(a,b)の写像を構築
        s = (-self.a+self.b)/2
        t = np.full(self.n,(self.b+self.a)/2) 
        x_modified = s*x +t
        w_modified = s*w
        # ステップ幅を仮想的に指定
        hsize = x_modified[1:]-x_modified[:-1]
        return  np.dot(w_modified,self.f(x_modified)),np.amax(hsize)
    
class q0:
    start = 0
    end = 1
    exact = np.exp(1.0)-1.0
    def function(x):
        return 3*(x**2)*np.exp(x**3)

class q1:
    start = 0
    end = 1
    exact = (np.pi**2 - 4)/np.pi**3
    def function(x):
        return (x**2)*np.sin(np.pi*x)

class q2:
    start = 1/np.sqrt(2)
    end = 1
    exact = np.pi/8 - 0.25
    def function(x):
        return np.sqrt(1-x**2)
    
class q3:
    start = np.arcsin(1/np.sqrt(2))
    end = np.arcsin(1)
    exact = np.pi/8 - 0.25
    def function(x):
        return (np.cos(x))**2
    
class q4:
    start = 0
    end = 1
    exact = np.pi*(13/16) - 23/15
    def horner(x,a:np.ndarray):
        # horner法による多項式計算
        val = 0
        for coef in a:
            val = val*x + coef
        return val
    
    def function(x):
        a = np.array([-1,-4,3,16,-11,-12,9])
        g = q4.horner(x,a)
        return (1-x)*np.sqrt(g)
          
class q5:
    start = 0
    end = 2*np.pi
    exact = 2*np.pi/3
    def function(x):
        return 1/(5-4*np.cos(x))

