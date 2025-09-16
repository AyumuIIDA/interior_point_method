import numpy as np
import matplotlib.pyplot as plt
from practice2 import gauss
    
class method:
    def __init__(self,x,f):
        self.x = x
        self.f = f

    def midpoint(self):
        # 任意の分割に基づく複合中点則
        h = self.x[1:] - self.x[:-1]
        mid = 0.5*(self.x[1:] + self.x[:-1])
        val = np.dot(self.f(mid),h)
        return val, np.amax(h)

    def trape(self):
        # 任意の分割に基づく複合台形則
        h = self.x[1:] - self.x[:-1]
        val = 0.5*np.dot(self.f(self.x[1:])+self.f(self.x[:-1]),h)
        return val, np.amax(h)

    def simpson(self):
        # 任意の分割に基づく複合Simpson則
        h = self.x[1:]-self.x[:-1]
        mid = (self.x[1:]+self.x[:-1])/2
        val = np.dot(self.f(self.x[1:])+self.f(self.x[:-1])+4*self.f(mid),h)/6
        return val , np.amax(h)
    
class evaluate:
    def __init__(self,f,max,min,step,a,b,exact):
        self.f = f
        self.max = max
        self.min = min
        self.a = a
        self.b = b
        self.step = step
        self.exact = exact
    
    def get_ConvergenceRate(self):
        error = []
        division = []
        exact_vec = np.full(4,self.exact)
        for m in range(self.min,self.max,self.step):
            x = np.linspace(self.a,self.b,m+1)
            meth = method(x,self.f)
            gau = gauss(self.a,self.b,self.f,m)
            val =np.array([meth.midpoint()[0],meth.trape()[0],meth.simpson()[0],gau.gaussian_quadratures()[0]])
            error.append(np.abs(val-exact_vec))
            hsize = np.array([meth.midpoint()[1],meth.trape()[1],meth.simpson()[1],gau.gaussian_quadratures()[1]])
            division.append(hsize)            
        error = np.array(error)
        division = np.array(division)
        rate = (np.log(error[1:]) - np.log(error[:-1]))/(np.log(division[1:]) - np.log(division[:-1]))
        
        return division, rate, error

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
      
if __name__ == "__main__":
    Q = [q0,q1,q2,q3,q4,q5]
    eval = []
    for q in Q: 
        e = evaluate(f=q.function,
                        max=200,
                        min=10,
                        step=10,
                        a=q.start,
                        b=q.end,
                        exact=q.exact)
        division, rate,error = e.get_ConvergenceRate()
        eval.append([division,rate,error])
    # for j in range(len(eval)):
    #     print(f"result of Q{j+1}:")  
    #     for i in range(eval[j][0].shape[0]-1):
    #         print(f"division:{division[i]}")
    #         print(f"{eval[j][0][i,0]:.3f},{eval[j][1][i,0]:.3f}")  # 中点則に対応
    #         print(f"{eval[j][0][i,1]:.3f},{eval[j][1][i,1]:.3f}")  # 台形則に対応
    #         print(f"{eval[j][0][i,2]:.3f},{eval[j][1][i,2]:.3f}")  # Simpson則に対応
    j = int(input("number of Q:"))
    for i in range(eval[j][0].shape[0]-1):
        print(f"itter = {i}")
        print(f"Midpoint   : r = {eval[j][1][i,0]:.3f}")  # 中点則に対応
        print(f"Trapezoidal: r = {eval[j][1][i,1]:.3f}")  # 台形則に対応
        print(f"Simpson    : r = {eval[j][1][i,2]:.3f}")  # Simpson則に対応
        print(f"Gaussian   : r = {eval[j][1][i,3]:.3f}")  # gauss型積分に対応
    
    
    # データ
    h1,e1 = eval[j][0][:,0], eval[j][2][:,0]
    h2,e2 = eval[j][0][:,1], eval[j][2][:,1]
    h3,e3 = eval[j][0][:,2], eval[j][2][:,2]
    h4,e4 = eval[j][0][:,3], eval[j][2][:,3]
    h_values = [h1, h2, h3, h4]  # 各手法の division
    error_values = [e1, e2, e3, e4]  # 各手法の error
    labels = ['midpoint rule', 'trapezoidal rule', 'Simpson rule',"Gaussian"]
    colors = ['bo-', 'rs-', 'g^-',"y*-"]  # 各手法のスタイル
    # プロット
    fig, ax = plt.subplots()
    for h, err, label, style in zip(h_values, error_values, labels, colors):
        ax.plot(h, err, style, label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('h')
    ax.set_ylabel('error')
    ax.legend()
    ax.grid(True)

    plt.show()
    
    print("\n[分割数200における各積分法の結果]")
    x = np.linspace(Q[j].start, Q[j].end, 201)
    f = Q[j].function
    meth = method(x, f)
    gau = gauss(Q[j].start, Q[j].end, f, 200)

    midpoint_val, midpoint_h = meth.midpoint()
    trape_val, trape_h = meth.trape()
    simpson_val, simpson_h = meth.simpson()
    gaussian_val, gaussian_h = gau.gaussian_quadratures()

    print(f"Midpoint:    h = {midpoint_h:.5f}, value = {midpoint_val:.10f}")
    print(f"Trapezoidal: h = {trape_h:.5e}, value = {trape_val:.10f}")
    print(f"Simpson:     h = {simpson_h:.5e}, value = {simpson_val:.10f}")
    print(f"Gaussian:    h = {gaussian_h:.5e}, value = {gaussian_val:.10f}")
