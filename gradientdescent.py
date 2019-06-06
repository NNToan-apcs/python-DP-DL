# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
# Note: eta = learning rate
#
def grad(w):
    print("w=", w)
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0]
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2

def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps 
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n))/(2*eps)
    return g 

def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False 




def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break 
        w.append(w_new)
    return (w, it) 



# check convergence
def has_converged(theta_new, grad):
    return np.linalg.norm(grad(theta_new)) / len(theta_new) < 1e-3

def GD_momentum(theta_init, grad, eta, gamma):
    # Suppose we want to store history of theta
    theta = [theta_init]
    v_old = np.zeros_like(theta_init)
    for it in range(100):
        v_new = gamma*v_old + eta*grad(theta[-1])
        theta_new = theta[-1] - v_new
        if has_converged(theta_new, grad):
            break 
        theta.append(theta_new)
        v_old = v_new
    return theta 
    # this variable includes all points in the path
    # if you just want the final answer, 
    # use `return theta[-1]`

# single point gradient
def sgrad(w, i, rd_id):
    true_i = rd_id[i]
    xi = Xbar[true_i, :]
    yi = y[true_i]
    a = np.dot(xi, w) - yi
    return (xi*a).reshape(2, 1)

def SGD(w_init, grad, eta):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    for it in range(10):
        # shuffle data 
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1 
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count % iter_check_w == 0:
                w_this_check = w_new                 
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:                                    
                    return w
                w_last_check = w_this_check
    return (w, it) 

if __name__ == "__main__":
    np.random.seed(2)

    X = np.random.rand(1000, 1)
    y = 4 + 3 * X + .2*np.random.randn(1000, 1) # noise added

    # Building Xbar 
    one = np.ones((X.shape[0],1))
    Xbar = np.concatenate((one, X), axis = 1)

    A = np.dot(Xbar.T, Xbar)
    b = np.dot(Xbar.T, y)
    w_lr = np.dot(np.linalg.pinv(A), b)
    print('Solution found by formula: w = ',w_lr.T)

    # Display result
    w = w_lr
    w_0 = w[0][0]
    w_1 = w[1][0]
    x0 = np.linspace(0, 1, 2, endpoint=True)
    y0 = w_0 + w_1*x0
    print( 'Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))

    # Draw the fitting line 
    plt.plot(X.T, y.T, 'b.')     # data 
    plt.plot(x0, y0, 'y', linewidth = 2)   # the fitting line
    plt.axis([0, 1, 0, 10])
    plt.show()

    # GD
    w_init = np.array([[2], [1]])
    
    (w1, it1) = myGD(w_init, grad, 100)
    print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))
    (w, it) = SGD(w_init, grad, 1)
    print('Solution found by SGD: w = ', w[-1].T, ',\nafter %d iterations.' %(it+1))