import numpy as np
import matplotlib.pyplot as plt

X = np.array([[2.5, 1],[5.1,1],[3.2,1],[8.5,1],[3.5,1], [1.5, 1], [9.2, 1], [5.5, 1],[8.3,1],[2.7,1],[7.7,1],[5.9,1],[4.5,1],[3.3,1],[1.1,1],[8.9,1],[2.5,1],[1.9,1],[6.1,1]])
Y = np.array([[21],[47],[27],[75],[30], [20], [88], [60],[81],[25],[85],[62],[41],[42],[17],[95],[30],[24],[67]])
W = np.zeros((X.shape[1],1))


def compute_cost(X, Y, W):
    m = X.shape[0]
    Y_hat = np.dot(X, W)
    error = Y_hat - Y
    cost = np.sum(error ** 2)
    total_cost = cost / (2 * m)
    return total_cost

plt.scatter(X.transpose()[0],Y.transpose(),marker='*',c='g')



def batch_grad_descent(X, Y, W, alpha, epoch):
    for i in range(epoch):
    
        Y_hat = np.dot(X, W)
        
        plt.plot(X.transpose()[0],Y_hat,color='b') #training lines of convergence
        error = Y_hat - Y
        gradient = np.dot(X.transpose(), error)
        W = W - alpha * gradient
        #print(i," ",compute_cost(X,Y,W))
    
    return W


w = batch_grad_descent(X, Y, W, 0.001, 60)

def predict(X,w):
    pred=np.dot(X,w)
    return pred[0,0]

x=np.array([[9.25,1]])

print("Score of a student while studying for",x[0,0],"hr/day is:",predict(x,w),"%")

result=np.dot(X,w)

plt.plot(X.transpose()[0],result,color='r') #final line of convergence
plt.title("Score v/s Studying hour")
plt.xlabel("Studying hours")
plt.ylabel("Score(%)")
plt.show()




