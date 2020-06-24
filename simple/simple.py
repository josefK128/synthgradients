# simple.py
# simple 2-layer feed-forward back-propagation neural net
# layer0 is input-constant data (4x3)
# layer1 is <changing> hidden layer estimate of output 
# synapse0 is <changing> vector of weights modifying layer1 to approx output
# y is constant target output
#
# usage: simple> py simple.py [iterations=4, [diagnostics=True]]
# realistic usage: simple> py simple.py 1000 False

# dependencies
import numpy as np
import sys



# activation sigmoid-function and its derivative 
# sigmoid f
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# dsigmoid f - derivative of sigmoid
def dsigmoid(sigma):
    return sigma*(1.0 - sigma)



# model-building synapse0 weights training function 
def action(iterations="4", diag=True):
    # input dataset x(4x3)
    x = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    
    # output dataset - target y(4x1)  NOTE: T is transpose
    y = np.array([[0,0,1,1]]).T   
    
    # input, output (same for all training iterations) diagnostics
    print("data input-constant x (layer0=x) (4x3)")
    print(x)
    print("\ntarget output-constant y (4x1)")
    print(y)


    # set seed for np.random so random values are 'repeatable' for consistency
    np.random.seed(1)
    
    # initialize hidden layer weights synapse0 - shape 3x1, mean=0
    synapse0 = 2.0*np.random.random((3,1)) - 1.0
    
    
    # training iteration
    for i in range(int(iterations)):
        #forward propagation
        layer0 = x   #input 4x3
        if diag == False:
            print("\n\n**** layer0=x(input), synapse0(weights): iteration=" + str(i))
        
        layer1 = sigmoid(np.dot(layer0, synapse0))  #(4x3)*(3x1) => 4x1
        if diag == False:
            print("hidden layer1=dot(layer0,synapse0) estmate of output (4x1)")
            print(layer1)
        
        error = y - layer1
        if diag == False:
            print("error between output and estimate layer1 (y-layer1) (4x1)")
            print(error)
        
        delta = error * dsigmoid(layer1)   #layer1 is points on sigmoid
        if diag == False:
            print("delta = error * dsigmoid(layer1) (4x1)")
            print(delta)
    
        #update weights
        synapse0 += np.dot(layer0.T, delta)
        if diag == False:
            print("synapse0(weights) += dot(layer0.T,delta) (3x1)")
            print(synapse0)
    
    
    # final diagnostics
    print("\n\n@@@@@@@@@@@ final delta = error * dsigmoid(layer1) (4x1)")
    print(delta)
    print("\nfinal synapse0(weights) += dot(layer0.T,delta) (3x1)")
    print(synapse0)
    print("\nfinal hidden layer1=dot(layer0,synapse0) op-est (4x3)*(3x1)=(4x1)")
    print(layer1)





if __name__ == "__main__": 
    print('\n+++++++++++ simple.py +++++++++++++++++++++')
    print("simple.py running as __main__")
    nargs = len(sys.argv) - 1
    position = 1
    iterations = 4
    diagnostics = True
    while nargs >= position:
        #print('simple: sys.argv[' + str(position) + '] = ' + str(sys.argv[position]))
        position += 1

    if nargs == 2:
        iterations = sys.argv[1]
        diagnostics = sys.argv[2]
        print("using iterations=" + str(iterations) + " and diagnostics=" + str(diagnostics))
    elif nargs == 1:
        iterations = sys.argv[1] 
        print("using iterations=" + str(iterations) + " and diagnostics=True")
    else:
        print("using iterations=4 and diagnostics=True")

    action(iterations, diagnostics)

else:
    print("simple.py module imported")
