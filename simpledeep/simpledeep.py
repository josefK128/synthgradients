# simpledeep.py
# simpledeep 2-layer feed-forward back-propagation neural net
# layer0 is input-constant data (4x3)
# layer1 is <changing> hidden layer 
# layer2 is <changing> hidden layer estimate of output 
# synapse0 is <changing> vector of weights transforming layer0 to layer1
# synapse1 is <changing> vector of weights modifying layer1 to layer2 appx outp
# y is constant target output
#
# usage: simpledeep> py simpledeep.py [iterations=4, [diagnostics=True]]
# realistic usage: simpledeep> py simpledeep.py 1000 False

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
    y = np.array([[0,1,1,0]]).T   
    
    # input, output (same for all training iterations) diagnostics
    print("data input-constant x (layer0=x) (4x3)")
    print(x)
    print("\ntarget output-constant y (4x1)")
    print(y)


    # set seed for np.random so random values are 'repeatable' for consistency
    np.random.seed(1)
    
    # initialize hidden layer weights synapse0 - shape 3x4, mean=0
    synapse0 = 2.0*np.random.random((3,4)) - 1.0
    
    # initialize hidden layer weights synapse1 - shape 4x1, mean=0
    synapse1 = 2.0*np.random.random((4,1)) - 1.0
    
    
    # training iteration
    for i in range(int(iterations)):
        #forward propagation
        layer0 = x   #input 4x3
        if diag == False:
            print("\n\n**** layer0=x(input), synapse0(weights): iteration=" + str(i))
        
        layer1 = sigmoid(np.dot(layer0, synapse0))  #(4x3)*(3x4) => 4x4
        if diag == False:
            print("hidden layer1=dot(layer0,synapse0) (4x1)")
            print(layer1)
        
        layer2 = sigmoid(np.dot(layer1, synapse1))  #(4x4)*(4x1) => 4x1
        if diag == False:
            print("hidden layer1=dot(layer0,synapse0) estmate of output (4x1)")
            print(layer1)
         
        error2 = y - layer2
        if diag == False:
            print("error2 between output and estimate layer2 (y-layer2) (4x1)")
            print(error2)
        
        delta2 = error2 * dsigmoid(layer2)   #layer2 is points on sigmoid
        if diag == False:
            print("delta = error * dsigmoid(layer1) (4x1)")
            print(delta)


        error1 = delta2.dot(synapse1.T)
        if diag == False:
            print("error1 - contribution of layer1 value to error2 (4x1)")
            print(error)

        delta1 = error1 * dsigmoid(layer1)   #layer1 is points on sigmoid
        if diag == False:
            print("delta = error * dsigmoid(layer1) (4x1)")
            print(delta)


        #update weights
        #synapse1 1x4
        synapse1 += layer1.T.dot(delta2)
        if diag == False:
            print("synapse1(weights) += layer1.T.dot(delta2) (1x4)")
            print(synapse1)
    
        #synapse0 3x4
        synapse0 += layer0.T.dot(delta1)
        if diag == False:
            print("synapse0(weights) += dot(layer0.T,delta) (3x4)")
            print(synapse0)
    
    
    # final diagnostics
    print("\n\n@@@@@@@@@@final synapse0(weights) += layer0.T.dot(delta1) (3x4)")
    print(synapse0)
    print("\nfinal synapse1(weights) += layer1.T.dot(delta2) (4x1)")
    print(synapse1)
    print("\nfinal hidden layer2=sigm(dot(layer1,synapse1)) op-est (4x4)*(4x1)=(4x1)")
    print(layer2)





if __name__ == "__main__": 
    print('\n+++++++++++ simpledeep.py +++++++++++++++++++++')
    print("simpledeep.py running as __main__")
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
    print("simpledeep.py module imported")
