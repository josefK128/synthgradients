# backprop.py
# backprop 3-layer feed-forward back propagation neural net corresponding
# to 'Backpropagation Algorithm' whitepaper analysis text:
# (input X - hidden Z - output Y)
# layer0 is input-constant data (4x3)
# layer1 is <changing> hidden layer 
# layer2 is <changing> hidden layer estimate of output 
# w0 is <changing> vector of weights transforming layer0 to layer1
# w1 is <changing> vector of weights modifying layer1 to layer2 appx outp
# y is constant target output
#
# usage: backprop> py backprop.py [iterations=4, [diagnostics=True]]
# realistic usage: backprop> py backprop.py 1000 False

# dependencies
import numpy as np
import sys



# activation sigmoid-function and its derivative 
# sigmoid f
# arg is real number x
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# dsigmoid f - derivative of sigmoid - slope of tangent to point on sigma curve
# arg is value on sigma curve, i.e. sigma = sigma(x)
def dsigmoid(sigma):
    return sigma*(1.0 - sigma)



# model-building w0 weights training function 
def action(iterations=4, diag=True):
    print("action: iterations = " + str(iterations) + " diag = " + str(diag))

    # constant input dataset _input(4x3)
    _input = np.array([[0,0,1],
                       [0,1,1],
                       [1,0,1],
                       [1,1,1]])
    
    # constant target output dataset - _output(4x1)  NOTE: T is transpose
    _output = np.array([[0,1,1,0]]).T   
    
    # input, output (same for all training iterations) diagnostics
    print("\nconstant data _input (layer0 =_input) for each training iteration) (4x3)")
    print(_input)
    print("constant target _output (4x1)")
    print(_output)


    # set seed for np.random so random values are 'repeatable' for consistency
    np.random.seed(1)
    
    # initialize hidden layer weights w0 - shape 3x4, mean=0, w0 in [-1,1)
    # these weights are between input layer0 (X) and layer1 (Z) (w[i][j])
    w0 = 2.0*np.random.random((3,4)) - 1.0
    
    # initialize hidden layer weights w1 - shape 4x1, mean=0, w1 in [-1,1)
    # these weights are between hidden layer1 (Z) and layer2 (Y) (w[j][k])
    w1 = 2.0*np.random.random((4,1)) - 1.0
    print("------------------------------------------------")
    
 

    # training iteration
    for i in range(int(iterations)):
        if diag == True:
            print("*** iteration = " + str(i) + "\n")

        ### forward propagation
        layer0 = _input   #input 4x3
        if diag == True:
            print("input layer0 = _input ")
            print(layer0)
        
        layer1 = sigmoid(np.dot(layer0, w0))  #(4x3)*(3x4) => 4x4
        if diag == True:
            print("hidden layer1=dot(layer0,w0) (4x1)")
            print(layer1)
        
        layer2 = sigmoid(np.dot(layer1, w1))  #(4x4)*(4x1) => 4x1
        if diag == True:
            print("prediction layer2=dot(layer1,w1) estimate of output (4x1)")
            print(layer2)
            print("constant target _output (4x1)")
            print(_output)

         
        ### back propagation
        error2 = _output - layer2
        if i%1000 == 0:
            print("\nerror2 at iteration " + str(i) + ":")
            print(error2)
        if diag == True:
            print("\nerror2 between output and estimate layer2 (y-layer2) (4x1)")
            print(error2)
        
        delta2 = error2 * dsigmoid(layer2)   #layer2 is points on sigmoid
        if diag == True:
            print("delta2 = error2 * dsigmoid(layer1) (4x1)")
            print(delta2)


        error1 = delta2.dot(w1.T)
        if diag == True:
            print("\nerror1 - contribution of layer1 value to error2 (4x1)")
            print(error1)

        delta1 = error1 * dsigmoid(layer1)   #layer1 is points on sigmoid
        if diag == True:
            print("delta1 = error1 * dsigmoid(layer1) (4x1)")
            print(delta1)


        #update weights
        #w1 1x4
        w1 += layer1.T.dot(delta2)
        if diag == True:
            print("\nw1(weights) += layer1.T.dot(delta2) (1x4)")
            print(w1)
    
        #w0 3x4
        w0 += layer0.T.dot(delta1)
        if diag == True:
            print("w0(weights) += dot(layer0.T,delta) (3x4)")
            print(w0)
            print("------------------------------------------------")
    
    


    # final diagnostics
    print("\n\n@@@@@@final ip hidden w0(weights) += layer0.T.dot(delta1) (3x4)")
    print(w0)
    print("\nfinal hidden layer1=dot(layer0,w0) (4x3)*(3x4)=(4x4)")
    print(layer1)
    print("\nfinal hidden output w1(weights) += layer1.T.dot(delta2) (4x1)")
    print(w1)
    print("\nfinal op prediction layer2=sigm(dot(layer1,w1)) (4x4)*(4x1)=(4x1)")
    print(layer2)
    print("\nfinal error2 = output-layer2 (4x1)")
    print(error2)





if __name__ == "__main__": 
    print('\n+++++++++++ backprop.py +++++++++++++++++++++')
    print("backprop.py running as __main__")
    nargs = len(sys.argv) - 1
    position = 1
    iterations = 4
    diagnostics = True

    print("nargs = " + str(nargs))
    while nargs >= position:
        #print('backprop: sys.argv[' + str(position) + '] = ' + str(sys.argv[position]))
        position += 1

    if nargs == 2:
        iterations = int(sys.argv[1])
        s = sys.argv[2].lower()
        if(s == "false" or s == "f"):
            diagnostics = False
        else:
            diagnostics = True
    elif nargs == 1:
        iterations = int(sys.argv[1]) 

    action(iterations, diagnostics)

else:
    print("backprop.py module imported")
