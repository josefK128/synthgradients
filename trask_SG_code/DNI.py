# DNI.py
import numpy as np
import sys

def generate_dataset(output_dim = 8,num_examples=1000):
    def int2vec(x,dim=output_dim):
        out = np.zeros(dim)
        binrep = np.array(list(np.binary_repr(x))).astype('int')
        out[-len(binrep):] = binrep
        return out

    x_left_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    x_right_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    y_int = x_left_int + x_right_int

    x = list()
    for i in range(len(x_left_int)):
        x.append(np.concatenate((int2vec(x_left_int[i]),int2vec(x_right_int[i]))))

    y = list()
    for i in range(len(y_int)):
        y.append(int2vec(y_int[i]))

    x = np.array(x)
    y = np.array(y)
    
    print("generate_data:")
    print("len(x) = " + str(len(x)))
    print("len(y) = " + str(len(y)))
    print("x[0] = " + str(x[0]))
    print("y[0] = " + str(y[0]))
    print("\n")
    
    return (x,y)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(out):
    return out * (1 - out)


#------------------------------------------------------------
class DNI(object):
    
    def __init__(self,input_dim, output_dim,nonlinear,dnonlinear,alpha = 0.1):
        dni = self

        print("DNI init:")
        print("input_dim = " + str(input_dim))
        print("output_dim = " + str(output_dim))
        print("alpha = " + str(alpha))
        print("\n")
        
        dni.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
        dni.sggen_weights = (np.random.randn(output_dim,output_dim) * 0.2) - 0.1
        dni.nonlinear = nonlinear
        dni.dnonlinear = dnonlinear
        dni.alpha = alpha
    
    def forward_and_synthetic_update(dni,input):
        dni.input = input
        dni.output = dni.nonlinear(dni.input.dot(dni.weights))
        
        dni.sg = dni.output.dot(dni.sggen_weights)
        dni.weighted_sg = dni.sg * dni.dnonlinear(dni.output)
        dni.weights += dni.input.T.dot(dni.weighted_sg) * dni.alpha
        
        return dni.weighted_sg.dot(dni.weights.T), dni.output
    
    def update_synthetic_weights(dni,true_gradient):
        dni.sg_delta = dni.sg - true_gradient
        dni.sggen_weights += dni.output.T.dot(dni.sg_delta) * dni.alpha
#------------------------------------------------------------

np.random.seed(1)

num_examples = 1000
output_dim = 12
iterations = 1000

x,y = generate_dataset(num_examples=num_examples, output_dim = output_dim)

batch_size = 1000
alpha = 0.0001

input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = len(y[0])

layer_1 = DNI(input_dim,layer_1_dim,sigmoid,dsigmoid,alpha)
layer_2 = DNI(layer_1_dim,layer_2_dim,sigmoid,dsigmoid,alpha)
layer_3 = DNI(layer_2_dim, output_dim,sigmoid, dsigmoid,alpha)

for iter in range(iterations):
    error = 0

    for batch_i in range(int(len(x) / batch_size)):
        batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
        batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]  
        
        _, layer_1_out = layer_1.forward_and_synthetic_update(batch_x)
        layer_1_delta, layer_2_out = layer_2.forward_and_synthetic_update(layer_1_out)
        layer_2_delta, layer_3_out = layer_3.forward_and_synthetic_update(layer_2_out)

        layer_3_delta = layer_3_out - batch_y
        layer_3.update_synthetic_weights(layer_3_delta)
        layer_2.update_synthetic_weights(layer_2_delta)
        layer_1.update_synthetic_weights(layer_1_delta)
        
        error += (np.sum(np.abs(layer_3_delta * layer_3_out * (1 - layer_3_out))))

    if(error < 0.1):
        sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
        print("")
        break       
        
    sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
    if(iter % 100 == 99):
        print("")
