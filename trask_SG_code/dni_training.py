# dni training

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

layer_1 = DNI(input_dim,layer_1_dim,sigmoid,sigmoid_out2deriv,alpha)
layer_2 = DNI(layer_1_dim,layer_2_dim,sigmoid,sigmoid_out2deriv,alpha)
layer_3 = DNI(layer_2_dim, output_dim,sigmoid, sigmoid_out2deriv,alpha)

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
