# layer training 

np.random.seed(1)

num_examples = 1000
output_dim = 12
iterations = 1000

x,y = generate_dataset(num_examples=num_examples, output_dim = output_dim)

batch_size = 10
alpha = 0.1

input_dim = len(x[0])
layer_1_dim = 128
layer_2_dim = 64
output_dim = len(y[0])

layer_1 = Layer(input_dim,layer_1_dim,sigmoid,sigmoid_out2deriv)
layer_2 = Layer(layer_1_dim,layer_2_dim,sigmoid,sigmoid_out2deriv)
layer_3 = Layer(layer_2_dim, output_dim,sigmoid, sigmoid_out2deriv)

for iter in range(iterations):
    error = 0

    for batch_i in range(int(len(x) / batch_size)):
        batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
        batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]  
        
        layer_1_out = layer_1.forward(batch_x)
        layer_2_out = layer_2.forward(layer_1_out)
        layer_3_out = layer_3.forward(layer_2_out)

        layer_3_delta = layer_3_out - batch_y
        layer_2_delta = layer_3.backward(layer_3_delta)
        layer_1_delta = layer_2.backward(layer_2_delta)
        layer_1.backward(layer_1_delta)
        
        layer_1.update()
        layer_2.update()
        layer_3.update()
        
        error += (np.sum(np.abs(layer_3_delta * layer_3_out * (1 - layer_3_out))))

    sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
    if(iter % 100 == 99):
        print("")
