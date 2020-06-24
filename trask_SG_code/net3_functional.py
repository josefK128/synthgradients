# net3 functional code containing class method code

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

weights_0_1 = (np.random.randn(input_dim,layer_1_dim) * 0.2) - 0.1
weights_1_2 = (np.random.randn(layer_1_dim,layer_2_dim) * 0.2) - 0.1
weights_2_3 = (np.random.randn(layer_2_dim,output_dim) * 0.2) - 0.1


for iter in range(iterations):
    error = 0

    for batch_i in range(int(len(x) / batch_size)):
        batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
        batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]    

        layer_0 = batch_x
        layer_1 = sigmoid(layer_0.dot(weights_0_1))
        layer_2 = sigmoid(layer_1.dot(weights_1_2))
        layer_3 = sigmoid(layer_2.dot(weights_2_3))

        layer_3_delta = (layer_3 - batch_y) * layer_3  * (1 - layer_3)
        layer_2_delta = layer_3_delta.dot(weights_2_3.T) * layer_2 * (1 - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * layer_1 * (1 - layer_1)

        weights_0_1 -= layer_0.T.dot(layer_1_delta) * alpha
        weights_1_2 -= layer_1.T.dot(layer_2_delta) * alpha
        weights_2_3 -= layer_2.T.dot(layer_3_delta) * alpha

        error += (np.sum(np.abs(layer_3_delta)))

    sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error))
    if(iter % 100 == 99):
        print("")
