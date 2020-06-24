class DNI(object):

    def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv,alpha = 0.1):

        print("DNI init:")
        print("input_dim = " + str(input_dim))
        print("output_dim = " + str(output_dim))
        print("alpha = " + str(alpha))
        print("\n")
        
        self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
        self.weights_synthetic_grads = 
            (np.random.randn(output_dim,output_dim) * 0.2) - 0.1
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        self.alpha = alpha
    


    def forward_and_synthetic_update(self,input):
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights))
        
        self.synthetic_gradient = self.output.dot(self.weights_synthetic_grads)
        
        self.weight_synthetic_gradient = self.synthetic_gradient * 
            self.nonlin_deriv(self.output)
        
        self.weights += self.input.T.dot(self.weight_synthetic_gradient) * 
            self.alpha
        
        return self.weight_synthetic_gradient.dot(self.weights.T), self.output



    def update_synthetic_weights(self,true_gradient):
        self.synthetic_gradient_delta = self.synthetic_gradient - true_gradient
        
        self.weights_synthetic_grads += 
            self.output.T.dot(self.synthetic_gradient_delta) * self.alpha

