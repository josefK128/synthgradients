class Layer(object):
    
    def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv):
        
        self.weights = (np.random.randn(input_dim, output_dim) * 0.2) - 0.1
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
    

    def forward(self,input):
        self.input = input
        self.output = self.nonlin(self.input.dot(self.weights))
        return self.output



    def backward(self,output_delta):
        self.weight_output_delta = output_delta * self.nonlin_deriv(self.output)

        return self.weight_output_delta.dot(self.weights.T)



    def update(self,alpha=0.1):
        self.weights -= self.input.T.dot(self.weight_output_delta) * alpha


