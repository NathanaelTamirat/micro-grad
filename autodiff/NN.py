import random 
from engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0
    def parameters(self):
        return []    
class Neuron(Module):
    def __init__(self,nin):
        self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b=Value(random.uniform(-1,1))
    def __call__(self,x):
        act=sum((wi*xi for wi,xi in zip(self.w,x)),self.b) # ,self.b means adding the bais  zip(self.w,x) is parining the w and x(it like concat)
        out=act.relu()
        return out   
    def parameters(self):
        return self.w+[self.b]   
    # def __repr__(self):
    #     return f"{'tanh' if self.nonlin else 'Linear'} Neuron({len(self.w)})"   
class Layer(Module):
    def __init__(self,nin,nout):
        self.neurons=[Neuron(nin) for _ in range(nout)]
    def __call__(self,x):
        outs=[n(x) for n in self.neurons]
        return outs  
    def parameters(self):
        # params=[]
        # for neuron in self.neurons:
        #     ps=neuron.parameters()
        #     params.extend(ps)
        # return params    
        return [p for n in self.neurons for p in n.parameters()]

    # def __rep__(self):
    #     return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"   
class MLP(Module):
    def __init__(self,nin,nouts):
        sz=[nin]+nouts
        self.layers=[Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    # def __repr__(self):
    #     return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    