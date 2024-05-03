import math
class Value:
    def __init__(self,data,_children=(),_op="",label=""):
        self.data=data
        self.grad=0     #initialize it to zero
        self._backward=lambda: None
        self._prev=set(_children)
        self._op=_op
        self.label=label
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other) 
        out =Value(self.data*other.data,(self,other),"*")      
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*other.grad
        out._backward=_backward
        return out
    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)  
        out=Value(self.data+other.data,(self,other),"+")
        def _backward():
            self.grad+=out.grad
            other.grad+=out.grad
        out._backward=_backward
        return out 
    def backward(self):
        #topological order of the childern graph
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        #go one variable at a time and apply the chain rule to get its gradient
        self.grad=1     
        for v in reversed(topo):
            v._backward()
    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out=Value(t,(self,),"tanh")
        def _backward():
            self.grad=(1-t**2)*out.grad
        out._backward=_backward
        return out 
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        x=self.data
        out=Value(math.exp(x),(self,),"exp")
        def _backward():
            self.grad=out.data*out.grad
        out._backward=_backward
        return out
    def __pow__(self,other):
        assert isinstance(other,(int,float)), "only supporting int/float powers"
        out=Value(self.data**other,(self,),f'**{other}')

        def _backward():
            self.grad+=(other*self.data**(other-1))*out.grad
        out._backward=_backward
        return out

    # to access reverse operations ex. self.__mul__(other)  or other.__mul__(self) 
    def __neg__(self):
        return self*-1
    def __radd__(self,other):
        return self+other
    def __sub__(self,other):
        return self+(-other)
    def __rsub__(self,other):
        return other+(-self)
    def __repr__(self):
        return f"Value:data={self.data} grad={self.grad}"
    def __truediv__(self,other):
        return self*(other**-1)
    def __rtruediv__(self,other):
        return other*(self**-1)