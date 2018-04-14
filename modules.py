import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv_weight(in_planes, planes, kernel_size=3, stride=1, padding=0, bias=False):
    " init convolutions parameters, necessary due to code architecture "
    params = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride,
                       padding=padding, bias=bias).weight.data
    return params

class Complex(nn.Module):
    def __init__(self, real=None, imag=None):
        super(Complex, self).__init__()
        self.real = real
        if imag is None and real is not None:
            self.imag = torch.zeros_like(self.real)
        elif imag is None and real is None:
            self.imag = None
        else:
            self.imag = imag

    def mag(self):
        return torch.sqrt(self.real**2 + self.imag**2)

    def phase(self):
        return torch.atan2(self.imag, self.real)

    def from_polar(self, mag, phase):
        self.real = mag*torch.cos(phase)
        self.imag = mag*torch.sin(phase)
        return

    def __repr__(self):
        print(f'Complex Variable containing:\nreal:\n{self.real}imaginary:\n{self.imag}')
        return ''

class C_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(C_conv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weight_real = nn.Parameter(conv_weight(in_channels, out_channels, kernel_size, stride, padding), requires_grad=True)
        self.weight_imag = nn.Parameter(conv_weight(in_channels, out_channels, kernel_size, stride, padding), requires_grad=True)

    def forward(self, complex):
        x_ = F.conv2d(complex.real, self.weight_real, stride=self.stride, padding=self.padding) - \
             F.conv2d(complex.imag, self.weight_imag, stride=self.stride, padding=self.padding)
        y_ = F.conv2d(complex.imag, self.weight_real, stride=self.stride, padding=self.padding) + \
             F.conv2d(complex.real, self.weight_imag, stride=self.stride, padding=self.padding)
        return Complex(x_, y_)

class C_ReLU(nn.Module):
    def __init__(self):
        super(C_ReLU, self).__init__()

    def forward(self, complex):
        return Complex(F.relu(complex.real), F.relu(complex.imag))


class Mod_ReLU(nn.Module):
    def __init__(self, channels):
        super(Mod_ReLU, self).__init__()
        self.b = nn.Parameter(torch.FloatTensor(channels).fill_(0),requires_grad=True)

    def forward(self, complex):
        mag = complex.mag()
        mag = F.relu(mag + self.b[None, :, None, None])
        res = Complex()
        res.from_polar(mag, complex.phase())
        return res

class C_BatchNorm2d(nn.Module):
    def __init__(self):
        super(C_BatchNorm2d, self).__init__()

    def forward(self, complex):
        # TODO
        raise NotImplementedError

def complex_weight_init():
    # TODO
    raise NotImplementedError

class Sample(nn.Module):
    """
    Foo model
    """
    def __init__(self):
        super(Sample, self).__init__()
        self.conv1 = C_conv2d(3, 3,3,1,1)
        self.relu = C_ReLU()
        self.conv2 = C_conv2d(3, 3,3,1,1)
        self.mod_relu = Mod_ReLU(3)
        self.conv3 = C_conv2d(3, 3,3,1,1)
    def forward(self, complex):
        complex = self.conv1(complex)
        complex = self.relu(complex)
        complex = self.conv2(complex)
        complex = self.mod_relu(complex)
        return self.conv3(complex)

def test_1():
    from torch import optim
    a = Variable(torch.rand(2,3,5,5),requires_grad=True)
    b = Variable(torch.rand(2,3,5,5), requires_grad=True)
    complex = Complex(a, b)
    conv1 = C_conv2d(3, 3,3,1,1)
    prev = list(conv1.parameters())[0].clone()
    res = conv1(complex)
    optimizer = optim.Adam(conv1.parameters(), 0.1)
    foo = Variable(torch.rand(2,3,5,5))
    for i in range(10):
        complex_res = conv1(complex)
        loss = F.mse_loss(complex_res.mag(), foo)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_2():
    from torch import optim
    import numpy as np
    a = Variable(torch.rand(2,3,5,5),requires_grad=True)
    b = Variable(torch.rand(2,3,5,5), requires_grad=True)
    complex = Complex(a, b)
    model = Sample()
    parameters_start = [p.clone() for p in model.parameters()]
    prev = list(model.parameters())[0].clone()
    optimizer = optim.Adam(model.parameters(), 0.1)
    foo = Variable(torch.rand(2,3,5,5))
    for i in range(10):
        complex_res = model(complex)
        loss = F.mse_loss(complex_res.mag(), foo)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for l1, l2 in zip(parameters_start,list(model.parameters())):
        assert np.array_equal(l1.cpu().data.numpy(), l2.cpu().data.numpy()) == False

test_1()
test_2()
