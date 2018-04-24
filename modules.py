import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def conv_weight(in_planes, planes, kernel_size=3, stride=1, padding=0, bias=False, transpose=False):
    " init convolutions parameters, necessary due to code architecture "
    if transpose:
        params = nn.ConvTranspose2d(in_planes, planes, kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=bias).weight.data
    else:
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

class C_convtranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(C_convtranspose2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.weight_real = nn.Parameter(conv_weight(in_channels, out_channels, kernel_size, stride, padding,transpose=True), requires_grad=True)
        self.weight_imag = nn.Parameter(conv_weight(in_channels, out_channels, kernel_size, stride, padding, transpose=True), requires_grad=True)

    def forward(self, complex):
        x_ = F.conv_transpose2d(complex.real, self.weight_real, stride=self.stride, padding=self.padding) - \
             F.conv_transpose2d(complex.imag, self.weight_imag, stride=self.stride, padding=self.padding)
        y_ = F.conv_transpose2d(complex.imag, self.weight_real, stride=self.stride, padding=self.padding) + \
             F.conv_transpose2d(complex.real, self.weight_imag, stride=self.stride, padding=self.padding)
        return Complex(x_, y_)

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


class C_Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(C_Linear, self).__init__()
        self.weight_real = nn.Parameter(nn.Linear(in_dim, out_dim).weight.data, requires_grad=True)
        self.weight_imag = nn.Parameter(nn.Linear(in_dim, out_dim).weight.data, requires_grad=True)
    def forward(self, complex):
        x_ = F.linear(complex.real, self.weight_real) - F.linear(complex.imag, self.weight_imag)
        y_ = F.linear(complex.real, self.weight_imag) + F.linear(complex.imag, self.weight_real)
        return Complex(x_, y_)

class C_ReLU(nn.Module):
    def __init__(self):
        super(C_ReLU, self).__init__()

    def forward(self, complex):
        return Complex(F.relu(complex.real), F.relu(complex.imag))

class C_LeakyReLU(nn.Module):
    def __init__(self, alpha=0.001):
        super(C_LeakyReLU, self).__init__()
        self.alpha = alpha
    def forward(self, complex):
        return Complex(F.leaky_relu(complex.real, self.alpha), F.leaky_relu(complex.imag, self.alpha))



class Mod_ReLU(nn.Module):
    def __init__(self, channels):
        super(Mod_ReLU, self).__init__()
        self.b = nn.Parameter(torch.FloatTensor(channels).fill_(0),requires_grad=True)

    def forward(self, complex):
        mag = complex.mag()
        if len(mag.shape) > 2:
            mag = F.relu(mag + self.b[None, :, None, None])
        else:
            mag = F.relu(mag + self.b[None, :])
        res = Complex()
        res.from_polar(mag, complex.phase())
        return res

class C_BatchNorm2d(nn.Module):
    def __init__(self):
        super(C_BatchNorm2d, self).__init__()

    def forward(self, complex):
        # TODO
        raise NotImplementedError

def complex_weight_init(m):
    classname = m.__class__.__name__
    if classname.find('C_Linear') != -1:
       # real weigths
        fan_in_real, fan_out_real = nn.init._calculate_fan_in_and_fan_out(m.weight_real.data)
        s_real = 1. / (fan_in_real + fan_out_real) # glorot or xavier criterion
        rng_real = np.random.RandomState(999)
        modulus_real = rng_real.rayleigh(scale=s_real, size=m.weight_real.data.shape)
        phase_real = rng_real.uniform(low=-np.pi, high=np.pi, size=m.weight_real.data.shape)
        weight_real = torch.from_numpy(modulus_real) * torch.cos(torch.from_numpy(phase_real))
        # imag weights
        fan_in_imag, fan_out_imag = nn.init._calculate_fan_in_and_fan_out(m.weight_imag.data)
        s_imag = 1. / (fan_in_imag + fan_out_imag) # glorot or xavier criterion
        rng_imag = np.random.RandomState(999)
        modulus_imag = rng_imag.rayleigh(scale=s_imag, size=m.weight_imag.data.shape)
        phase_imag = rng_imag.uniform(low=-np.pi, high=np.pi, size=m.weight_imag.data.shape)
        weight_imag = torch.from_numpy(modulus_imag) * torch.cos(torch.from_numpy(phase_imag))

    if classname.find('C_conv2d') != -1:
        # real weigths
        fan_in_real, fan_out_real = nn.init._calculate_fan_in_and_fan_out(m.weight_real.data)
        s_real = 1. / (fan_in_real + fan_out_real) # glorot or xavier criterion
        rng_real = np.random.RandomState(999)
        modulus_real = rng_real.rayleigh(scale=s_real, size=m.weight_real.data.shape)
        phase_real = rng_real.uniform(low=-np.pi, high=np.pi, size=m.weight_real.data.shape)
        weight_real = torch.from_numpy(modulus_real) * torch.cos(torch.from_numpy(phase_real))
        # imag weights
        fan_in_imag, fan_out_imag = nn.init._calculate_fan_in_and_fan_out(m.weight_imag.data)
        s_imag = 1. / (fan_in_imag + fan_out_imag) # glorot or xavier criterion
        rng_imag = np.random.RandomState(999)
        modulus_imag = rng_imag.rayleigh(scale=s_imag, size=m.weight_imag.data.shape)
        phase_imag = rng_imag.uniform(low=-np.pi, high=np.pi, size=m.weight_imag.data.shape)
        weight_imag = torch.from_numpy(modulus_imag) * torch.cos(torch.from_numpy(phase_imag))

    if classname.find('C_BatchNorm2d') != -1:
        # real weigths
        fan_in_real, fan_out_real = nn.init._calculate_fan_in_and_fan_out(m.weight_real.data)
        s_real = 1. / (fan_in_real + fan_out_real) # glorot or xavier criterion
        rng_real = np.random.RandomState(999)
        modulus_real = rng_real.rayleigh(scale=s_real, size=m.weight_real.data.shape)
        phase_real = rng_real.uniform(low=-np.pi, high=np.pi, size=m.weight_real.data.shape)
        weight_real = torch.from_numpy(modulus_real) * torch.cos(torch.from_numpy(phase_real))
        # imag weights
        fan_in_imag, fan_out_imag = nn.init._calculate_fan_in_and_fan_out(m.weight_imag.data)
        s_imag = 1. / (fan_in_imag + fan_out_imag) # glorot or xavier criterion
        rng_imag = np.random.RandomState(999)
        modulus_imag = rng_imag.rayleigh(scale=s_imag, size=m.weight_imag.data.shape)
        phase_imag = rng_imag.uniform(low=-np.pi, high=np.pi, size=m.weight_imag.data.shape)
        weight_imag = torch.from_numpy(modulus_imag) * torch.cos(torch.from_numpy(phase_imag))

class Sample(nn.Module):
    """
    Foo model
    """
    def __init__(self):
        super(Sample, self).__init__()
        self.conv1 = C_convtranspose2d(3, 3,3,1,1)
        self.relu = C_ReLU()
        self.conv2 = C_convtranspose2d(3, 3,3,1,1)
        self.mod_relu = Mod_ReLU(3)
        self.conv3 = C_convtranspose2d(3, 3,3,1,1)
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
    complex_weight_init(conv1) # conv layer weight init.
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
    model.apply(complex_weight_init) # apply complex weights initialization
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
