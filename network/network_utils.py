from torch.nn import init
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.cuda.amp.autocast_mode import autocast
import numpy as np
import torch
import torch.nn as nn


def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    # return np.prod(a - np.arange(k)) / np.math.factorial(k)
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
        l: associated Legendre polynomial degree.
        m: associated Legendre polynomial order.
        k: power of cos(theta).

    Returns:
        A float, the coefficient of the term corresponding to the inputs.
    """
    return (
            (-1) ** m
            * 2 ** l
            * np.math.factorial(l)
            / np.math.factorial(k)
            / np.math.factorial(l - k - m)
            * generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l)
    )


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    # return (np.sqrt(
    #     (2.0 * l + 1.0) * np.math.factorial(l - m) /
    #     (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))
    return np.sqrt(
        (2.0 * l + 1.0)
        * np.math.factorial(l - m)
        / (4.0 * np.pi * np.math.factorial(l + m))
    ) * assoc_legendre_coeff(l, m, k)


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2 ** i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    ml_array = np.array(ml_list).T
    return ml_array


class IntegratedDirEncoder(nn.Module):
    """Module for integrated directional encoding (IDE).
        from Equations 6-8 of arxiv.org/abs/2112.03907.
    """

    def __init__(self, input_dim=3, deg_view=4):
        """Initialize integrated directional encoding (IDE) module.

        Args:
            deg_view: number of spherical harmonics degrees to use.
        
        Raises:
            ValueError: if deg_view is larger than 5.

        """
        super().__init__()
        self.deg_view = deg_view

        if deg_view > 5:
            raise ValueError("Only deg_view of at most 5 is numerically stable.")

        ml_array = get_ml_array(deg_view)
        l_max = 2 ** (deg_view - 1)

        # Create a matrix corresponding to ml_array holding all coefficients, which,
        # when multiplied (from the right) by the z coordinate Vandermonde matrix,
        # results in the z component of the encoding.
        mat = np.zeros((l_max + 1, ml_array.shape[1]))
        for i, (m, l) in enumerate(ml_array.T):
            for k in range(l - m + 1):
                mat[k, i] = sph_harm_coeff(l, m, k)

        sigma = 0.5 * ml_array[1, :] * (ml_array[1, :] + 1)

        self.register_buffer("mat", torch.Tensor(mat), False)
        self.register_buffer("ml_array", torch.Tensor(ml_array), False)
        self.register_buffer("pow_level", torch.arange(l_max + 1), False)
        self.register_buffer("sigma", torch.Tensor(sigma), False)

        self.output_dim = (2 ** deg_view - 1 + deg_view) * 2

    @autocast(enabled=False)
    def forward(self, xyz, roughness=0, **kwargs):
        """Compute integrated directional encoding (IDE).

        Args:
            xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
            kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
                Mises-Fisher distribution.

        Returns:
            An array with the resulting IDE.
        """
        kappa_inv = roughness
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]
        # avoid 0 + 0j exponentiation
        zero_xy = torch.logical_and(x == 0, y == 0)
        y = y + zero_xy

        vmz = z ** self.pow_level
        vmxy = (x + 1j * y) ** self.ml_array[0, :]

        sph_harms = vmxy * torch.matmul(vmz, self.mat)

        ide = sph_harms * torch.exp(-self.sigma * kappa_inv)

        # check whether Nan appears
        if torch.isnan(ide).any():
            print('Nan appears in IDE')
            raise ValueError('Nan appears in IDE')

        return torch.cat([torch.real(ide), torch.imag(ide)], dim=-1)

    def forward_wo_j(self, xyz, roughness=0, **kwargs):  # a non-complex version for web demo
        """Compute integrated directional encoding (IDE).

        Args:
            xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
            kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
                Mises-Fisher distribution.

        Returns:
            An array with the resulting IDE.
        """
        kappa_inv = roughness
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]
        # avoid 0 + 0j exponentiation
        zero_xy = torch.logical_and(x == 0, y == 0)
        y = y + zero_xy

        vmz = z ** self.pow_level
        # vmxy = (x + 1j * y) ** self.ml_array[0, :]
        # euler's formula: e^(i theta) = cos(theta) + i sin(theta)
        vmxy_r = torch.pow(x ** 2 + y ** 2, self.ml_array[0, :] / 2)
        vmxy_theta = torch.atan2(y, x) * self.ml_array[0, :]
        vmxy_x = vmxy_r * torch.cos(vmxy_theta)  # real part
        vmxy_y = vmxy_r * torch.sin(vmxy_theta)  # imaginary part

        z_component = torch.matmul(vmz, self.mat)
        sph_harms_x = vmxy_x * z_component
        sph_harms_y = vmxy_y * z_component

        exp_scale = torch.exp(-self.sigma * kappa_inv)
        ide_x = sph_harms_x * exp_scale
        ide_y = sph_harms_y * exp_scale

        return torch.cat([ide_x, ide_y], dim=-1)


class FreqEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs, log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos), is_numpy=True):

        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        if is_numpy:
            self.freq_bands = freq_bands.numpy().tolist()
        else:
            self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, input, **kwargs):
        out = []
        if self.include_input:
            out.append(input)

        for freq in self.freq_bands:
            out += [p_fn(input * freq) for p_fn in self.periodic_fns]

        out = torch.cat(out, dim=-1)
        return out


def get_xavier_multiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[
            1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    std = get_xavier_multiplier(m, gain)
    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))


def init_weights(net, init_type='xavier_uniform', gain=1):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'xavier_uniform':
                xavier_uniform_(m, gain)
            elif init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [{}] is not implemented'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def init_seq(s, init_type='xavier_uniform', activation=nn.ReLU):
    '''initialize sequential model
    only for linear.
    '''
    for layer in s:
        if isinstance(layer, nn.Linear):
            if isinstance(activation, nn.ReLU):
                init_weights(layer, init_type, nn.init.calculate_gain('relu'))
            elif isinstance(activation, nn.LeakyReLU):
                init_weights(layer, init_type, nn.init.calculate_gain('leaky_relu', activation.negative_slope))
            else:
                init_weights(layer, init_type)
    if isinstance(s[-1], nn.Linear):
        init_weights(s[-1])


class trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


# trunc_exp = _trunc_exp.apply


# https://github.com/CompVis/stable-diffusion/tree/main/ldm/modules/diffusionmodules
def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
