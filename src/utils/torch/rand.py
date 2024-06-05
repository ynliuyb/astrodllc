import torch
import numpy as np
import utils.torch.modules as modules

# function to transform "noise" using a given mean and scale
def transform(eps, mu, scale):
    sample = mu + scale * eps
    return sample

# function to sample from a Logistic distribution (mu=0, scale=1)
def logistic_eps(shape, device, bound=1e-5):
    # sample from a Gaussian
    u = torch.rand(shape, device=device)

    # clamp between two bounds to ensure numerical stability
    u = torch.clamp(u, min=bound, max=1 - bound)

    # transform to a sample from the Logistic distribution 转化为Logistic分布中的样本
    eps = torch.log(u) - torch.log1p(-u)
    return eps

# function to calculate the log-probability of x under a Logistic(mu, scale) distribution
def logistic_logp(mu, scale, x):
    _y = -(x - mu) / scale
    _logp = -_y - torch.log(scale) - 2 * modules.softplus(-_y)
    logp = _logp.flatten(2)
    return logp

# function to calculate the log-probability of x under a discretized Logistic(mu, scale) distribution
# 计算离散Logistic（μ，尺度）分布下x的对数概率的函数
# heavily based on discretized_mix_logistic_loss() in https://github.com/openai/pixel-cnn
def discretized_logistic_logp(mu, scale, x, n):
    cen = (2**n-1)/2
    # [ 0 , 2^n - 1 ] -> [-1.1] (this means bin sizes of 2./255.)
    x_rescaled = (x - cen) / cen  #重新缩放
    invscale = 1. / scale

    x_centered = x_rescaled - mu

    plus_in = invscale * (x_centered + 1. / cen*2. )
    cdf_plus = torch.sigmoid(plus_in)
    min_in = invscale * (x_centered - 1. / cen*2.)
    cdf_min = torch.sigmoid(min_in)

    # log-probability for edge case of 0
    log_cdf_plus = plus_in - modules.softplus(plus_in)

    # log-probability for edge case of 255
    log_one_minus_cdf_min = - modules.softplus(min_in)

    # other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = invscale * x_centered

    # log-probability in the center of the bin, to be used in extreme cases
    log_pdf_mid = mid_in - torch.log(scale) - 2. * modules.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal case, extremely low-probability case
    cond1 = torch.where(cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-12, max=None)),
                        log_pdf_mid - np.log(cen))
    cond2 = torch.where(x_rescaled > .999, log_one_minus_cdf_min, cond1)
    logps = torch.where(x_rescaled < -.999, log_cdf_plus, cond2)

    logp = logps.flatten(1)
    return logp

# function to calculate the CDF of the Logistic(mu, scale) distribution evaluated under x
# x->y 计算在x下评估的Logistic(mu, scale)分布的CDF
def logistic_cdf(x, mu, scale):
    return torch.sigmoid((x - mu) / scale)

# function to calculate the inverse CDF (quantile function) of the Logistic(mu, scale) distribution evaluated under x
# 计算在x下评估的Logistic(mu，尺度)分布的逆CDF(分位数函数)
# y->x
def logistic_icdf(p, mu, scale):
    return mu + scale * torch.log(p / (1. - p))

# class that is used to determine endpoints and centers of discretization bins用于确定离散化容器的端点和中心的类
# in which every bin has equal mass under some given Logistic(mu, scale) distribution.
# note: the first (-inf) and last (inf) endpoint are not created here, but rather
# accounted for in the compression/decompression loop注意:这里没有创建第一个(-inf)端点和最后一个(inf)端点，而是在压缩/解压循环中考虑
class Bins:
    def __init__(self, mu, scale, precision):
        # number of bits used 使用的位数
        self.precision = precision

        # the resulting number of bins from the amount of bits used 根据使用的比特数量得出的二进制数
        self.nbins = 1 << precision

        # parameters of the Logistic distribution
        self.mu, self.scale = mu, scale

        # datatype used
        self.type = self.mu.dtype

        # device used (GPU/CPU)
        self.device = self.mu.device
        self.shape = list(self.mu.shape)

    def endpoints(self):
        # first uniformly between [0,1] 首先在[0,1]之间均匀
        # shape: (self.nbins-1, )
        # endpoint_probs = [1/self.nbins , 2/self.nbins , ... , (self.nbins-1)/self.nbins]
        endpoint_probs = torch.arange(1., self.nbins, dtype=self.type, device=self.device) / self.nbins

        # reshape
        endpoint_probs = endpoint_probs[(None,) * len(self.shape)] # shape: [1, 1, 1, self.nbins-1]
        endpoint_probs = endpoint_probs.permute([-1] + list(range(len(self.shape)))) # shape: [self.nbins-1, 1, 1, 1]
        endpoint_probs = endpoint_probs.expand([-1] + self.shape) # shape: [self.nbins-1, 1, 1, np.prod(model.zdim)]

        # put those samples through the inverse CDF
        endpoints = logistic_icdf(endpoint_probs, self.mu, self.scale)

        # reshape
        endpoints = endpoints.permute(list(range(1, len(self.shape) + 1)) + [0]) # self.shape + [1 << bits]
        return endpoints

    def centres(self):
        # first uniformly between [0,1]
        # shape: (self.nbins, )
        centre_probs = (torch.arange(end=self.nbins, dtype=self.type, device=self.device) + .5) / self.nbins

        # reshape
        centre_probs = centre_probs[(None,) * len(self.shape)] # shape: [1, 1, 1, self.nbins]
        centre_probs = centre_probs.permute([-1] + list(range(len(self.shape)))) # shape: [self.nbins, 1, 1, 1]
        centre_probs = centre_probs.expand([-1] + self.shape) # shape: [self.nbins , self.shape]

        # put those samples through the inverse CDF
        centres = logistic_icdf(centre_probs, self.mu, self.scale)

        # reshape
        centres = centres.permute(list(range(1, len(self.shape) + 1)) + [0]) # [self.shape, self.nbins]
        return centres

# class that is used to determine the endpoints and center of bins discretized uniformly between [0,1]
# these bins are used for the discretized Logistic distribution during compression/decompression
# note: the first (-inf) and last (inf) endpoint are not created here, but rather
# accounted for in the compression/decompression loop
# 类，用于确定在[0,1]之间均匀离散的容器的端点和中心
# 这些箱子用于压缩/解压期间的离散Logistic分布
# 注意:第一个(-inf)和最后一个(inf)端点不在这里创建，而是在压缩/解压循环中考虑
class ImageBins:
    def __init__(self, type, device, shape, n):  # max=2^n
        # datatype used
        self.type = type

        # device used (CPU/GPU)
        self.device = device
        self.shape = [shape]
        self.max = 2**n-1

    def endpoints(self):
        cen = self.max/2
        endpoints = torch.arange(1, self.max+1, dtype=self.type, device=self.device)
        endpoints = ((endpoints - cen) / cen) - 1./cen*2.
        endpoints = endpoints[None,].expand(self.shape + [-1])
        return endpoints  #[xdim, 255]

    def centres(self):
        cen = self.max / 2
        centres = torch.arange(0, self.max+1, dtype=self.type, device=self.device)
        centres = (centres - cen) / cen
        centres = centres[None,].expand(self.shape + [-1])
        return centres    #[xdim, 256]