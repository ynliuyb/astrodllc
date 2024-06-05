# from model.astronomic_train import Model
# from discretization import *
from benchmark_compress import *
from torchvision import transforms
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from os.path import isfile, join
# import astro_constants as ac
import sys
from PIL import Image
from terminaltables import AsciiTable

class ANS:
    def __init__(self, pmfs, bits=16-1, quantbits=8):
        self.device = pmfs.device
        self.bits = bits
        self.quantbits = quantbits

        # mask of 2**bits - 1 bits
        self.mask = (1 << bits) - 1

        # normalization constants
        self.lbound = 1 << 16
        self.tail_bits = (1 << 16) - 1

        self.seq_len, self.support = pmfs.shape

        # compute pmf's and cdf's scaled up by 2**n
        multiplier = (1 << self.bits) - (1 << self.quantbits)
        self.pmfs = (pmfs * multiplier).long()

        # add ones to counter zero probabilities
        self.pmfs += torch.ones_like(self.pmfs)

        # add remnant to the maximum value of the probabilites
        self.pmfs[torch.arange(0, self.seq_len),torch.argmax(self.pmfs, dim=1)] += ((1 << self.bits) - self.pmfs.sum(1))

        # compute cdf's
        self.cdfs = torch.cumsum(self.pmfs, dim=1) # compute CDF (scaled up to 2**n)
        self.cdfs = torch.cat([torch.zeros([self.cdfs.shape[0], 1], dtype=torch.long, device=self.device), self.cdfs], dim=1) # pad with 0 at the beginning

        # move cdf's and pmf's the cpu for faster encoding and decoding
        self.cdfs = self.cdfs.cpu().numpy()
        self.pmfs = self.pmfs.cpu().numpy()

        assert self.cdfs.shape == (self.seq_len, self.support + 1)
        assert np.all(self.cdfs[:,-1] == (1 << bits))

    def encode(self, x, symbols):
        for i, s in enumerate(symbols):
            pmf = int(self.pmfs[i,s])
            if x[-1] >= ((self.lbound >> self.bits) << ac.ANS_bits) * pmf:
                x.append(x[-1] >> ac.ANS_bits)
                x[-2] = x[-2] & self.tail_bits
            x[-1] = ((x[-1] // pmf) << self.bits) + (x[-1] % pmf) + int(self.cdfs[i, s])
        return x

    def decode(self, x):
        sequence = np.zeros((self.seq_len,), dtype=np.int64)
        for i in reversed(range(self.seq_len)):
            masked_x = x[-1] & self.mask
            s = np.searchsorted(self.cdfs[i,:-1], masked_x, 'right') - 1
            sequence[i] = s
            x[-1] = int(self.pmfs[i,s]) * (x[-1] >> self.bits) + masked_x - int(self.cdfs[i, s])
            if x[-1] < self.lbound:
                x[-1] = (x[-1] << ac.ANS_bits) | x.pop(-2)
        sequence = torch.from_numpy(sequence).to(self.device)
        return x, sequence

def compress(quantbits, nz, gpu, blocks):
    # model and compression params
    zdim = np.prod(ac.zdim)
    zrange = torch.arange(zdim)
    xdim = np.prod(ac.xs)
    xrange = torch.arange(xdim)
    ansbits = ac.ANS_bits # ANS precision
    type = torch.float64 # datatype throughout compression
    device = "cpu" if gpu < 0 else f"cuda:{gpu}" # gpu

    # <=== MODEL ===>
    model = Model(xs = ac.xs,
                  nz=ac.astro_nz,
                  zdim=ac.zdim,
                  nprocessing=4,
                  kernel_size=3,
                  resdepth=ac.astro_blocks,
                  reswidth=ac.width).to(device)
    model.load_state_dict(
        torch.load(f'model/params/nz'+str(ac.astro_nz),
                   map_location=lambda storage, location: storage
                   )
    )
    model.eval()

    # get discretization bins for latent variables 获取潜在变量的离散化箱
    zendpoints, zcentres = discretize(nz, quantbits, type, device, model)

    # get discretization bins for discretized logistic
    xbins = ImageBins(type, device, xdim, ac.astro_bits)
    xendpoints = xbins.endpoints()
    xcentres = xbins.centres()

    # compression experiment params
    nblocks = blocks.shape[0]

    # < ===== COMPRESSION ===>
    # initialize compression
    model.compress()
    excess_state_len = 30000
    state = list(map(int, np.random.randint(low=1 << 8, high=(1 << 16) - 1, size=excess_state_len, dtype=np.uint16))) # fill state list with 'random' bits
    state[-1] = state[-1] << ac.ANS_bits

    # <===== SENDER =====>
    iterator = tqdm(range(nblocks), desc="Bit-Swap")
    for xi in iterator:
        x = ac.transform_ops_astro(Image.fromarray(blocks[xi])).to(device).view(xdim)

        # < ===== Bit-Swap ====>
        # inference and generative model
        for zi in range(nz):
            # inference model
            input = zcentres[zi - 1, zrange, zsym] if zi > 0 else xcentres[xrange, x.long()]
            mu, scale = model.infer(zi)(given=input)
            cdfs = logistic_cdf(zendpoints[zi].t(), mu, scale).t()
            pmfs = cdfs[:, 1:] - cdfs[:, :-1]
            pmfs = torch.cat((cdfs[:,0].unsqueeze(1), pmfs, 1. - cdfs[:,-1].unsqueeze(1)), dim=1)

            # decode z
            state, zsymtop = ANS(pmfs, bits=ansbits, quantbits=quantbits).decode(state)

            # save excess state length for calculations
            # print("initial bits taken") if len(state) < excess_state_len else None
            excess_state_len = len(state) if len(state) < excess_state_len else excess_state_len

            # generative model
            z = zcentres[zi, zrange, zsymtop]
            mu, scale = model.generate(zi)(given=z)
            cdfs = logistic_cdf((zendpoints[zi - 1] if zi > 0 else xendpoints).t(), mu, scale).t() # most expensive calculation?
            pmfs = cdfs[:, 1:] - cdfs[:, :-1]
            pmfs = torch.cat((cdfs[:,0].unsqueeze(1), pmfs, 1. - cdfs[:,-1].unsqueeze(1)), dim=1)

            # encode z or x
            state = ANS(pmfs, bits=ansbits, quantbits=(quantbits if zi > 0 else 8)).encode(state, zsym if zi > 0 else x.long())

            zsym = zsymtop

        # prior
        cdfs = logistic_cdf(zendpoints[-1].t(), torch.zeros(1, device=device, dtype=type), torch.ones(1, device=device, dtype=type)).t()
        pmfs = cdfs[:, 1:] - cdfs[:, :-1]
        pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

        # encode prior
        state = ANS(pmfs, bits=ansbits, quantbits=quantbits).encode(state, zsymtop)

    # remove excess streams
    del state[0:excess_state_len - 1]

    return state

def input_image(path):

        if not isinstance(path, str):
            print("Path must be string.")
        if not os.path.exists(path):
            print("Path does not exist.")
        if not isfile(path):
            print("Path does not point to a file.")

        dir, file = os.path.split(os.path.abspath(path))
        filename, file_ext = os.path.splitext(file)

        img = plt.imread(path)
        img = img.astype('uint8')

        old_h, old_w = img.shape
        blocks, h, w = extract_blocks(img, block_size=(54, 64))
        cropped = True if (old_h != h or old_w != w) else False
        return blocks, old_h, old_w, h, w, cropped, dir, filename, file_ext

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def input_gpu():
    print("We highly recommend using GPU's.")
    print("Give GPU index (0, 1, 2 etc.) or CPU (-1) if that is the only option.")
    while True:
        sys.stdout.write("Index: ")
        gpu = '-1'
        # gpu = input()

        if not RepresentsInt(gpu):
            print("Index must be an integer.")
            continue

        return int(gpu)

def mainf(iimage):
    # retrieve GPU index
    # gpu = input_gpu()
    gpu = int(-1)

    blocks, old_h, old_w, h, w, cropped, \
    dir, filename, file_ext = input_image(iimage)

    # seed for replicating experiment and stability
    # np.random.seed(100)
    # random.seed(50)
    # torch.manual_seed(50)
    # torch.cuda.manual_seed(50)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # reconstruct from crop
    img_uncompressed = unextract_blocks(blocks, h, w)

    # save uncompressed crop
    np.save(join(dir, f"{filename}_uncompressed"), img_uncompressed)
    size_uncompressed = os.path.getsize(join(dir, f"{filename}_uncompressed.npy")) * 8

    # save uncompressed back to file extension if cropped version is smaller than original
    if cropped:
        print("Error!!!")

    # verbose results
    image_data = [
        ['Property', 'Value'],
        ['Filename', filename],
        ['Directory', dir],
        ['Original shape' if cropped else 'Shape', f"({old_h}, {old_w})"]
    ]

    image_data.append(['Raw size', f"{size_uncompressed} bits"])
    table = AsciiTable(image_data)
    table.title = "Image data"
    print("")
    print(table.table)

    # compress with Bit-Swap
    print("")
    state = compress(quantbits=ac.astro_quantbits, nz=ac.astro_nz, gpu=gpu, blocks=blocks)

    # move tail bits (everything after 32 bits) to new state stream
    state.append(state[-1] >> ac.ANS_bits)
    state[-2] = state[-2] & ((1 << ac.ANS_bits) - 1)

    # append number of blocks, height, the width and file extension of the image to the state
    state.append(blocks.shape[0])
    state.append(h)
    state.append(w)

    # save compressed image
    state_array = np.array(state, dtype=np.uint16)
    np.save(join(dir, f"{filename}_bitswap"), state_array)
    size_bitswap = os.path.getsize(join(dir, f"{filename}_bitswap.npy")) * 8

    # other compressors
    print("")
    print("Gzip, bzip2, LZMA, PNG and WebP...")
    state_gzip = np.array(list(gzip.compress(img_uncompressed.tobytes())), dtype=np.uint8)
    np.save(join(dir, f"{filename}_gzip"), state_gzip)
    size_gzip = os.path.getsize(join(dir, f"{filename}_gzip.npy")) * 8

    state_bz2 = np.array(list(bz2.compress(img_uncompressed.tobytes())), dtype=np.uint8)
    np.save(join(dir, f"{filename}_bz2"), state_bz2)
    size_bz2 = os.path.getsize(join(dir, f"{filename}_bz2.npy")) * 8

    state_lzma = np.array(list(lzma.compress(img_uncompressed.tobytes())), dtype=np.uint8)
    np.save(join(dir, f"{filename}_lzma"), state_lzma)
    size_lzma = os.path.getsize(join(dir, f"{filename}_lzma.npy")) * 8

    state_png = io.BytesIO()
    Image.fromarray(img_uncompressed).save(state_png, format='PNG', optimize=True)
    state_png = np.array(list(state_png.getvalue()), dtype=np.uint8)
    np.save(join(dir, f"{filename}_png"), state_png)
    size_png = os.path.getsize(join(dir, f"{filename}_png.npy")) * 8

    # state_webp = io.BytesIO()
    # Image.fromarray(img_uncompressed).save(state_webp, format='WebP', lossless=True, quality=100)
    # state_webp = np.array(list(state_webp.getvalue()), dtype=np.uint8)
    # np.save(join(dir, f"{filename}_webp"), state_webp)
    # size_webp = os.path.getsize(join(dir, f"{filename}_webp.npy")) * 8

    # verbose results
    compression_data = [
        ['Compression Scheme', 'Filename', 'Size (bits)', 'Ratio'],
        ['Uncompressed', f"{filename}_uncompressed.npy", size_uncompressed, '1'],

        ['GNU Gzip', f"{filename}_gzip.npy", size_gzip, f'{(size_uncompressed / size_gzip) :.2f}'],

        ['bzip2', f"{filename}_bz2.npy", size_bz2, f'{(size_uncompressed / size_bz2) :.2f}'],

        ['LZMA', f"{filename}_lzma.npy", size_lzma, f'{(size_uncompressed / size_lzma) :.2f}'],

        ['PNG', f"{filename}_png.npy", size_png, f'{(size_uncompressed / size_png) :.2f}']

        # ['WebP', f"{filename}_webp.npy", size_webp, f'{(size_uncompressed / size_webp) :.2f}'],

        # ['Bit-Swap', f"{filename}_bitswap.npy", size_bitswap, f'{(size_uncompressed/size_bitswap) :.2f}']
    ]
    table = AsciiTable(compression_data)
    table.title = "Results"
    print("")
    print(table.table)

mainf('001901.jpg')