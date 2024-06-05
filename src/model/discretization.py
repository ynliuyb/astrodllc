from utils.torch.rand import *
from tqdm import tqdm
import os
import astro_constants as ac
from torchvision import datasets, transforms
from torch.utils.data import *
from sklearn.preprocessing import KBinsDiscretizer

# function that returns discretization bin endpoints and centres返回离散化箱的端点和中心
def discretize(nz, quantbits, type, device, model):
    # number of samples per bin
    ppb = 30

    # total number of samples (ppb * number of bins)
    nsamples = ppb * (1 << quantbits)

    with torch.no_grad():
        if not os.path.exists(f'bins/astronomic_nz{nz}_zendpoints{quantbits}.pt'):
          # set up an empty tensor for all the bins (number of latent variables, total dimension of latent, number of bins)
          # 为所有的箱子建立一个空张量(潜在变量的数量，潜在的总维数，箱子的数量)
          # note that we do not include the first and last endpoints, because those will always be -inf and inf
          # 请注意，我们不包括第一个端点和最后一个端点，因为它们总是-inf和inf
          zendpoints = np.zeros((nz, np.prod(model.zdim), (1 << quantbits) - 1))
          zcentres = np.zeros((nz, np.prod(model.zdim), (1 << quantbits)))

          # top latent is fixed, so we can calculate the discretization bins without samples
          # 顶潜是固定的，所以我们可以在没有样本的情况下计算离散化箱
          zbins = Bins(torch.zeros((1, 1, np.prod(model.zdim))), torch.ones((1, 1, np.prod(model.zdim))), quantbits)
          zendpoints[nz - 1] = zbins.endpoints().numpy()
          zcentres[nz - 1] = zbins.centres().numpy()

          full_data = ac.full_data
          train_size = ac.train_size
          test_size = len(full_data) - train_size
          train_set, test_set = torch.utils.data.random_split(full_data, [train_size, test_size])


            # set-up a batch-loader for the dataset
          train_loader = DataLoader(
                dataset=train_set,
                batch_size=ac.astro_batch, shuffle=True, drop_last=True)
          datapoints = list(train_loader)

          # concatenate the dataset with itself if the length is not sufficient
          # 如果长度不够，则将数据集与其本身连接 eg: a=[1,2]--->a=[1,2,1,2]
          while len(datapoints) < nsamples:
              datapoints += datapoints

          bs = ac.astro_batch # batch size to iterate over
          batches = nsamples // bs # number of batches

          # use 16-bit values to reduce memory usage 减少内存使用
          gen_samples = np.zeros((nz, nsamples) + model.zdim, dtype=np.float32)   #(nz, nsamples, model.zdim)
          gen_samples[-1] = logistic_eps((nsamples,) + model.zdim, device="cpu", bound=1e-30).numpy()
          inf_samples = np.zeros((nz, nsamples) + model.zdim, dtype=np.float32)

          # iterate over the latent variables 迭代潜在变量
          for zi in reversed(range(1, nz)):
              # obtain samples from the generative model
              iterator = tqdm(range(batches), desc=f"sampling z{zi} from gen")
              for bi in iterator:
                  mu, scale = model.generate(zi)(given=torch.from_numpy(gen_samples[zi][bi * bs: bi * bs + bs]).to(device).float())
                  gen_samples[zi - 1][bi * bs: bi * bs + bs] = transform(logistic_eps(mu.shape, device=device, bound=1e-30), mu, scale).to("cpu")

              # obtain samples from the inference model (using the dataset)
              iterator = tqdm(range(batches), desc=f"sampling z{nz - zi} from inf")
              for bi in iterator:
                  datapoint = datapoints[bi][0]
                  given = (datapoint if nz - zi - 1 == 0 else torch.from_numpy(inf_samples[nz - zi - 2][bi * bs: bi * bs + bs])).to(device).float()
                  mu, scale = model.infer(nz - zi - 1)(given=given)
                  inf_samples[nz - zi - 1][bi * bs: bi * bs + bs] = transform(logistic_eps(mu.shape, device=device, bound=1e-30), mu, scale).cpu().numpy()

            # get the discretization bins
          for zi in range(nz - 1):
              samples = np.concatenate([gen_samples[zi], inf_samples[zi]], axis=0) #拼接
              zendpoints[zi], zcentres[zi] = discretize_kbins(model, samples, quantbits, strategy='uniform')

          # move the discretization bins to the GPU
          zendpoints = torch.from_numpy(zendpoints)
          zcentres = torch.from_numpy(zcentres)

          # save the bins for reproducibility and speed purposes
          torch.save(zendpoints, f'bins/astronomic_nz{nz}_zendpoints{quantbits}.pt')
          torch.save(zcentres, f'bins/astronomic_nz{nz}_zcentres{quantbits}.pt')
        else:
            zendpoints = torch.load(f'bins/astronomic_nz{nz}_zendpoints{quantbits}.pt',
                                    map_location=lambda storage, location: storage)
            zcentres = torch.load(f'bins/astronomic_nz{nz}_zcentres{quantbits}.pt',
                                  map_location=lambda storage, location: storage)




    # cast the bins to the appropriate type (in our experiments always float64)
    # 将容器转换为适当的类型(在我们的实验中总是float64)
    return zendpoints.type(type).to(device), zcentres.type(type).to(device)

# function that exploits the KBinsDiscretizer from scikit-learn 利用scikit-learn中的KBinsDiscretizer的函数
# two strategy are available 有两种策略
# 1. uniform: bins with equal width 均布:箱子宽度相等
# 2. quantile: bins with equal frequency 分位数:频率相等的箱子
def  discretize_kbins(model, samples, quantbits, strategy):
    # reshape samples
    samples = samples.reshape(-1, np.prod(model.zdim))

    # apply discretization   strategy= uniform/ quantile/ kmeans
    est = KBinsDiscretizer(n_bins=1 << quantbits, strategy=strategy)
    est.fit(samples)

    # obtain the discretization bins endpoints获取离散化箱端点
    endpoints = np.array([np.array(ar) for ar in est.bin_edges_]).transpose()
    centres = (endpoints[:-1,:] + endpoints[1:,:]) / 2
    endpoints = endpoints[1:-1]

    return endpoints.transpose(), centres.transpose()