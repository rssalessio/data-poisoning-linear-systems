from utils import collect_data
import scipy.signal as scipysig
import pickle


if __name__ == '__main__':
    dt = 0.05
    num = [0.28261, 0.50666]
    den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
    sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
    dim_x, dim_u = sys.B.shape


    NUM_SIMS = 100
    timesteps = [100, 500]
    STD_U = 1
    STD_W = 0.1

    for i in range(NUM_SIMS):
        for T in timesteps:
            data = collect_data(T, STD_U, STD_W, sys)
            with open(f'data/datasets/data_{T}timesteps_{i}.pkl', 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
