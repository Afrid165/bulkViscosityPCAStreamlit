from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PCATransformation import PCATransformation

#this file generates bulk viscosity samples in 2D (T, mu_B) space 

def transformation(zeta_s):
    zeta_s_max = 0.4 #scaling factor here taken as 0.4 to ensure that zeta/s does not exceed 0.4 
    scale = 1.0
    return zeta_s_max/2.*(1. + np.tanh(scale*zeta_s))

def zeta_s_file_writer(T, mu_B, zeta_s, filename) -> None:
    """ This function writes the zeta_s to a pickle file with a dictionary
        for each zeta_s. The different columns are: T, mu_B, zeta_s
    """
    zeta_s_dict = {}
    for zs in range(len(zeta_s)):
        # create a 2D grid of T and mu_B values
        T_grid, mu_B_grid = np.meshgrid(T, mu_B)
        # making a 3D array by stacking T_muB and zeta_s[zs]
        data = np.stack([T_grid, mu_B_grid, zeta_s[zs]], axis=2)
        print(f"Shape of data for zeta_s {zs:04}: {data.shape}")
        # Store the data in the dictionary
        zeta_s_dict[f'{zs:04}'] = data
    with open(filename, 'wb') as f:
        pickle.dump(zeta_s_dict, f)

def main(ranSeed: int, number_of_zeta_s: int) -> None:
    T_min = 0.00
    T_max = 0.50
    mu_B_min = 0.00
    mu_B_max = 0.50
    print(f"Minimum of the datapoints is: {T_min}")
    print(f"Maximum of the datapoints is: {T_max}")
    print(f"Minimum of the mu_B is: {mu_B_min}")
    print(f"Maximum of the mu_B is: {mu_B_max}")

    # set the random seed
    if ranSeed >= 0:
        randomness = np.random.seed(ranSeed)
    else:
        randomness = np.random.seed()

    # make a 2D grid of T and mu_B values
    print("Generating T_muB grid ...")
    T_plot = np.linspace(T_min, T_max, 15)
    mu_B_plot = np.linspace(mu_B_min, mu_B_max, 10)
    T_grid, mu_B_grid = np.meshgrid(T_plot, mu_B_plot)
    T_muB = np.column_stack((T_grid.ravel(), mu_B_grid.ravel()))
    # print out the shape of T_muB
    print(f"Shape of T_muB: {T_muB.shape}")

    # putting the constraints in for anchor points
    Tlow = np.array([T_min])
    Thigh = np.array([T_max])
    T_GP = np.concatenate((Tlow, Thigh))
    T_GP, mu_B_GP = np.meshgrid(T_GP, mu_B_plot)
    T_muB_GP = np.column_stack((T_GP.ravel(), mu_B_GP.ravel()))
    print(f"Shape of T_muB_GP: {T_muB_GP.shape}")
    expon_low = np.array([-4])
    expon_high = np.array([-4])
    training_data = np.concatenate((expon_low, expon_high))
    #make training data shape consistent with T_muB_GP
    training_data = np.tile(training_data, len(mu_B_plot))
    print(f"Shape of training_data: {training_data.shape}")
    
  
    correlation_length_min = 0.10
    correlation_length_max = 0.20
    
    zeta_s_set = []
    nsamples_per_batch = max(1, int(number_of_zeta_s/100)) # number of samples to generate in each batch
    progress = 0
    while len(zeta_s_set) < number_of_zeta_s:
        correlation_length = np.random.uniform(correlation_length_min,
                                               correlation_length_max)
        print(f"Progress {progress}%, corr len = {correlation_length:.2f} ...")

        # generate a 2D GP in T and mu_B direction with the anchor points
        kernel = RBF(length_scale=correlation_length,
                     length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel, optimizer=None)
        gp.fit(T_muB_GP, training_data)
        zeta_s_vs_T_GP = gp.sample_y(T_muB, nsamples_per_batch,
                                      random_state=randomness).transpose()
        print(f"Shape of zeta_s_vs_T_GP: {zeta_s_vs_T_GP.shape}")
        
        
        # apply the transformation to the GP to make sure zeta/s is positive and bounded between 0 and 0.4
        zeta_s_vs_T_GP = transformation(zeta_s_vs_T_GP)

        #adding the samples to the set
        for sample_i in zeta_s_vs_T_GP:
            print(f"Sample i shape: {sample_i.shape}")
            # reshape the sample to 2D grid
            sample_i = sample_i.reshape(len(mu_B_plot), len(T_plot))
            print(f"Reshaped sample i shape: {sample_i.shape}")
            zeta_s_set.append(sample_i)
            if (sample_i < 0).any() or (sample_i > 0.4).any():
                print("\033[91m[Error: zeta/s out of bounds!\033[0m")
                return
            else:
                print("zeta/s within bounds.")
        progress += 1

    # make verification plots
    plt.figure().add_subplot(111, projection='3d')
    ax = plt.subplot(111, projection='3d')
    print(f"Shape of first zeta_s sample: {zeta_s_set[0].shape}")
    ax.plot_surface(T_grid, mu_B_grid, zeta_s_set[0], cmap='viridis', alpha=0.5)
    plt.xlim([T_min, T_max])
    plt.ylim([mu_B_min, mu_B_max])
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$\mu_B$ [GeV]")
    plt.show()
    plt.title("Bulk viscosity samples")
    plt.savefig("zeta_s_samples_3D.png")
    plt.clf()
   
    # write the EoS to a file
    zeta_s_file_writer(T_plot, mu_B_plot, zeta_s_set, "zeta_s.pkl")

    # check PCA
    plt.figure()
    #varianceList = [0.9, 0.95, 0.99]
    varianceList = [0.9]
    for var_i in varianceList:
        scaler = StandardScaler()
        pca = PCA(n_components=var_i)
        zeta_s_set = np.array(zeta_s_set)
        zeta_s_set_2d = zeta_s_set.reshape(zeta_s_set.shape[0], -1)  #flatten the 3D data into a 2d one so that we can use Scaler and PCA
        print(f"Shape of zeta_s_set: {zeta_s_set.shape}")
        print(f"Shape of zeta_s_set_2d: {zeta_s_set_2d.shape}")
        scaled = scaler.fit_transform(zeta_s_set_2d)
        PCA_fitted = pca.fit(scaled)
        print(f"Number of components = {pca.n_components_}")
        PCs = PCA_fitted.transform(scaled)
        # perform the inverse transform to get the original data
        zeta_s_reconstructed = PCA_fitted.inverse_transform(PCs)
        zeta_s_reconstructed = scaler.inverse_transform(zeta_s_reconstructed)
        zeta_s_reconstructed = zeta_s_reconstructed.reshape(zeta_s_set.shape[0], zeta_s_set.shape[1], zeta_s_set.shape[2]) # reshape back to the original shape
        RMS_error = np.sqrt(
            np.mean((zeta_s_set - zeta_s_reconstructed)**2, axis=0))
        ax = plt.subplot(111, projection='3d')
        ax.plot_surface(T_grid, mu_B_grid, RMS_error, label=f"var = {var_i:.2f}, nPC = {pca.n_components_}")
       
        
    plt.legend()
    plt.xlim([T_min, T_max])
    plt.ylim([mu_B_min, mu_B_max])
    plt.xlabel(r"$T$ [GeV]")
    plt.ylabel(r"$\mu_B$ [GeV]")
    plt.title(r"$\zeta/s$ RMS error")
    plt.savefig("RMS_errors_bulk_viscosity.png")
    plt.clf()

    # check the distribution for PCs
    pca = PCATransformation(0.95)
    PCs = pca.fit_transform(zeta_s_set_2d)
    print(f"Number of components = {PCs.shape[1]}")
    print(PCs.min(axis=0), PCs.max(axis=0))
    for i in range(PCs.shape[1]):
        plt.figure()
        plt.hist(PCs[:, i], bins=17, density=True)
        plt.savefig(f"PC{i}.png")
        plt.clf()

    with open("bulkPCA.pickle", "wb") as f:
        pickle.dump(pca, f)



if __name__ == "__main__":
    ranSeed = -1
    number_of_zeta_s = 100
    main(ranSeed, number_of_zeta_s)
