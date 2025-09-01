import pickle

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

@st.cache_resource
def loadPCA(pcaFile):
    with open(pcaFile, 'rb') as f:
        pca = pickle.load(f)
    return pca

def main():
    pcaFile = "bulkPCA.pickle"
    pca = loadPCA(pcaFile)

    st.sidebar.header('Model Parameters:')
    params = []     # record the model parameter values
    for iPC in range(pca.n_components):
        parVal = st.sidebar.slider(label=f"PC: {iPC}",
                                   min_value=round(pca.pcMin[iPC], 2),
                                   max_value=round(pca.pcMax[iPC], 2),
                                   value=0.,
                                   step=(pca.pcMax[iPC] - pca.pcMin[iPC])/1000.,
                                   format='%f')
        params.append(parVal)
    params = np.array([params,])

    T_plot = np.linspace(0., 0.5, 15)
    mu_B_plot = np.linspace(0., 0.5, 10)
    bulk = pca.inverse_transform(params).reshape(len(mu_B_plot), len(T_plot))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    T_grid, mu_B_grid = np.meshgrid(T_plot, mu_B_plot)
    bulk = bulk.reshape(len(mu_B_plot), len(T_plot))
    ax.plot_surface(T_grid, mu_B_grid, bulk, cmap='viridis')
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 0.5])
    ax.set_zlim([-0.2, 0.4])
    ax.set_xlabel(r"T (GeV)")
    ax.set_ylabel(r"$\mu_B$ (GeV)")
    ax.set_zlabel(r"$\zeta/s$")
    plt.title(r"$\zeta/s$ from PCA")
    st.pyplot(fig)


if __name__ == '__main__':
    main()
