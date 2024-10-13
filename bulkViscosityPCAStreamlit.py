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

    T_plot = np.linspace(0., 0.5, 100)
    shear = pca.inverse_transform(params).flatten()

    fig = plt.figure()
    plt.plot(T_plot, shear, '-r')
    plt.xlim([0, 0.5])
    plt.ylim([0, 0.4])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$\zeta/s$")
    st.pyplot(fig)


if __name__ == '__main__':
    main()
