import numpy as np
from sklearn.cluster import KMeans

# R6
def construir_bag_of_poses(treino, teste, k):
    poses_treino = np.vstack(treino["Skeleton_Sequence"])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(poses_treino)

    def histogramas(sequencias):
        return np.vstack([
            np.bincount(kmeans.predict(seq), minlength=k) / len(seq)
            for seq in sequencias["Skeleton_Sequence"]
        ])

    Xtrain_bop = histogramas(treino)
    Xtest_bop = histogramas(teste)

    return Xtrain_bop, Xtest_bop
