import numpy as np
import pandas as pd 

# R1.a) 
def regressao_linear(X, Y):
    theta = np.linalg.pinv(X.T @ X) @ (X.T @ Y)   # entender melhor essa parte depois

    beta = theta[:-1] 
    beta0  = theta[-1]

    return beta, beta0

# R1.b)  
def separar_sequencias(df):
    sequencias = df[(df["Patient_ID"] == 4) & (df["Exercise_Id"] == "E5")]

    n_treino = int(0.8 * len(sequencias))

    seq_treino = sequencias.iloc[:n_treino]
    seq_teste = sequencias.iloc[n_treino:]

    return seq_treino, seq_teste

# R1.c) 
def construir_matrizes(sequencias):
    X = []
    Y = []

    for seq in sequencias["Skeleton_Sequence"]:
        X.append(seq[:-1])
        Y.append(seq[1:])

    X = np.vstack(X)
    Y = np.vstack(Y)

    X = np.column_stack((X, np.ones(len(X))))   # adiciona coluna de 1s

    return X, Y

# R1.d) Basta chamar nossa função construir_matrizes() passando seq_teste como argumento

# R1.e) 
def avaliar_regressao(Y, Y_previsto):
    sse = np.sum((Y - Y_previsto)**2)

    media = np.mean(Y, axis=0)
    sst = np.sum((Y - media)**2)

    r2 = 1 - sse / sst if sst > 0 else np.nan

    return sse, r2





