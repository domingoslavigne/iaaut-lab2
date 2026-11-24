import numpy as np
from funcoes_aux_proj_apr_supervisionada import extrair_atributos

# R3.a) 
class ArvoreDecisao:
    class Node:
        def __init__(self, feature=None, threshold=None, esquerda=None, direita=None, *, valor=None):
            self.feature = feature
            self.limiar = threshold
            self.esquerda = esquerda
            self.direita = direita
            self.valor = valor

        def is_leaf_node(self):
            return self.valor is not None

    def __init__(self, profundidade_maxima=None):
        self.profundidade_maxima = profundidade_maxima
        self.raiz = None

    def _entropy(self, y):
        _, contagens = np.unique(y, return_counts=True)
        p = contagens / len(y)
        return -np.sum(p * np.log2(p))
    
    def _mutual_information(self, y, y_esquerda, y_direita):

        h = self._entropy(y)
        h_esquerda = self._entropy(y_esquerda)
        h_direita = self._entropy(y_direita)

        return h - (len(y_esquerda) / len(y) * h_esquerda + len(y_direita) / len(y) * h_direita)
        
    def _best_split(self, X, y):
        melhor_ganho = 0
        melhor_feature = None
        melhor_limiar = None

        for feature in range(X.shape[1]):
            valores = np.unique(X[:, feature])
            limiares = (valores[:-1] + valores[1:]) / 2

            for limiar in limiares:
                mascara = X[:, feature] <= limiar

                ganho = self._mutual_information(y, y[mascara], y[~mascara])

                if ganho > melhor_ganho:
                    melhor_ganho = ganho
                    melhor_feature = feature
                    melhor_limiar = limiar

        return melhor_feature, melhor_limiar     

    def _build_tree(self, X, y, profundidade=0):
        classes, contagens = np.unique(y, return_counts=True)
        classe = classes[contagens == contagens.max()][-1]

        if (len(classes) == 1 or (self.profundidade_maxima is not None and profundidade >= self.profundidade_maxima)):
            return self.Node(valor=classe)

        feature, limiar = self._best_split(X, y)

        if feature is None:
            return self.Node(valor=classe)

        mascara = X[:, feature] <= limiar

        esquerda = self._build_tree(X[mascara], y[mascara], profundidade + 1)
        direita = self._build_tree(X[~mascara], y[~mascara], profundidade + 1)

        return self.Node(feature, limiar, esquerda, direita)

    def fit(self, X, y):
        self.raiz = self._build_tree(X, y)
        return self

    def predict(self, X):
        previsoes = []

        for linha in X:
            no = self.raiz

            while not no.is_leaf_node():
                if linha[no.feature] <= no.limiar:
                    no = no.esquerda
                else:
                    no = no.direita

            previsoes.append(no.valor)

        return np.array(previsoes)

# R3.b)
def separar_sequencias_arvore(df):
    sequencias = df[df["Patient_ID"] == 4]

    n_treino = int(0.8 * len(sequencias))

    treino = sequencias.iloc[:n_treino]
    teste = sequencias.iloc[n_treino:]

    classes = {"E1": 0, "E2": 1, "E3": 2, "E4": 3, "E5": 4}

    ytrain = treino["Exercise_Id"].map(classes).to_numpy()
    ytest = teste["Exercise_Id"].map(classes).to_numpy()

    return treino, teste, ytrain, ytest


# R3.c)
def construir_matrizes_arvore(treino, teste):
    Xtrain = np.vstack([extrair_atributos(seq) for seq in treino["Skeleton_Sequence"]])
    Xtest = np.vstack([extrair_atributos(seq) for seq in teste["Skeleton_Sequence"]])

    return Xtrain, Xtest

