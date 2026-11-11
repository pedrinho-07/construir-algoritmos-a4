import numpy as np
import pandas as pd


def calcular_coeficientes(caminho_csv):
    tabela = pd.read_csv(caminho_csv)

    areas = tabela["area_util_m2"].to_list()
    quartos = tabela["quantidade_quartos"].to_list()
    precos = tabela["preco_venda"].to_list()


    quantidade = len(areas)
    X = np.column_stack([np.ones(quantidade), areas, quartos])

    Y = np.array(precos).reshape(-1, 1)

    beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)

    return beta.flatten().tolist()

b0, b1, b2 = calcular_coeficientes("imoveis_lar_perfeito.csv")
print(f"β0 (valor base): {b0:.2f}")
print(f"β1 (cada m² a mais): {b1:.2f}")
print(f"β2 (cada quarto a mais): {b2:.2f}")