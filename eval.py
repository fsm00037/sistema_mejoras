import pandas as pd
import pandas.api.types
import numpy as np

def apk(actual, predicted, k=10):
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    actual = list(map(lambda x: x.split(' '), solution['rel_docs'].astype(str).to_list()))
    predicted = list(map(lambda x: x.split(' '), submission['rel_docs'].astype(str).to_list()))
    return mapk(actual, predicted, 10)

# Crear DataFrames de Pandas
solution_df = pd.read_csv("data\\train-gold.csv")
submission_df = pd.read_csv("resultados\\train-output.csv")

# Llamar a la función score
resultado = score(solution_df, submission_df, 'rel_docs')
print("Resultado del score:", resultado)
