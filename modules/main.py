import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(os.path.abspath('data/Skyrim_Weapons.csv'))
dataset.fillna(0, inplace=True)
plt.scatter(dataset[['Weight']], dataset[['Damage']])
plt.xlabel('Tamanho')
plt.ylabel('Dano')
plt.title('Relação Tamanho x Dano')
plt.show()

X = dataset[['Weight']]
Y = dataset[['Damage']]
X['Weight'] = X['Weight'] * 0.453592  # Converte para quilogramas

escala = StandardScaler()
escala.fit(X)
X_norm = escala.transform(X)

X_norm_train, X_norm_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.3)

rna = MLPRegressor(hidden_layer_sizes=(10, 5),max_iter=2000,tol=0.0000001,learning_rate_init=0.1,solver='sgd',activation='logistic',learning_rate='constant',verbose=2)

rna.fit(X_norm_train, Y_train)

Y_rna_prevision = rna.predict(X_norm_test)

X_test = escala.inverse_transform(X_norm_test)
plt.scatter(X_test, Y_test, alpha=0.5, label='Reais')
plt.scatter(X_test, Y_rna_prevision, alpha=0.5, label='MLP')
plt.xlabel('Tamanho')
plt.ylabel('Dano')
plt.title('COmparação algoritmo MLP com dados reais')
plt.legend(loc=1)
plt.show()