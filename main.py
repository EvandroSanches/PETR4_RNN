import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model


#Realiza previsão de preço e tendencia da ação PETR4 baseado nos ultimos 90 dias
#Modelo 02 foi treinado na base de dados para fazer previsão em época de greve
#para avaliar como o modelo se comporta em crises

epochs = 100
batch_size = 32
normalizador = MinMaxScaler()

#Carrega e trata dados
def CarregaDados(arquivo):
    #Carregando base de dados e eliminando dados nulos
    dados = pd.read_csv(arquivo)
    dados = dados.dropna()
    dados = dados['Open']

    #Alterando formato para array numpy 2D
    dados = np.asarray(dados)
    dados = np.expand_dims(dados, axis=1)

    #Normalizando dados em uma escala de 0 a 1 para agilizar processamento dos dados
    global normalizador

    try:
        normalizador = joblib.load('normalizador')
        dados = normalizador.fit_transform(dados)
    except:
        dados = normalizador.fit_transform(dados)
        joblib.dump(normalizador,'normalizador')


    #Criando variaveis previsores e preco_real para comparação
    #O input de previsores deve estar no modelo de array 3D (batch, timesteps, feature)
    # 'previsores' irá receber os 90 registros anteriores ao valor de previsão
    # 'preco_real' irá receber  o registro adjacente aos registros previsores
    # O loop for irá percorrer toda base dados com essa lógica criando uma base de treinamento
    previsores = []
    preco_real = []
    for i in range(90, dados.shape[0]):
        previsores.append(dados[i - 90 : i, 0])
        preco_real.append(dados[i, 0])

    previsores = np.asarray(previsores)
    preco_real = np.asarray(preco_real)

    previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))
    preco_real = np.expand_dims(preco_real, axis=1)

    return previsores, preco_real

#Cria modelo RNN
def CriaRede():
    previsores, preco_real = CarregaDados('PETR4.SA.csv')

    modelo = Sequential()

    #Input_shape = ('shape timesteps', 'quantidade de features')
    modelo.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 1)))
    modelo.add(Dropout(0.3))
    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.3))
    modelo.add(LSTM(units=80, return_sequences=True))
    modelo.add(Dropout(0.3))
    modelo.add(LSTM(units=80))
    modelo.add(Dropout(0.3))

    modelo.add(Dense(units=1, activation='linear'))

    modelo.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return modelo

#Treinamento com base nos dados carregados 'PETR4.SA.csv'
def Treinamento():
    previsores, preco_real = CarregaDados('Dados_Greve/PETR4.SA_Greve_Treinamento.csv')
    previsores_teste, preco_real_teste = CarregaDados('Dados_Greve/PETR4_Greve_Teste.csv')

    modelo = CriaRede()

    resultado = modelo.fit(x=previsores, y=preco_real, batch_size=batch_size, epochs=epochs, validation_data=(previsores_teste, preco_real_teste))

    modelo.save('Modelo.0.2')

    plt.plot(resultado.history['loss'])
    plt.plot(resultado.history['val_loss'])
    plt.title('Relação de Perda')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend(('Treinamento', 'Teste'))
    plt.show()

    plt.plot(resultado.history['mae'])
    plt.plot(resultado.history['val_mae'])
    plt.title('Margem de Erro')
    plt.xlabel('Épocas')
    plt.ylabel('Erro')
    plt.legend(('Treinamento', 'Teste'))
    plt.show()


# Prevendo e validação base da base de dados 'PETR4.SA_Teste.csv'
def Previsao(caminho):
    previsores, preco_real = CarregaDados(caminho)

    modelo = load_model('Modelo.0.2')

    resultado = modelo.predict(previsores)
    global normalizador

    normalizador = joblib.load('normalizador')

    preco_real = normalizador.inverse_transform(preco_real)
    resultado = normalizador.inverse_transform(resultado)

    plt.plot(resultado)
    plt.plot(preco_real)
    plt.title('Relação Predição e Preço Real')
    plt.xlabel('Maio a Dezembro 2023 PETR4')
    plt.ylabel('Preço')
    plt.legend(('Predição', 'Preço Real'))
    plt.show()

    plt.plot(resultado - preco_real)
    plt.title('Taxa de Erro do Modelo em R$')
    plt.xlabel('Maio a Dezembro 2023 PETR4')
    plt.ylabel('Preço')
    plt.show()



Previsao('Dados_Greve/PETR4_Greve_Teste.csv')

