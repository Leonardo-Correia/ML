import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Cria a função de gerar dados sintéticos
def generate_data(n_samples = 3000, n_features = 4):
    """"
    Gera dados sintéticos para regressão linear
    :param (int) n_samples: número de amostras
    :param (int) n_features: número de recursos
    :return (torch.tensor) x, y

    raise
    ------
    ValueError
        Se n_samples ou n_features não forem inteiros positivos
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError('n_samples deve ser um inteiro positivo')

    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError('n_features deve ser um inteiro positivo')

    x = np.random.rand(n_samples, n_features)
    y = 1*x[:, 0:1]  + 0.0 * np.random.randn(n_samples, n_features)  
    
    x = torch.FloatTensor(x)
    x = x/torch.max(x)
    y = torch.FloatTensor(y)
    return x, y

# Defina a arquitetura do autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 60),  # Camada de entrada com 3 recursos (x)
            nn.ReLU(),
            nn.Linear(60, 56),  # Camada oculta
            nn.ReLU(),
            nn.Linear(56, 52),  # Camada oculta
            nn.ReLU(),
            nn.Linear(52, 48),  # Camada oculta
            nn.ReLU(),
            nn.Linear(48, 44),  # Camada oculta
            nn.ReLU(),
            nn.Linear(44, 40),  # Camada oculta
            nn.ReLU(),
            nn.Linear(40, 36)  # Camada de saída na representação latente
        )
        self.decoder = nn.Sequential(
            nn.Linear(36, 40),  # Camada de entrada com 16 neurônios
            nn.ReLU(),
            nn.Linear(40, 44),  # Camada oculta
            nn.ReLU(),
            nn.Linear(44, 48),  # Camada oculta
            nn.ReLU(),
            nn.Linear(48, 52),  # Camada oculta
            nn.ReLU(),
            nn.Linear(52, 56),  # Camada oculta
            nn.ReLU(),
            nn.Linear(56, 60),  # Camada oculta
            nn.ReLU(),
            nn.Linear(60, 64)  # Camada de saída com 1 neurônio (para previsão de y)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



# Gerar função de parâmetros da Rede Neural

def training_network(lr = 0.005, num_epochs = 10000):
    """
    Função que treina a rede neural
    :param (float) lr: taxa de aprendizado
    :param (int) num_epochs: número de épocas
    :return (torch.tensor) predicted_output

    raise
    ------
    ValueError
        Se lr não for um float positivo

    ValueError
        Se num_epochs não for um inteiro positivo
    """

    if not isinstance(lr, float) or lr <= 0:
        raise ValueError('lr deve ser um float positivo')
    
    if not isinstance(num_epochs, int) or num_epochs <= 0:
        raise ValueError('num_epochs deve ser um inteiro positivo')

# Crie uma instância do modelo e defina a função de perda e otimizador
    autoencoder = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Treine o autoencoder
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        recon = autoencoder(x)
        loss = criterion(recon, x)  # Usamos x como alvo, pois estamos tentando reconstruir x
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')

    return autoencoder
# Use o autoencoder para fazer previsões

def make_predictions(input_data,test_data, outpu_latent = True):
    """"
    Função que gera previsões com o autoencoder ou gera a representação com menor dimensionalidade com o encoder
    :param (list, np.array or torch.tensor) input_data: vetor com os dados de entrada
    :param (list, np.array or torch.tensor) test_data: vetor com os dados de teste
    :param (boll) outpu_latent: se True, gera a previsão com o autoencoder, se False, gera a representação com menor dimensionalidade com o encoder
    :return (torch.tensor): predicted_output ou encoded

    raise
    ------
    ValueError
        Se input_data não for uma lista, np.array ou torch.tensor

    ValueError
        Se input_data não for da dim correta
    """
    if not isinstance(test_data, (list, np.ndarray, torch.Tensor)):
        raise ValueError('input_data deve ser uma lista, np.array ou torch.tensor')

    if len(test_data) != input_data.shape[1]:
        raise ValueError('test_data deve ter a dimensão do input_data')

    if outpu_latent:
        with torch.no_grad():
            test_input = torch.FloatTensor([test_data]) 
            predicted_output = autoencoder(test_input)
            print(f'Input: {test_input.tolist()}, Predicted Output: {predicted_output.tolist()}')
            return predicted_output

    if outpu_latent == False:
        with torch.no_grad():
            test_input = torch.FloatTensor([test_data])
            encoded = autoencoder.encoder(test_input)
            print(f'Input: {test_input.tolist()}, Encoded: {encoded.tolist()}')
            return encoded


if __name__ == '__main__':
    x, y = generate_data(10000, 64)
    autoencoder =  training_network(lr=0.1, num_epochs=10000)
    make_predictions(x,np.linspace(0,8,64), outpu_latent = True)
    make_predictions(x,np.linspace(0,8,64), outpu_latent = False)