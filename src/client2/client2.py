import flwr as fl  
from typing import Dict, List, Tuple  
import numpy as np  
import torch 
from torch import nn, optim  
from torch.utils.data import DataLoader, TensorDataset 
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  

# Carregar dados do arquivo CSV
csv_file_path = "sleep_deprivation_dataset_detailed2.csv"  # Caminho para o arquivo CSV
print(f"Carregando dados do arquivo CSV: {csv_file_path}")
df = pd.read_csv(csv_file_path)  # Carrega os dados do CSV para um DataFrame
print(f"Dados carregados. Número de linhas: {len(df)}")

# Preprocessamento dos dados
# Seleciona as características e os alvos do DataFrame
features = df[["Sleep_Hours", "Sleep_Quality_Score", "Daytime_Sleepiness"]].values
targets = df[["Stroop_Task_Reaction_Time", "PVT_Reaction_Time", "N_Back_Accuracy"]].values

print("Normalizando as características e os alvos...")
# Normaliza as características (inputs) e os alvos (outputs) usando StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)
scaler_target = StandardScaler()
targets = scaler_target.fit_transform(targets)

# Divide os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
print(f"Tamanho do conjunto de treinamento: {len(X_train)}, Tamanho do conjunto de teste: {len(X_test)}")

# Converte os dados de treino e teste para tensores PyTorch
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Definição do modelo MLP (Perceptron Multicamadas)
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()  # Chama o inicializador da classe base
        # Definição das camadas do modelo
        self.fc1 = nn.Linear(input_size, hidden_size)  # Camada de entrada
        self.relu = nn.ReLU()  # Função de ativação ReLU
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)  # Camada oculta
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)  # Camada oculta
        self.fc4 = nn.Linear(hidden_size, output_size)  # Camada de saída
        self.dropout = nn.Dropout(p=0.5)  # Camada de Dropout para regularização

    def forward(self, x):
        # Passagem dos dados através das camadas
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x  # Retorna a saída final

# Inicializa o modelo MLP
input_size = 3  # Número de entradas (características)
hidden_size = 50  # Número de neurônios na camada oculta
output_size = 3  # Número de saídas (alvos)
model = MLP(input_size, hidden_size, output_size)  # Cria o modelo com os parâmetros definidos
print(f"Modelo MLP criado com {input_size} entradas, {hidden_size} neurônios ocultos e {output_size} saídas.")

# Função de agregação personalizada para calcular as métricas de avaliação (MSE)
def aggregate_evaluate_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    print("Agregando métricas de avaliação...")
    try:
        # Calcula a média dos valores MSE recebidos dos clientes
        mse_values = [m["mse"] for m in metrics if "mse" in m]
        average_mse = np.mean(mse_values)  # Calcula a média do MSE
        print(f"Média das métricas MSE: {average_mse}")
        return {"mse": average_mse}  # Retorna o MSE médio
    except Exception as e:
        # Se houver erro ao agregar, exibe a mensagem e retorna um dicionário vazio
        print(f"Erro ao agregar métricas de avaliação: {e}")
        return {}

# Classe de cliente federado personalizada
class MyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        print("Obtendo parâmetros do modelo...")
        # Retorna os parâmetros do modelo atual (como um vetor numpy)
        return [param.detach().numpy() for param in model.parameters()]

    def set_parameters(self, parameters):
        print("Configurando parâmetros no modelo...")
        # Define os parâmetros no modelo a partir dos valores recebidos
        params = zip(model.parameters(), parameters)
        for param, new_param in params:
            param.data = torch.from_numpy(new_param)

    def fit(self, parameters, config):
        print("Iniciando treinamento do modelo...")
        self.set_parameters(parameters)  # Atualiza os parâmetros do modelo
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)  # Carrega os dados de treino
        model.train()  # Coloca o modelo em modo de treino
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Define o otimizador
        criterion = nn.MSELoss()  # Define a função de perda (Erro Quadrático Médio)

        # Treinamento do modelo em batches
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # Zera os gradientes
            output = model(data)  # Passa os dados pelo modelo
            loss = criterion(output, target)  # Calcula a perda
            loss.backward()  # Propaga o erro para os parâmetros
            optimizer.step()  # Atualiza os parâmetros
            if batch_idx % 10 == 0:  # Exibe a cada 10 batches
                print(f"Treinamento: Batch {batch_idx}, Perda (loss): {loss.item()}")

        print("Treinamento concluído.")
        # Retorna os parâmetros treinados
        return self.get_parameters(config), len(X_train), {}

    def evaluate(self, parameters, config):
        print("Iniciando avaliação do modelo...")
        self.set_parameters(parameters)  # Atualiza os parâmetros do modelo
        model.eval()  # Coloca o modelo em modo de avaliação
        with torch.no_grad():  # Desabilita o cálculo de gradientes para avaliação
            output = model(X_test)  # Faz a previsão nos dados de teste
            mse = nn.MSELoss()(output, y_test)  # Calcula o MSE na avaliação
            print(f"Erro Quadrático Médio (MSE) na avaliação: {mse.item()}")

        # Retorna o MSE para o servidor federado
        return mse.item(), len(X_test), {"mse": mse.item()}

# Estratégia personalizada para aprendizado federado
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,  # Número mínimo de clientes para iniciar o treinamento
    min_evaluate_clients=2,  # Número mínimo de clientes para avaliar
    min_available_clients=2,  # Número mínimo de clientes disponíveis para começar
    evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,  # Função de agregação para métricas de avaliação
)

# Iniciar o cliente federado
if __name__ == "__main__":
    client = MyClient()  # Cria uma instância do cliente
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())  # Conecta o cliente ao servidor
