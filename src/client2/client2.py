import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

load_dotenv()

# Carregar os dados 
csv_file_path = os.getenv("CSV_FILE_PATH_2")
df = pd.read_csv(csv_file_path)
print("Carregando dados...")


# Selecionar features e target
print("Selecionando features e target...")
features = df[["Sleep_Hours", "Age", "Sleep_Quality_Score", "Daytime_Sleepiness"]].values
targets = df[["Stroop_Task_Reaction_Time","PVT_Reaction_Time", "N_Back_Accuracy"]].values

# Normalizar os dados
print("Normalizando dados...")
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

# Dividir em treino e teste
print("Dividindo dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Converter para tensores PyTorch
print("Convertendo dados para tensores PyTorch...")
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# Definição do modelo MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Função de treinamento
def train_model(model, data, targets, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Época {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Cliente FL com treinamento e avaliação
class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = MLP(input_size=4, hidden_size=10, output_size=1)
        print("Modelo MLP inicializado.")

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        print("Configurando parâmetros do modelo...")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("Iniciando treinamento...")
        self.set_parameters(parameters)
        train_model(self.model, X_train, y_train)
        print("Treinamento concluído.")
        return self.get_parameters(), len(X_train), {}

    def evaluate(self, parameters, config):
        print("Iniciando avaliação...")
        self.set_parameters(parameters)
        
        # Fazer previsão nos dados de teste
        with torch.no_grad():
            predictions = self.model(X_test)
        
        # Calcular erro médio
        criterion = nn.MSELoss()
        loss = criterion(predictions, y_test)
        
        # Calcular erro absoluto médio (MAE)
        mae = torch.mean(torch.abs(predictions - y_test))

        print(f"Avaliação: Loss={loss.item():.4f}, MAE={mae.item():.4f}")
        
        return loss.item(), len(X_test), {"mae": mae.item()}

# Rodar o cliente FL
print("Iniciando cliente federado...")
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FLClient().to_client()  # FLClient deve ser do tipo flwr.client.NumPyClient
)