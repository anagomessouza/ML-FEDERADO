import os
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

load_dotenv()

# Carregar os dados reais
csv_file_path = os.getenv("CSV_FILE_PATH_1")
df = pd.read_csv(csv_file_path)
print("Carregando dados...")

# Selecionar features e target
print("Selecionando features e target...")
features = df[["Sleep_Hours"]].values
targets = df[["Stroop_Task_Reaction_Time","PVT_Reaction_Time", "N_Back_Accuracy"]].values

# Normalizar os dados
print("Normalizando dados...")
scaler = StandardScaler()
features = scaler.fit_transform(features)
scaler_target = StandardScaler()
targets = scaler_target.fit_transform(targets)

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
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # Camada adicional
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# Função de treinamento
def train_model(model, data, targets, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.L1Loss()  # MAE

    for epoch in range(epochs):
        print(f"Iniciando época {epoch+1}/{epochs}...")
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Época {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Treinar e avaliar o modelo diretamente
# Treinar e avaliar o modelo diretamente
def main():
    # Inicializar modelo
    model = MLP(input_size=1, hidden_size=10, output_size=3) # Ajustar o input_size para 1 e o output_size para 3

    # Treinar o modelo
    print("Iniciando o treinamento do modelo...")
    train_model(model, X_train, y_train, epochs=50)
    print("Treinamento concluído.")

    # Avaliar o modelo
    print("Iniciando a avaliação do modelo...")
    model.eval()  # Mudar o modelo para o modo de avaliação
    with torch.no_grad():
        predictions = model(X_test)

    # Calcular o erro médio quadrático (MSE)
    criterion = nn.L1Loss()  # MAE
    loss = criterion(predictions, y_test)
    print(f"Erro médio quadrático (MSE): {loss.item():.4f}")

    # Calcular erro absoluto médio (MAE)
    mae = torch.mean(torch.abs(predictions - y_test))
    print(f"Erro absoluto médio (MAE): {mae.item():.4f}")

# Executar o treino e teste direto
if __name__ == "__main__":
    main()
