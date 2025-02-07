import flwr as fl
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Estratégia de agregação multithread
def aggregate(models):
    print("Agregando modelos dos clientes...")

    # Multi-thread para agregação de pesos
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        aggregated_weights = list(executor.map(np.mean, zip(*models)))

    print("Agregação concluída!")
    return aggregated_weights

# Função para iniciar o servidor federado
def start_federated_server():
    print("Iniciando servidor federado...")

    strategy = fl.server.strategy.FedAvg(aggregate_fn=aggregate)  # Usando estratégia personalizada de agregação
    
    # O server agora será gerido pelo flower-supernode
    print("Flower SuperNode gerenciado.")
    return strategy

if __name__ == "__main__":
    # Iniciar a estratégia
    strategy = start_federated_server()
    
    # O código agora deve estar esperando o flower-supernode gerenciar a comunicação e execução
    print("Servidor federado configurado para uso com Flower SuperNode.")
