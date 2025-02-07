import flwr as fl  
from typing import Dict, List, Tuple  
import numpy as np  
import time  

# Classe personalizada para manter o servidor ativo
class PersistentFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, min_available_clients: int = 2, **kwargs): 
        # Inicializa a classe base FedAvg e define o número mínimo de clientes necessários
        super().__init__(**kwargs)
        self.min_available_clients = min_available_clients

    def configure_fit(
        self, server_round: int, parameters: List[np.ndarray], client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, Dict[str, any]]]:
        # Este método é chamado antes de começar uma rodada de treinamento
        
        available_clients = client_manager.num_available()  # Verifica o número de clientes disponíveis
        print(f"[SERVER] Rodada {server_round}: Clientes disponíveis: {available_clients}")
        
        # Se o número de clientes disponíveis for menor que o mínimo necessário, o servidor espera
        if available_clients < self.min_available_clients:
            print(f"Esperando por mais clientes. Clientes disponíveis: {available_clients}")
            return []  # Retorna uma lista vazia, não inicia a rodada
        print(f"[SERVER] Iniciando rodada {server_round} com {available_clients} clientes disponíveis.")
        # Se houver clientes suficientes, inicia a rodada de treinamento
        return super().configure_fit(server_round, parameters, client_manager)

# Função para agregar métricas de treinamento
def aggregate_fit_metrics(metrics):
    try:
        # Calcula a média da perda ('loss') dos clientes
        print(f"[AGGREGATE] Agregando métricas: {metrics}")
        return {"loss": np.mean([m["loss"] for m in metrics if "loss" in m])}
    except Exception as e:
        # Caso ocorra erro, exibe a mensagem e retorna um dicionário vazio
        print(f"Erro ao agregar métricas de treinamento: {e}")
        return {}

# Cria a estratégia de aprendizado federado com a função personalizada de agregação de métricas
strategy = PersistentFedAvg(
    min_fit_clients=1,  # Número mínimo de clientes para iniciar o treinamento
    min_evaluate_clients=1,  # Número mínimo de clientes para avaliação
    min_available_clients=1,  # Número mínimo de clientes disponíveis para iniciar a rodada
    fit_metrics_aggregation_fn=aggregate_fit_metrics,  # Função para agregar as métricas
)

# Função para iniciar o servidor federado de forma persistente
def start_persistent_server():
    while True:  # Loop infinito para tentar reiniciar o servidor caso ocorra algum erro
        try:
            print("[SERVER] Iniciando servidor federado...")
            # Inicia o servidor federado com a estratégia definida
            fl.server.start_server(
                server_address="0.0.0.0:8080",  # Escuta em todas as interfaces de rede
                strategy=strategy,  # Usa a estratégia personalizada
                config=fl.server.ServerConfig(num_rounds=3),  # Define o número de rodadas
                grpc_max_message_length=1024 * 1024 * 1024,  # Define o tamanho máximo da mensagem
            )
        except Exception as e:
            # Se ocorrer um erro, exibe a mensagem de erro e tenta reiniciar após 15 segundos
            print(f"Erro detectado: {e}")
            print("Reiniciando o servidor em 15 segundos...")
            time.sleep(15)  # Espera 15 segundos antes de reiniciar

# Inicia o servidor se o script for executado diretamente
if __name__ == "__main__":
    start_persistent_server()

