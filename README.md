# Descrição do Sistema de Aprendizado Federado: Cliente e Servidor

Este sistema implementa um modelo de aprendizado federado utilizando a biblioteca Flower (`flwr`). O aprendizado federado permite que vários clientes treinem um modelo de aprendizado de máquina de forma colaborativa sem compartilhar dados entre si, promovendo a privacidade e a segurança dos dados.

O código é composto por duas partes principais: o **cliente federado** e o **servidor federado**. Cada parte tem uma função específica no processo de aprendizado, como descrito abaixo.

---

## **Cliente Federado**

O cliente federado é responsável por treinar e avaliar o modelo localmente. Em vez de enviar os dados para um servidor centralizado, o cliente envia apenas os **parâmetros** do modelo após o treinamento, o que permite preservar a privacidade dos dados.

### Fluxo de Trabalho do Cliente

1. **Carregamento e Pré-processamento dos Dados**:
   - O cliente carrega os dados de um arquivo CSV contendo informações sobre privação de sono e desempenho cognitivo.
   - As características (entradas) e os alvos (saídas) são extraídos e normalizados usando `StandardScaler` do scikit-learn. Isso é feito para garantir que as variáveis tenham a mesma escala e que o modelo treine de maneira mais eficiente.

2. **Divisão dos Dados**:
   - O conjunto de dados é dividido em conjuntos de **treinamento** e **teste** (80% para treino e 20% para teste) usando `train_test_split`.

3. **Definição do Modelo**:
   - O modelo é uma rede neural **MLP (Multilayer Perceptron)**, com camadas totalmente conectadas (Fully Connected Layers), ativação ReLU e regularização **Dropout** para evitar o overfitting.
   - O modelo é projetado para prever 3 variáveis de desempenho cognitivo com base nas características de privação de sono.

4. **Treinamento do Modelo**:
   - O cliente treina o modelo localmente usando os dados de treinamento.
   - O modelo é otimizado com o **otimizador Adam** e a função de perda utilizada é o **Erro Quadrático Médio (MSE)**.
   - O processo de treinamento é feito em **batches**, o que melhora a eficiência e o desempenho do treinamento.

5. **Avaliação do Modelo**:
   - Após o treinamento, o cliente avalia o modelo utilizando os dados de teste.
   - A avaliação é feita com a mesma função de perda (MSE), que calcula o erro entre as previsões do modelo e os valores reais.

6. **Interação com o Servidor**:
   - O cliente envia os parâmetros do modelo para o servidor após cada rodada de treinamento e recebe os parâmetros agregados do servidor. Esse processo é realizado por meio do protocolo de aprendizado federado do Flower.
   
7. **Função de Agregação de Métricas**:
   - O cliente também contribui para a função de agregação das métricas de avaliação (MSE) recebidas dos diferentes clientes. A função calcula a média das métricas de todos os clientes participantes.

---

## **Servidor Federado**

O servidor federado é responsável por coordenar o processo de aprendizado federado. Ele agrega os parâmetros do modelo enviados pelos clientes e executa a avaliação global. O servidor não possui acesso direto aos dados dos clientes, o que mantém a privacidade.

### Fluxo de Trabalho do Servidor

1. **Estratégia de Aprendizado Federado**:
   - O servidor utiliza a estratégia **FedAvg** (Federação de Média), que agrega os parâmetros do modelo de cada cliente de forma eficiente. Nessa estratégia, os parâmetros são atualizados com a média ponderada dos modelos locais treinados pelos clientes.
   
2. **Configuração de Parâmetros**:
   - O servidor especifica as condições para o treinamento, como o número mínimo de clientes para começar o treinamento, o número mínimo de clientes para avaliação, e a função de agregação de métricas. No caso deste código, a **função `aggregate_evaluate_metrics`** calcula a média do MSE retornado pelos clientes.

3. **Execução do Cliente**:
   - O servidor também gerencia a comunicação entre os clientes e mantém o processo de aprendizado federado em andamento. Ele aguarda os parâmetros do modelo de todos os clientes participantes, realiza a agregação e envia os parâmetros atualizados para cada cliente.

4. **Métricas de Avaliação**:
   - Durante o processo de avaliação, o servidor pode acessar as métricas de desempenho dos modelos locais (MSE), agregando essas informações para fornecer uma visão geral do desempenho global do modelo.

---

## **Execução do Sistema**

1. **Cliente**:
   - O cliente é iniciado executando o código do cliente federado, que se conecta ao servidor federado (esteja atento, primeiro o servidor deve estar rodando). O cliente localmente treina o modelo e envia os parâmetros para o servidor após cada rodada de treinamento.

2. **Servidor**:
   - O servidor federado é configurado para rodar com o endereço especificado e começa a aceitar clientes. Ele coordena o processo de agregação e avaliação do modelo.

---

## **Principais Benefícios do Aprendizado Federado**:

- **Privacidade dos Dados**: Os dados dos clientes não são compartilhados, o que garante a privacidade.
- **Treinamento Descentralizado**: O modelo é treinado de forma colaborativa entre vários clientes, aproveitando os dados de forma eficiente sem centralizar tudo em um único servidor.
- **Escalabilidade**: A abordagem federada pode ser escalada para incluir muitos clientes, permitindo o treinamento em grandes conjuntos de dados distribuídos.

---
**Database utilizada**:

https://www.kaggle.com/datasets/sacramentotechnology/sleep-deprivation-and-cognitive-performance/data

Uma segunda database foi criada com base nessa para o cliente 2 

---
Este sistema é um exemplo de como utilizar o aprendizado federado com a biblioteca **Flower** para criar um modelo de aprendizado de máquina eficiente e seguro, preservando a privacidade dos dados.
