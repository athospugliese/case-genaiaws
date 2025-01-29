# 🧠 Case Desenvolvedor GenAI AWS

## 🚀 **Desafio**
O objetivo deste desafio é criar uma solução utilizando as bibliotecas **Langchain** e **Langgraph**, com a seguinte estrutura:

### 🎯 **Requisitos**
1. **Agente Conversacional**: 
   - Interage com o usuário em qualquer tema, exceto **Engenharia Civil**.
   - A resposta final ao usuário deve ser proveniente deste agente.

2. **Agente de Busca**:
   - Realiza buscas com um limite de **10 retornos** por consulta.
   - É acionado apenas quando necessário pelo Agente Conversacional.

3. **API Pública**:
   - Criada com **FastAPI**, esta API:
     - Recebe perguntas do usuário.
     - Retorna respostas apenas do Agente Conversacional.
   - Deve rodar em um ambiente com **Kubernetes** e **Docker**.

4. **Modelos Disponíveis**:
   - Suporte aos seguintes modelos:
     - 🟢 **GPT-4o-mini**: Respostas rápidas e precisas.
     - 🟢 **GPT-4o**: Modelo avançado para tarefas complexas.
     - 🔵 **Llama-3.1-8B-Instant**: Modelo leve para respostas rápidas.
     - 🔵 **Llama-3.3-70B-Versatile**: Versátil para demandas complexas.
     - 🟡 **Llama-Guard-3-8B**: Focado em moderação de conteúdo.
     - 🟡 **DeepSeek-R1-Distill-Llama-70B**: Para análises profundas e buscas.

---

## 📂 **Estrutura de Diretórios**

### 📦 Diretório `/src`

#### 🧩 **Agentes (`/src/agents`)**
- **`agents.py`**: Implementação dos agentes conversacional e de busca.
- **`llama_guard.py`**: Moderação de conteúdo com LlamaGuard.
- **`supervisor.py`**: Gerenciamento da comunicação entre agentes.
- **`utils.py`**: Utilitários para operação dos agentes.

#### 💻 **Cliente (`/src/client`)**
- **`client.py`**: Gerencia conexões entre a API e os agentes.

#### 🔧 **Core (`/src/core`)**
- **`llm.py`**: Configura e integra os modelos disponíveis.
- **`settings.py`**: Configurações principais do sistema.

#### 🗂 **Schemas (`/src/schema`)**
- **`models.py`**: Define modelos e dados estruturados.
- **`task_data.py`**: Estrutura para gerenciamento de dados.

#### 🔌 **Serviços (`/src/service`)**
- **`service.py`**: Lógica principal do serviço.
- **`run_agent.py`**: Executa os agentes.
- **`run_service.py`**: Inicializa o serviço.

---


### 🔒 **Moderação com LlamaGuard** ✋ **Prevenção de Conteúdo Inseguro**
O **LlamaGuard** assegura que o conteúdo gerado seja seguro, classificando respostas em categorias de risco.  
**Principais Categorias de Violação**:
- **S1**: Crimes violentos.
- **S2**: Crimes não violentos.
- **S3**: Crimes sexuais.
- **S4**: Exploração infantil.
- **S5**: Difamação.
- **S6**: Aconselhamento especializado.
- **S7**: Privacidade.
- **S8**: Propriedade intelectual.
- **S9**: Armas indiscriminadas.
- **S10**: Discurso de ódio.
- **S11**: Auto-mutilação.
- **S12**: Conteúdo sexual.
- **S13**: Eleições.
- **S14**: Abuso de interpretadores de código.

### 📊 **Saída de Moderação**
A classe `LlamaGuardOutput` fornece:
- **safety_assessment**: Avaliação geral da segurança do conteúdo.
- **unsafe_categories**: Lista de categorias inseguras detectadas.

## Demonstração

![App Screenshot](https://i.imgur.com/QkMF6Ln.png)

![App Screenshot](https://i.imgur.com/xUdmSv5.png)

---

## 📈 Feedback e Tracing com LangSmith e 📌 Funcionalidades
#### Tracing:
Monitora fluxos de execução dos agentes.
Identifica gargalos e erros em tempo real.
#### Feedback:
Captura avaliações do usuário sobre a qualidade das respostas.
Dados são usados para melhorar continuamente os agentes.
#### 🔍 Visualização de Logs
Logs detalhados são gerados pelo LangSmith para análise e otimização do desempenho.
##



## ⚙️ **Como Executar**

#### **📋 Pré-requisitos**
- Docker e Kubernetes instalados.
- Python 3.12+.
- Conta AWS configurada (opcional).

### **▶️ Passos**
1. Clone o repositório:
   ```bash
   git clone https://github.com/athospugliese/case-genaiaws
   cd case-genaiaws
   python3 -m venv venv
   source venv/bin/activate  # Linux/MacOS
   .\venv\Scripts\activate  # Windows

   pip install -r requirements.txt

## 🚀 **Instruções de Docker e Kubernetes**

#### **🐳 Criando a Imagem Docker**

Para criar a imagem Docker do seu projeto, siga estas etapas:

**Criar a Imagem Docker**:
   
   No diretório do projeto, crie a imagem Docker com o seguinte comando:

    docker build -t nome-da-imagem .
    docker run -p 8000:8000 nome-da-imagem


#### **🚢 Implantando no Kubernetes**

Para criar a imagem Docker do seu projeto, siga estas etapas:

**Configuração do Kubernetes**:

Certifique-se de ter o kubectl instalado e configurado corretamente. Você deve criar um arquivo chamado deployment.yaml no diretório do projeto com o seguinte conteúdo:

    apiVersion: apps/v1
    kind: Deployment
    metadata:
    name: nome-do-app
    spec:
    replicas: 3
    selector:
        matchLabels:
        app: nome-do-app
    template:
        metadata:
        labels:
            app: nome-do-app
        spec:
        containers:
        - name: nome-do-app
            image: nome-da-imagem:latest  # Nome da imagem Docker que você criou
            ports:
            - containerPort: 8000


**Criar o Deployment no Kubernetes**:
   
   Aplique o arquivo de configuração para criar o Deployment no seu cluster Kubernetes com o primeiro comando
   Verifique se os PODs estão rodando com o segundo comando

    kubectl apply -f deployment.yaml
    kubectl get pods

**Expor o acesso**:
   
   Se você deseja expor seu serviço fora do cluster Kubernetes, crie um serviço com o seguinte comando:

    kubectl expose deployment nome-do-app --type=LoadBalancer --port=80 --target-port=8000





## Arquitetura AWS

![App Screenshot](https://i.imgur.com/VDusCAn.png)

![App Screenshot](https://i.imgur.com/PFF5eUX.png)
