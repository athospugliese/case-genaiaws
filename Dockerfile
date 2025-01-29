# Use uma imagem base do Python
FROM python:3.12-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto para o contêiner
COPY . /app

# Atualiza o pip e instala as dependências
RUN pip install --upgrade pip && pip install -r requirements.txt

# Define o PYTHONPATH para incluir o diretório src
ENV PYTHONPATH=/app/src

# Exponha as portas necessárias
EXPOSE 8501 8000

# Comando para iniciar Streamlit e FastAPI
CMD ["sh", "-c", "streamlit run src/app.py --server.port=8501 & uvicorn service.service:app --host 0.0.0.0 --port 8000"]
