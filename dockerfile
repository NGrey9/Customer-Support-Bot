FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

ENV PATH /opt/conda/envs/ai_agent/bin:$PATH

RUN apt-get update && apt-get install -y curl wget && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY .env /app

COPY . /app

RUN echo "#!/bin/bash\n\
set -e\n\
echo \"Activating conda environment...\"\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate ai_agent\n\
\n\
echo \"Downloading Ollama model...\"\n\
ollama pull deepseek-r1\n\
ollama pull llama3.1:8b\n\
\n\
echo \"Starting Ollama service...\"\n\
ollama serve &\n\
\n\
sleep 10\n\
\n\
echo \"Starting main application...\"\n\
exec python main.py" > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 8088 11434 27017 3306

ENTRYPOINT ["/app/entrypoint.sh"]
