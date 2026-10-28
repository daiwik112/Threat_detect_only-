# API image — build context is the project root (sharavathi-theft/), see docker-compose.yml
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# versions pinned to the environment the models in data/models were trained with (joblib / torch artifacts)
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install pandapower==3.5.6 numpy==2.4.6 pandas==2.3.3 pyarrow==26.0.0 scipy==1.18.1 \
                scikit-learn==1.9.1 xgboost==3.4.1 shap==0.54.1 networkx==3.7 pyyaml==6.0.3 \
                pydantic==2.14.1 joblib==1.6.0 fastapi==0.162.0 "uvicorn[standard]==0.54.0" \
                httpx==0.28.1 anthropic==2.3.0 torch-geometric==2.8.1
COPY grid ./grid
COPY simdata ./simdata
COPY ml ./ml
COPY rag ./rag
COPY vectordb ./vectordb
COPY future ./future
COPY config ./config
COPY backend ./backend
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
