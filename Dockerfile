FROM ghcr.io/mlflow/mlflow:v2.21.3

RUN pip install psycopg2-binary sqlalchemy

CMD [ "mlflow", "server", "--host=0.0.0.0", "--backend-store-uri", "postgresql://mlflow_user:mlflow_password@postgres:5432/mlflow_db", "--artifacts-destination file:///ml/mlartifacts"]