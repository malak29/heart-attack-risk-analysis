# Heart Attack Prediction System 🏥

A production-ready machine learning system for predicting heart attack risk using advanced ML models and modern MLOps practices.

## 🌟 Features

- **Multiple ML Models**: Random Forest, Gradient Boosting, XGBoost, and Ensemble methods
- **Real-time Predictions**: Fast API with sub-second response times
- **Model Versioning**: Complete model registry with version control
- **A/B Testing**: Compare model performances in production
- **Data Drift Detection**: Automatic monitoring for data distribution changes
- **Auto-scaling**: Kubernetes HPA for automatic scaling based on load
- **Comprehensive Monitoring**: Prometheus + Grafana dashboards
- **API Documentation**: Auto-generated OpenAPI/Swagger docs
- **Security**: JWT authentication, API keys, and rate limiting


## 🏗️ System Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web Application]
        MOB[Mobile App]
        API_CLIENT[API Clients]
    end
    
    subgraph "API Gateway"
        NGINX[Nginx/Load Balancer]
    end
    
    subgraph "Application Layer"
        API1[FastAPI Instance 1]
        API2[FastAPI Instance 2]
        API3[FastAPI Instance 3]
    end
    
    subgraph "ML Services"
        MODEL_REG[Model Registry]
        PREDICTOR[Prediction Service]
        TRAINER[Training Service]
        MONITOR[Monitoring Service]
    end
    
    subgraph "Data Layer"
        POSTGRES[(PostgreSQL)]
        REDIS[(Redis Cache)]
        S3[Model Storage]
    end
    
    subgraph "Monitoring Stack"
        PROM[Prometheus]
        GRAF[Grafana]
        ALERT[AlertManager]
    end
    
    subgraph "ML Platform"
        MLFLOW[MLflow]
        JUPYTER[Jupyter Hub]
    end
    
    WEB --> NGINX
    MOB --> NGINX
    API_CLIENT --> NGINX
    
    NGINX --> API1
    NGINX --> API2
    NGINX --> API3
    
    API1 --> MODEL_REG
    API1 --> PREDICTOR
    API1 --> TRAINER
    API1 --> MONITOR
    
    API2 --> MODEL_REG
    API2 --> PREDICTOR
    API2 --> TRAINER
    API2 --> MONITOR
    
    API3 --> MODEL_REG
    API3 --> PREDICTOR
    API3 --> TRAINER
    API3 --> MONITOR
    
    MODEL_REG --> S3
    MODEL_REG --> POSTGRES
    
    PREDICTOR --> REDIS
    PREDICTOR --> MODEL_REG
    
    TRAINER --> MLFLOW
    TRAINER --> S3
    
    MONITOR --> PROM
    PROM --> GRAF
    PROM --> ALERT
    
    TRAINER --> POSTGRES
    PREDICTOR --> POSTGRES
    
    style WEB fill:#e1f5fe
    style MOB fill:#e1f5fe
    style API_CLIENT fill:#e1f5fe
    style API1 fill:#c8e6c9
    style API2 fill:#c8e6c9
    style API3 fill:#c8e6c9
    style POSTGRES fill:#fff3e0
    style REDIS fill:#fff3e0
    style S3 fill:#fff3e0
    style PROM fill:#f3e5f5
    style GRAF fill:#f3e5f5
    style MLFLOW fill:#fce4ec
```

### Data Flow Architecture

```mermaid
flowchart LR
    subgraph "Data Ingestion"
        RAW[Raw Data]
        UPLOAD[File Upload]
        STREAM[Stream Data]
    end
    
    subgraph "Data Processing"
        CLEAN[Data Cleaning]
        VALID[Validation]
        FEAT[Feature Engineering]
    end
    
    subgraph "ML Pipeline"
        TRAIN[Model Training]
        EVAL[Evaluation]
        REG[Model Registry]
        DEPLOY[Deployment]
    end
    
    subgraph "Serving"
        PRED[Prediction API]
        BATCH[Batch Processing]
        CACHE[Cache Layer]
    end
    
    subgraph "Monitoring"
        DRIFT[Drift Detection]
        PERF[Performance Metrics]
        ALERT[Alerting]
    end
    
    RAW --> CLEAN
    UPLOAD --> CLEAN
    STREAM --> CLEAN
    
    CLEAN --> VALID
    VALID --> FEAT
    
    FEAT --> TRAIN
    TRAIN --> EVAL
    EVAL --> REG
    REG --> DEPLOY
    
    DEPLOY --> PRED
    DEPLOY --> BATCH
    PRED --> CACHE
    
    PRED --> DRIFT
    PRED --> PERF
    DRIFT --> ALERT
    PERF --> ALERT
    
    style RAW fill:#e3f2fd
    style UPLOAD fill:#e3f2fd
    style STREAM fill:#e3f2fd
    style TRAIN fill:#f3e5f5
    style EVAL fill:#f3e5f5
    style PRED fill:#e8f5e9
    style DRIFT fill:#fff3e0
    style PERF fill:#fff3e0
```

### ML Model Lifecycle

```mermaid
stateDiagram-v2
    [*] --> DataCollection
    DataCollection --> DataPreparation
    DataPreparation --> FeatureEngineering
    FeatureEngineering --> ModelTraining
    ModelTraining --> ModelEvaluation
    ModelEvaluation --> ModelSelection: Meets Threshold
    ModelEvaluation --> ModelTraining: Below Threshold
    ModelSelection --> ModelRegistry
    ModelRegistry --> Staging
    Staging --> ABTesting
    ABTesting --> Production: Better Performance
    ABTesting --> Staging: Worse Performance
    Production --> Monitoring
    Monitoring --> Retraining: Drift Detected
    Monitoring --> Production: Normal
    Retraining --> DataPreparation
    Production --> [*]
```

### API Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API Gateway
    participant FastAPI
    participant Auth Service
    participant Model Registry
    participant Prediction Service
    participant Cache
    participant Database
    participant Monitoring
    
    Client->>API Gateway: POST /predict
    API Gateway->>FastAPI: Forward Request
    FastAPI->>Auth Service: Validate Token
    Auth Service-->>FastAPI: Token Valid
    
    FastAPI->>Cache: Check Cache
    alt Cache Hit
        Cache-->>FastAPI: Return Cached Result
    else Cache Miss
        FastAPI->>Model Registry: Get Active Model
        Model Registry-->>FastAPI: Model v1.2.3
        FastAPI->>Prediction Service: Process Features
        Prediction Service->>Prediction Service: Engineer Features
        Prediction Service->>Prediction Service: Make Prediction
        Prediction Service-->>FastAPI: Risk Score
        FastAPI->>Cache: Store Result
        FastAPI->>Database: Log Prediction
    end
    
    FastAPI->>Monitoring: Send Metrics
    FastAPI-->>API Gateway: Return Response
    API Gateway-->>Client: JSON Response
```


## 🚀 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/malak29/heart-attack-prediction.git
cd heart-attack-prediction

# Install dependencies
make install

# Run the application
make run

# Access the API
curl http://localhost:8000/api/v1/health
```

### Docker Deployment

```bash
# Build and run with Docker Compose
make docker-run

# Access services
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000 (admin/admin)
```

### API Documentation

Once running, access the interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📊 API Endpoints

### Prediction Endpoints
- `POST /api/v1/predict/` - Single patient prediction
- `POST /api/v1/predict/batch` - Batch predictions
- `POST /api/v1/predict/assess` - Detailed risk assessment

### Model Management
- `GET /api/v1/models/` - List all models
- `POST /api/v1/models/{version}/activate` - Activate model
- `POST /api/v1/models/compare` - Compare models

### Training
- `POST /api/v1/train/` - Train new model
- `POST /api/v1/train/upload-data` - Upload training data
- `GET /api/v1/train/jobs` - List training jobs

### Health & Monitoring
- `GET /api/v1/health/` - Basic health check
- `GET /api/v1/health/detailed` - Detailed system status
- `GET /api/v1/health/metrics` - Performance metrics

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FastAPI   │────▶│   ML Models │────▶│  Monitoring │
│     API     │     │   Registry  │     │  (Prometheus)│
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                    │
       ▼                   ▼                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PostgreSQL │     │    Redis    │     │   Grafana   │
│   Database  │     │    Cache    │     │  Dashboard  │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run integration tests
make test-integration
```

## 📈 Model Training

### Using the API

```python
import requests

# Upload training data
with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/train/upload-data',
        files={'file': f}
    )

# Train model
response = requests.post(
    'http://localhost:8000/api/v1/train/',
    json={
        'model_type': 'random_forest',
        'auto_tune': True,
        'validation_split': 0.2
    }
)
```

### Using CLI

```bash
# Train with default settings
make train

# Train with auto-tuning
make train-auto
```

## 🚢 Deployment

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes/deployment.yaml

# Check deployment status
kubectl get pods -n heart-attack-prediction

# Access logs
kubectl logs -f deployment/heart-attack-api -n heart-attack-prediction
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## 📊 Monitoring

### Metrics Available
- Request count and latency
- Model prediction confidence
- Data drift indicators
- System resource usage
- Error rates

### Grafana Dashboards
1. Navigate to http://localhost:3000
2. Login with admin/admin
3. Import dashboards from `monitoring/grafana/dashboards/`

## 🔒 Security

### Authentication
- JWT tokens for user authentication
- API keys for service-to-service communication
- Rate limiting to prevent abuse

### Environment Variables
```bash
# Copy example env file
cp .env.example .env

# Update with your values
vim .env
```
