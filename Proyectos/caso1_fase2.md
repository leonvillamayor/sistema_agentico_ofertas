# Caso de Uso 1: Yield Predictivo - Fase II: Operativización y MLOps

## Introducción a la Fase II

La Fase II del proyecto Yield Predictivo de SSMAS marca la transición del MVP validado en la Fase I hacia un sistema de producción robusto, escalable y completamente operativo. Esta fase transforma el entorno de desarrollo en una plataforma de machine learning de clase empresarial que puede manejar la volumetría completa de SSMAS: 50 millones de registros diarios y más de 7.000 millones de impresiones mensuales.

El enfoque principal de esta fase es la implementación de pipelines de MLOps automatizados, la integración con el sistema de subastas en tiempo real de SSMAS, y el establecimiento de un framework de monitorización avanzada que garantice la calidad y rendimiento continuos del modelo predictivo.

## Objetivos de la Fase II

### Objetivo Principal
Operativizar el sistema Yield Predictivo en el entorno de producción de SSMAS, implementando pipelines de MLOps completamente automatizados que permitan el entrenamiento, despliegue y monitorización continua del modelo XGBoost para optimización de CPM en tiempo real.

### Objetivos Específicos

1. **Escalabilidad de Producción**: Migrar la arquitectura para soportar 50M+ registros diarios con latencia sub-50ms
2. **MLOps Automatizado**: Implementar pipelines CI/CD para machine learning con reentrenamiento automático
3. **Integración en Tiempo Real**: Conectar el sistema con la plataforma de subastas SSMAS existente
4. **Monitorización Avanzada**: Establecer sistema completo de observabilidad para model drift y performance
5. **Alta Disponibilidad**: Garantizar 99.9% SLA con redundancia multi-AZ y auto-recovery
6. **Seguridad Empresarial**: Implementar controles de seguridad y compliance para datos de producción

## Arquitectura de Producción

### Migración Multi-Cuenta

#### Estructura de Cuentas de Producción
```
SSMAS Production Workload
├── ML Production Account (123456789012)
│   ├── Region Principal: eu-west-1 (Dublin)
│   ├── Region DR: eu-central-1 (Frankfurt)
│   └── VPC: 10.0.0.0/16
├── Data Production Account (123456789013)
│   ├── Data Lake S3: 50M+ registros/día
│   ├── Streaming: Kinesis Data Streams
│   └── VPC: 10.0.32.0/19
└── AdOps Production Account (123456789014)
    ├── API Gateway: Integración SSMAS
    ├── Application Load Balancer
    └── VPC: 10.0.64.0/19
```

#### Conectividad Multi-Cuenta
**AWS Transit Gateway**:
- TGW Hub en Network Account
- Cross-account VPC attachments
- Route tables específicas por entorno
- Bandwidth: 50 Gbps entre cuentas

**VPC Peering Directo**:
- ML ↔ Data: Transferencia de features
- ML ↔ AdOps: Predicciones en tiempo real
- Latencia < 1ms entre AZ en misma región

### Arquitectura de Datos Escalable

#### Data Lake Optimizado para ML

**Amazon S3 - Estructura Avanzada**:
```
s3://ssmas-yield-prod-datalake-eu-west-1/
├── raw/
│   ├── year=2024/month=10/day=22/hour=14/
│   │   ├── impressions/partition=device_category/
│   │   ├── auctions/partition=country/
│   │   └── users/partition=traffic_source/
├── processed/
│   ├── features/
│   │   ├── training_features/version=v2.1/
│   │   ├── inference_features/latest/
│   │   └── feature_store/
│   └── aggregated/
│       ├── hourly_stats/
│       ├── daily_metrics/
│       └── weekly_trends/
├── models/
│   ├── production/
│   │   ├── current/yield-predictor-v2.1/
│   │   ├── champion/yield-predictor-v2.0/
│   │   └── challenger/yield-predictor-v2.2/
│   ├── experiments/
│   └── archived/
└── outputs/
    ├── predictions/
    │   ├── real_time/partition=hour/
    │   └── batch/partition=day/
    ├── model_metrics/
    └── monitoring/
```

**Optimizaciones de Performance**:
- **Particionado inteligente**: Por device_category, country, hour
- **Formato columnar**: Parquet con compresión GZIP
- **S3 Transfer Acceleration**: Para ingesta global
- **S3 Intelligent Tiering**: Optimización automática de costos
- **Multipart Upload**: Para archivos > 100MB

#### Real-Time Data Streaming

**Amazon Kinesis Data Streams**:
```python
# Configuración de Kinesis para 50M registros/día
kinesis_config = {
    "StreamName": "ssmas-impressions-stream",
    "ShardCount": 100,  # ~500 registros/segundo por shard
    "RetentionPeriod": 168,  # 7 días
    "ShardLevelMetrics": ["IncomingRecords", "OutgoingRecords"],
    "EncryptionType": "KMS",
    "KMSKeyId": "alias/ssmas-kinesis-key"
}
```

**Amazon Kinesis Data Analytics**:
- **Real-time feature engineering**: Agregaciones en ventanas temporales
- **Anomaly detection**: Detección de patrones inusuales en tiempo real
- **Data quality monitoring**: Validación continua de datos entrantes

**AWS Kinesis Data Firehose**:
- **Delivery a S3**: Batch cada 60 segundos o 128MB
- **Format conversion**: JSON a Parquet automático
- **Data transformation**: Lambda para limpieza de datos
- **Error record handling**: S3 bucket separado para registros erróneos

### Machine Learning Platform Escalable

#### Amazon SageMaker - Configuración de Producción

**SageMaker Feature Store**:
```python
# Feature Store para features en tiempo real
feature_group_config = {
    "FeatureGroupName": "yield-predictor-features",
    "RecordIdentifierFeatureName": "impression_id",
    "EventTimeFeatureName": "event_time",
    "OnlineStoreConfig": {
        "EnableOnlineStore": True,
        "SecurityConfig": {
            "KmsKeyId": "alias/ssmas-feature-store-key"
        }
    },
    "OfflineStoreConfig": {
        "S3StorageConfig": {
            "S3Uri": "s3://ssmas-yield-prod-datalake-eu-west-1/feature_store/",
            "KmsKeyId": "alias/ssmas-feature-store-key"
        },
        "DisableGlueTableCreation": False
    }
}
```

**SageMaker Model Registry**:
- **Model Versioning**: Versionado automático con MLflow
- **Model Approval**: Workflow de aprobación automática basada en métricas
- **A/B Testing**: Deployment gradual con split de tráfico
- **Model Lineage**: Trazabilidad completa desde datos hasta predicciones

**Multi-Model Endpoints**:
```python
# Configuración de endpoint multi-modelo
multi_model_config = {
    "EndpointName": "yield-predictor-multi-model",
    "EndpointConfigName": "yield-predictor-config-v2",
    "ProductionVariants": [
        {
            "VariantName": "champion-model",
            "ModelName": "yield-predictor-v2-1",
            "InitialInstanceCount": 3,
            "InstanceType": "ml.c5.xlarge",
            "InitialVariantWeight": 80,
            "AcceleratorType": "ml.eia2.medium"
        },
        {
            "VariantName": "challenger-model",
            "ModelName": "yield-predictor-v2-2",
            "InitialInstanceCount": 1,
            "InstanceType": "ml.c5.xlarge",
            "InitialVariantWeight": 20,
            "AcceleratorType": "ml.eia2.medium"
        }
    ],
    "DataCaptureConfig": {
        "EnableCapture": True,
        "InitialSamplingPercentage": 10,
        "DestinationS3Uri": "s3://ssmas-yield-prod-datalake-eu-west-1/model_data_capture/"
    }
}
```

### MLOps Pipeline Automatizado

#### CI/CD para Machine Learning

**AWS CodePipeline - ML Pipeline**:
```yaml
# ml-pipeline.yml
stages:
  - name: Source
    actions:
      - name: SourceAction
        actionTypeId:
          category: Source
          owner: AWS
          provider: S3
        configuration:
          S3Bucket: ssmas-ml-code-repo
          S3ObjectKey: ml-code.zip
        outputArtifacts:
          - name: SourceOutput

  - name: DataValidation
    actions:
      - name: ValidateData
        actionTypeId:
          category: Invoke
          owner: AWS
          provider: Lambda
        configuration:
          FunctionName: validate-training-data
        inputArtifacts:
          - name: SourceOutput

  - name: Training
    actions:
      - name: TrainModel
        actionTypeId:
          category: Invoke
          owner: AWS
          provider: SageMaker
        configuration:
          TrainingJobName: yield-predictor-training-${codepipeline.PipelineExecutionId}
          TrainingImage: 246618743249.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest
          RoleArn: arn:aws:iam::123456789012:role/SageMakerExecutionRole
          InputDataConfig:
            - ChannelName: training
              DataSource:
                S3DataSource:
                  S3DataType: S3Prefix
                  S3Uri: s3://ssmas-yield-prod-datalake-eu-west-1/processed/training_features/latest/
                  S3DataDistributionType: FullyReplicated
          OutputDataConfig:
            S3OutputPath: s3://ssmas-yield-prod-datalake-eu-west-1/models/training_output/

  - name: ModelEvaluation
    actions:
      - name: EvaluateModel
        actionTypeId:
          category: Invoke
          owner: AWS
          provider: Lambda
        configuration:
          FunctionName: evaluate-model-performance

  - name: ModelApproval
    actions:
      - name: ApproveModel
        actionTypeId:
          category: Approval
          owner: AWS
          provider: Manual
        configuration:
          CustomData: "Review model metrics and approve for production deployment"

  - name: Deploy
    actions:
      - name: DeployToProduction
        actionTypeId:
          category: Invoke
          owner: AWS
          provider: SageMaker
        configuration:
          EndpointName: yield-predictor-prod
          EndpointConfigName: yield-predictor-config-${codepipeline.PipelineExecutionId}
```

#### Automated Retraining Pipeline

**AWS Step Functions - Retraining Workflow**:
```json
{
  "Comment": "Automated ML model retraining workflow",
  "StartAt": "CheckDataDrift",
  "States": {
    "CheckDataDrift": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:eu-west-1:123456789012:function:check-data-drift",
      "Next": "EvaluateDriftLevel"
    },
    "EvaluateDriftLevel": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.drift_score",
          "NumericGreaterThan": 0.7,
          "Next": "TriggerRetraining"
        },
        {
          "Variable": "$.drift_score",
          "NumericGreaterThan": 0.5,
          "Next": "ScheduleRetraining"
        }
      ],
      "Default": "MonitoringOnly"
    },
    "TriggerRetraining": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Parameters": {
        "TrainingJobName.$": "$.training_job_name",
        "RoleArn": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
        "AlgorithmSpecification": {
          "TrainingImage": "246618743249.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest",
          "TrainingInputMode": "File"
        },
        "InputDataConfig": [
          {
            "ChannelName": "training",
            "DataSource": {
              "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": "s3://ssmas-yield-prod-datalake-eu-west-1/processed/training_features/latest/",
                "S3DataDistributionType": "FullyReplicated"
              }
            }
          }
        ],
        "OutputDataConfig": {
          "S3OutputPath": "s3://ssmas-yield-prod-datalake-eu-west-1/models/retraining_output/"
        },
        "ResourceConfig": {
          "InstanceType": "ml.m5.xlarge",
          "InstanceCount": 1,
          "VolumeSizeInGB": 30
        },
        "StoppingCondition": {
          "MaxRuntimeInSeconds": 3600
        }
      },
      "Next": "EvaluateNewModel"
    },
    "EvaluateNewModel": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:eu-west-1:123456789012:function:evaluate-retrained-model",
      "Next": "DecideDeployment"
    },
    "DecideDeployment": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.new_model_better",
          "BooleanEquals": true,
          "Next": "DeployNewModel"
        }
      ],
      "Default": "KeepCurrentModel"
    },
    "DeployNewModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createEndpointConfig",
      "Next": "UpdateEndpoint"
    },
    "UpdateEndpoint": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:updateEndpoint",
      "Next": "NotifySuccess"
    },
    "ScheduleRetraining": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:eu-west-1:123456789012:function:schedule-future-retraining",
      "End": true
    },
    "MonitoringOnly": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:eu-west-1:123456789012:function:continue-monitoring",
      "End": true
    },
    "KeepCurrentModel": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:eu-west-1:123456789012:function:log-retraining-decision",
      "End": true
    },
    "NotifySuccess": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:eu-west-1:123456789012:function:notify-deployment-success",
      "End": true
    }
  }
}
```

#### Model Drift Detection

**AWS Lambda - Drift Detection Function**:
```python
import json
import boto3
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Detecta data drift comparando distribuciones de features
    entre datos de entrenamiento y datos de producción
    """

    # Clientes AWS
    s3_client = boto3.client('s3')
    cloudwatch = boto3.client('cloudwatch')

    try:
        # Configuración
        bucket = 'ssmas-yield-prod-datalake-eu-west-1'
        training_data_path = 'processed/training_features/baseline/'
        current_data_path = 'processed/inference_features/latest/'

        # Cargar datos de referencia (baseline)
        baseline_data = load_data_from_s3(s3_client, bucket, training_data_path)

        # Cargar datos actuales
        current_data = load_data_from_s3(s3_client, bucket, current_data_path)

        # Calcular drift para cada feature
        drift_scores = calculate_feature_drift(baseline_data, current_data)

        # Calcular drift score global
        global_drift_score = np.mean(list(drift_scores.values()))

        # Publicar métricas a CloudWatch
        publish_drift_metrics(cloudwatch, drift_scores, global_drift_score)

        # Determinar nivel de alerta
        alert_level = determine_alert_level(global_drift_score)

        # Respuesta
        response = {
            'drift_score': global_drift_score,
            'alert_level': alert_level,
            'feature_drift_scores': drift_scores,
            'timestamp': pd.Timestamp.now().isoformat(),
            'requires_retraining': global_drift_score > 0.5
        }

        logger.info(f"Drift detection completed. Global score: {global_drift_score}")

        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }

    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def load_data_from_s3(s3_client, bucket: str, prefix: str) -> pd.DataFrame:
    """Carga datos de S3 y los convierte a DataFrame"""

    # Listar objetos en el prefijo
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if 'Contents' not in response:
        raise ValueError(f"No data found in {prefix}")

    # Tomar el archivo más reciente
    latest_object = max(response['Contents'], key=lambda x: x['LastModified'])

    # Descargar y cargar
    obj = s3_client.get_object(Bucket=bucket, Key=latest_object['Key'])
    data = pd.read_parquet(obj['Body'])

    return data

def calculate_feature_drift(baseline: pd.DataFrame, current: pd.DataFrame) -> Dict[str, float]:
    """Calcula drift score para cada feature usando test de Kolmogorov-Smirnov"""

    drift_scores = {}

    # Features numéricas para comparar
    numeric_features = baseline.select_dtypes(include=[np.number]).columns

    for feature in numeric_features:
        if feature in current.columns:
            # Test de Kolmogorov-Smirnov
            ks_statistic, p_value = stats.ks_2samp(
                baseline[feature].dropna(),
                current[feature].dropna()
            )

            # Drift score (0-1, donde 1 = máximo drift)
            drift_scores[feature] = ks_statistic

    return drift_scores

def publish_drift_metrics(cloudwatch, drift_scores: Dict[str, float], global_score: float):
    """Publica métricas de drift a CloudWatch"""

    # Métrica global
    cloudwatch.put_metric_data(
        Namespace='SSMAS/YieldPredictor/Drift',
        MetricData=[
            {
                'MetricName': 'GlobalDriftScore',
                'Value': global_score,
                'Unit': 'None'
            }
        ]
    )

    # Métricas por feature
    for feature, score in drift_scores.items():
        cloudwatch.put_metric_data(
            Namespace='SSMAS/YieldPredictor/Drift',
            MetricData=[
                {
                    'MetricName': 'FeatureDriftScore',
                    'Value': score,
                    'Unit': 'None',
                    'Dimensions': [
                        {
                            'Name': 'FeatureName',
                            'Value': feature
                        }
                    ]
                }
            ]
        )

def determine_alert_level(drift_score: float) -> str:
    """Determina nivel de alerta basado en drift score"""

    if drift_score > 0.7:
        return "CRITICAL"
    elif drift_score > 0.5:
        return "WARNING"
    elif drift_score > 0.3:
        return "INFO"
    else:
        return "OK"
```

## Integración con Sistema de Subastas SSMAS

### API de Predicción en Tiempo Real

#### Arquitectura de API Gateway Avanzada

**API Gateway REST - Configuración de Producción**:
```yaml
# api-gateway-prod.yml
apiVersion: v2
info:
  title: SSMAS Yield Predictor Production API
  version: "2.0"

servers:
  - url: https://api-yield.ssmas.com/v2
    description: Production server

paths:
  /predict:
    post:
      summary: Get CPM prediction for impression opportunity
      operationId: predictCPM
      security:
        - ApiKeyAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/PredictionRequest'
      responses:
        '200':
          description: Successful prediction
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PredictionResponse'
        '400':
          description: Invalid request
        '429':
          description: Rate limit exceeded
        '500':
          description: Internal server error

  /predict/batch:
    post:
      summary: Get CPM predictions for multiple impressions
      operationId: predictBatchCPM
      security:
        - ApiKeyAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: array
              items:
                $ref: '#/components/schemas/PredictionRequest'
              maxItems: 1000
      responses:
        '200':
          description: Successful batch prediction
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/PredictionResponse'

  /health:
    get:
      summary: Health check endpoint
      responses:
        '200':
          description: Service is healthy

components:
  schemas:
    PredictionRequest:
      type: object
      required:
        - impression_id
        - hour_of_day
        - day_of_week
        - country_tier
        - device_category
        - format
        - inventory_type
      properties:
        impression_id:
          type: string
          description: Unique identifier for the impression
        hour_of_day:
          type: integer
          minimum: 0
          maximum: 23
        day_of_week:
          type: integer
          minimum: 1
          maximum: 7
        country_tier:
          type: integer
          minimum: 1
          maximum: 3
        device_category:
          type: string
          enum: [mobile, tablet, desktop]
        format:
          type: string
          enum: [banner, video, native, interstitial]
        inventory_type:
          type: string
          enum: [premium, standard, remnant]
        bidder:
          type: string
        demand_channel:
          type: string
        domain:
          type: string
        user_engagement_score:
          type: number
          minimum: 0
          maximum: 1
        device_performance_score:
          type: number
          minimum: 0
        historical_ctr_avg:
          type: number
          minimum: 0
          maximum: 1

    PredictionResponse:
      type: object
      properties:
        impression_id:
          type: string
        predicted_cpm:
          type: number
          description: Predicted CPM in EUR
        confidence_interval:
          type: object
          properties:
            lower:
              type: number
            upper:
              type: number
        model_version:
          type: string
        prediction_timestamp:
          type: string
          format: date-time
        latency_ms:
          type: number

  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
```

#### Lambda de Integración Optimizada

**Lambda Function - Predicción en Tiempo Real**:
```python
import json
import boto3
import time
import logging
from typing import Dict, List, Any
import numpy as np
from botocore.exceptions import ClientError

# Configuración de logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clientes AWS globales (reutilización de conexiones)
sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='eu-west-1')
cloudwatch = boto3.client('cloudwatch', region_name='eu-west-1')

# Configuración
ENDPOINT_NAME = 'yield-predictor-multi-model'
FEATURE_STORE_NAME = 'yield-predictor-features'

def lambda_handler(event, context):
    """
    Handler principal para predicciones de CPM en tiempo real
    Optimizado para baja latencia y alta volumetría
    """

    start_time = time.time()

    try:
        # Parsear request
        if 'body' in event:
            request_body = json.loads(event['body'])
        else:
            request_body = event

        # Determinar si es batch o single prediction
        if isinstance(request_body, list):
            # Batch prediction
            predictions = process_batch_predictions(request_body)
            response_body = predictions
        else:
            # Single prediction
            prediction = process_single_prediction(request_body)
            response_body = prediction

        # Calcular latencia
        latency_ms = (time.time() - start_time) * 1000

        # Publicar métricas
        publish_latency_metric(latency_ms)

        # Respuesta
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'X-Request-ID': context.aws_request_id,
                'X-Latency-MS': str(round(latency_ms, 2))
            },
            'body': json.dumps(response_body, default=str)
        }

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return create_error_response(400, "Invalid request format", str(e))

    except ClientError as e:
        logger.error(f"AWS service error: {str(e)}")
        return create_error_response(500, "Service unavailable", "Prediction service temporarily unavailable")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return create_error_response(500, "Internal server error", "An unexpected error occurred")

def process_single_prediction(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Procesa una predicción individual"""

    # Validar datos de entrada
    validate_prediction_request(request_data)

    # Enriquecer con features adicionales
    enriched_features = enrich_features(request_data)

    # Realizar predicción
    prediction_result = call_sagemaker_endpoint(enriched_features)

    # Formatear respuesta
    response = format_prediction_response(
        request_data.get('impression_id'),
        prediction_result,
        enriched_features
    )

    return response

def process_batch_predictions(request_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Procesa múltiples predicciones en batch"""

    if len(request_batch) > 1000:
        raise ValueError("Batch size cannot exceed 1000 predictions")

    predictions = []

    # Procesar en paralelo (usando threading para I/O bound operations)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Enviar todas las predicciones
        future_to_request = {
            executor.submit(process_single_prediction, request): request
            for request in request_batch
        }

        # Recoger resultados
        for future in as_completed(future_to_request):
            try:
                prediction = future.result()
                predictions.append(prediction)
            except Exception as e:
                # En caso de error, agregar respuesta de error
                request = future_to_request[future]
                error_response = {
                    'impression_id': request.get('impression_id', 'unknown'),
                    'error': str(e),
                    'predicted_cpm': None
                }
                predictions.append(error_response)

    return predictions

def validate_prediction_request(request_data: Dict[str, Any]) -> None:
    """Valida los datos de entrada de la predicción"""

    required_fields = [
        'hour_of_day', 'day_of_week', 'country_tier',
        'device_category', 'format', 'inventory_type'
    ]

    # Verificar campos requeridos
    for field in required_fields:
        if field not in request_data:
            raise ValueError(f"Missing required field: {field}")

    # Validaciones específicas
    if not (0 <= request_data['hour_of_day'] <= 23):
        raise ValueError("hour_of_day must be between 0 and 23")

    if not (1 <= request_data['day_of_week'] <= 7):
        raise ValueError("day_of_week must be between 1 and 7")

    if not (1 <= request_data['country_tier'] <= 3):
        raise ValueError("country_tier must be between 1 and 3")

    valid_device_categories = ['mobile', 'tablet', 'desktop']
    if request_data['device_category'] not in valid_device_categories:
        raise ValueError(f"device_category must be one of: {valid_device_categories}")

def enrich_features(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enriquece los datos con features adicionales del Feature Store"""

    # Copiar datos originales
    enriched = request_data.copy()

    # Agregar timestamp
    enriched['prediction_timestamp'] = time.time()

    # Calcular features temporales adicionales
    hour = request_data['hour_of_day']
    day = request_data['day_of_week']

    enriched['is_weekend'] = 1 if day in [6, 7] else 0
    enriched['hour_category'] = categorize_hour(hour)
    enriched['is_prime_time'] = 1 if 19 <= hour <= 23 else 0

    # Features geográficas
    enriched['is_eu'] = 1 if request_data.get('country_tier', 1) == 1 else 0

    # Features de dispositivo
    device_category = request_data['device_category']
    enriched['device_category_encoded'] = encode_device_category(device_category)

    # Features adicionales con valores por defecto
    enriched.setdefault('user_engagement_score', 0.5)
    enriched.setdefault('device_performance_score', 100.0)
    enriched.setdefault('historical_ctr_avg', 0.02)

    return enriched

def categorize_hour(hour: int) -> str:
    """Categoriza la hora en franjas temporales"""
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

def encode_device_category(category: str) -> int:
    """Codifica categoría de dispositivo a valor numérico"""
    encoding = {
        'mobile': 1,
        'tablet': 2,
        'desktop': 3
    }
    return encoding.get(category, 1)

def call_sagemaker_endpoint(features: Dict[str, Any]) -> Dict[str, Any]:
    """Llama al endpoint de SageMaker para obtener predicción"""

    # Preparar datos para el modelo
    model_input = prepare_model_input(features)

    # Llamar endpoint
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(model_input),
        TargetVariant='champion-model'  # Usar modelo campeón por defecto
    )

    # Parsear respuesta
    result = json.loads(response['Body'].read().decode())

    return result

def prepare_model_input(features: Dict[str, Any]) -> Dict[str, Any]:
    """Prepara los datos en el formato esperado por el modelo"""

    # Lista de features en el orden esperado por el modelo
    model_features = [
        'hour_of_day', 'day_of_week', 'is_weekend', 'country_tier',
        'device_category_encoded', 'is_eu', 'user_engagement_score',
        'device_performance_score', 'historical_ctr_avg', 'is_prime_time'
    ]

    # Crear array de features
    feature_values = []
    for feature in model_features:
        value = features.get(feature, 0)
        feature_values.append(float(value))

    return {
        'instances': [feature_values]
    }

def format_prediction_response(impression_id: str, prediction_result: Dict, features: Dict) -> Dict[str, Any]:
    """Formatea la respuesta de predicción"""

    predicted_cpm = prediction_result.get('predictions', [0])[0]

    # Calcular intervalo de confianza (aproximación)
    confidence_margin = predicted_cpm * 0.1  # 10% margin

    response = {
        'impression_id': impression_id,
        'predicted_cpm': round(float(predicted_cpm), 4),
        'confidence_interval': {
            'lower': round(float(predicted_cpm - confidence_margin), 4),
            'upper': round(float(predicted_cpm + confidence_margin), 4)
        },
        'model_version': 'v2.1',
        'prediction_timestamp': features.get('prediction_timestamp'),
        'features_used': len([k for k in features.keys() if not k.startswith('_')])
    }

    return response

def publish_latency_metric(latency_ms: float) -> None:
    """Publica métrica de latencia a CloudWatch"""

    try:
        cloudwatch.put_metric_data(
            Namespace='SSMAS/YieldPredictor/Production',
            MetricData=[
                {
                    'MetricName': 'PredictionLatency',
                    'Value': latency_ms,
                    'Unit': 'Milliseconds'
                }
            ]
        )
    except Exception as e:
        logger.warning(f"Failed to publish latency metric: {str(e)}")

def create_error_response(status_code: int, error_type: str, message: str) -> Dict[str, Any]:
    """Crea respuesta de error estandarizada"""

    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'error': error_type,
            'message': message,
            'timestamp': time.time()
        })
    }
```

### Integración con Plataforma de Subastas

#### Modificación del Ad Server SSMAS

**Pseudo-código de Integración**:
```javascript
// Integración en el Ad Server SSMAS
class YieldOptimizer {
    constructor() {
        this.apiUrl = 'https://api-yield.ssmas.com/v2';
        this.apiKey = process.env.YIELD_PREDICTOR_API_KEY;
        this.fallbackCPM = 2.0; // EUR
        this.timeoutMs = 25; // Timeout muy agresivo para RTB
    }

    async optimizeFloorPrice(impressionData) {
        try {
            // Preparar datos para predicción
            const predictionRequest = {
                impression_id: impressionData.id,
                hour_of_day: new Date().getHours(),
                day_of_week: new Date().getDay() + 1,
                country_tier: this.getCountryTier(impressionData.country),
                device_category: impressionData.device.category,
                format: impressionData.adUnit.format,
                inventory_type: impressionData.adUnit.type,
                bidder: impressionData.bidder,
                domain: impressionData.domain,
                user_engagement_score: impressionData.user.engagementScore,
                device_performance_score: impressionData.device.performanceScore
            };

            // Llamada a API con timeout
            const prediction = await this.callPredictionAPI(predictionRequest);

            // Aplicar lógica de floor price
            const dynamicFloor = this.calculateDynamicFloor(
                prediction.predicted_cpm,
                impressionData
            );

            // Logging para análisis
            this.logFloorPriceDecision(impressionData.id, {
                original_floor: impressionData.floorPrice,
                predicted_cpm: prediction.predicted_cpm,
                dynamic_floor: dynamicFloor,
                model_version: prediction.model_version
            });

            return dynamicFloor;

        } catch (error) {
            console.error('Yield optimization failed:', error);
            // Fallback a floor price estático
            return impressionData.floorPrice || this.fallbackCPM;
        }
    }

    async callPredictionAPI(requestData) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

        try {
            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.apiKey
                },
                body: JSON.stringify(requestData),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`API returned ${response.status}`);
            }

            return await response.json();

        } catch (error) {
            clearTimeout(timeoutId);
            throw error;
        }
    }

    calculateDynamicFloor(predictedCPM, impressionData) {
        // Lógica de negocio para floor price dinámico

        // Factor de confianza basado en historical data
        const confidenceFactor = this.getConfidenceFactor(impressionData);

        // Ajuste por competition level
        const competitionMultiplier = this.getCompetitionMultiplier(impressionData);

        // Floor price ajustado
        let dynamicFloor = predictedCPM * confidenceFactor * competitionMultiplier;

        // Límites mínimos y máximos
        const minFloor = 0.5; // EUR
        const maxFloor = 10.0; // EUR

        dynamicFloor = Math.max(minFloor, Math.min(maxFloor, dynamicFloor));

        return parseFloat(dynamicFloor.toFixed(4));
    }

    getCountryTier(country) {
        const tier1Countries = ['ES', 'FR', 'DE', 'IT', 'UK', 'US'];
        const tier2Countries = ['PT', 'NL', 'BE', 'CH', 'AT', 'SE'];

        if (tier1Countries.includes(country)) return 1;
        if (tier2Countries.includes(country)) return 2;
        return 3;
    }

    getConfidenceFactor(impressionData) {
        // Más datos históricos = más confianza
        const dataPoints = impressionData.historical?.dataPoints || 0;
        if (dataPoints > 1000) return 0.95;
        if (dataPoints > 100) return 0.85;
        if (dataPoints > 10) return 0.75;
        return 0.60;
    }

    getCompetitionMultiplier(impressionData) {
        // Ajuste basado en competencia esperada
        const hour = new Date().getHours();
        const isPrimeTime = hour >= 19 && hour <= 23;
        const isPremiumInventory = impressionData.adUnit.type === 'premium';

        if (isPrimeTime && isPremiumInventory) return 1.1;
        if (isPrimeTime || isPremiumInventory) return 1.05;
        return 1.0;
    }

    logFloorPriceDecision(impressionId, decision) {
        // Log estructurado para análisis posterior
        console.log(JSON.stringify({
            timestamp: new Date().toISOString(),
            impression_id: impressionId,
            event_type: 'floor_price_decision',
            ...decision
        }));
    }
}

// Uso en el auction handler
const yieldOptimizer = new YieldOptimizer();

async function handleAuctionRequest(impressionData) {
    // Optimizar floor price en tiempo real
    const optimizedFloor = await yieldOptimizer.optimizeFloorPrice(impressionData);

    // Aplicar nuevo floor price
    impressionData.floorPrice = optimizedFloor;

    // Continuar con proceso de subasta normal
    return processAuction(impressionData);
}
```

## Monitorización Avanzada y Observabilidad

### Dashboard de Producción

#### CloudWatch Dashboard Completo

**Dashboard Configuration**:
```json
{
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["SSMAS/YieldPredictor/Production", "PredictionLatency", {"stat": "Average"}],
                    [".", ".", {"stat": "p95"}],
                    [".", ".", {"stat": "p99"}]
                ],
                "period": 300,
                "stat": "Average",
                "region": "eu-west-1",
                "title": "Prediction Latency (ms)",
                "yAxis": {
                    "left": {
                        "min": 0,
                        "max": 100
                    }
                },
                "annotations": {
                    "horizontal": [
                        {
                            "value": 50,
                            "label": "SLA Limit",
                            "color": "#d62728"
                        }
                    ]
                }
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["SSMAS/YieldPredictor/Production", "ModelRMSE"],
                    [".", "ModelMAE"],
                    [".", "ModelR2Score"]
                ],
                "period": 3600,
                "stat": "Average",
                "region": "eu-west-1",
                "title": "Model Performance Metrics",
                "yAxis": {
                    "left": {
                        "min": 0
                    }
                }
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "Invocations", "EndpointName", "yield-predictor-multi-model"],
                    [".", "InvocationErrors", ".", "."]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "eu-west-1",
                "title": "SageMaker Endpoint Usage"
            }
        },
        {
            "type": "metric",
            "x": 8,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["SSMAS/YieldPredictor/Drift", "GlobalDriftScore"]
                ],
                "period": 3600,
                "stat": "Average",
                "region": "eu-west-1",
                "title": "Data Drift Score",
                "yAxis": {
                    "left": {
                        "min": 0,
                        "max": 1
                    }
                },
                "annotations": {
                    "horizontal": [
                        {
                            "value": 0.5,
                            "label": "Warning Threshold",
                            "color": "#ff7f0e"
                        },
                        {
                            "value": 0.7,
                            "label": "Critical Threshold",
                            "color": "#d62728"
                        }
                    ]
                }
            }
        },
        {
            "type": "metric",
            "x": 16,
            "y": 6,
            "width": 8,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/ApiGateway", "Count", "ApiName", "ssmas-yield-api-prod"],
                    [".", "4XXError", ".", "."],
                    [".", "5XXError", ".", "."]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "eu-west-1",
                "title": "API Gateway Metrics"
            }
        },
        {
            "type": "log",
            "x": 0,
            "y": 12,
            "width": 24,
            "height": 6,
            "properties": {
                "query": "SOURCE '/aws/lambda/yield-predictor-integration' | fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 100",
                "region": "eu-west-1",
                "title": "Recent Errors"
            }
        }
    ]
}
```

### Alertas y Notificaciones

#### CloudWatch Alarms Configuration

**Lambda Function - Alert Manager**:
```python
import json
import boto3
import os
from typing import Dict, Any

# Clientes AWS
sns = boto3.client('sns')
slack_webhook = os.environ.get('SLACK_WEBHOOK_URL')

def lambda_handler(event, context):
    """
    Maneja alertas de CloudWatch y las envía a múltiples canales
    """

    try:
        # Parsear mensaje de SNS
        message = json.loads(event['Records'][0]['Sns']['Message'])

        # Determinar tipo de alerta
        alert_type = determine_alert_type(message)

        # Formatear mensaje
        formatted_message = format_alert_message(message, alert_type)

        # Enviar notificaciones
        send_notifications(formatted_message, alert_type)

        return {'statusCode': 200}

    except Exception as e:
        print(f"Error processing alert: {str(e)}")
        return {'statusCode': 500}

def determine_alert_type(message: Dict[str, Any]) -> str:
    """Determina el tipo de alerta basado en el mensaje"""

    alarm_name = message.get('AlarmName', '')

    if 'Latency' in alarm_name:
        return 'PERFORMANCE'
    elif 'Error' in alarm_name:
        return 'ERROR'
    elif 'Drift' in alarm_name:
        return 'MODEL_DRIFT'
    elif 'Throughput' in alarm_name:
        return 'CAPACITY'
    else:
        return 'GENERAL'

def format_alert_message(message: Dict[str, Any], alert_type: str) -> Dict[str, Any]:
    """Formatea el mensaje de alerta"""

    severity_emoji = {
        'PERFORMANCE': '🚨',
        'ERROR': '❌',
        'MODEL_DRIFT': '📊',
        'CAPACITY': '⚡',
        'GENERAL': '⚠️'
    }

    formatted = {
        'alert_type': alert_type,
        'severity': message.get('NewStateValue', 'UNKNOWN'),
        'alarm_name': message.get('AlarmName'),
        'alarm_description': message.get('AlarmDescription'),
        'reason': message.get('NewStateReason'),
        'timestamp': message.get('StateChangeTime'),
        'emoji': severity_emoji.get(alert_type, '⚠️'),
        'aws_account': message.get('AWSAccountId'),
        'region': message.get('Region')
    }

    return formatted

def send_notifications(message: Dict[str, Any], alert_type: str):
    """Envía notificaciones a múltiples canales"""

    # Slack notification
    if slack_webhook:
        send_slack_notification(message)

    # Email notification para alertas críticas
    if message['severity'] == 'ALARM':
        send_email_notification(message)

    # PagerDuty para alertas de producción críticas
    if alert_type in ['PERFORMANCE', 'ERROR'] and message['severity'] == 'ALARM':
        send_pagerduty_notification(message)

def send_slack_notification(message: Dict[str, Any]):
    """Envía notificación a Slack"""

    import requests

    color_map = {
        'ALARM': 'danger',
        'OK': 'good',
        'INSUFFICIENT_DATA': 'warning'
    }

    slack_message = {
        "attachments": [
            {
                "color": color_map.get(message['severity'], 'warning'),
                "title": f"{message['emoji']} {message['alarm_name']}",
                "text": message['alarm_description'],
                "fields": [
                    {
                        "title": "Severity",
                        "value": message['severity'],
                        "short": True
                    },
                    {
                        "title": "Type",
                        "value": message['alert_type'],
                        "short": True
                    },
                    {
                        "title": "Reason",
                        "value": message['reason'],
                        "short": False
                    }
                ],
                "footer": f"AWS Account: {message['aws_account']} | Region: {message['region']}",
                "ts": message['timestamp']
            }
        ]
    }

    try:
        response = requests.post(slack_webhook, json=slack_message)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send Slack notification: {str(e)}")

def send_email_notification(message: Dict[str, Any]):
    """Envía notificación por email para alertas críticas"""

    subject = f"🚨 CRITICAL: {message['alarm_name']}"
    body = f"""
    CRITICAL ALERT - SSMAS Yield Predictor

    Alarm: {message['alarm_name']}
    Severity: {message['severity']}
    Type: {message['alert_type']}

    Description: {message['alarm_description']}
    Reason: {message['reason']}

    Time: {message['timestamp']}
    AWS Account: {message['aws_account']}
    Region: {message['region']}

    Please investigate immediately.

    Dashboard: https://console.aws.amazon.com/cloudwatch/home?region=eu-west-1#dashboards:name=SSMAS-Yield-Predictor
    """

    try:
        sns.publish(
            TopicArn=os.environ['CRITICAL_ALERTS_TOPIC'],
            Subject=subject,
            Message=body
        )
    except Exception as e:
        print(f"Failed to send email notification: {str(e)}")

def send_pagerduty_notification(message: Dict[str, Any]):
    """Envía alerta a PagerDuty para incidentes críticos"""

    import requests

    pagerduty_payload = {
        "routing_key": os.environ.get('PAGERDUTY_ROUTING_KEY'),
        "event_action": "trigger",
        "payload": {
            "summary": f"SSMAS Yield Predictor: {message['alarm_name']}",
            "severity": "critical",
            "source": message['aws_account'],
            "component": "yield-predictor",
            "group": "ml-platform",
            "class": message['alert_type'],
            "custom_details": {
                "alarm_description": message['alarm_description'],
                "reason": message['reason'],
                "region": message['region']
            }
        }
    }

    try:
        response = requests.post(
            'https://events.pagerduty.com/v2/enqueue',
            json=pagerduty_payload
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send PagerDuty notification: {str(e)}")
```

#### CloudWatch Alarms Configuration

**Terraform Configuration**:
```hcl
# CloudWatch Alarms for Production Monitoring

# Latency Alarm
resource "aws_cloudwatch_metric_alarm" "prediction_latency_high" {
  alarm_name          = "YieldPredictor-HighLatency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "PredictionLatency"
  namespace           = "SSMAS/YieldPredictor/Production"
  period              = "300"
  statistic           = "Average"
  threshold           = "50"
  alarm_description   = "Prediction latency is above 50ms"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]

  tags = {
    Environment = "production"
    Service     = "yield-predictor"
  }
}

# Error Rate Alarm
resource "aws_cloudwatch_metric_alarm" "prediction_errors_high" {
  alarm_name          = "YieldPredictor-HighErrorRate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "4XXError"
  namespace           = "AWS/ApiGateway"
  period              = "300"
  statistic           = "Sum"
  threshold           = "100"
  alarm_description   = "High error rate in prediction API"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    ApiName = "ssmas-yield-api-prod"
  }
}

# Model Drift Alarm
resource "aws_cloudwatch_metric_alarm" "model_drift_critical" {
  alarm_name          = "YieldPredictor-CriticalDrift"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "GlobalDriftScore"
  namespace           = "SSMAS/YieldPredictor/Drift"
  period              = "3600"
  statistic           = "Average"
  threshold           = "0.7"
  alarm_description   = "Critical data drift detected - model retraining required"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# SageMaker Endpoint Health
resource "aws_cloudwatch_metric_alarm" "sagemaker_endpoint_errors" {
  alarm_name          = "YieldPredictor-EndpointErrors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = "300"
  statistic           = "Average"
  threshold           = "30000"  # 30 seconds
  alarm_description   = "SageMaker endpoint latency is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    EndpointName = "yield-predictor-multi-model"
    VariantName  = "AllTraffic"
  }
}

# SNS Topic for Alerts
resource "aws_sns_topic" "alerts" {
  name = "ssmas-yield-predictor-alerts"
}

resource "aws_sns_topic_subscription" "slack_alerts" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.alert_manager.arn
}
```

## Seguridad y Compliance en Producción

### Cifrado y Gestión de Claves

#### AWS KMS - Key Management Strategy

**KMS Keys Configuration**:
```yaml
# KMS Keys para diferentes componentes
KMSKeys:
  YieldPredictorMasterKey:
    Type: AWS::KMS::Key
    Properties:
      Description: "Master key for SSMAS Yield Predictor encryption"
      KeyPolicy:
        Statement:
          - Sid: Enable IAM root permissions
            Effect: Allow
            Principal:
              AWS: !Sub "arn:aws:iam::${AWS::AccountId}:root"
            Action: "kms:*"
            Resource: "*"
          - Sid: Allow SageMaker service
            Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action:
              - "kms:Decrypt"
              - "kms:GenerateDataKey"
            Resource: "*"
          - Sid: Allow Lambda service
            Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action:
              - "kms:Decrypt"
              - "kms:GenerateDataKey"
            Resource: "*"

  YieldPredictorKeyAlias:
    Type: AWS::KMS::Alias
    Properties:
      AliasName: alias/ssmas-yield-predictor-master
      TargetKeyId: !Ref YieldPredictorMasterKey

  FeatureStoreKey:
    Type: AWS::KMS::Key
    Properties:
      Description: "Encryption key for SageMaker Feature Store"
      KeyPolicy:
        Statement:
          - Sid: Enable IAM root permissions
            Effect: Allow
            Principal:
              AWS: !Sub "arn:aws:iam::${AWS::AccountId}:root"
            Action: "kms:*"
            Resource: "*"
          - Sid: Allow SageMaker Feature Store
            Effect: Allow
            Principal:
              Service: sagemaker.amazonaws.com
            Action:
              - "kms:Decrypt"
              - "kms:GenerateDataKey"
              - "kms:CreateGrant"
            Resource: "*"

  DataLakeKey:
    Type: AWS::KMS::Key
    Properties:
      Description: "Encryption key for S3 Data Lake"
      KeyPolicy:
        Statement:
          - Sid: Enable IAM root permissions
            Effect: Allow
            Principal:
              AWS: !Sub "arn:aws:iam::${AWS::AccountId}:root"
            Action: "kms:*"
            Resource: "*"
          - Sid: Allow S3 service
            Effect: Allow
            Principal:
              Service: s3.amazonaws.com
            Action:
              - "kms:Decrypt"
              - "kms:GenerateDataKey"
            Resource: "*"
          - Sid: Allow Glue service
            Effect: Allow
            Principal:
              Service: glue.amazonaws.com
            Action:
              - "kms:Decrypt"
              - "kms:GenerateDataKey"
            Resource: "*"
```

### Gestión de Secretos

#### AWS Secrets Manager Integration

**Lambda Function - Secret Retrieval**:
```python
import boto3
import json
import os
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class SecretManager:
    """Gestión centralizada de secretos para Yield Predictor"""

    def __init__(self, region_name='eu-west-1'):
        self.secrets_client = boto3.client('secretsmanager', region_name=region_name)
        self.cache = {}  # Cache en memoria para secretos

    def get_secret(self, secret_name: str) -> dict:
        """Obtiene un secreto de AWS Secrets Manager con cache"""

        # Verificar cache primero
        if secret_name in self.cache:
            return self.cache[secret_name]

        try:
            response = self.secrets_client.get_secret_value(SecretId=secret_name)
            secret_value = json.loads(response['SecretString'])

            # Guardar en cache
            self.cache[secret_name] = secret_value

            logger.info(f"Successfully retrieved secret: {secret_name}")
            return secret_value

        except ClientError as e:
            logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in secret {secret_name}")
            raise

    def get_api_credentials(self) -> dict:
        """Obtiene credenciales de API para servicios externos"""
        return self.get_secret('ssmas/yield-predictor/api-credentials')

    def get_database_credentials(self) -> dict:
        """Obtiene credenciales de base de datos"""
        return self.get_secret('ssmas/yield-predictor/database-credentials')

    def get_slack_webhook(self) -> str:
        """Obtiene URL del webhook de Slack"""
        credentials = self.get_secret('ssmas/yield-predictor/slack-credentials')
        return credentials.get('webhook_url')

# Uso en Lambda functions
secret_manager = SecretManager()

def lambda_handler(event, context):
    """Handler que utiliza secretos de manera segura"""

    try:
        # Obtener credenciales necesarias
        api_creds = secret_manager.get_api_credentials()

        # Usar credenciales sin exponerlas en logs
        api_key = api_creds.get('external_api_key')

        # Procesar request...

        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'success'})
        }

    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal server error'})
        }
```

### Auditoría y Compliance

#### CloudTrail Configuration

**CloudTrail Setup for ML Platform**:
```json
{
  "CloudWatchLogsLogGroupArn": "arn:aws:logs:eu-west-1:123456789012:log-group:ssmas-yield-predictor-audit:*",
  "CloudWatchLogsRoleArn": "arn:aws:iam::123456789012:role/CloudTrail-CloudWatchLogs-Role",
  "IncludeGlobalServiceEvents": true,
  "IsLogging": true,
  "IsMultiRegionTrail": true,
  "S3BucketName": "ssmas-yield-predictor-audit-logs",
  "S3KeyPrefix": "cloudtrail-logs/",
  "TrailName": "ssmas-yield-predictor-audit",
  "EventSelectors": [
    {
      "ReadWriteType": "All",
      "IncludeManagementEvents": true,
      "DataResources": [
        {
          "Type": "AWS::S3::Object",
          "Values": [
            "arn:aws:s3:::ssmas-yield-prod-datalake-eu-west-1/*"
          ]
        },
        {
          "Type": "AWS::SageMaker::Model",
          "Values": ["*"]
        },
        {
          "Type": "AWS::SageMaker::EndpointConfig",
          "Values": ["*"]
        },
        {
          "Type": "AWS::SageMaker::Endpoint",
          "Values": ["*"]
        }
      ]
    }
  ]
}
```

## Estimación de Costos de Producción

### Desglose Detallado de Costos

#### Servicios de Compute y ML
```yaml
# Costos mensuales estimados para producción
CostBreakdown:
  SageMaker:
    Training:
      # Retraining semanal
      InstanceType: ml.m5.2xlarge
      HoursPerWeek: 4
      CostPerHour: 0.461
      MonthlyCost: 8.00

    Endpoints:
      Production:
        InstanceType: ml.c5.xlarge
        InstanceCount: 3
        HoursPerMonth: 2160
        CostPerHour: 0.192
        MonthlyCost: 1244.16

      A/B_Testing:
        InstanceType: ml.c5.large
        InstanceCount: 1
        HoursPerMonth: 2160
        CostPerHour: 0.096
        MonthlyCost: 207.36

    FeatureStore:
      OnlineStore: 50.00
      OfflineStore: 25.00

    Total: 1334.52

  DataServices:
    S3:
      Storage: 500GB
      StandardStorage: 11.50
      Requests: 20.00
      Transfer: 15.00
      Total: 46.50

    Kinesis:
      DataStreams:
        Shards: 100
        HoursPerMonth: 744
        CostPerShardHour: 0.015
        MonthlyCost: 1116.00

      DataFirehose:
        DataVolume: 150GB
        CostPer1000Records: 0.029
        MonthlyCost: 87.00

      DataAnalytics:
        KPUs: 4
        HoursPerMonth: 744
        CostPerKPUHour: 0.11
        MonthlyCost: 327.36

      Total: 1530.36

    Glue:
      ETLJobs:
        DPUHours: 200
        CostPerDPUHour: 0.44
        MonthlyCost: 88.00

      DataCatalog:
        Requests: 10M
        CostPer1MRequests: 1.00
        MonthlyCost: 10.00

      Total: 98.00

  APIAndLambda:
    ApiGateway:
      Requests: 100M
      CostPer1MRequests: 3.50
      MonthlyCost: 350.00

    Lambda:
      Invocations: 100M
      Duration: 200ms
      Memory: 1GB
      MonthlyCost: 166.67

    Total: 516.67

  MonitoringAndSecurity:
    CloudWatch:
      CustomMetrics: 100
      LogsIngestion: 50GB
      DashboardHours: 744
      MonthlyCost: 85.00

    KMS:
      KeyUsage: 1M
      CostPer10KRequests: 0.03
      MonthlyCost: 3.00

    SecretsManager:
      Secrets: 10
      CostPerSecret: 0.40
      MonthlyCost: 4.00

    CloudTrail:
      DataEvents: 10M
      CostPer100KEvents: 0.10
      MonthlyCost: 10.00

    Total: 102.00

  NetworkingAndSecurity:
    VPCEndpoints:
      Endpoints: 5
      HoursPerMonth: 3720
      CostPerEndpointHour: 0.01
      MonthlyCost: 37.20

    NATGateways:
      Gateways: 2
      HoursPerMonth: 1488
      CostPerGatewayHour: 0.045
      DataProcessing: 500GB
      CostPerGB: 0.045
      MonthlyCost: 89.46

    ApplicationLoadBalancer:
      LoadBalancers: 2
      HoursPerMonth: 1488
      CostPerHour: 0.0225
      LCUHours: 2976
      CostPerLCUHour: 0.008
      MonthlyCost: 57.31

    Total: 183.97

# Total mensual estimado
TotalMonthlyCost: 3811.52  # EUR
```

### Optimizaciones de Costo

#### Reserved Instances y Savings Plans
```yaml
CostOptimizations:
  SageMakerSavingsPlans:
    CurrentCost: 1334.52
    SavingsPercentage: 64%
    OptimizedCost: 480.43
    AnnualSavings: 10249.08

  EC2ReservedInstances:
    # Para instancias subyacentes de servicios
    CurrentCost: 183.97
    SavingsPercentage: 40%
    OptimizedCost: 110.38
    AnnualSavings: 883.08

  S3IntelligentTiering:
    CurrentCost: 46.50
    SavingsPercentage: 30%
    OptimizedCost: 32.55
    AnnualSavings: 167.40

  SpotInstancesForTraining:
    CurrentCost: 8.00
    SavingsPercentage: 60%
    OptimizedCost: 3.20
    AnnualSavings: 57.60

# Costo optimizado total
OptimizedMonthlyCost: 2157.13  # EUR
TotalAnnualSavings: 19835.16   # EUR
```

## Cronograma de la Fase II

### Planning Detallado (8 semanas)

#### Semanas 1-2: Infrastructure Scaling
**Semana 1: Multi-Account Setup**
- Configuración de cuentas de producción
- Setup de Transit Gateway y networking
- Implementación de KMS keys y encryption
- Configuración de IAM roles y políticas

**Semana 2: Data Platform Scaling**
- Configuración de Kinesis Data Streams (100 shards)
- Setup de S3 Data Lake con particionado optimizado
- Implementación de Glue ETL jobs escalables
- Testing de ingesta de alta volumetría

#### Semanas 3-4: ML Platform Production
**Semana 3: SageMaker Production Setup**
- Configuración de Feature Store
- Setup de Model Registry y versioning
- Implementación de multi-model endpoints
- Configuración de auto-scaling

**Semana 4: MLOps Pipeline Implementation**
- Desarrollo de CI/CD pipeline con CodePipeline
- Implementación de automated retraining
- Setup de model drift detection
- Testing de deployment automático

#### Semanas 5-6: Integration and API
**Semana 5: API Gateway Production**
- Configuración de API Gateway con caching
- Implementación de rate limiting y throttling
- Setup de Lambda functions optimizadas
- Testing de latencia y throughput

**Semana 6: SSMAS Integration**
- Integración con sistema de subastas existente
- Implementación de fallback mechanisms
- Testing de integración end-to-end
- Validation de dynamic floor pricing

#### Semanas 7-8: Monitoring and Go-Live
**Semana 7: Monitoring and Alerting**
- Configuración de CloudWatch dashboards
- Setup de alertas y notificaciones
- Implementación de logging estructurado
- Testing de incident response

**Semana 8: Production Deployment**
- Deployment gradual con blue-green strategy
- Testing de carga en producción
- Monitoring intensivo y ajustes finales
- Handover y documentación

## Criterios de Éxito de la Fase II

### Métricas Técnicas
1. **Performance en Producción**:
   - Latencia P95 < 50ms para predicciones individuales
   - Throughput > 10,000 predicciones/segundo
   - Uptime > 99.9% medido por mes
   - RMSE del modelo < 0.45 EUR en datos de producción

2. **Escalabilidad Demostrada**:
   - Procesamiento exitoso de 50M+ registros diarios
   - Auto-scaling funcional bajo carga variable
   - Retraining automático sin impacto en servicio
   - Drift detection funcionando correctamente

3. **Integración Completa**:
   - API integrada con sistema SSMAS sin errores
   - Dynamic floor pricing operativo
   - Fallback mechanisms validados
   - Logging y monitoring completamente funcional

### Métricas de Negocio
1. **Impacto en Revenue**:
   - Incremento medible en CPM promedio
   - Mantenimiento o mejora del fill rate
   - ROI positivo del sistema vs. costos de infraestructura
   - Validación A/B testing vs. floor prices estáticos

2. **Operacional**:
   - Reducción en tiempo de respuesta a incidentes
   - Automatización completa de retraining
   - Dashboards operacionales utilizados diariamente
   - Compliance con auditoría de seguridad

## Conclusión de la Fase II

La Fase II del proyecto Yield Predictivo representa la materialización completa de la visión estratégica de SSMAS para optimización publicitaria predictiva. A través de la implementación de una plataforma de machine learning de clase empresarial, SSMAS estará posicionada para procesar sus 50 millones de registros diarios con predicciones de CPM en tiempo real, estableciendo un nuevo estándar en el mercado AdTech español.

La arquitectura implementada no solo cumple con los requisitos actuales de escalabilidad y performance, sino que proporciona la base tecnológica para futuras innovaciones en optimización publicitaria. El framework de MLOps automatizado asegura que el modelo mantenga su precisión y relevancia a medida que evoluciona el comportamiento del mercado publicitario.

Los beneficios empresariales de esta fase incluyen no solo la optimización directa de ingresos a través de dynamic floor pricing, sino también la construcción de capacidades internas que posicionan a SSMAS como líder tecnológico en el sector, atrayendo nuevos clientes y reteniendo los existentes mediante resultados superiores y una propuesta de valor diferenciada.