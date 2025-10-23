# Arquitectura Técnica AWS - Agente de AdOps Autónomo

## 1. Introducción

### 1.1 Propósito del Documento
Este documento describe la arquitectura técnica detallada en AWS para implementar el sistema de Agente de AdOps Autónomo de SSMAS. Define los componentes de infraestructura, servicios gestionados, integraciones y flujos de datos necesarios para construir una solución robusta, escalable y segura en la nube de AWS.

### 1.2 Principios de Diseño Técnico
- **Serverless-First**: Maximizar el uso de servicios gestionados para minimizar overhead operativo
- **Event-Driven Architecture**: Arquitectura basada en eventos para máxima flexibilidad y escalabilidad
- **Microservicios**: Componentes desacoplados y especializados
- **Security by Design**: Seguridad integrada en cada capa
- **High Availability**: Diseño multi-AZ para resiliencia
- **Cost Optimization**: Uso eficiente de recursos con auto-scaling

### 1.3 AWS Well-Architected Framework
La arquitectura se alinea con los cinco pilares del AWS Well-Architected Framework:
- **Excelencia Operacional**: Automatización y observabilidad completa
- **Seguridad**: Defense in depth con múltiples capas de protección
- **Fiabilidad**: Diseño tolerante a fallos con recuperación automática
- **Eficiencia del Rendimiento**: Uso óptimo de recursos computacionales
- **Optimización de Costos**: Pay-as-you-go con reserved capacity donde aplique

## 2. Landing Zone y Estructura Organizacional

### 2.1 AWS Control Tower Landing Zone

#### 2.1.1 Configuración Base
```
Root Organization (SSMAS)
├── Security OU
│   ├── Log Archive Account
│   └── Audit Account
├── Production OU
│   ├── AdOps Production Account
│   └── Data Production Account
├── Non-Production OU
│   ├── Development Account
│   ├── Staging Account
│   └── Testing Account
├── Shared Services OU
│   ├── Network Account
│   └── Shared Tools Account
└── Sandbox OU
    └── Innovation Sandbox Account
```

#### 2.1.2 Cuentas y Responsabilidades

**Security OU**:
- **Log Archive Account**: Almacenamiento centralizado de logs (CloudTrail, Config, VPC Flow Logs)
- **Audit Account**: Acceso read-only cross-account para auditoría y compliance

**Production OU**:
- **AdOps Production Account**: Ambiente productivo del agente y servicios relacionados
- **Data Production Account**: Data Lake y servicios de analytics

**Non-Production OU**:
- **Development Account**: Desarrollo y pruebas unitarias
- **Staging Account**: Pruebas de integración y UAT
- **Testing Account**: Pruebas de carga y seguridad

**Shared Services OU**:
- **Network Account**: Transit Gateway, Direct Connect, VPN
- **Shared Tools Account**: CI/CD, herramientas compartidas

### 2.2 Guardrails y Políticas

#### 2.2.1 Preventive Controls (SCPs)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": [
        "ec2:TerminateInstances",
        "rds:DeleteDBInstance"
      ],
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "aws:PrincipalOrgID": "${aws:PrincipalOrgID}"
        }
      }
    }
  ]
}
```

#### 2.2.2 Detective Controls
- AWS Config Rules para compliance continuo
- Amazon GuardDuty para detección de amenazas
- AWS Security Hub para visibilidad centralizada
- Amazon Macie para protección de datos sensibles

## 3. Arquitectura de Red

### 3.1 Diseño de VPC Multi-Capa

#### 3.1.1 Estructura de VPC
```
Production VPC (10.0.0.0/16)
├── Public Subnets
│   ├── AZ-1: 10.0.1.0/24 (NAT Gateway, ALB)
│   └── AZ-2: 10.0.2.0/24 (NAT Gateway, ALB)
├── Private Subnets - App Tier
│   ├── AZ-1: 10.0.10.0/24 (Lambda, ECS)
│   └── AZ-2: 10.0.11.0/24 (Lambda, ECS)
├── Private Subnets - Data Tier
│   ├── AZ-1: 10.0.20.0/24 (RDS, ElastiCache)
│   └── AZ-2: 10.0.21.0/24 (RDS, ElastiCache)
└── Private Subnets - Endpoints
    ├── AZ-1: 10.0.30.0/24 (VPC Endpoints)
    └── AZ-2: 10.0.31.0/24 (VPC Endpoints)
```

#### 3.1.2 Componentes de Red
- **Internet Gateway**: Salida a internet para subnets públicas
- **NAT Gateways**: Salida a internet para subnets privadas (HA en múltiples AZs)
- **VPC Endpoints**: Acceso privado a servicios AWS (S3, DynamoDB, Bedrock, etc.)
- **Transit Gateway**: Conectividad entre VPCs y on-premises
- **AWS PrivateLink**: Conexiones privadas con servicios de partners

### 3.2 Conectividad Híbrida

#### 3.2.1 AWS Direct Connect
- **Virtual Interfaces (VIFs)**:
  - Production VIF: 1 Gbps dedicado
  - Backup VIF: 500 Mbps redundancia
- **BGP Configuration**: AS paths optimizados para routing
- **Connection Redundancy**: Dual connections en diferentes ubicaciones

#### 3.2.2 Site-to-Site VPN
- **Backup Path**: Respaldo para Direct Connect
- **IPSec Tunnels**: Encriptación AES-256
- **Route Priority**: Menor prioridad que Direct Connect

### 3.3 Seguridad de Red

#### 3.3.1 Security Groups
```yaml
# Bedrock Agent Security Group
bedrock-agent-sg:
  ingress:
    - protocol: https
      port: 443
      source: private-subnet-cidr
  egress:
    - protocol: all
      destination: 0.0.0.0/0

# Lambda Security Group
lambda-sg:
  ingress: []  # No ingress needed
  egress:
    - protocol: https
      port: 443
      destination: vpc-endpoint-sg
    - protocol: tcp
      port: 5432
      destination: rds-sg
```

#### 3.3.2 Network ACLs
- **Stateless rules** para defensa adicional
- **Deny lists** para IPs maliciosas conocidas
- **Rate limiting** en capa de red

## 4. Arquitectura de Servicios Core

### 4.1 Capa de Monitorización y Detección

#### 4.1.1 Amazon CloudWatch
```yaml
CloudWatch:
  Metrics:
    Custom_Metrics:
      - Namespace: SSMAS/AdOps
        Metrics:
          - RPM_by_Publisher
          - Fill_Rate
          - Bid_Latency
          - Error_Rate
    Alarms:
      - Name: LowRPMAlarm
        MetricName: RPM_by_Publisher
        Statistic: Average
        Period: 300
        EvaluationPeriods: 2
        Threshold: 0.8  # 20% drop
        ComparisonOperator: LessThanThreshold
        Actions:
          - SNS_Topic: agent-trigger-topic

  Logs:
    LogGroups:
      - /aws/lambda/diagnostic-tools
      - /aws/bedrock/agent-logs
      - /aws/applicationinsights/ssmas

  Insights:
    Queries:
      - ErrorPatternDetection
      - LatencyAnalysis
      - TrafficAnomalies
```

#### 4.1.2 Amazon EventBridge
```yaml
EventBridge:
  Rules:
    - Name: MetricAnomalyRule
      EventPattern:
        source: aws.cloudwatch
        detail-type: CloudWatch Alarm State Change
        detail:
          state:
            value: ALARM
      Targets:
        - Arn: !GetAtt BedrockAgentInvoker.Arn
          RetryPolicy:
            MaximumRetryAttempts: 3
            MaximumEventAge: 3600
```

#### 4.1.3 AWS X-Ray
- **Distributed Tracing**: End-to-end request tracking
- **Service Map**: Visualización de dependencias
- **Performance Analysis**: Identificación de bottlenecks

### 4.2 Capa de Orquestación con Amazon Bedrock

#### 4.2.1 Amazon Bedrock Agent Configuration
```python
# Agent Configuration
agent_config = {
    "agentName": "AdOpsAutonomousAgent",
    "agentResourceRoleArn": "arn:aws:iam::xxx:role/BedrockAgentRole",
    "foundationModel": "anthropic.claude-3-sonnet-20240229-v1:0",
    "instruction": """
    You are an expert AdOps engineer responsible for monitoring and maintaining
    the advertising platform. When an anomaly is detected:
    1. Analyze the symptoms and context
    2. Identify the root cause
    3. Execute appropriate diagnostic tools
    4. Implement remediation actions
    5. Validate the resolution
    6. Document the incident
    """,
    "idleSessionTTLInSeconds": 600,
    "agentVersion": "DRAFT"
}

# Action Groups
action_groups = [
    {
        "actionGroupName": "DiagnosticTools",
        "actionGroupExecutor": {
            "lambda": "arn:aws:lambda:region:account:function:diagnostic-executor"
        },
        "apiSchema": {
            "s3": {
                "s3BucketName": "ssmas-agent-schemas",
                "s3ObjectKey": "diagnostic-api-schema.yaml"
            }
        }
    },
    {
        "actionGroupName": "RemediationActions",
        "actionGroupExecutor": {
            "lambda": "arn:aws:lambda:region:account:function:remediation-executor"
        },
        "apiSchema": {
            "s3": {
                "s3BucketName": "ssmas-agent-schemas",
                "s3ObjectKey": "remediation-api-schema.yaml"
            }
        }
    }
]

# Knowledge Base Association
knowledge_base = {
    "knowledgeBaseId": "kb-ssmas-runbooks",
    "description": "AdOps runbooks and troubleshooting guides",
    "knowledgeBaseState": "ENABLED"
}
```

#### 4.2.2 Knowledge Base con Amazon Bedrock
```yaml
KnowledgeBase:
  DataSource:
    Type: S3
    Configuration:
      BucketArn: arn:aws:s3:::ssmas-knowledge-base
      InclusionPrefixes:
        - runbooks/
        - troubleshooting/
        - best-practices/

  VectorDatabase:
    Type: Amazon OpenSearch Serverless
    Configuration:
      CollectionArn: arn:aws:aoss:region:account:collection/ssmas-vectors
      VectorIndexName: adops-knowledge
      FieldMapping:
        vectorField: embedding
        textField: content
        metadataField: metadata

  EmbeddingModel:
    ModelId: amazon.titan-embed-text-v1
    Dimensions: 1536
```

### 4.3 Capa de Diagnóstico y Herramientas

#### 4.3.1 AWS Lambda Functions
```python
# Diagnostic Lambda Function Example
import json
import boto3
import os

def verify_publisher_config_handler(event, context):
    """
    Verifica la configuración de un publisher específico
    """
    publisher_id = event['publisher_id']

    # Clients
    dynamodb = boto3.resource('dynamodb')
    ssm = boto3.client('ssm')

    # Get publisher configuration
    table = dynamodb.Table(os.environ['PUBLISHER_TABLE'])
    response = table.get_item(Key={'publisher_id': publisher_id})

    if 'Item' not in response:
        return {
            'statusCode': 404,
            'diagnostic': 'Publisher not found',
            'severity': 'HIGH',
            'recommended_action': 'verify_publisher_exists'
        }

    config = response['Item']

    # Validate configuration
    issues = []
    if not config.get('ad_units'):
        issues.append('Missing ad units configuration')
    if config.get('status') != 'active':
        issues.append(f"Publisher status is {config.get('status')}")
    if config.get('rpm_threshold', 0) < 0.5:
        issues.append('RPM threshold below minimum')

    return {
        'statusCode': 200,
        'diagnostic': 'Configuration analysis complete',
        'issues': issues,
        'severity': 'HIGH' if issues else 'NONE',
        'recommended_action': 'fix_configuration' if issues else 'none',
        'details': config
    }

# Lambda Layer for Common Utilities
layer_config = {
    "LayerName": "adops-common-utils",
    "Description": "Common utilities for AdOps Lambda functions",
    "Content": {
        "S3Bucket": "ssmas-lambda-layers",
        "S3Key": "adops-utils-layer.zip"
    },
    "CompatibleRuntimes": ["python3.11"],
    "LicenseInfo": "MIT"
}
```

#### 4.3.2 AWS Step Functions para Orquestación Compleja
```json
{
  "Comment": "AdOps Incident Resolution Workflow",
  "StartAt": "DetectAnomaly",
  "States": {
    "DetectAnomaly": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:region:account:function:detect-anomaly",
      "Next": "ClassifyIncident"
    },
    "ClassifyIncident": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.severity",
          "StringEquals": "CRITICAL",
          "Next": "ImmediateRemediation"
        },
        {
          "Variable": "$.severity",
          "StringEquals": "HIGH",
          "Next": "DiagnosticAnalysis"
        }
      ],
      "Default": "MonitorOnly"
    },
    "DiagnosticAnalysis": {
      "Type": "Parallel",
      "Branches": [
        {
          "StartAt": "CheckConfiguration",
          "States": {
            "CheckConfiguration": {
              "Type": "Task",
              "Resource": "arn:aws:lambda:region:account:function:check-config",
              "End": true
            }
          }
        },
        {
          "StartAt": "AnalyzeLogs",
          "States": {
            "AnalyzeLogs": {
              "Type": "Task",
              "Resource": "arn:aws:lambda:region:account:function:analyze-logs",
              "End": true
            }
          }
        }
      ],
      "Next": "DetermineAction"
    },
    "DetermineAction": {
      "Type": "Task",
      "Resource": "arn:aws:states:::bedrock:invokeModel",
      "Parameters": {
        "ModelId": "anthropic.claude-3-sonnet",
        "Body": {
          "prompt": "Based on diagnostics, determine remediation action",
          "context.$": "$.diagnostics"
        }
      },
      "Next": "ExecuteRemediation"
    },
    "ImmediateRemediation": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:region:account:function:emergency-action",
      "Next": "NotifyTeam"
    },
    "ExecuteRemediation": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:region:account:function:execute-remediation",
      "Retry": [
        {
          "ErrorEquals": ["States.TaskFailed"],
          "IntervalSeconds": 2,
          "MaxAttempts": 3,
          "BackoffRate": 2.0
        }
      ],
      "Next": "ValidateResolution"
    },
    "ValidateResolution": {
      "Type": "Wait",
      "Seconds": 60,
      "Next": "CheckMetrics"
    },
    "CheckMetrics": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:region:account:function:validate-metrics",
      "Next": "ResolutionSuccess"
    },
    "ResolutionSuccess": {
      "Type": "Succeed"
    },
    "MonitorOnly": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:region:account:function:log-event",
      "End": true
    },
    "NotifyTeam": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Parameters": {
        "TopicArn": "arn:aws:sns:region:account:adops-alerts",
        "Message.$": "$.notification"
      },
      "End": true
    }
  }
}
```

### 4.4 Capa de Datos y Almacenamiento

#### 4.4.1 Amazon DynamoDB para Estado y Configuración
```yaml
DynamoDB:
  Tables:
    - TableName: publisher-configurations
      PartitionKey: publisher_id (String)
      SortKey: timestamp (Number)
      GlobalSecondaryIndexes:
        - IndexName: status-index
          PartitionKey: status
          SortKey: last_updated
      StreamSpecification:
        StreamEnabled: true
        StreamViewType: NEW_AND_OLD_IMAGES
      PointInTimeRecovery: true

    - TableName: incident-history
      PartitionKey: incident_id (String)
      SortKey: timestamp (Number)
      TimeToLive:
        AttributeName: ttl
        Enabled: true

    - TableName: agent-decisions
      PartitionKey: decision_id (String)
      Attributes:
        - context
        - action_taken
        - outcome
        - feedback
```

#### 4.4.2 Amazon S3 para Data Lake
```yaml
S3:
  Buckets:
    - Name: ssmas-adops-data-lake
      Structure:
        - /raw/
          - /cloudwatch-metrics/
          - /application-logs/
          - /partner-data/
        - /processed/
          - /aggregated-metrics/
          - /anomaly-reports/
        - /curated/
          - /dashboards/
          - /ml-training-data/

      Lifecycle:
        - Id: archive-old-data
          Status: Enabled
          Transitions:
            - Days: 30
              StorageClass: INTELLIGENT_TIERING
            - Days: 90
              StorageClass: GLACIER_IR
            - Days: 365
              StorageClass: DEEP_ARCHIVE

      Replication:
        Role: arn:aws:iam::account:role/s3-replication-role
        Rules:
          - Status: Enabled
            Priority: 1
            Destination:
              Bucket: arn:aws:s3:::ssmas-adops-dr-bucket
              StorageClass: STANDARD_IA
```

#### 4.4.3 Amazon RDS para Datos Transaccionales
```yaml
RDS:
  Engine: PostgreSQL
  Version: 15.4
  MultiAZ: true
  InstanceClass: db.r6g.xlarge
  Storage:
    Type: gp3
    Size: 500GB
    IOPS: 12000
    Throughput: 500

  BackupRetention: 30 days
  BackupWindow: "03:00-04:00"
  MaintenanceWindow: "sun:04:00-sun:05:00"

  PerformanceInsights:
    Enabled: true
    RetentionPeriod: 731  # 2 years

  Security:
    Encryption: true
    KmsKeyId: arn:aws:kms:region:account:key/xxx
    IAMAuthentication: true

  ReadReplicas:
    - Region: us-east-1
      InstanceClass: db.r6g.large
    - Region: eu-west-1
      InstanceClass: db.r6g.large
```

### 4.5 Capa de Procesamiento y Analytics

#### 4.5.1 Amazon Kinesis para Streaming
```yaml
Kinesis:
  DataStreams:
    - Name: adops-event-stream
      ShardCount: 10
      RetentionPeriod: 168  # 7 days
      ShardLevelMetrics:
        - IncomingRecords
        - OutgoingRecords
      Encryption:
        Type: KMS
        KeyId: arn:aws:kms:region:account:key/xxx

  DataFirehose:
    - Name: metrics-to-s3
      Source: adops-event-stream
      Destination:
        Type: ExtendedS3
        Configuration:
          BucketARN: arn:aws:s3:::ssmas-adops-data-lake
          Prefix: raw/streaming/
          ErrorOutputPrefix: error/
          BufferingHints:
            IntervalInSeconds: 60
            SizeInMBs: 128
          CompressionFormat: GZIP
          DataFormatConversion:
            Enabled: true
            OutputFormatConfiguration:
              Serializer:
                ParquetSerDe: {}

  Analytics:
    ApplicationName: adops-real-time-analytics
    Inputs:
      - NamePrefix: SOURCE_SQL_STREAM
        KinesisStreamsInput:
          ResourceARN: arn:aws:kinesis:region:account:stream/adops-event-stream
    Outputs:
      - Name: DESTINATION_SQL_STREAM
        LambdaOutput:
          ResourceARN: arn:aws:lambda:region:account:function:process-anomalies
```

#### 4.5.2 Amazon Athena para Queries Ad-Hoc
```sql
-- Example Athena Table Definition
CREATE EXTERNAL TABLE IF NOT EXISTS adops_metrics (
  timestamp bigint,
  publisher_id string,
  metric_name string,
  metric_value double,
  dimensions map<string, string>,
  year int,
  month int,
  day int
)
STORED AS PARQUET
LOCATION 's3://ssmas-adops-data-lake/processed/aggregated-metrics/'
PARTITIONED BY (year int, month int, day int)
TBLPROPERTIES ('has_encrypted_data'='true');

-- Query Example for Anomaly Detection
WITH baseline AS (
  SELECT
    publisher_id,
    AVG(metric_value) as avg_rpm,
    STDDEV(metric_value) as std_rpm
  FROM adops_metrics
  WHERE metric_name = 'RPM'
    AND timestamp > unix_timestamp(current_timestamp - interval '7' day)
  GROUP BY publisher_id
),
current_metrics AS (
  SELECT
    publisher_id,
    metric_value as current_rpm
  FROM adops_metrics
  WHERE metric_name = 'RPM'
    AND timestamp > unix_timestamp(current_timestamp - interval '1' hour)
)
SELECT
  c.publisher_id,
  c.current_rpm,
  b.avg_rpm,
  (c.current_rpm - b.avg_rpm) / b.std_rpm as z_score
FROM current_metrics c
JOIN baseline b ON c.publisher_id = b.publisher_id
WHERE ABS((c.current_rpm - b.avg_rpm) / b.std_rpm) > 3;
```

## 5. Seguridad y Compliance

### 5.1 Identity and Access Management (IAM)

#### 5.1.1 Roles y Políticas
```json
{
  "BedrockAgentExecutionRole": {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "Service": "bedrock.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
      }
    ],
    "Policies": [
      {
        "PolicyName": "BedrockAgentPolicy",
        "Statement": [
          {
            "Effect": "Allow",
            "Action": [
              "bedrock:InvokeModel",
              "bedrock:RetrieveAndGenerate"
            ],
            "Resource": "*"
          },
          {
            "Effect": "Allow",
            "Action": [
              "lambda:InvokeFunction"
            ],
            "Resource": "arn:aws:lambda:*:*:function:adops-*"
          },
          {
            "Effect": "Allow",
            "Action": [
              "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::ssmas-knowledge-base/*"
          },
          {
            "Effect": "Allow",
            "Action": [
              "aoss:APIAccessAll"
            ],
            "Resource": "arn:aws:aoss:*:*:collection/ssmas-vectors"
          }
        ]
      }
    ]
  }
}
```

#### 5.1.2 AWS IAM Identity Center (SSO)
```yaml
IdentityCenter:
  IdentitySource:
    Type: External
    Provider: AzureAD

  PermissionSets:
    - Name: AdOpsEngineer
      SessionDuration: PT4H
      ManagedPolicies:
        - arn:aws:iam::aws:policy/ReadOnlyAccess
      InlinePolicy:
        Statement:
          - Effect: Allow
            Action:
              - bedrock:InvokeAgent
              - lambda:InvokeFunction
            Resource: "*"

    - Name: AdOpsAdmin
      SessionDuration: PT2H
      ManagedPolicies:
        - arn:aws:iam::aws:policy/PowerUserAccess

  AccountAssignments:
    - PrincipalType: GROUP
      PrincipalId: AzureAD-AdOpsTeam
      PermissionSet: AdOpsEngineer
      TargetAccounts:
        - Production
        - Staging
```

### 5.2 Encryption and Key Management

#### 5.2.1 AWS KMS Configuration
```yaml
KMS:
  CustomerMasterKeys:
    - Alias: alias/ssmas-adops-cmk
      KeyPolicy:
        Version: '2012-10-17'
        Statement:
          - Sid: Enable IAM User Permissions
            Effect: Allow
            Principal:
              AWS: arn:aws:iam::account:root
            Action: kms:*
            Resource: '*'
          - Sid: Allow services to use the key
            Effect: Allow
            Principal:
              Service:
                - s3.amazonaws.com
                - dynamodb.amazonaws.com
                - rds.amazonaws.com
            Action:
              - kms:Decrypt
              - kms:GenerateDataKey
            Resource: '*'

      KeyRotation: true
      MultiRegion: true

  DataKeys:
    GenerateDataKeyWithoutPlaintext: true
    GrantTokens: []
```

### 5.3 Compliance y Auditoría

#### 5.3.1 AWS CloudTrail
```yaml
CloudTrail:
  Trail:
    Name: ssmas-adops-audit-trail
    S3BucketName: ssmas-audit-logs
    IncludeGlobalServiceEvents: true
    IsMultiRegionTrail: true
    EnableLogFileValidation: true

    EventSelectors:
      - ReadWriteType: All
        IncludeManagementEvents: true
        DataResources:
          - Type: AWS::S3::Object
            Values:
              - arn:aws:s3:::ssmas-*/
          - Type: AWS::Bedrock::Agent
            Values:
              - arn:aws:bedrock:*:*:agent/*

    InsightSelectors:
      - InsightType: ApiCallRateInsight
```

#### 5.3.2 AWS Config Rules
```yaml
Config:
  Rules:
    - Name: s3-bucket-encryption
      Source:
        Owner: AWS
        SourceIdentifier: S3_BUCKET_SERVER_SIDE_ENCRYPTION_ENABLED

    - Name: rds-encryption-enabled
      Source:
        Owner: AWS
        SourceIdentifier: RDS_STORAGE_ENCRYPTED

    - Name: lambda-dlq-check
      Source:
        Owner: AWS
        SourceIdentifier: LAMBDA_DLQ_CHECK

    - Name: bedrock-agent-logging
      Source:
        Owner: CUSTOM
        SourceDetails:
          - MessageType: ConfigurationItemChangeNotification
            EventSource: aws.config

  RemediationConfiguration:
    AutomaticRemediation: true
    MaximumAutomaticAttempts: 3
    RetryAttemptSeconds: 60
```

## 6. Observabilidad y Monitorización

### 6.1 Amazon CloudWatch Dashboard
```json
{
  "DashboardName": "AdOps-Autonomous-Agent",
  "DashboardBody": {
    "widgets": [
      {
        "type": "metric",
        "properties": {
          "metrics": [
            ["SSMAS/AdOps", "AgentInvocations", {"stat": "Sum"}],
            [".", "SuccessfulRemediations", {"stat": "Sum"}],
            [".", "FailedRemediations", {"stat": "Sum"}],
            [".", "AverageResolutionTime", {"stat": "Average"}]
          ],
          "period": 300,
          "stat": "Average",
          "region": "us-east-1",
          "title": "Agent Performance Metrics"
        }
      },
      {
        "type": "metric",
        "properties": {
          "metrics": [
            ["AWS/Lambda", "Invocations", {"stat": "Sum"}],
            [".", "Errors", {"stat": "Sum"}],
            [".", "Duration", {"stat": "Average"}],
            [".", "ConcurrentExecutions", {"stat": "Maximum"}]
          ],
          "period": 60,
          "stat": "Average",
          "region": "us-east-1",
          "title": "Lambda Functions Performance"
        }
      },
      {
        "type": "log",
        "properties": {
          "query": "SOURCE '/aws/bedrock/agent-logs' | fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc | limit 20",
          "region": "us-east-1",
          "title": "Recent Agent Errors"
        }
      }
    ]
  }
}
```

### 6.2 Application Performance Monitoring

#### 6.2.1 AWS X-Ray Configuration
```python
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# Patch all supported libraries
patch_all()

@xray_recorder.capture('diagnostic_analysis')
def analyze_publisher_metrics(publisher_id):
    subsegment = xray_recorder.current_subsegment()
    subsegment.put_annotation('publisher_id', publisher_id)
    subsegment.put_metadata('analysis_type', 'metrics')

    # Analysis logic
    metrics = fetch_metrics(publisher_id)

    subsegment.put_metadata('metrics_count', len(metrics))

    return analyze_metrics(metrics)
```

#### 6.2.2 Custom Metrics
```python
import boto3
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

def put_custom_metric(metric_name, value, unit='None', dimensions=None):
    """
    Envía métricas personalizadas a CloudWatch
    """
    metric_data = {
        'MetricName': metric_name,
        'Value': value,
        'Unit': unit,
        'Timestamp': datetime.utcnow()
    }

    if dimensions:
        metric_data['Dimensions'] = [
            {'Name': k, 'Value': v} for k, v in dimensions.items()
        ]

    cloudwatch.put_metric_data(
        Namespace='SSMAS/AdOps',
        MetricData=[metric_data]
    )

# Usage example
put_custom_metric(
    'RemediationSuccess',
    1.0,
    'Count',
    {'ActionType': 'cache_clear', 'Publisher': 'publisher123'}
)
```

## 7. CI/CD y Automatización

### 7.1 AWS CodePipeline
```yaml
Pipeline:
  Name: adops-agent-deployment
  RoleArn: arn:aws:iam::account:role/CodePipelineRole

  Stages:
    - Name: Source
      Actions:
        - Name: SourceAction
          ActionTypeId:
            Category: Source
            Owner: ThirdParty
            Provider: GitHub
            Version: 2
          Configuration:
            Owner: SSMAS
            Repo: adops-autonomous-agent
            Branch: main
            OAuthToken: !Ref GitHubToken

    - Name: Build
      Actions:
        - Name: BuildAction
          ActionTypeId:
            Category: Build
            Owner: AWS
            Provider: CodeBuild
            Version: 1
          Configuration:
            ProjectName: adops-agent-build

    - Name: Test
      Actions:
        - Name: UnitTests
          ActionTypeId:
            Category: Test
            Owner: AWS
            Provider: CodeBuild
            Version: 1
          Configuration:
            ProjectName: adops-unit-tests

        - Name: IntegrationTests
          ActionTypeId:
            Category: Test
            Owner: AWS
            Provider: CodeBuild
            Version: 1
          Configuration:
            ProjectName: adops-integration-tests

    - Name: DeployToStaging
      Actions:
        - Name: DeployAction
          ActionTypeId:
            Category: Deploy
            Owner: AWS
            Provider: CloudFormation
            Version: 1
          Configuration:
            ActionMode: CREATE_UPDATE
            StackName: adops-agent-staging
            TemplatePath: BuildArtifact::template.yaml
            Capabilities: CAPABILITY_IAM

    - Name: Approval
      Actions:
        - Name: ManualApproval
          ActionTypeId:
            Category: Approval
            Owner: AWS
            Provider: Manual
            Version: 1
          Configuration:
            CustomData: Please review staging deployment

    - Name: DeployToProduction
      Actions:
        - Name: ProductionDeploy
          ActionTypeId:
            Category: Deploy
            Owner: AWS
            Provider: CloudFormation
            Version: 1
          Configuration:
            ActionMode: CREATE_UPDATE
            StackName: adops-agent-production
            TemplatePath: BuildArtifact::template.yaml
            Capabilities: CAPABILITY_IAM
```

### 7.2 Infrastructure as Code con AWS CDK
```python
from aws_cdk import (
    Stack,
    aws_bedrock as bedrock,
    aws_lambda as lambda_,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    aws_iam as iam,
    Duration
)
from constructs import Construct

class AdOpsAgentStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # S3 Bucket for Knowledge Base
        knowledge_bucket = s3.Bucket(
            self, "KnowledgeBucket",
            bucket_name="ssmas-knowledge-base",
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="delete-old-versions",
                    noncurrent_version_expiration=Duration.days(90)
                )
            ]
        )

        # DynamoDB Table for Configuration
        config_table = dynamodb.Table(
            self, "ConfigTable",
            table_name="publisher-configurations",
            partition_key=dynamodb.Attribute(
                name="publisher_id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery=True,
            stream=dynamodb.StreamViewType.NEW_AND_OLD_IMAGES
        )

        # Lambda Function for Diagnostics
        diagnostic_lambda = lambda_.Function(
            self, "DiagnosticFunction",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="diagnostic.handler",
            code=lambda_.Code.from_asset("lambda/diagnostic"),
            environment={
                "CONFIG_TABLE": config_table.table_name
            },
            timeout=Duration.minutes(5),
            memory_size=1024,
            tracing=lambda_.Tracing.ACTIVE
        )

        # Grant permissions
        config_table.grant_read_data(diagnostic_lambda)

        # Bedrock Agent Role
        agent_role = iam.Role(
            self, "BedrockAgentRole",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
            inline_policies={
                "AgentPolicy": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                "bedrock:InvokeModel",
                                "lambda:InvokeFunction"
                            ],
                            resources=["*"]
                        )
                    ]
                )
            }
        )

        # Note: Bedrock Agent creation would be done via API
        # as CDK support might be limited
```

## 8. Disaster Recovery y Business Continuity

### 8.1 Estrategia Multi-Región

#### 8.1.1 Configuración Active-Passive
```yaml
Regions:
  Primary: us-east-1
  Secondary: us-west-2

  Replication:
    DynamoDB:
      GlobalTables: true
      Regions: [us-east-1, us-west-2]

    S3:
      CrossRegionReplication: true
      ReplicationTime:
        Status: Enabled
        Time:
          Minutes: 15

    RDS:
      CrossRegionReadReplica: true
      AutomatedBackups:
        ReplicationEnabled: true
        RetentionPeriod: 7
```

#### 8.1.2 Route 53 Health Checks
```yaml
Route53:
  HealthChecks:
    - Type: HTTPS
      ResourcePath: /health
      FullyQualifiedDomainName: adops-api.ssmas.com
      Port: 443
      RequestInterval: 30
      FailureThreshold: 3

  RecordSets:
    - Name: adops-api.ssmas.com
      Type: A
      SetIdentifier: Primary
      Failover: PRIMARY
      HealthCheckId: !Ref PrimaryHealthCheck
      AliasTarget:
        DNSName: !GetAtt PrimaryALB.DNSName

    - Name: adops-api.ssmas.com
      Type: A
      SetIdentifier: Secondary
      Failover: SECONDARY
      AliasTarget:
        DNSName: !GetAtt SecondaryALB.DNSName
```

### 8.2 Backup y Recuperación

#### 8.2.1 AWS Backup Plan
```yaml
BackupPlan:
  BackupPlanName: adops-comprehensive-backup
  Rules:
    - RuleName: DailyBackups
      TargetBackupVault: Default
      ScheduleExpression: cron(0 5 ? * * *)
      StartWindowMinutes: 60
      CompletionWindowMinutes: 120
      Lifecycle:
        MoveToColdStorageAfterDays: 30
        DeleteAfterDays: 365
      RecoveryPointTags:
        Environment: Production
        Application: AdOps

    - RuleName: HourlyBackups
      TargetBackupVault: Critical
      ScheduleExpression: rate(1 hour)
      StartWindowMinutes: 10
      CompletionWindowMinutes: 60
      Lifecycle:
        DeleteAfterDays: 7
      RecoveryPointTags:
        Type: Continuous
        Criticality: High

  BackupSelection:
    Resources:
      - arn:aws:dynamodb:*:*:table/publisher-*
      - arn:aws:rds:*:*:db:adops-*
      - arn:aws:s3:::ssmas-*
```

## 9. Optimización de Costos

### 9.1 Reserved Capacity y Savings Plans
```yaml
CostOptimization:
  ComputeSavingsPlan:
    Type: Compute
    Term: 1 Year
    PaymentOption: All Upfront
    HourlyCommitment: $100

  ReservedInstances:
    RDS:
      - InstanceType: db.r6g.xlarge
        Term: 1 Year
        PaymentOption: Partial Upfront
        Count: 2

  S3IntelligentTiering:
    Enabled: true
    ArchiveConfiguration:
      ArchiveAccess:
        Days: 90
      DeepArchiveAccess:
        Days: 180
```

### 9.2 Cost Allocation Tags
```yaml
TaggingStrategy:
  Required:
    - Environment: [Development, Staging, Production]
    - Application: AdOps
    - CostCenter: Engineering
    - Owner: TeamName
    - Project: AutonomousAgent

  CostAllocation:
    - Tag: Environment
      Status: Active
    - Tag: Application
      Status: Active
    - Tag: CostCenter
      Status: Active
```

## 10. Conclusiones y Mejores Prácticas

### 10.1 Principios Clave de la Arquitectura
1. **Automatización Total**: Mínima intervención humana en operaciones rutinarias
2. **Resiliencia**: Capacidad de recuperación automática ante fallos
3. **Escalabilidad**: Crecimiento sin degradación del rendimiento
4. **Seguridad**: Protección en profundidad en todas las capas
5. **Observabilidad**: Visibilidad completa del sistema

### 10.2 Checklist de Implementación
- [ ] Landing Zone configurado con AWS Control Tower
- [ ] VPCs y networking establecidos
- [ ] Roles IAM y políticas definidas
- [ ] Servicios core desplegados (Bedrock, Lambda, DynamoDB)
- [ ] Monitorización y alertas configuradas
- [ ] CI/CD pipeline operativo
- [ ] Backup y DR probados
- [ ] Documentación completa
- [ ] Runbooks actualizados
- [ ] Equipo capacitado

### 10.3 Evolución Futura
- **Machine Learning Avanzado**: Modelos personalizados en SageMaker
- **Multi-Cloud**: Expansión a Azure/GCP para redundancia
- **Edge Computing**: Procesamiento en CloudFront Edge Locations
- **Quantum Ready**: Preparación para AWS Braket
- **Sostenibilidad**: Optimización para carbon neutrality

### 10.4 Referencias y Recursos
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS Control Tower Best Practices](https://docs.aws.amazon.com/controltower/latest/userguide/best-practices.html)
- [AWS Security Best Practices](https://aws.amazon.com/security/best-practices/)
- [AWS Cost Optimization](https://aws.amazon.com/aws-cost-management/)