# Arquitectura Técnica - AWS Landing Zone para SSMAS

## Introducción

La arquitectura técnica de la AWS Landing Zone para SSMAS detalla la implementación específica de servicios AWS, configuraciones de red, patrones de seguridad y integraciones técnicas necesarias para soportar los casos de uso de Inteligencia Artificial y AdTech. Esta arquitectura está diseñada para la región primaria de Dublin (eu-west-1) con capacidades de expansión multi-región.

## Especificaciones Técnicas Generales

### Regiones AWS
- **Región Primaria**: Europe (Dublin) - eu-west-1
- **Región Secundaria**: Europe (Frankfurt) - eu-central-1
- **Availability Zones**: Mínimo 3 AZs por región para alta disponibilidad

### Naming Convention
```
<organization>-<environment>-<service>-<region>-<purpose>
Ejemplo: ssmas-prod-sagemaker-euw1-yield-prediction
```

### Tagging Strategy
```json
{
  "Organization": "SSMAS",
  "Environment": "Production|Staging|Development|Sandbox",
  "Project": "AI-AdTech",
  "CostCenter": "Technology",
  "Owner": "team-email@ssmas.com",
  "BackupSchedule": "Daily|Weekly|None",
  "MANAGED_BY": "CCAPI-MCP-SERVER",
  "MCP_SERVER_SOURCE_CODE": "https://github.com/awslabs/mcp/tree/main/src/ccapi-mcp-server",
  "MCP_SERVER_VERSION": "1.0.0"
}
```

## Arquitectura de Red Técnica

### CIDR Allocation Strategy

```
Root Organization CIDR: 10.0.0.0/8

Security OU:
├── Log Archive Account: 10.1.0.0/16
└── Audit Account: 10.2.0.0/16

Infrastructure OU:
├── Network Account: 10.10.0.0/16
├── Shared Services Account: 10.11.0.0/16
└── DNS Account: 10.12.0.0/16

Workloads OU:
├── Production OU:
│   ├── ML Production: 10.20.0.0/16
│   ├── AdOps Production: 10.21.0.0/16
│   └── Data Production: 10.22.0.0/16
├── Staging OU:
│   ├── ML Staging: 10.30.0.0/16
│   ├── AdOps Staging: 10.31.0.0/16
│   └── Data Staging: 10.32.0.0/16
└── Development OU:
    ├── ML Development: 10.40.0.0/16
    ├── AdOps Development: 10.41.0.0/16
    └── Data Development: 10.42.0.0/16

Sandbox OU:
├── AI/ML Sandbox: 10.50.0.0/16
└── General Sandbox: 10.51.0.0/16
```

### VPC Design Pattern per Account

```
Account VPC: /16 network
├── Public Subnets: /24 per AZ
│   ├── eu-west-1a: x.x.1.0/24
│   ├── eu-west-1b: x.x.2.0/24
│   └── eu-west-1c: x.x.3.0/24
├── Private Subnets: /24 per AZ
│   ├── eu-west-1a: x.x.11.0/24
│   ├── eu-west-1b: x.x.12.0/24
│   └── eu-west-1c: x.x.13.0/24
├── Database Subnets: /24 per AZ
│   ├── eu-west-1a: x.x.21.0/24
│   ├── eu-west-1b: x.x.22.0/24
│   └── eu-west-1c: x.x.23.0/24
└── Management Subnets: /28 per AZ
    ├── eu-west-1a: x.x.31.0/28
    ├── eu-west-1b: x.x.31.16/28
    └── eu-west-1c: x.x.31.32/28
```

### Transit Gateway Configuration

```yaml
Transit Gateway (Network Account):
  ASN: 64512
  CIDR: 10.0.0.0/8
  Default Route Table Association: disable
  Default Route Table Propagation: disable
  DNS Support: enable
  Multicast Support: disable

Route Tables:
  - Production-RT:
      Associated: [ML-Prod, AdOps-Prod, Data-Prod]
      Propagated: [Shared-Services, Network]
  - Staging-RT:
      Associated: [ML-Stage, AdOps-Stage, Data-Stage]
      Propagated: [Shared-Services, Network]
  - Development-RT:
      Associated: [ML-Dev, AdOps-Dev, Data-Dev]
      Propagated: [Shared-Services, Network]
  - Sandbox-RT:
      Associated: [AI-Sandbox, General-Sandbox]
      Propagated: [Shared-Services]
  - Security-RT:
      Associated: [Log-Archive, Audit]
      Propagated: [Shared-Services, Network]
```

### Direct Connect Configuration

```yaml
Direct Connect Gateway:
  Virtual Interfaces:
    - Primary VIF:
        VLAN: 100
        BGP ASN: 65000
        Connection Speed: 10 Gbps
        Redundancy: Active-Active
    - Backup VIF:
        VLAN: 200
        BGP ASN: 65000
        Connection Speed: 10 Gbps
        Redundancy: Standby

BGP Configuration:
  Customer ASN: 65000
  AWS ASN: 64512
  MD5 Authentication: enabled
  BFD: enabled
```

## Arquitectura de Seguridad Técnica

### AWS Control Tower Guardrails

**Mandatory Guardrails (Preventive):**
```yaml
- Disallow Changes to Encryption Configuration for S3 Buckets
- Disallow Changes to Logging Configuration for AWS S3 Buckets
- Disallow Changes to CloudTrail
- Disallow Creation of Access Keys for the Root User
- Disallow Actions as a Root User
- Detect Whether MFA for the Root User is Enabled
- Detect Whether the Administrator Access Policy is Attached to IAM Roles
```

**Strongly Recommended Guardrails (Detective):**
```yaml
- Detect Whether Public Read Access to S3 Buckets is Allowed
- Detect Whether Public Write Access to S3 Buckets is Allowed
- Detect Whether MFA is Enabled for IAM Users
- Detect Whether Any Resources Allow Unrestricted SSH Access
- Detect Whether Any Security Groups Allow Unrestricted Internet Access
```

**Elective Guardrails (Custom):**
```yaml
- Detect Whether EBS Volumes are Encrypted
- Detect Whether RDS Storage is Encrypted
- Detect Whether Lambda Functions are Using Supported Runtimes
- Detect Whether EC2 Instances are Using Instance Metadata Service Version 2
```

### IAM Identity Center Configuration

```yaml
Identity Source: External Identity Provider (Active Directory)
Permission Sets:
  - SSMASAdministrator:
      ManagedPolicies: [AdministratorAccess]
      SessionDuration: 4 hours
      Accounts: [All]

  - SSMASDataScientist:
      ManagedPolicies: [SageMakerFullAccess, S3ReadOnlyAccess]
      CustomPolicies: [BedrockAccess, GlueAccess]
      SessionDuration: 8 hours
      Accounts: [ML-Prod, ML-Stage, Data-Prod, Data-Stage]

  - SSMASAdOpsEngineer:
      ManagedPolicies: [CloudWatchReadOnlyAccess, LambdaFullAccess]
      CustomPolicies: [BedrockAgentsAccess]
      SessionDuration: 8 hours
      Accounts: [AdOps-Prod, AdOps-Stage]

  - SSMASDeveloper:
      ManagedPolicies: [PowerUserAccess]
      DeniedActions: [organizations:*, account:*]
      SessionDuration: 8 hours
      Accounts: [Development OU, Sandbox OU]

  - SSMASReadOnly:
      ManagedPolicies: [ReadOnlyAccess]
      SessionDuration: 4 hours
      Accounts: [All]
```

### Encryption Strategy

**KMS Key Architecture:**
```yaml
Organization Master Key (Root Account):
  KeyId: alias/ssmas-organization-master
  Usage: Cross-account encryption
  Policy: Organization-wide access

Account-Specific Keys:
  - alias/ssmas-{account}-{service}:
      ML Production: alias/ssmas-ml-prod-sagemaker
      Data Production: alias/ssmas-data-prod-s3
      AdOps Production: alias/ssmas-adops-prod-lambda

Service-Specific Keys:
  - S3: Customer-managed KMS keys per bucket
  - RDS: Dedicated key for database encryption
  - EBS: Account-level key for volume encryption
  - Lambda: Environment-specific keys
  - SageMaker: Model and training data encryption
```

## Arquitectura de Datos Técnica

### S3 Data Lake Architecture

```yaml
Bucket Structure (Data Production Account):
Primary Data Lake:
  Bucket: ssmas-datalake-prod-euw1
  Versioning: Enabled
  Encryption: SSE-KMS (CMK)
  Lifecycle:
    - IA Transition: 30 days
    - Glacier Transition: 90 days
    - Deep Archive: 365 days
    - Delete: 2555 days (7 years)

  Folder Structure:
    /raw/
      /year=2024/month=01/day=15/hour=14/
        /impressions/
        /auctions/
        /users/
        /events/
    /processed/
      /feature-engineering/
        /daily-aggregates/
        /hourly-metrics/
      /ml-ready/
        /training-data/
        /inference-data/
    /models/
      /yield-prediction/
        /v1.0/
        /v1.1/
      /anomaly-detection/
    /archives/
      /compliance/
      /audit-logs/

Staging Buckets:
  Bucket: ssmas-datalake-stage-euw1
  Configuration: Same as production

Development Buckets:
  Bucket: ssmas-datalake-dev-euw1
  Lifecycle: Aggressive (delete after 30 days)
```

### Kinesis Streaming Architecture

```yaml
Kinesis Data Streams:
  ssmas-impressions-stream-prod:
    Shards: 100 (auto-scaling enabled)
    Retention: 168 hours (7 days)
    Encryption: Server-side with KMS

  ssmas-auctions-stream-prod:
    Shards: 50
    Retention: 24 hours
    Encryption: Server-side with KMS

  ssmas-events-stream-prod:
    Shards: 20
    Retention: 24 hours
    Encryption: Server-side with KMS

Kinesis Data Firehose:
  ssmas-to-s3-delivery-prod:
    Destination: S3 Data Lake
    Buffer Size: 128 MB
    Buffer Interval: 60 seconds
    Compression: GZIP
    Error Records: S3 error bucket

Kinesis Analytics:
  ssmas-realtime-analytics-prod:
    Application: Real-time metrics calculation
    Input: Kinesis Data Streams
    Output: CloudWatch Metrics, S3 aggregates
```

### Glue ETL Configuration

```yaml
Glue Data Catalog:
  Database: ssmas_adtech_catalog
  Tables:
    - impressions_raw
    - auctions_processed
    - user_segments
    - ml_features

Glue Jobs:
  feature-engineering-daily:
    Type: Python Shell
    Schedule: Daily at 02:00 UTC
    Parallelism: 10 DPUs
    Timeout: 2 hours

  data-quality-validation:
    Type: Spark
    Schedule: Hourly
    Parallelism: 5 DPUs

  ml-preprocessing:
    Type: Python Shell
    Trigger: On-demand
    Parallelism: 20 DPUs
```

## Arquitectura de Machine Learning Técnica

### Amazon SageMaker Configuration

```yaml
SageMaker Domain (ML Production):
  DomainName: ssmas-ml-production
  VPC: ML Production VPC
  Subnets: Private subnets only
  SecurityGroups: SageMaker-security-group

  UserProfiles:
    - DataScientist:
        ExecutionRole: SageMakerExecutionRole
        InstanceTypes: [ml.t3.medium, ml.m5.large, ml.m5.xlarge]
    - MLEngineer:
        ExecutionRole: SageMakerMLOpsRole
        InstanceTypes: [ml.m5.large, ml.m5.2xlarge]

Training Jobs Configuration:
  InstanceTypes:
    - Development: [ml.m5.large, ml.m5.xlarge]
    - Production: [ml.m5.2xlarge, ml.m5.4xlarge, ml.p3.2xlarge]

  VolumeEncryption: KMS encrypted
  NetworkIsolation: Enabled for production
  VPC: Private subnets only

Inference Endpoints:
  Yield Prediction:
    InstanceType: ml.c5.2xlarge
    InitialInstanceCount: 3
    AutoScaling:
      MinCapacity: 2
      MaxCapacity: 10
      TargetValue: 70% CPU utilization

  Multi-Model Endpoint:
    InstanceType: ml.m5.2xlarge
    Models: [yield-prediction, anomaly-detection]
```

### Amazon Bedrock Configuration

```yaml
Bedrock Agents (AdOps Production):
  YieldOptimizationAgent:
    Model: Claude 3 Sonnet
    Instructions: |
      You are an AdOps automation agent. Monitor metrics,
      diagnose issues, and execute remediation actions.
    Tools:
      - verify_publisher_config
      - check_demand_partner_latency
      - disable_partner_temporarily
      - clear_cache
      - analyze_error_logs

  AnomalyDetectionAgent:
    Model: Claude 3 Haiku
    Instructions: |
      Detect anomalies in AdTech metrics and escalate
      or remediate based on severity.
    Tools:
      - check_fill_rate
      - analyze_rpm_trends
      - alert_operations_team

Knowledge Bases:
  AdOpsKnowledgeBase:
    EmbeddingModel: Titan Embeddings G1 - Text
    VectorDatabase: OpenSearch Serverless
    DataSource: S3 runbooks and procedures
```

## Arquitectura de Aplicaciones Técnica

### API Gateway Configuration

```yaml
API Gateway (AdOps Production):
  Type: Regional
  EndpointConfiguration: Private

  APIs:
    yield-prediction-api:
      Stage: prod
      Throttling: 10000 requests/second
      Caching: Enabled (300 seconds TTL)
      Authentication: IAM + API Keys

    adops-monitoring-api:
      Stage: prod
      Throttling: 1000 requests/second
      Authentication: Cognito User Pools

    data-ingestion-api:
      Stage: prod
      Throttling: 50000 requests/second
      Caching: Disabled

Custom Authorizers:
  ssmas-jwt-authorizer:
    Type: Lambda
    TokenSource: Authorization header
    CacheTTL: 300 seconds
```

### Lambda Functions Architecture

```yaml
Lambda Layers:
  ssmas-common-layer:
    Runtime: Python 3.11
    Libraries: [boto3, pandas, numpy, requests]

  ssmas-ml-layer:
    Runtime: Python 3.11
    Libraries: [sagemaker, bedrock, scikit-learn]

Lambda Functions (AdOps Production):
  yield-prediction-inference:
    Runtime: Python 3.11
    Memory: 3008 MB
    Timeout: 30 seconds
    VPC: AdOps Production VPC
    Environment:
      SAGEMAKER_ENDPOINT: yield-prediction-endpoint

  adops-automation-handler:
    Runtime: Python 3.11
    Memory: 512 MB
    Timeout: 900 seconds
    EventSource: CloudWatch Alarms
    Environment:
      BEDROCK_AGENT_ID: adops-agent-id

  data-processing-trigger:
    Runtime: Python 3.11
    Memory: 1024 MB
    Timeout: 300 seconds
    EventSource: S3 Events
```

### Step Functions Workflows

```yaml
ML Pipeline Workflow:
  StateMachine: ssmas-ml-pipeline-prod
  Definition:
    StartAt: DataValidation
    States:
      DataValidation:
        Type: Task
        Resource: lambda:data-quality-check
        Next: FeatureEngineering

      FeatureEngineering:
        Type: Task
        Resource: glue:feature-engineering-job
        Next: ModelTraining

      ModelTraining:
        Type: Task
        Resource: sagemaker:training-job
        Next: ModelEvaluation

      ModelEvaluation:
        Type: Task
        Resource: lambda:model-evaluation
        Next: ModelDeployment

      ModelDeployment:
        Type: Task
        Resource: sagemaker:create-endpoint
        End: true

AdOps Automation Workflow:
  StateMachine: ssmas-adops-automation-prod
  Definition:
    StartAt: AlertReceived
    States:
      AlertReceived:
        Type: Task
        Resource: lambda:alert-processor
        Next: DiagnosticChoice

      DiagnosticChoice:
        Type: Choice
        Choices:
          - Variable: $.severity
            StringEquals: HIGH
            Next: ImmediateRemediation
          - Variable: $.severity
            StringEquals: MEDIUM
            Next: AnalysisRequired
        Default: LogAndContinue
```

## Monitorización y Observabilidad Técnica

### CloudWatch Configuration

```yaml
Log Groups:
  /aws/lambda/yield-prediction-inference:
    RetentionDays: 30
    KMSEncryption: Enabled

  /aws/sagemaker/TrainingJobs:
    RetentionDays: 90

  /aws/apigateway/yield-prediction-api:
    RetentionDays: 14

Custom Metrics:
  SSMAS/AdTech:
    - FillRate (Percentage)
    - RPM (Revenue per Mille)
    - LatencyP99 (Milliseconds)
    - ErrorRate (Percentage)

  SSMAS/ML:
    - ModelAccuracy (Percentage)
    - PredictionLatency (Milliseconds)
    - ModelDrift (Score)

Alarms:
  FillRateDropAlert:
    MetricName: FillRate
    Threshold: 85
    ComparisonOperator: LessThanThreshold
    EvaluationPeriods: 2
    Actions: [SNS Topic, Lambda Function]

  HighLatencyAlert:
    MetricName: LatencyP99
    Threshold: 100
    ComparisonOperator: GreaterThanThreshold
    EvaluationPeriods: 3
```

### X-Ray Tracing

```yaml
X-Ray Configuration:
  Services:
    - API Gateway: Enabled
    - Lambda: Active tracing
    - SageMaker: Custom instrumentation

  Sampling Rules:
    - RuleName: AdOpsHighVolume
      Priority: 9000
      FixedRate: 0.1
      ReservoirSize: 1
      ServiceName: yield-prediction-api

    - RuleName: MLInference
      Priority: 5000
      FixedRate: 0.5
      ReservoirSize: 2
      ServiceName: sagemaker-endpoint
```

## Backup y Disaster Recovery Técnico

### RDS Backup Configuration

```yaml
RDS Instances:
  ssmas-adops-metadata-prod:
    Engine: PostgreSQL 14
    InstanceClass: db.r6g.2xlarge
    MultiAZ: true
    BackupRetention: 35 days
    BackupWindow: "03:00-04:00 UTC"
    MaintenanceWindow: "sun:04:00-sun:05:00 UTC"
    Encryption: KMS

    Read Replicas:
      - ssmas-adops-metadata-replica-1 (eu-west-1b)
      - ssmas-adops-metadata-replica-2 (eu-west-1c)

  Cross-Region Replica:
    Region: eu-central-1
    Purpose: Disaster Recovery
```

### EBS Snapshot Configuration

```yaml
EBS Snapshot Lifecycle:
  DailySnapshots:
    Schedule: Daily at 02:00 UTC
    Retention: 7 days
    Tags:
      - Backup: Daily
      - Environment: Production

  WeeklySnapshots:
    Schedule: Sunday at 01:00 UTC
    Retention: 4 weeks
    CrossRegionCopy: eu-central-1

  MonthlySnapshots:
    Schedule: First Sunday of month
    Retention: 12 months
    CrossRegionCopy: eu-central-1
```

### S3 Cross-Region Replication

```yaml
Replication Configuration:
  Source: ssmas-datalake-prod-euw1
  Destination: ssmas-datalake-dr-euc1

  Rules:
    - CriticalData:
        Prefix: /models/
        Status: Enabled
        Priority: 1

    - ComplianceData:
        Prefix: /archives/compliance/
        Status: Enabled
        Priority: 2

    - RawData:
        Prefix: /raw/
        Status: Disabled (Cost optimization)
```

## Automatización y CI/CD Técnico

### CodePipeline Configuration

```yaml
ML Model Pipeline:
  Pipeline: ssmas-ml-model-pipeline
  Source: CodeCommit repository

  Stages:
    - Source:
        Action: CodeCommit
        Repository: ssmas-ml-models
        Branch: main

    - Build:
        Action: CodeBuild
        Project: ssmas-model-build
        Environment: Python 3.11

    - TestInStaging:
        Action: SageMaker
        Endpoint: staging-endpoint

    - ApprovalGate:
        Action: Manual Approval

    - DeployToProduction:
        Action: SageMaker
        Endpoint: production-endpoint

Infrastructure Pipeline:
  Pipeline: ssmas-infrastructure-pipeline
  Source: CodeCommit repository

  Stages:
    - Source:
        Repository: ssmas-infrastructure
        Branch: main

    - Validate:
        Action: CloudFormation
        Template: validate-only

    - DeployToStaging:
        Action: CloudFormation
        StackName: ssmas-staging-stack

    - DeployToProduction:
        Action: CloudFormation
        StackName: ssmas-production-stack
```

### CodeBuild Projects

```yaml
ML Model Build:
  Project: ssmas-model-build
  Environment:
    Type: Linux
    Image: aws/codebuild/amazonlinux2-x86_64-standard:3.0
    ComputeType: BUILD_GENERAL1_LARGE

  BuildSpec:
    phases:
      pre_build:
        commands:
          - pip install -r requirements.txt
          - python -m pytest tests/
      build:
        commands:
          - python train_model.py
          - python package_model.py
      post_build:
        commands:
          - aws s3 cp model.tar.gz s3://ssmas-ml-artifacts/

Infrastructure Build:
  Project: ssmas-infrastructure-build
  Environment:
    Type: Linux
    Image: aws/codebuild/amazonlinux2-x86_64-standard:3.0

  BuildSpec:
    phases:
      build:
        commands:
          - cfn-lint templates/*.yaml
          - aws cloudformation validate-template --template-body file://template.yaml
```

## Seguridad Técnica Avanzada

### AWS Config Rules

```yaml
Custom Config Rules:
  sagemaker-endpoint-encrypted:
    Source: AWS Config Managed Rules
    Parameters:
      kmsKeyId: alias/ssmas-ml-prod-sagemaker

  s3-bucket-ssl-requests-only:
    Source: AWS Config Managed Rules

  lambda-function-public-access-prohibited:
    Source: AWS Config Managed Rules

  ec2-instance-managed-by-ssm:
    Source: AWS Config Managed Rules

Remediation Actions:
  s3-bucket-public-read-prohibited:
    AutoRemediation: true
    Action: Lambda function to block public access

  security-group-ssh-check:
    AutoRemediation: true
    Action: Lambda function to remove 0.0.0.0/0 rules
```

### GuardDuty Configuration

```yaml
GuardDuty (Security Tooling Account):
  Master Account: Security Tooling
  Member Accounts: All organization accounts

  Findings Export:
    Destination: S3 bucket in Log Archive account
    KMSEncryption: Enabled

  Threat Intelligence:
    - AWS threat intelligence
    - Custom threat intelligence feeds

  Malware Protection:
    EBS Volume Scanning: Enabled
    Lambda Function Scanning: Enabled
```

### Security Hub Configuration

```yaml
Security Hub (Security Tooling Account):
  Standards:
    - AWS Foundational Security Standard
    - CIS AWS Foundations Benchmark
    - PCI DSS

  Custom Insights:
    - High Severity Findings by Account
    - Unresolved Findings Trending
    - Compliance Score by Business Unit

  Integration:
    - GuardDuty findings
    - Config rule compliance
    - Inspector findings
    - Access Analyzer findings
```

## Costos y Optimización Técnica

### Reserved Instance Strategy

```yaml
EC2 Reserved Instances:
  SageMaker Training:
    InstanceType: ml.m5.2xlarge
    Count: 5
    Term: 1 year
    PaymentOption: All Upfront

  SageMaker Inference:
    InstanceType: ml.c5.2xlarge
    Count: 3
    Term: 1 year
    PaymentOption: Partial Upfront

RDS Reserved Instances:
  Primary Database:
    InstanceClass: db.r6g.2xlarge
    Count: 1
    Term: 3 years
    PaymentOption: All Upfront
```

### Savings Plans

```yaml
Compute Savings Plans:
  Commitment: $50,000/year
  Term: 1 year
  PaymentOption: No Upfront

SageMaker Savings Plans:
  Commitment: $25,000/year
  Term: 1 year
  PaymentOption: Partial Upfront
```

### Cost Allocation Tags

```yaml
Mandatory Tags:
  - CostCenter: [Technology, Marketing, Operations]
  - Project: [AI-AdTech, Infrastructure, Security]
  - Environment: [Production, Staging, Development, Sandbox]
  - Owner: [team-email]
  - Purpose: [ML-Training, ML-Inference, Data-Processing, API]

Cost Categories:
  - Function: [Compute, Storage, Network, ML, Security]
  - Team: [Data-Science, DevOps, Security, AdOps]
  - Geography: [Dublin, Frankfurt]
```

## Especificaciones de Performance

### SLA Targets

```yaml
Service Level Agreements:
  API Gateway Response Time:
    P50: < 50ms
    P95: < 200ms
    P99: < 500ms

  SageMaker Inference:
    P50: < 20ms
    P95: < 100ms
    P99: < 250ms

  S3 Data Lake Access:
    First Byte: < 100ms
    Throughput: > 1 GB/s

  Availability:
    Production Services: 99.9%
    Development Services: 99.0%
```

### Auto Scaling Configuration

```yaml
Lambda Auto Scaling:
  Concurrent Executions:
    Account Limit: 10,000
    Function Reserved: 1,000 (yield-prediction)

API Gateway:
  Throttling Limits:
    Account Level: 50,000 requests/second
    API Level: 10,000 requests/second

SageMaker Auto Scaling:
  yield-prediction-endpoint:
    MinCapacity: 2
    MaxCapacity: 10
    TargetValue: 70 (InvocationsPerInstance)
    ScaleOutCooldown: 300 seconds
    ScaleInCooldown: 300 seconds
```

## Compliance y Auditoría Técnica

### CloudTrail Configuration

```yaml
Organization Trail (Log Archive Account):
  Name: ssmas-organization-trail
  S3Bucket: ssmas-cloudtrail-logs-euw1
  S3KeyPrefix: AWSLogs/
  IncludeGlobalServiceEvents: true
  IsMultiRegionTrail: true
  EnableLogFileValidation: true
  KMSKeyId: alias/ssmas-cloudtrail-key

  EventSelectors:
    - ReadWriteType: All
      IncludeManagementEvents: true
      DataResources:
        - Type: AWS::S3::Object
          Values: ["arn:aws:s3:::ssmas-datalake-*/*"]
        - Type: AWS::Lambda::Function
          Values: ["*"]

Insight Selectors:
  - InsightType: ApiCallRateInsight
  - InsightType: ApiErrorRateInsight
```

### AWS Config Organization Configuration

```yaml
Configuration Recorder (All Accounts):
  RecordingGroup:
    AllSupported: true
    IncludeGlobalResourceTypes: true
    ResourceTypes: [All AWS Resources]

  DeliveryChannel:
    S3BucketName: ssmas-config-bucket-euw1
    S3KeyPrefix: AWSLogs/
    ConfigSnapshotDeliveryProperties:
      DeliveryFrequency: Daily
```

Esta arquitectura técnica proporciona la base detallada para implementar la AWS Landing Zone de SSMAS, asegurando que todos los componentes estén correctamente configurados para soportar los casos de uso de IA y AdTech con los más altos estándares de seguridad, performance y compliance.