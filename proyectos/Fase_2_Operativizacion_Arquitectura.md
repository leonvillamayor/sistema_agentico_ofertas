# Fase 2: Operativización y Aplicación MLOps - Arquitectura Técnica AWS

## Visión General de la Fase 2

La Fase 2 se enfoca en la operativización completa del Agente de AdOps Autónomo, expandiendo las capacidades validadas en el MVP hacia un sistema de producción robusto, escalable y que maneja múltiples tipos de problemas operativos. Esta fase implementa arquitectura completa de microservicios, pipelines MLOps, y monitorización avanzada.

### Objetivos Principales
- **Escalabilidad Productiva**: Sistema capaz de manejar todos los editores y tipos de problemas
- **Arquitectura MLOps**: Pipelines de CI/CD para modelos y agentes
- **Multi-Agente**: Especialización por tipo de problema
- **Observabilidad Completa**: Monitorización, alerting y analytics avanzados
- **Alta Disponibilidad**: Despliegue multi-AZ con redundancia

### Alcance Expandido
- **Todos los Editores**: Cobertura completa de la base de editores SSMAS
- **5-7 Tipos de Problemas**: Performance, configuración, demand partners, infraestructura, seguridad
- **20-25 Herramientas**: Biblioteca completa de diagnóstico y remediación
- **Modo Activo**: Operación autónoma con supervisión

## Arquitectura AWS de Producción

### Componentes Avanzados de Detección

#### 1. Sistema de Ingesta Multi-Fuente

```yaml
Kinesis Data Streams Configuration:
  Streams:
    - Name: adops-metrics-stream
      ShardCount: 20
      RetentionPeriod: 168 hours  # 7 days
      ShardLevelMetrics:
        - IncomingRecords
        - OutgoingRecords
        - IncomingBytes
        - OutgoingBytes
      Encryption:
        Type: KMS
        KeyId: alias/adops-kinesis-key

    - Name: adops-events-stream
      ShardCount: 10
      RetentionPeriod: 24 hours
      StreamModeDetails:
        StreamMode: ON_DEMAND

  DataAnalytics:
    Application: adops-realtime-processor
    Runtime: Apache Flink 1.15

    SQL_Queries:
      RPM_Anomaly_Detection: |
        CREATE TABLE rpm_metrics (
          editor_id VARCHAR(50),
          rpm DOUBLE,
          timestamp TIMESTAMP(3),
          WATERMARK FOR timestamp AS timestamp - INTERVAL '30' SECOND
        ) WITH (
          'connector' = 'kinesis',
          'stream' = 'adops-metrics-stream',
          'aws.region' = 'us-east-1',
          'format' = 'json'
        );

        CREATE TABLE rpm_anomalies AS
        SELECT
          editor_id,
          AVG(rpm) as avg_rpm,
          STDDEV(rpm) as stddev_rpm,
          COUNT(*) as data_points,
          TUMBLE_START(timestamp, INTERVAL '5' MINUTE) as window_start
        FROM rpm_metrics
        GROUP BY editor_id, TUMBLE(timestamp, INTERVAL '5' MINUTE)
        HAVING AVG(rpm) < (
          SELECT AVG(rpm) - 2 * STDDEV(rpm)
          FROM rpm_metrics
          WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '1' HOUR
        );

      Fill_Rate_Correlation: |
        CREATE TABLE correlation_analysis AS
        SELECT
          m1.editor_id,
          m1.timestamp,
          m1.rpm,
          m2.fill_rate,
          m3.latency,
          CASE
            WHEN m1.rpm < 30 AND m2.fill_rate < 0.8 THEN 'DEMAND_ISSUE'
            WHEN m1.rpm < 30 AND m3.latency > 3000 THEN 'LATENCY_ISSUE'
            WHEN m1.rpm < 30 AND m2.fill_rate > 0.9 THEN 'PRICING_ISSUE'
            ELSE 'UNKNOWN'
          END as problem_type
        FROM rpm_metrics m1
        JOIN fill_rate_metrics m2 ON m1.editor_id = m2.editor_id
          AND m1.timestamp = m2.timestamp
        JOIN latency_metrics m3 ON m1.editor_id = m3.editor_id
          AND m1.timestamp = m3.timestamp;

  DataFirehose:
    DeliveryStreams:
      - Name: adops-data-lake-delivery
        Destination: ExtendedS3
        ExtendedS3Configuration:
          BucketARN: arn:aws:s3:::adops-data-lake-prod
          BufferingHints:
            SizeInMBs: 64
            IntervalInSeconds: 60
          CompressionFormat: GZIP
          DataFormatConversionConfiguration:
            Enabled: true
            OutputFormatConfiguration:
              Serializer:
                ParquetSerDe: {}
          DynamicPartitioning:
            Enabled: true
          ProcessingConfiguration:
            Enabled: true
            Processors:
              - Type: Lambda
                Parameters:
                  - ParameterName: LambdaArn
                    ParameterValue: arn:aws:lambda:us-east-1:account:function:data-enrichment
```

#### 2. EventBridge Avanzado con Reglas Complejas

```yaml
EventBridge Configuration:
  CustomEventBus:
    Name: adops-production-events
    Description: "Production event bus for AdOps operations"

  Rules:
    - Name: multi-metric-anomaly
      EventPattern:
        source: ["adops.analytics"]
        detail-type: ["Composite Anomaly Detected"]
        detail:
          severity: ["CRITICAL", "HIGH"]
          metrics_affected:
            - exists: true
            - numeric:
                ">": 2
      Targets:
        - Id: "1"
          Arn: arn:aws:states:us-east-1:account:stateMachine:critical-incident-workflow
          RoleArn: arn:aws:iam::account:role/EventBridgeStepFunctionsRole
        - Id: "2"
          Arn: arn:aws:sns:us-east-1:account:critical-alerts
          MessageGroupId: critical-incidents

    - Name: demand-partner-degradation
      EventPattern:
        source: ["adops.monitoring"]
        detail-type: ["Partner Health Check"]
        detail:
          partner_status: ["DEGRADED", "UNHEALTHY"]
          impact_assessment:
            revenue_impact:
              numeric:
                ">": 1000
      Targets:
        - Id: "1"
          Arn: arn:aws:lambda:us-east-1:account:function:partner-remediation-orchestrator
        - Id: "2"
          Arn: arn:aws:events:us-east-1:account:event-bus/partner-management-events

    - Name: security-incident-detected
      EventPattern:
        source: ["aws.guardduty", "aws.securityhub"]
        detail-type: ["GuardDuty Finding", "Security Hub Findings - Imported"]
        detail:
          severity:
            numeric:
              ">=": 7.0
      Targets:
        - Id: "1"
          Arn: arn:aws:lambda:us-east-1:account:function:security-agent-trigger
          InputTransformer:
            InputPathsMap:
              severity: "$.detail.severity"
              finding_type: "$.detail.type"
            InputTemplate: |
              {
                "agent_type": "security",
                "priority": "immediate",
                "context": {
                  "severity": "<severity>",
                  "finding_type": "<finding_type>",
                  "source_event": <aws.events.event>
                }
              }

  Archive:
    Name: adops-events-archive
    SourceArn: arn:aws:events:us-east-1:account:event-bus/adops-production-events
    RetentionDays: 30
    Description: "Archive for compliance and debugging"

  Replay:
    Enabled: true
    Use_Cases:
      - Debugging incident response workflows
      - Testing new agent configurations
      - Training data generation for ML models
```

### Arquitectura Multi-Agente con Bedrock

#### 1. Orquestador Central

```yaml
Bedrock Agent Orchestrator:
  Model: anthropic.claude-3-sonnet

  Agent:
    Name: adops-orchestrator-agent
    Description: "Central orchestrator for multi-agent AdOps system"

    Instructions: |
      You are the central orchestrator for the AdOps autonomous system. Your responsibilities:

      1. INCIDENT TRIAGE
         - Analyze incoming alerts and classify problem types
         - Determine severity and urgency
         - Select appropriate specialist agent(s)
         - Coordinate multi-agent responses when needed

      2. RESOURCE MANAGEMENT
         - Prevent agent conflicts and resource contention
         - Manage escalation pathways
         - Coordinate with human operators when needed

      3. DECISION OVERSIGHT
         - Validate high-impact remediation decisions
         - Ensure compliance with safety constraints
         - Maintain system-wide context and state

      Classification Framework:
      - PERFORMANCE: Latency, throughput, response time issues
      - CONFIGURATION: Settings, parameters, feature flags
      - DEMAND: Partner connectivity, bidding, fill rates
      - INFRASTRUCTURE: Servers, scaling, resource availability
      - SECURITY: Threats, vulnerabilities, compliance violations

      Always provide reasoning for agent selection and coordinate handoffs clearly.

    ActionGroups:
      - Name: agent-coordination
        Description: "Tools for managing other agents"
        Lambda: arn:aws:lambda:us-east-1:account:function:agent-coordinator

      - Name: incident-management
        Description: "Tools for incident lifecycle management"
        Lambda: arn:aws:lambda:us-east-1:account:function:incident-manager

  KnowledgeBase:
    Name: orchestrator-knowledge
    DataSource:
      Type: S3
      Bucket: adops-orchestrator-knowledge
      Includes:
        - Agent capabilities matrix
        - Escalation procedures
        - System topology
        - Incident response runbooks
```

#### 2. Agentes Especializados

```yaml
Specialized_Agents:
  PerformanceAgent:
    Model: anthropic.claude-3-sonnet
    Name: adops-performance-agent

    Instructions: |
      You are a performance optimization specialist for AdOps systems. Focus on:

      - Latency analysis and optimization
      - Throughput bottleneck identification
      - Cache performance and optimization
      - Load balancing and traffic distribution
      - Resource utilization analysis

      Use systematic diagnosis: metrics → traces → logs → root cause → remediation.

    ActionGroups:
      - performance-diagnostics
      - cache-management
      - load-balancing
      - resource-scaling

    KnowledgeBase:
      Includes:
        - Performance baselines by editor
        - Optimization playbooks
        - Capacity planning guidelines

  ConfigurationAgent:
    Model: anthropic.claude-3-sonnet
    Name: adops-configuration-agent

    Instructions: |
      You are a configuration management specialist. Handle:

      - Feature flag validation and rollback
      - Header bidding configuration optimization
      - Ad unit setup verification
      - Consent management platform (CMP) configuration
      - Integration parameter validation

      Always backup configurations before changes and validate against baselines.

    ActionGroups:
      - config-validation
      - feature-flag-management
      - header-bidding-optimization
      - integration-management

  DemandPartnerAgent:
    Model: anthropic.claude-3-sonnet
    Name: adops-demand-agent

    Instructions: |
      You specialize in demand partner relationship management:

      - Partner connectivity monitoring
      - Bid response analysis
      - Fill rate optimization
      - Revenue impact assessment
      - Partner performance comparison

      Prioritize revenue impact and maintain partner relationships.

    ActionGroups:
      - partner-health-monitoring
      - bid-analysis
      - revenue-optimization
      - partner-communication

  InfrastructureAgent:
    Model: anthropic.claude-3-haiku  # Cost optimization for frequent scaling operations
    Name: adops-infrastructure-agent

    Instructions: |
      You manage infrastructure scaling and reliability:

      - Auto-scaling policy optimization
      - Server health monitoring
      - Network connectivity issues
      - Database performance tuning
      - CDN optimization

      Focus on availability and cost optimization.

    ActionGroups:
      - auto-scaling-management
      - server-health-monitoring
      - network-diagnostics
      - database-optimization

  SecurityAgent:
    Model: anthropic.claude-3-sonnet
    Name: adops-security-agent

    Instructions: |
      You are responsible for security incident response:

      - Threat detection and analysis
      - Vulnerability assessment
      - Compliance monitoring
      - Access control validation
      - Fraud detection in ad traffic

      Security always takes priority over performance or revenue.

    ActionGroups:
      - threat-analysis
      - vulnerability-scanning
      - compliance-checking
      - fraud-detection
```

### Biblioteca Completa de Herramientas Lambda

#### 1. Herramientas de Diagnóstico Avanzadas

```python
# Advanced diagnostic tools for production

import boto3
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

class AdvancedDiagnosticTools:
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.logs = boto3.client('logs')
        self.xray = boto3.client('xray')
        self.athena = boto3.client('athena')
        self.opensearch = boto3.client('opensearch')

    def analyze_metric_correlations(self, editor_id: str,
                                   time_range_hours: int = 2) -> Dict:
        """
        Advanced correlation analysis between multiple metrics
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_range_hours)

        metrics = [
            'RPM', 'FillRate', 'BidLatency', 'ErrorRate',
            'PartnerTimeouts', 'CacheHitRate'
        ]

        metric_data = {}

        for metric in metrics:
            response = self.cloudwatch.get_metric_statistics(
                Namespace='SSMAS/AdOps',
                MetricName=metric,
                Dimensions=[{'Name': 'EditorId', 'Value': editor_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )

            if response['Datapoints']:
                df = pd.DataFrame(response['Datapoints'])
                df = df.sort_values('Timestamp')
                metric_data[metric] = df['Average'].values

        # Calculate correlation matrix
        if len(metric_data) >= 2:
            correlation_matrix = np.corrcoef(list(metric_data.values()))

            # Identify strong correlations (>0.7 or <-0.7)
            strong_correlations = []
            for i, metric1 in enumerate(metrics):
                for j, metric2 in enumerate(metrics):
                    if i < j and abs(correlation_matrix[i][j]) > 0.7:
                        strong_correlations.append({
                            'metric1': metric1,
                            'metric2': metric2,
                            'correlation': float(correlation_matrix[i][j]),
                            'strength': 'strong_positive' if correlation_matrix[i][j] > 0 else 'strong_negative'
                        })

            return {
                'editor_id': editor_id,
                'analysis_window': f"{time_range_hours} hours",
                'strong_correlations': strong_correlations,
                'correlation_matrix': correlation_matrix.tolist(),
                'insights': self._generate_correlation_insights(strong_correlations)
            }

        return {'error': 'Insufficient data for correlation analysis'}

    def trace_request_path(self, trace_id: str) -> Dict:
        """
        Use X-Ray to trace complete request path and identify bottlenecks
        """
        try:
            response = self.xray.get_trace_summaries(
                TimeRangeType='TraceId',
                TraceIds=[trace_id]
            )

            if not response['TraceSummaries']:
                return {'error': f'Trace {trace_id} not found'}

            trace_summary = response['TraceSummaries'][0]

            # Get detailed trace
            trace_detail = self.xray.batch_get_traces(
                TraceIds=[trace_id]
            )

            if trace_detail['Traces']:
                segments = trace_detail['Traces'][0]['Segments']

                # Analyze segments for performance bottlenecks
                bottlenecks = []
                total_duration = 0

                for segment in segments:
                    segment_data = json.loads(segment['Document'])
                    duration = segment_data.get('end_time', 0) - segment_data.get('start_time', 0)

                    if duration > 1.0:  # More than 1 second
                        bottlenecks.append({
                            'service': segment_data.get('name'),
                            'duration': duration,
                            'error': segment_data.get('error', False),
                            'fault': segment_data.get('fault', False)
                        })

                    total_duration += duration

                return {
                    'trace_id': trace_id,
                    'total_duration': total_duration,
                    'bottlenecks': sorted(bottlenecks, key=lambda x: x['duration'], reverse=True),
                    'recommendations': self._generate_trace_recommendations(bottlenecks)
                }

        except Exception as e:
            return {'error': f'Trace analysis failed: {str(e)}'}

    def analyze_log_patterns(self, log_group: str,
                           search_pattern: str,
                           time_range_minutes: int = 30) -> Dict:
        """
        Advanced log pattern analysis using CloudWatch Logs Insights
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=time_range_minutes)

        query = f"""
        fields @timestamp, @message
        | filter @message like /{search_pattern}/
        | stats count() by bin(5m)
        | sort @timestamp desc
        """

        try:
            response = self.logs.start_query(
                logGroupName=log_group,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query
            )

            query_id = response['queryId']

            # Wait for query completion
            import time
            while True:
                result = self.logs.get_query_results(queryId=query_id)
                if result['status'] == 'Complete':
                    break
                elif result['status'] == 'Failed':
                    return {'error': 'Log query failed'}
                time.sleep(1)

            # Process results
            results = result['results']
            if results:
                pattern_counts = []
                for row in results:
                    timestamp = row[0]['value']
                    count = int(row[1]['value'])
                    pattern_counts.append({
                        'timestamp': timestamp,
                        'count': count
                    })

                # Detect anomalies in pattern frequency
                counts = [item['count'] for item in pattern_counts]
                if len(counts) > 3:
                    mean_count = np.mean(counts)
                    std_count = np.std(counts)

                    anomalies = []
                    for item in pattern_counts:
                        if abs(item['count'] - mean_count) > 2 * std_count:
                            anomalies.append(item)

                    return {
                        'log_group': log_group,
                        'pattern': search_pattern,
                        'time_range_minutes': time_range_minutes,
                        'total_matches': sum(counts),
                        'pattern_frequency': pattern_counts,
                        'anomalies': anomalies,
                        'baseline_mean': mean_count,
                        'baseline_std': std_count
                    }

            return {'message': 'No patterns found matching criteria'}

        except Exception as e:
            return {'error': f'Log analysis failed: {str(e)}'}

    def comprehensive_health_check(self, editor_id: str) -> Dict:
        """
        Comprehensive system health check across all dimensions
        """
        health_report = {
            'editor_id': editor_id,
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'UNKNOWN',
            'health_score': 0,
            'checks': {}
        }

        # Performance Check
        perf_check = self._check_performance_health(editor_id)
        health_report['checks']['performance'] = perf_check

        # Configuration Check
        config_check = self._check_configuration_health(editor_id)
        health_report['checks']['configuration'] = config_check

        # Demand Partner Check
        partner_check = self._check_partner_health(editor_id)
        health_report['checks']['demand_partners'] = partner_check

        # Infrastructure Check
        infra_check = self._check_infrastructure_health(editor_id)
        health_report['checks']['infrastructure'] = infra_check

        # Calculate overall health score
        scores = [check.get('score', 0) for check in health_report['checks'].values()]
        health_report['health_score'] = sum(scores) / len(scores) if scores else 0

        # Determine overall status
        if health_report['health_score'] >= 90:
            health_report['overall_status'] = 'HEALTHY'
        elif health_report['health_score'] >= 70:
            health_report['overall_status'] = 'DEGRADED'
        elif health_report['health_score'] >= 50:
            health_report['overall_status'] = 'UNHEALTHY'
        else:
            health_report['overall_status'] = 'CRITICAL'

        return health_report

    def predictive_anomaly_detection(self, editor_id: str,
                                   metric_name: str,
                                   prediction_horizon_minutes: int = 60) -> Dict:
        """
        ML-based predictive anomaly detection
        """
        # Get historical data for training
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)

        response = self.cloudwatch.get_metric_statistics(
            Namespace='SSMAS/AdOps',
            MetricName=metric_name,
            Dimensions=[{'Name': 'EditorId', 'Value': editor_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,
            Statistics=['Average']
        )

        if len(response['Datapoints']) < 50:
            return {'error': 'Insufficient historical data for prediction'}

        # Prepare data
        df = pd.DataFrame(response['Datapoints'])
        df = df.sort_values('Timestamp')
        df['timestamp_numeric'] = pd.to_datetime(df['Timestamp']).astype(int) / 10**9

        # Simple trend analysis and prediction
        X = df['timestamp_numeric'].values.reshape(-1, 1)
        y = df['Average'].values

        # Use simple linear regression for trend
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)

        # Generate future timestamps
        future_start = end_time.timestamp()
        future_end = (end_time + timedelta(minutes=prediction_horizon_minutes)).timestamp()
        future_timestamps = np.linspace(future_start, future_end,
                                       prediction_horizon_minutes // 5)

        # Predict future values
        future_predictions = model.predict(future_timestamps.reshape(-1, 1))

        # Calculate prediction confidence intervals
        residuals = y - model.predict(X)
        std_error = np.std(residuals)

        predictions = []
        for i, (timestamp, prediction) in enumerate(zip(future_timestamps, future_predictions)):
            predictions.append({
                'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                'predicted_value': float(prediction),
                'confidence_lower': float(prediction - 2 * std_error),
                'confidence_upper': float(prediction + 2 * std_error),
                'anomaly_risk': 'HIGH' if prediction < (np.mean(y) - 2 * np.std(y)) else 'LOW'
            })

        return {
            'editor_id': editor_id,
            'metric': metric_name,
            'prediction_horizon_minutes': prediction_horizon_minutes,
            'current_value': float(y[-1]),
            'trend': 'INCREASING' if model.coef_[0] > 0 else 'DECREASING',
            'trend_strength': abs(float(model.coef_[0])),
            'predictions': predictions,
            'model_accuracy': float(model.score(X, y))
        }
```

#### 2. Herramientas de Remediación Inteligente

```python
class IntelligentRemediationTools:
    def __init__(self):
        self.autoscaling = boto3.client('autoscaling')
        self.elbv2 = boto3.client('elbv2')
        self.route53 = boto3.client('route53')
        self.ssm = boto3.client('ssm')
        self.dynamodb = boto3.resource('dynamodb')

    def intelligent_traffic_shifting(self, editor_id: str,
                                   unhealthy_partners: List[str],
                                   shift_percentage: float = 50.0) -> Dict:
        """
        Intelligently shift traffic away from unhealthy partners
        """
        try:
            # Get current traffic distribution
            config_table = self.dynamodb.Table('adops-traffic-config')
            current_config = config_table.get_item(
                Key={'editor_id': editor_id}
            )

            if 'Item' not in current_config:
                return {'error': f'No traffic config found for editor {editor_id}'}

            traffic_config = current_config['Item']['partner_weights']

            # Calculate new weights
            total_unhealthy_weight = sum(
                traffic_config.get(partner, 0)
                for partner in unhealthy_partners
            )

            healthy_partners = [
                partner for partner in traffic_config.keys()
                if partner not in unhealthy_partners
            ]

            if not healthy_partners:
                return {'error': 'No healthy partners available for traffic shifting'}

            # Redistribute traffic
            new_config = traffic_config.copy()
            weight_to_redistribute = 0

            for partner in unhealthy_partners:
                if partner in new_config:
                    weight_reduction = new_config[partner] * (shift_percentage / 100)
                    new_config[partner] -= weight_reduction
                    weight_to_redistribute += weight_reduction

            # Distribute to healthy partners proportionally
            healthy_total = sum(new_config[partner] for partner in healthy_partners)
            for partner in healthy_partners:
                proportion = new_config[partner] / healthy_total
                new_config[partner] += weight_to_redistribute * proportion

            # Update configuration
            config_table.update_item(
                Key={'editor_id': editor_id},
                UpdateExpression='SET partner_weights = :new_weights, last_updated = :timestamp',
                ExpressionAttributeValues={
                    ':new_weights': new_config,
                    ':timestamp': datetime.utcnow().isoformat()
                }
            )

            return {
                'success': True,
                'editor_id': editor_id,
                'action': 'traffic_shifted',
                'unhealthy_partners': unhealthy_partners,
                'shift_percentage': shift_percentage,
                'old_config': traffic_config,
                'new_config': new_config,
                'weight_redistributed': weight_to_redistribute
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def adaptive_auto_scaling(self, editor_id: str,
                            predicted_load_increase: float,
                            time_horizon_minutes: int = 30) -> Dict:
        """
        Proactively scale infrastructure based on predicted load
        """
        try:
            # Get current auto scaling configuration
            asg_name = f"adops-{editor_id}-asg"

            response = self.autoscaling.describe_auto_scaling_groups(
                AutoScalingGroupNames=[asg_name]
            )

            if not response['AutoScalingGroups']:
                return {'error': f'Auto Scaling Group {asg_name} not found'}

            current_asg = response['AutoScalingGroups'][0]
            current_capacity = current_asg['DesiredCapacity']
            current_max = current_asg['MaxSize']

            # Calculate required capacity based on predicted load
            scaling_factor = 1 + (predicted_load_increase / 100)
            required_capacity = int(current_capacity * scaling_factor)

            # Ensure we don't exceed max capacity
            new_capacity = min(required_capacity, current_max)

            if new_capacity > current_capacity:
                # Scale up proactively
                self.autoscaling.set_desired_capacity(
                    AutoScalingGroupName=asg_name,
                    DesiredCapacity=new_capacity,
                    HonorCooldown=False  # Override cooldown for predictive scaling
                )

                # Schedule scale-down if this is temporary
                if time_horizon_minutes < 120:  # Less than 2 hours
                    self._schedule_scale_down(
                        asg_name, current_capacity, time_horizon_minutes + 30
                    )

                return {
                    'success': True,
                    'action': 'proactive_scale_up',
                    'editor_id': editor_id,
                    'asg_name': asg_name,
                    'old_capacity': current_capacity,
                    'new_capacity': new_capacity,
                    'predicted_load_increase': predicted_load_increase,
                    'scheduled_scale_down': time_horizon_minutes < 120
                }
            else:
                return {
                    'success': True,
                    'action': 'no_scaling_needed',
                    'message': 'Current capacity sufficient for predicted load'
                }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def circuit_breaker_management(self, service_name: str,
                                 failure_threshold: float = 50.0,
                                 action: str = 'activate') -> Dict:
        """
        Manage circuit breakers for failing services
        """
        try:
            # Update circuit breaker configuration in Parameter Store
            parameter_name = f"/adops/circuit-breakers/{service_name}"

            if action == 'activate':
                config = {
                    'enabled': True,
                    'failure_threshold': failure_threshold,
                    'timeout_duration': 30000,  # 30 seconds
                    'activated_at': datetime.utcnow().isoformat(),
                    'auto_recovery': True,
                    'recovery_timeout': 300000  # 5 minutes
                }
            else:  # deactivate
                config = {
                    'enabled': False,
                    'deactivated_at': datetime.utcnow().isoformat()
                }

            self.ssm.put_parameter(
                Name=parameter_name,
                Value=json.dumps(config),
                Type='String',
                Overwrite=True,
                Tags=[
                    {'Key': 'Service', 'Value': service_name},
                    {'Key': 'Component', 'Value': 'CircuitBreaker'},
                    {'Key': 'Environment', 'Value': 'production'}
                ]
            )

            return {
                'success': True,
                'action': f'circuit_breaker_{action}',
                'service_name': service_name,
                'configuration': config,
                'parameter_name': parameter_name
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def intelligent_cache_management(self, editor_id: str,
                                   cache_type: str = 'adaptive') -> Dict:
        """
        Intelligent cache warming and invalidation
        """
        try:
            if cache_type == 'adaptive':
                # Analyze cache hit rates and warm frequently accessed content
                cache_stats = self._analyze_cache_performance(editor_id)

                if cache_stats['hit_rate'] < 0.8:  # Below 80% hit rate
                    # Identify popular content for warming
                    popular_content = self._identify_popular_content(editor_id)

                    # Warm cache with popular content
                    warming_result = self._warm_cache(editor_id, popular_content)

                    return {
                        'success': True,
                        'action': 'adaptive_cache_warming',
                        'editor_id': editor_id,
                        'current_hit_rate': cache_stats['hit_rate'],
                        'content_warmed': len(popular_content),
                        'warming_result': warming_result
                    }
                else:
                    return {
                        'success': True,
                        'action': 'no_warming_needed',
                        'message': f"Cache hit rate {cache_stats['hit_rate']:.2%} is satisfactory"
                    }

            elif cache_type == 'invalidate':
                # Intelligent cache invalidation
                invalidation_result = self._selective_cache_invalidation(editor_id)

                return {
                    'success': True,
                    'action': 'selective_cache_invalidation',
                    'editor_id': editor_id,
                    'invalidation_result': invalidation_result
                }

        except Exception as e:
            return {'success': False, 'error': str(e)}
```

### Arquitectura MLOps para Mejora Continua

#### 1. Pipeline de Entrenamiento SageMaker

```yaml
SageMaker Pipeline Configuration:
  Pipeline:
    Name: adops-ml-training-pipeline
    Description: "Continuous training pipeline for AdOps ML models"

    Parameters:
      - Name: InputDataPath
        Type: String
        DefaultValue: s3://adops-data-lake/curated/training-data/

      - Name: ModelApprovalStatus
        Type: String
        DefaultValue: PendingManualApproval

    Steps:
      - Name: DataPreprocessing
        Type: Processing
        ProcessingStep:
          ProcessorArgs:
            ProcessingJobName: adops-data-preprocessing
            ProcessingInputs:
              - InputName: raw-data
                S3Input:
                  S3Uri: !Ref InputDataPath
                  LocalPath: /opt/ml/processing/input
            ProcessingOutputs:
              - OutputName: preprocessed-data
                S3Output:
                  S3Uri: s3://adops-ml-artifacts/preprocessed/
                  LocalPath: /opt/ml/processing/output
            AppSpecification:
              ImageUri: 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker
              ContainerEntrypoint: ["python3", "preprocessing.py"]

      - Name: ModelTraining
        Type: Training
        TrainingStep:
          TrainingJobName: adops-anomaly-detection-training
          AlgorithmSpecification:
            TrainingImage: 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker
            TrainingInputMode: File
          InputDataConfig:
            - ChannelName: training
              DataSource:
                S3DataSource:
                  S3DataType: S3Prefix
                  S3Uri: !Join ['/', [!GetAtt DataPreprocessing.ProcessingOutputConfig.Outputs.preprocessed-data.S3Output.S3Uri, 'train']]
            - ChannelName: validation
              DataSource:
                S3DataSource:
                  S3DataType: S3Prefix
                  S3Uri: !Join ['/', [!GetAtt DataPreprocessing.ProcessingOutputConfig.Outputs.preprocessed-data.S3Output.S3Uri, 'validation']]
          OutputDataConfig:
            S3OutputPath: s3://adops-ml-artifacts/models/
          ResourceConfig:
            InstanceType: ml.g4dn.xlarge
            InstanceCount: 1
            VolumeSizeInGB: 30
          StoppingCondition:
            MaxRuntimeInSeconds: 7200
          HyperParameters:
            epochs: 50
            batch-size: 32
            learning-rate: 0.001

      - Name: ModelEvaluation
        Type: Processing
        ProcessingStep:
          ProcessorArgs:
            ProcessingJobName: adops-model-evaluation
            ProcessingInputs:
              - InputName: model
                S3Input:
                  S3Uri: !GetAtt ModelTraining.ModelArtifacts.S3ModelArtifacts
                  LocalPath: /opt/ml/processing/model
              - InputName: test-data
                S3Input:
                  S3Uri: !Join ['/', [!GetAtt DataPreprocessing.ProcessingOutputConfig.Outputs.preprocessed-data.S3Output.S3Uri, 'test']]
                  LocalPath: /opt/ml/processing/test
            ProcessingOutputs:
              - OutputName: evaluation-report
                S3Output:
                  S3Uri: s3://adops-ml-artifacts/evaluation/
                  LocalPath: /opt/ml/processing/evaluation
            AppSpecification:
              ImageUri: 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker
              ContainerEntrypoint: ["python3", "evaluation.py"]

      - Name: ModelApprovalCondition
        Type: Condition
        ConditionStep:
          Conditions:
            - Type: ConditionGreaterThanOrEqualTo
              LeftValue: !GetAtt ModelEvaluation.ProcessingOutputConfig.Outputs.evaluation-report.OutputValue.accuracy
              RightValue: 0.85  # Minimum 85% accuracy
          IfSteps:
            - Name: RegisterModel
              Type: RegisterModel
              RegisterModelStep:
                ModelName: adops-anomaly-detection
                ModelPackageGroupName: adops-models
                ModelApprovalStatus: !Ref ModelApprovalStatus
                ModelMetrics:
                  ModelQuality:
                    Statistics:
                      S3Uri: !Join ['/', [!GetAtt ModelEvaluation.ProcessingOutputConfig.Outputs.evaluation-report.S3Output.S3Uri, 'statistics.json']]
                InferenceSpecification:
                  Containers:
                    - Image: 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker
                      ModelDataUrl: !GetAtt ModelTraining.ModelArtifacts.S3ModelArtifacts
          ElseSteps:
            - Name: ModelRejection
              Type: Processing
              ProcessingStep:
                ProcessorArgs:
                  ProcessingJobName: model-rejection-notification
                  AppSpecification:
                    ImageUri: 763104351884.dkr.ecr.us-east-1.amazonaws.com/python:3.9-slim
                    ContainerEntrypoint: ["python3", "-c", "import boto3; sns = boto3.client('sns'); sns.publish(TopicArn='arn:aws:sns:us-east-1:account:model-alerts', Message='Model training failed quality checks')"]
```

#### 2. Continuous Deployment Pipeline

```yaml
CodePipeline Configuration:
  Pipeline:
    Name: adops-agent-deployment-pipeline
    RoleArn: arn:aws:iam::account:role/CodePipelineServiceRole

    Stages:
      - Name: Source
        Actions:
          - Name: SourceAction
            ActionTypeId:
              Category: Source
              Owner: AWS
              Provider: CodeCommit
              Version: 1
            Configuration:
              RepositoryName: adops-agent-code
              BranchName: main
            OutputArtifacts:
              - Name: SourceOutput

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
            InputArtifacts:
              - Name: SourceOutput
            OutputArtifacts:
              - Name: BuildOutput

      - Name: TestStaging
        Actions:
          - Name: DeployToStaging
            ActionTypeId:
              Category: Deploy
              Owner: AWS
              Provider: CloudFormation
              Version: 1
            Configuration:
              ActionMode: CREATE_UPDATE
              StackName: adops-agent-staging
              TemplatePath: BuildOutput::packaged-template.yaml
              Capabilities: CAPABILITY_IAM
              ParameterOverrides: |
                {
                  "Environment": "staging",
                  "AgentMode": "testing"
                }

          - Name: IntegrationTests
            ActionTypeId:
              Category: Test
              Owner: AWS
              Provider: CodeBuild
              Version: 1
            Configuration:
              ProjectName: adops-integration-tests
            InputArtifacts:
              - Name: BuildOutput
            RunOrder: 2

      - Name: ProductionApproval
        Actions:
          - Name: ManualApproval
            ActionTypeId:
              Category: Approval
              Owner: AWS
              Provider: Manual
              Version: 1
            Configuration:
              NotificationArn: arn:aws:sns:us-east-1:account:deployment-approvals
              CustomData: "Please review staging test results before production deployment"

      - Name: ProductionDeploy
        Actions:
          - Name: DeployToProduction
            ActionTypeId:
              Category: Deploy
              Owner: AWS
              Provider: CloudFormation
              Version: 1
            Configuration:
              ActionMode: CREATE_UPDATE
              StackName: adops-agent-production
              TemplatePath: BuildOutput::packaged-template.yaml
              Capabilities: CAPABILITY_IAM
              ParameterOverrides: |
                {
                  "Environment": "production",
                  "AgentMode": "active"
                }

          - Name: ProductionValidation
            ActionTypeId:
              Category: Test
              Owner: AWS
              Provider: CodeBuild
              Version: 1
            Configuration:
              ProjectName: adops-production-validation
            InputArtifacts:
              - Name: BuildOutput
            RunOrder: 2
```

## Monitorización y Observabilidad Avanzada

### 1. CloudWatch Dashboard Operacional

```yaml
CloudWatch Dashboard Production:
  Name: AdOps-Production-Operations

  Widgets:
    - RealTimeMetrics:
        Type: Line
        Width: 12
        Height: 6
        Properties:
          Metrics:
            - [SSMAS/AdOps, RPM, EditorId, ALL]
            - [., FillRate, ., .]
            - [., BidLatency, ., .]
          Period: 300
          Stat: Average
          Region: us-east-1
          Title: "Real-time Performance Metrics"
          YAxis:
            Left:
              Min: 0

    - AgentActivity:
        Type: Number
        Width: 6
        Height: 3
        Properties:
          Metrics:
            - [AWS/Bedrock, InvocationsCount, AgentId, adops-orchestrator-agent]
          Period: 3600
          Stat: Sum
          Title: "Agent Activations (Last Hour)"

    - ResolutionEfficiency:
        Type: Pie
        Width: 6
        Height: 3
        Properties:
          Metrics:
            - Expression: "SEARCH('{SSMAS/AdOps} resolution_status=\"resolved\"', 'Sum')"
            - Expression: "SEARCH('{SSMAS/AdOps} resolution_status=\"escalated\"', 'Sum')"
            - Expression: "SEARCH('{SSMAS/AdOps} resolution_status=\"failed\"', 'Sum')"
          Title: "Resolution Distribution (24h)"

    - SystemHealth:
        Type: SingleValue
        Width: 12
        Height: 3
        Properties:
          Metrics:
            - [SSMAS/AdOps, SystemHealthScore]
          Period: 300
          Stat: Average
          Title: "Overall System Health"
          Color: "Green"

    - CostTracking:
        Type: Line
        Width: 6
        Height: 4
        Properties:
          Metrics:
            - [AWS/Billing, EstimatedCharges, ServiceName, AmazonBedrock]
            - [., ., ., AWSLambda]
            - [., ., ., AmazonDynamoDB]
          Period: 86400
          Stat: Maximum
          Title: "Daily Cost Tracking"

    - ErrorRates:
        Type: Line
        Width: 6
        Height: 4
        Properties:
          Metrics:
            - [AWS/Lambda, Errors, FunctionName, adops-diagnostic-tools]
            - [., ., ., adops-remediation-tools]
            - [AWS/Bedrock, ModelInvocationErrors, ModelId, anthropic.claude-3-sonnet]
          Period: 300
          Stat: Sum
          Title: "Error Rates by Component"
```

### 2. Alerting Inteligente

```yaml
Intelligent Alerting Configuration:
  CompositeAlarms:
    - Name: SystemDegradation
      Description: "Multi-dimensional system degradation detection"
      AlarmRule: |
        (ALARM("RPMDropAlarm") OR ALARM("FillRateDropAlarm"))
        AND ALARM("HighLatencyAlarm")
      ActionsEnabled: true
      AlarmActions:
        - arn:aws:sns:us-east-1:account:critical-incidents
      TreatMissingData: notBreaching

    - Name: AgentMalfunction
      Description: "Agent performance degradation"
      AlarmRule: |
        ALARM("HighAgentLatency") AND
        (ALARM("LowResolutionRate") OR ALARM("HighEscalationRate"))
      ActionsEnabled: true
      AlarmActions:
        - arn:aws:sns:us-east-1:account:agent-alerts

  AnomalyDetectors:
    - MetricName: RPM
      Namespace: SSMAS/AdOps
      Dimensions:
        - Name: EditorId
          Value: "*"
      Stat: Average
      DetectorConfiguration:
        MetricMathAnomalyDetector:
          MetricDataQueries:
            - Id: m1
              MetricStat:
                Metric:
                  MetricName: RPM
                  Namespace: SSMAS/AdOps
                Period: 300
                Stat: Average

  CustomMetrics:
    - Name: MTTR
      Unit: Seconds
      Description: "Mean Time To Resolution"

    - Name: AutomationRate
      Unit: Percent
      Description: "Percentage of incidents resolved automatically"

    - Name: CostPerIncident
      Unit: None
      Description: "Cost per incident resolution"
```

## Conclusión de la Fase 2

La Fase 2 establece una arquitectura de producción robusta que amplía significativamente las capacidades del MVP. Con múltiples agentes especializados, herramientas avanzadas de diagnóstico y remediación, y pipelines MLOps completos, el sistema está preparado para manejar las operaciones complejas de AdTech a escala empresarial.

### Logros Esperados:
- **Cobertura Completa**: 90%+ de problemas operativos manejados automáticamente
- **Múltiples Tipos de Problemas**: 5-7 categorías diferentes de issues
- **Escalabilidad**: Soporte para todos los editores SSMAS
- **Mejora Continua**: Pipelines MLOps para evolución automática

### Preparación para Fase 3:
- Base sólida para capacidades predictivas avanzadas
- Arquitectura extensible para nuevos casos de uso
- Datos históricos ricos para entrenamiento de modelos
- Patrones establecidos para automatización completa