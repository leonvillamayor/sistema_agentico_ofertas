# Caso de Uso 1: Yield Predictivo - Fase III: Optimización Continua y Evolución

## Introducción a la Fase III

La Fase III del proyecto Yield Predictivo de SSMAS representa la evolución del sistema hacia una plataforma de optimización publicitaria autoadaptativa e inteligente. Tras el éxito de la operativización en la Fase II, esta fase se centra en la mejora continua, optimización avanzada y la expansión de capacidades que mantengan a SSMAS a la vanguardia tecnológica del sector AdTech.

Esta fase implementa un ecosistema de machine learning que no solo predice, sino que aprende y evoluciona continuamente, incorporando técnicas avanzadas como AutoML, optimización multi-objetivo, y capacidades de explicabilidad que permiten insights profundos sobre el comportamiento del mercado publicitario.

## Objetivos de la Fase III

### Objetivo Principal
Transformar el sistema Yield Predictivo en una plataforma de optimización publicitaria autoadaptativa que mejore continuamente su rendimiento, incorpore nuevas técnicas de machine learning, y proporcione insights avanzados para decisiones estratégicas de negocio.

### Objetivos Específicos

1. **Optimización Autónoma**: Implementar AutoML para optimización continua de hiperparámetros y arquitectura de modelos
2. **Multi-Objective Optimization**: Balancear múltiples métricas (CPM, Fill Rate, Revenue Total) simultáneamente
3. **Explicabilidad e Insights**: Desarrollar capacidades de interpretabilidad para insights de negocio
4. **Expansion de Features**: Incorporar datos externos y señales de mercado avanzadas
5. **Advanced Analytics**: Implementar análisis predictivo de tendencias y forecasting de mercado
6. **Cost Optimization**: Optimizar costos de infraestructura y operación continua

## Arquitectura Avanzada de ML

### AutoML Pipeline

#### Amazon SageMaker Autopilot Integration

**Configuración de AutoML**:
```python
import boto3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

class AutoMLManager:
    """Gestión automatizada de experimentos de ML y optimización de modelos"""

    def __init__(self, region='eu-west-1'):
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.bucket = 'ssmas-yield-prod-datalake-eu-west-1'

    def launch_autopilot_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """Lanza experimento de AutoML con SageMaker Autopilot"""

        experiment_name = f"yield-predictor-automl-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        autopilot_config = {
            'AutoMLJobName': experiment_name,
            'InputDataConfig': [
                {
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f's3://{self.bucket}/processed/automl_training/',
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'TargetAttributeName': 'cpm',
                    'ContentType': 'text/csv',
                    'CompressionType': 'None'
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f's3://{self.bucket}/automl_output/{experiment_name}/'
            },
            'RoleArn': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole',
            'AutoMLJobObjective': {
                'MetricName': 'MSE'  # Mean Squared Error para regresión
            },
            'AutoMLJobConfig': {
                'CompletionCriteria': {
                    'MaxCandidates': 50,
                    'MaxRuntimePerTrainingJobInSeconds': 3600,
                    'MaxAutoMLJobRuntimeInSeconds': 28800  # 8 horas
                },
                'SecurityConfig': {
                    'VolumeKmsKeyId': 'alias/ssmas-yield-predictor-master',
                    'EnableInterContainerTrafficEncryption': True,
                    'VpcConfig': {
                        'SecurityGroupIds': ['sg-12345678'],
                        'Subnets': ['subnet-12345678', 'subnet-87654321']
                    }
                },
                'DataSplitConfig': {
                    'ValidationFraction': 0.2
                }
            },
            'Tags': [
                {'Key': 'Project', 'Value': 'YieldPredictor'},
                {'Key': 'Environment', 'Value': 'Production'},
                {'Key': 'Phase', 'Value': 'AutoML'},
                {'Key': 'Purpose', 'Value': 'ContinuousOptimization'}
            ]
        }

        # Lanzar experimento
        response = self.sagemaker.create_auto_ml_job(**autopilot_config)

        print(f"AutoML experiment launched: {experiment_name}")
        return experiment_name

    def monitor_autopilot_experiment(self, job_name: str) -> Dict[str, Any]:
        """Monitorea el progreso del experimento AutoML"""

        response = self.sagemaker.describe_auto_ml_job(AutoMLJobName=job_name)

        status = response['AutoMLJobStatus']

        if status == 'Completed':
            # Obtener el mejor modelo
            best_candidate = response['BestCandidate']

            metrics = {
                'best_candidate_name': best_candidate['CandidateName'],
                'objective_metric_value': best_candidate['FinalAutoMLJobObjectiveMetric']['Value'],
                'algorithm': best_candidate['InferenceContainers'][0]['Image'].split('/')[-1],
                'training_time': response.get('CreationTime'),
                'completion_time': response.get('EndTime')
            }

            return {
                'status': status,
                'metrics': metrics,
                'model_uri': best_candidate['InferenceContainers'][0]['ModelDataUrl']
            }

        return {
            'status': status,
            'progress': response.get('AutoMLJobSecondaryStatus', 'Unknown')
        }

    def compare_with_current_model(self, automl_model_uri: str) -> Dict[str, Any]:
        """Compara el modelo AutoML con el modelo actual en producción"""

        # Configurar endpoint de prueba para el nuevo modelo
        test_endpoint_name = f"yield-predictor-automl-test-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Crear configuración de endpoint para el modelo AutoML
        endpoint_config = {
            'EndpointConfigName': f"{test_endpoint_name}-config",
            'ProductionVariants': [
                {
                    'VariantName': 'automl-variant',
                    'ModelName': f"{test_endpoint_name}-model",
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large',
                    'InitialVariantWeight': 100
                }
            ]
        }

        # Registrar modelo AutoML
        model_config = {
            'ModelName': f"{test_endpoint_name}-model",
            'PrimaryContainer': {
                'Image': '246618743249.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
                'ModelDataUrl': automl_model_uri,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                }
            },
            'ExecutionRoleArn': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'
        }

        try:
            # Crear modelo
            self.sagemaker.create_model(**model_config)

            # Crear configuración de endpoint
            self.sagemaker.create_endpoint_config(**endpoint_config)

            # Crear endpoint de prueba
            self.sagemaker.create_endpoint(
                EndpointName=test_endpoint_name,
                EndpointConfigName=endpoint_config['EndpointConfigName']
            )

            # Esperar a que el endpoint esté listo
            waiter = self.sagemaker.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=test_endpoint_name)

            # Ejecutar comparación de performance
            comparison_results = self._run_model_comparison(test_endpoint_name)

            # Limpiar endpoint de prueba
            self._cleanup_test_endpoint(test_endpoint_name)

            return comparison_results

        except Exception as e:
            print(f"Error during model comparison: {str(e)}")
            self._cleanup_test_endpoint(test_endpoint_name)
            raise

    def _run_model_comparison(self, test_endpoint_name: str) -> Dict[str, Any]:
        """Ejecuta comparación entre modelo AutoML y modelo actual"""

        # Cargar datos de test
        test_data = self._load_test_dataset()

        # Realizar predicciones con ambos modelos
        automl_predictions = self._predict_batch(test_endpoint_name, test_data)
        current_predictions = self._predict_batch('yield-predictor-multi-model', test_data)

        # Calcular métricas de comparación
        automl_metrics = self._calculate_metrics(test_data['true_values'], automl_predictions)
        current_metrics = self._calculate_metrics(test_data['true_values'], current_predictions)

        # Comparación de rendimiento
        improvement = {
            'rmse_improvement': (current_metrics['rmse'] - automl_metrics['rmse']) / current_metrics['rmse'],
            'mae_improvement': (current_metrics['mae'] - automl_metrics['mae']) / current_metrics['mae'],
            'r2_improvement': automl_metrics['r2'] - current_metrics['r2']
        }

        return {
            'automl_metrics': automl_metrics,
            'current_metrics': current_metrics,
            'improvement': improvement,
            'recommendation': self._generate_recommendation(improvement)
        }

    def _generate_recommendation(self, improvement: Dict[str, float]) -> str:
        """Genera recomendación basada en mejoras de métricas"""

        rmse_threshold = 0.05  # 5% mejora mínima
        overall_improvement = (
            improvement['rmse_improvement'] +
            improvement['mae_improvement'] +
            improvement['r2_improvement']
        ) / 3

        if overall_improvement > rmse_threshold:
            return "DEPLOY - Significant improvement detected"
        elif overall_improvement > 0:
            return "CONSIDER - Marginal improvement, evaluate business impact"
        else:
            return "REJECT - No significant improvement"
```

#### Multi-Objective Optimization

**Implementación de Optimización Multi-Objetivo**:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict

class MultiObjectiveOptimizer:
    """Optimizador multi-objetivo para balancear CPM, Fill Rate y Revenue Total"""

    def __init__(self):
        self.cpm_model = None
        self.fill_rate_model = None
        self.revenue_model = None
        self.pareto_frontier = None

    def train_multi_objective_models(self, training_data: Dict[str, np.ndarray]):
        """Entrena modelos separados para cada objetivo"""

        X = training_data['features']
        y_cpm = training_data['cpm']
        y_fill_rate = training_data['fill_rate']
        y_revenue = training_data['revenue']

        # Modelo para CPM
        self.cpm_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.cpm_model.fit(X, y_cpm)

        # Modelo para Fill Rate
        self.fill_rate_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.fill_rate_model.fit(X, y_fill_rate)

        # Modelo para Revenue Total
        self.revenue_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.revenue_model.fit(X, y_revenue)

        print("Multi-objective models trained successfully")

    def predict_all_objectives(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Predice todos los objetivos para las features dadas"""

        predictions = {
            'cpm': self.cpm_model.predict(features),
            'fill_rate': self.fill_rate_model.predict(features),
            'revenue': self.revenue_model.predict(features)
        }

        return predictions

    def find_pareto_optimal_prices(self,
                                   features: np.ndarray,
                                   price_range: Tuple[float, float] = (0.5, 10.0),
                                   num_prices: int = 100) -> Dict[str, Any]:
        """Encuentra precios Pareto-óptimos para balancear objetivos"""

        price_candidates = np.linspace(price_range[0], price_range[1], num_prices)
        pareto_solutions = []

        for price in price_candidates:
            # Añadir precio como feature adicional
            features_with_price = np.column_stack([features, np.full(len(features), price)])

            # Predecir objetivos
            predictions = self.predict_all_objectives(features_with_price)

            # Calcular métricas agregadas
            avg_cpm = np.mean(predictions['cpm'])
            avg_fill_rate = np.mean(predictions['fill_rate'])
            total_revenue = np.sum(predictions['revenue'])

            solution = {
                'price': price,
                'cpm': avg_cpm,
                'fill_rate': avg_fill_rate,
                'revenue': total_revenue,
                'utility_score': self._calculate_utility_score(avg_cpm, avg_fill_rate, total_revenue)
            }

            pareto_solutions.append(solution)

        # Filtrar soluciones Pareto-óptimas
        pareto_optimal = self._filter_pareto_optimal(pareto_solutions)

        return {
            'pareto_solutions': pareto_optimal,
            'recommended_price': self._select_best_solution(pareto_optimal),
            'trade_offs': self._analyze_trade_offs(pareto_optimal)
        }

    def _calculate_utility_score(self, cpm: float, fill_rate: float, revenue: float) -> float:
        """Calcula score de utilidad combinando múltiples objetivos"""

        # Pesos configurables para cada objetivo
        weights = {
            'cpm': 0.3,
            'fill_rate': 0.3,
            'revenue': 0.4
        }

        # Normalización aproximada (basada en rangos típicos)
        normalized_cpm = min(cpm / 5.0, 1.0)  # CPM máximo esperado: 5 EUR
        normalized_fill_rate = fill_rate  # Ya está en [0,1]
        normalized_revenue = min(revenue / 10000, 1.0)  # Revenue máximo esperado

        utility = (
            weights['cpm'] * normalized_cpm +
            weights['fill_rate'] * normalized_fill_rate +
            weights['revenue'] * normalized_revenue
        )

        return utility

    def _filter_pareto_optimal(self, solutions: List[Dict]) -> List[Dict]:
        """Filtra soluciones que están en la frontera de Pareto"""

        pareto_optimal = []

        for i, solution_a in enumerate(solutions):
            is_dominated = False

            for j, solution_b in enumerate(solutions):
                if i != j:
                    # Verificar si solution_b domina a solution_a
                    if (solution_b['cpm'] >= solution_a['cpm'] and
                        solution_b['fill_rate'] >= solution_a['fill_rate'] and
                        solution_b['revenue'] >= solution_a['revenue'] and
                        (solution_b['cpm'] > solution_a['cpm'] or
                         solution_b['fill_rate'] > solution_a['fill_rate'] or
                         solution_b['revenue'] > solution_a['revenue'])):
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_optimal.append(solution_a)

        return sorted(pareto_optimal, key=lambda x: x['utility_score'], reverse=True)

    def _select_best_solution(self, pareto_solutions: List[Dict]) -> Dict[str, Any]:
        """Selecciona la mejor solución basada en utility score"""

        if not pareto_solutions:
            return None

        best_solution = max(pareto_solutions, key=lambda x: x['utility_score'])
        return best_solution

    def _analyze_trade_offs(self, pareto_solutions: List[Dict]) -> Dict[str, Any]:
        """Analiza trade-offs entre objetivos en las soluciones Pareto-óptimas"""

        if len(pareto_solutions) < 2:
            return {"analysis": "Insufficient solutions for trade-off analysis"}

        # Calcular correlaciones entre objetivos
        cpms = [sol['cpm'] for sol in pareto_solutions]
        fill_rates = [sol['fill_rate'] for sol in pareto_solutions]
        revenues = [sol['revenue'] for sol in pareto_solutions]

        cpm_fillrate_corr = np.corrcoef(cpms, fill_rates)[0, 1]
        cpm_revenue_corr = np.corrcoef(cpms, revenues)[0, 1]
        fillrate_revenue_corr = np.corrcoef(fill_rates, revenues)[0, 1]

        return {
            'cpm_fillrate_correlation': cpm_fillrate_corr,
            'cpm_revenue_correlation': cpm_revenue_corr,
            'fillrate_revenue_correlation': fillrate_revenue_corr,
            'trade_off_analysis': {
                'high_cpm_low_fillrate': cpm_fillrate_corr < -0.5,
                'high_cpm_high_revenue': cpm_revenue_corr > 0.5,
                'balanced_optimization_possible': abs(cpm_fillrate_corr) < 0.3
            }
        }
```

### Explicabilidad e Insights

#### SHAP Integration para Interpretabilidad

**Implementación de SHAP para Explicaciones de Modelo**:
```python
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import boto3
import json
from typing import Dict, List, Any, Optional

class ModelExplainabilityEngine:
    """Motor de explicabilidad para insights de modelo y decisiones de negocio"""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.s3_client = boto3.client('s3')
        self.bucket = 'ssmas-yield-prod-datalake-eu-west-1'

    def initialize_explainer(self, background_data: np.ndarray):
        """Inicializa el explainer SHAP con datos de fondo"""

        # Usar TreeExplainer para modelos basados en árboles (XGBoost)
        self.explainer = shap.TreeExplainer(self.model)

        # Alternativamente, usar KernelExplainer para modelos más complejos
        # self.explainer = shap.KernelExplainer(
        #     self.model.predict,
        #     background_data[:100]  # Muestra pequeña para eficiencia
        # )

        print("SHAP explainer initialized successfully")

    def explain_predictions(self, X: np.ndarray, save_results: bool = True) -> Dict[str, Any]:
        """Genera explicaciones SHAP para las predicciones"""

        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")

        # Calcular valores SHAP
        self.shap_values = self.explainer.shap_values(X)

        # Crear DataFrame para análisis
        explanations_df = pd.DataFrame(
            self.shap_values,
            columns=self.feature_names
        )

        # Análisis de importancia de features
        feature_importance = self._calculate_feature_importance()

        # Análisis de interacciones
        feature_interactions = self._analyze_feature_interactions(X)

        # Insights de negocio
        business_insights = self._generate_business_insights(explanations_df)

        results = {
            'feature_importance': feature_importance,
            'feature_interactions': feature_interactions,
            'business_insights': business_insights,
            'shap_values': self.shap_values.tolist(),
            'predictions': self.model.predict(X).tolist()
        }

        if save_results:
            self._save_explanations_to_s3(results)

        return results

    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calcula importancia de features basada en valores SHAP"""

        if self.shap_values is None:
            raise ValueError("SHAP values not calculated")

        # Importancia global (promedio de valores absolutos)
        importance_scores = np.mean(np.abs(self.shap_values), axis=0)

        feature_importance = {
            feature: float(score)
            for feature, score in zip(self.feature_names, importance_scores)
        }

        # Ordenar por importancia
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance

    def _analyze_feature_interactions(self, X: np.ndarray) -> Dict[str, Any]:
        """Analiza interacciones entre features importantes"""

        # Seleccionar top 5 features más importantes
        importance = self._calculate_feature_importance()
        top_features = list(importance.keys())[:5]
        top_indices = [self.feature_names.index(f) for f in top_features]

        interactions = {}

        # Calcular interacciones por pares
        for i, feature1 in enumerate(top_features):
            for j, feature2 in enumerate(top_features[i+1:], i+1):
                interaction_strength = self._calculate_interaction_strength(
                    X, top_indices[i], top_indices[j]
                )
                interactions[f"{feature1}_x_{feature2}"] = interaction_strength

        return {
            'top_features': top_features,
            'pairwise_interactions': interactions,
            'strongest_interaction': max(interactions.items(), key=lambda x: abs(x[1]))
        }

    def _calculate_interaction_strength(self, X: np.ndarray,
                                       feature1_idx: int,
                                       feature2_idx: int) -> float:
        """Calcula fuerza de interacción entre dos features"""

        # Crear variaciones de los features
        X_modified = X.copy()

        # Variar feature1 manteniendo feature2 constante
        feature1_variations = np.linspace(
            X[:, feature1_idx].min(),
            X[:, feature1_idx].max(),
            10
        )

        interaction_effects = []

        for variation in feature1_variations:
            X_temp = X_modified.copy()
            X_temp[:, feature1_idx] = variation

            # Predecir con feature1 variado
            pred_varied = self.model.predict(X_temp)

            # Calcular efecto de interacción
            interaction_effect = np.std(pred_varied)
            interaction_effects.append(interaction_effect)

        # Retornar varianza del efecto de interacción
        return float(np.var(interaction_effects))

    def _generate_business_insights(self, explanations_df: pd.DataFrame) -> Dict[str, Any]:
        """Genera insights de negocio basados en explicaciones SHAP"""

        insights = {}

        # 1. Análisis temporal
        temporal_features = ['hour_of_day', 'day_of_week', 'is_weekend']
        temporal_insights = {}

        for feature in temporal_features:
            if feature in explanations_df.columns:
                avg_impact = explanations_df[feature].mean()
                temporal_insights[feature] = {
                    'average_impact': float(avg_impact),
                    'interpretation': self._interpret_temporal_impact(feature, avg_impact)
                }

        insights['temporal_patterns'] = temporal_insights

        # 2. Análisis geográfico
        geo_features = ['country_tier', 'is_eu']
        geo_insights = {}

        for feature in geo_features:
            if feature in explanations_df.columns:
                avg_impact = explanations_df[feature].mean()
                geo_insights[feature] = {
                    'average_impact': float(avg_impact),
                    'interpretation': self._interpret_geo_impact(feature, avg_impact)
                }

        insights['geographic_patterns'] = geo_insights

        # 3. Análisis de dispositivo
        device_features = ['device_category_encoded', 'device_performance_score']
        device_insights = {}

        for feature in device_features:
            if feature in explanations_df.columns:
                avg_impact = explanations_df[feature].mean()
                device_insights[feature] = {
                    'average_impact': float(avg_impact),
                    'interpretation': self._interpret_device_impact(feature, avg_impact)
                }

        insights['device_patterns'] = device_insights

        # 4. Recomendaciones estratégicas
        insights['strategic_recommendations'] = self._generate_strategic_recommendations(
            temporal_insights, geo_insights, device_insights
        )

        return insights

    def _interpret_temporal_impact(self, feature: str, impact: float) -> str:
        """Interpreta el impacto temporal en términos de negocio"""

        if feature == 'hour_of_day':
            if impact > 0.1:
                return "Prime time hours significantly increase CPM predictions"
            elif impact < -0.1:
                return "Off-peak hours reduce CPM predictions"
            else:
                return "Hour of day has moderate impact on CPM"

        elif feature == 'day_of_week':
            if impact > 0.05:
                return "Weekdays show higher CPM potential"
            elif impact < -0.05:
                return "Weekends show lower CPM potential"
            else:
                return "Day of week has balanced impact on CPM"

        elif feature == 'is_weekend':
            if impact > 0:
                return "Weekend periods increase CPM predictions"
            else:
                return "Weekend periods decrease CPM predictions"

        return "Temporal pattern identified"

    def _interpret_geo_impact(self, feature: str, impact: float) -> str:
        """Interpreta el impacto geográfico en términos de negocio"""

        if feature == 'country_tier':
            if impact > 0.2:
                return "Premium markets (Tier 1) drive significantly higher CPMs"
            elif impact < -0.2:
                return "Emerging markets show lower CPM potential"
            else:
                return "Geographic segmentation has moderate impact"

        elif feature == 'is_eu':
            if impact > 0:
                return "EU markets show premium CPM potential"
            else:
                return "Non-EU markets may offer volume opportunities"

        return "Geographic pattern identified"

    def _interpret_device_impact(self, feature: str, impact: float) -> str:
        """Interpreta el impacto de dispositivo en términos de negocio"""

        if feature == 'device_category_encoded':
            if impact > 0.15:
                return "Premium device categories command higher CPMs"
            elif impact < -0.15:
                return "Basic device categories show lower CPM potential"
            else:
                return "Device category has balanced impact on CPM"

        elif feature == 'device_performance_score':
            if impact > 0.1:
                return "High-performance devices increase CPM potential"
            else:
                return "Device performance has limited impact on CPM"

        return "Device pattern identified"

    def _generate_strategic_recommendations(self,
                                           temporal: Dict,
                                           geo: Dict,
                                           device: Dict) -> List[str]:
        """Genera recomendaciones estratégicas basadas en insights"""

        recommendations = []

        # Análisis temporal
        if 'hour_of_day' in temporal and temporal['hour_of_day']['average_impact'] > 0.1:
            recommendations.append(
                "Consider dynamic pricing strategies with premium rates during peak hours"
            )

        # Análisis geográfico
        if 'country_tier' in geo and geo['country_tier']['average_impact'] > 0.2:
            recommendations.append(
                "Focus premium inventory acquisition in Tier 1 markets for higher yields"
            )

        # Análisis de dispositivo
        if 'device_performance_score' in device and device['device_performance_score']['average_impact'] > 0.1:
            recommendations.append(
                "Target high-performance device users for premium ad placements"
            )

        # Recomendación general
        recommendations.append(
            "Implement feature-specific floor price adjustments based on SHAP insights"
        )

        return recommendations

    def generate_explanation_dashboard(self, X: np.ndarray, output_path: str = None):
        """Genera dashboard visual de explicaciones"""

        if self.shap_values is None:
            self.explain_predictions(X, save_results=False)

        # Crear visualizaciones SHAP
        plt.figure(figsize=(15, 10))

        # 1. Summary plot
        plt.subplot(2, 2, 1)
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, show=False)
        plt.title("Feature Importance Summary")

        # 2. Feature importance bar plot
        plt.subplot(2, 2, 2)
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names,
                         plot_type="bar", show=False)
        plt.title("Feature Importance Ranking")

        # 3. Waterfall plot para primera predicción
        plt.subplot(2, 2, 3)
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[0],
                base_values=self.explainer.expected_value,
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title("Prediction Explanation (Sample 1)")

        # 4. Dependence plot para feature más importante
        top_feature_idx = np.argmax(np.mean(np.abs(self.shap_values), axis=0))
        plt.subplot(2, 2, 4)
        shap.dependence_plot(
            top_feature_idx, self.shap_values, X,
            feature_names=self.feature_names, show=False
        )
        plt.title(f"Dependence Plot: {self.feature_names[top_feature_idx]}")

        plt.tight_layout()

        # Guardar dashboard
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Explanation dashboard saved to {output_path}")

        plt.show()

    def _save_explanations_to_s3(self, explanations: Dict[str, Any]):
        """Guarda explicaciones en S3 para análisis posterior"""

        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        key = f"model_explanations/shap_analysis_{timestamp}.json"

        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(explanations, indent=2),
                ContentType='application/json'
            )
            print(f"Explanations saved to s3://{self.bucket}/{key}")
        except Exception as e:
            print(f"Error saving explanations to S3: {str(e)}")
```

### Advanced Analytics y Forecasting

#### Market Trend Analysis

**Implementación de Análisis de Tendencias de Mercado**:
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import boto3
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

class MarketTrendAnalyzer:
    """Analizador de tendencias de mercado publicitario y forecasting"""

    def __init__(self, region='eu-west-1'):
        self.s3_client = boto3.client('s3', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.bucket = 'ssmas-yield-prod-datalake-eu-west-1'
        self.scaler = StandardScaler()

    def analyze_market_trends(self, lookback_days: int = 90) -> Dict[str, Any]:
        """Analiza tendencias de mercado en los últimos N días"""

        # Cargar datos históricos
        market_data = self._load_market_data(lookback_days)

        # Análisis de tendencias por dimensión
        trends_analysis = {
            'overall_market': self._analyze_overall_trends(market_data),
            'device_trends': self._analyze_device_trends(market_data),
            'geo_trends': self._analyze_geographic_trends(market_data),
            'temporal_trends': self._analyze_temporal_trends(market_data),
            'format_trends': self._analyze_format_trends(market_data)
        }

        # Forecasting
        forecasts = self._generate_forecasts(market_data)

        # Alertas de mercado
        market_alerts = self._detect_market_anomalies(market_data, trends_analysis)

        return {
            'analysis_period': {
                'start_date': (datetime.now() - timedelta(days=lookback_days)).isoformat(),
                'end_date': datetime.now().isoformat(),
                'days_analyzed': lookback_days
            },
            'trends_analysis': trends_analysis,
            'forecasts': forecasts,
            'market_alerts': market_alerts,
            'summary_insights': self._generate_summary_insights(trends_analysis, forecasts)
        }

    def _load_market_data(self, lookback_days: int) -> pd.DataFrame:
        """Carga datos de mercado de S3 para análisis"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Simular carga de datos (en implementación real, cargar de S3)
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')

        # Generar datos sintéticos para demo
        np.random.seed(42)
        n_hours = len(date_range)

        market_data = pd.DataFrame({
            'timestamp': date_range,
            'avg_cpm': 2.5 + 0.5 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) +
                      0.2 * np.random.normal(0, 1, n_hours),
            'impressions': 1000000 + 200000 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) +
                          50000 * np.random.normal(0, 1, n_hours),
            'fill_rate': 0.75 + 0.1 * np.sin(np.arange(n_hours) * 2 * np.pi / 24) +
                        0.05 * np.random.normal(0, 1, n_hours),
            'revenue': np.zeros(n_hours),  # Calculado después
            'device_mobile_share': 0.6 + 0.1 * np.random.normal(0, 1, n_hours),
            'device_desktop_share': 0.3 + 0.05 * np.random.normal(0, 1, n_hours),
            'tier1_countries_share': 0.4 + 0.05 * np.random.normal(0, 1, n_hours),
            'video_format_share': 0.25 + 0.05 * np.random.normal(0, 1, n_hours)
        })

        # Calcular revenue
        market_data['revenue'] = market_data['avg_cpm'] * market_data['impressions'] / 1000

        # Agregar features temporales
        market_data['hour'] = market_data['timestamp'].dt.hour
        market_data['day_of_week'] = market_data['timestamp'].dt.dayofweek
        market_data['is_weekend'] = market_data['day_of_week'].isin([5, 6])

        return market_data

    def _analyze_overall_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza tendencias generales del mercado"""

        # Agregar por día para análisis de tendencias
        daily_data = data.groupby(data['timestamp'].dt.date).agg({
            'avg_cpm': 'mean',
            'impressions': 'sum',
            'fill_rate': 'mean',
            'revenue': 'sum'
        }).reset_index()

        trends = {}

        for metric in ['avg_cpm', 'impressions', 'fill_rate', 'revenue']:
            values = daily_data[metric].values

            # Calcular tendencia (regresión lineal simple)
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)

            # Calcular porcentaje de cambio
            pct_change = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0

            # Volatilidad
            volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0

            trends[metric] = {
                'slope': float(slope),
                'percent_change': float(pct_change),
                'volatility': float(volatility),
                'current_value': float(values[-1]),
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            }

        return trends

    def _analyze_device_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza tendencias por tipo de dispositivo"""

        device_trends = {}

        # Analizar share de mobile vs desktop
        mobile_trend = self._calculate_trend(data['device_mobile_share'].values)
        desktop_trend = self._calculate_trend(data['device_desktop_share'].values)

        device_trends['mobile'] = {
            'current_share': float(data['device_mobile_share'].iloc[-1]),
            'trend': mobile_trend,
            'interpretation': self._interpret_device_trend('mobile', mobile_trend)
        }

        device_trends['desktop'] = {
            'current_share': float(data['device_desktop_share'].iloc[-1]),
            'trend': desktop_trend,
            'interpretation': self._interpret_device_trend('desktop', desktop_trend)
        }

        return device_trends

    def _analyze_geographic_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza tendencias geográficas"""

        geo_trends = {}

        tier1_trend = self._calculate_trend(data['tier1_countries_share'].values)

        geo_trends['tier1_countries'] = {
            'current_share': float(data['tier1_countries_share'].iloc[-1]),
            'trend': tier1_trend,
            'interpretation': self._interpret_geo_trend(tier1_trend)
        }

        return geo_trends

    def _analyze_temporal_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza patrones temporales"""

        temporal_trends = {}

        # Análisis por hora del día
        hourly_patterns = data.groupby('hour').agg({
            'avg_cpm': 'mean',
            'impressions': 'mean',
            'fill_rate': 'mean'
        })

        # Encontrar horas pico
        peak_cpm_hours = hourly_patterns['avg_cpm'].nlargest(3).index.tolist()
        peak_traffic_hours = hourly_patterns['impressions'].nlargest(3).index.tolist()

        # Análisis weekend vs weekday
        weekend_metrics = data[data['is_weekend']].agg({
            'avg_cpm': 'mean',
            'impressions': 'mean',
            'fill_rate': 'mean'
        })

        weekday_metrics = data[~data['is_weekend']].agg({
            'avg_cpm': 'mean',
            'impressions': 'mean',
            'fill_rate': 'mean'
        })

        temporal_trends = {
            'peak_cpm_hours': peak_cpm_hours,
            'peak_traffic_hours': peak_traffic_hours,
            'weekend_vs_weekday': {
                'cpm_ratio': float(weekend_metrics['avg_cpm'] / weekday_metrics['avg_cpm']),
                'traffic_ratio': float(weekend_metrics['impressions'] / weekday_metrics['impressions']),
                'fill_rate_ratio': float(weekend_metrics['fill_rate'] / weekday_metrics['fill_rate'])
            }
        }

        return temporal_trends

    def _analyze_format_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza tendencias por formato publicitario"""

        format_trends = {}

        video_trend = self._calculate_trend(data['video_format_share'].values)

        format_trends['video'] = {
            'current_share': float(data['video_format_share'].iloc[-1]),
            'trend': video_trend,
            'interpretation': self._interpret_format_trend('video', video_trend)
        }

        return format_trends

    def _calculate_trend(self, values: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de tendencia para una serie temporal"""

        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)

        pct_change = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
        volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0

        return {
            'slope': float(slope),
            'percent_change': float(pct_change),
            'volatility': float(volatility),
            'direction': 'increasing' if slope > 0.001 else 'decreasing' if slope < -0.001 else 'stable'
        }

    def _generate_forecasts(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Genera forecasts para métricas clave"""

        forecasts = {}

        # Preparar datos diarios para forecasting
        daily_data = data.groupby(data['timestamp'].dt.date).agg({
            'avg_cpm': 'mean',
            'impressions': 'sum',
            'revenue': 'sum'
        }).reset_index()

        # Forecast para próximos 7 días
        forecast_horizon = 7

        for metric in ['avg_cpm', 'impressions', 'revenue']:
            try:
                forecast = self._forecast_metric(daily_data[metric].values, forecast_horizon)
                forecasts[metric] = forecast
            except Exception as e:
                print(f"Error forecasting {metric}: {str(e)}")
                forecasts[metric] = {'error': str(e)}

        return forecasts

    def _forecast_metric(self, values: np.ndarray, horizon: int) -> Dict[str, Any]:
        """Genera forecast para una métrica específica usando Exponential Smoothing"""

        try:
            # Usar Exponential Smoothing
            model = ExponentialSmoothing(values, trend='add', seasonal=None)
            fitted_model = model.fit()

            # Generar forecast
            forecast = fitted_model.forecast(steps=horizon)

            # Calcular intervalos de confianza aproximados
            std_error = np.std(fitted_model.resid)
            confidence_interval = 1.96 * std_error  # 95% CI

            return {
                'forecast_values': forecast.tolist(),
                'confidence_interval': float(confidence_interval),
                'model_type': 'ExponentialSmoothing',
                'forecast_horizon_days': horizon,
                'last_actual_value': float(values[-1]),
                'predicted_change': float((forecast[0] - values[-1]) / values[-1] * 100)
            }

        except Exception as e:
            # Fallback a forecast simple
            trend = np.mean(np.diff(values[-7:]))  # Tendencia últimos 7 días
            last_value = values[-1]

            forecast = [last_value + trend * (i + 1) for i in range(horizon)]

            return {
                'forecast_values': forecast,
                'confidence_interval': float(np.std(values) * 0.2),
                'model_type': 'LinearTrend',
                'forecast_horizon_days': horizon,
                'last_actual_value': float(last_value),
                'predicted_change': float(trend / last_value * 100) if last_value != 0 else 0
            }

    def _detect_market_anomalies(self, data: pd.DataFrame, trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detecta anomalías en el mercado"""

        alerts = []

        # Verificar volatilidad alta
        for metric, trend_data in trends['overall_market'].items():
            if trend_data['volatility'] > 0.2:  # Umbral del 20%
                alerts.append({
                    'type': 'HIGH_VOLATILITY',
                    'metric': metric,
                    'severity': 'WARNING',
                    'message': f"High volatility detected in {metric}: {trend_data['volatility']:.2%}",
                    'value': trend_data['volatility']
                })

        # Verificar cambios drásticos
        for metric, trend_data in trends['overall_market'].items():
            if abs(trend_data['percent_change']) > 10:  # Cambio mayor al 10%
                severity = 'CRITICAL' if abs(trend_data['percent_change']) > 20 else 'WARNING'
                alerts.append({
                    'type': 'SIGNIFICANT_CHANGE',
                    'metric': metric,
                    'severity': severity,
                    'message': f"Significant change in {metric}: {trend_data['percent_change']:.1f}%",
                    'value': trend_data['percent_change']
                })

        return alerts

    def _generate_summary_insights(self, trends: Dict[str, Any], forecasts: Dict[str, Any]) -> List[str]:
        """Genera insights resumidos para el equipo de negocio"""

        insights = []

        # Insight sobre tendencia general del mercado
        cpm_trend = trends['overall_market']['avg_cpm']['trend_direction']
        if cpm_trend == 'increasing':
            insights.append("Market CPMs are trending upward - favorable conditions for yield optimization")
        elif cpm_trend == 'decreasing':
            insights.append("Market CPMs are trending downward - focus on volume and fill rate optimization")

        # Insight sobre dispositivos
        if 'mobile' in trends['device_trends']:
            mobile_trend = trends['device_trends']['mobile']['trend']['direction']
            if mobile_trend == 'increasing':
                insights.append("Mobile traffic share is growing - ensure mobile-optimized ad experiences")

        # Insight sobre geografía
        if 'tier1_countries' in trends['geo_trends']:
            tier1_trend = trends['geo_trends']['tier1_countries']['trend']['direction']
            if tier1_trend == 'increasing':
                insights.append("Premium market share is expanding - opportunity for higher yields")

        # Insight sobre forecast
        if 'avg_cpm' in forecasts and 'predicted_change' in forecasts['avg_cpm']:
            cpm_forecast_change = forecasts['avg_cpm']['predicted_change']
            if cpm_forecast_change > 5:
                insights.append(f"CPM forecast shows {cpm_forecast_change:.1f}% increase - optimize floor prices accordingly")
            elif cpm_forecast_change < -5:
                insights.append(f"CPM forecast shows {cmp_forecast_change:.1f}% decrease - consider volume strategies")

        return insights

    def _interpret_device_trend(self, device: str, trend: Dict[str, float]) -> str:
        """Interpreta tendencias de dispositivo"""

        direction = trend['direction']
        change = trend['percent_change']

        if device == 'mobile':
            if direction == 'increasing':
                return f"Mobile traffic share growing by {change:.1f}% - mobile-first strategy recommended"
            elif direction == 'decreasing':
                return f"Mobile traffic share declining by {abs(change):.1f}% - diversification opportunity"

        return f"{device.title()} trend: {direction} ({change:.1f}%)"

    def _interpret_geo_trend(self, trend: Dict[str, float]) -> str:
        """Interpreta tendencias geográficas"""

        direction = trend['direction']
        change = trend['percent_change']

        if direction == 'increasing':
            return f"Premium markets expanding by {change:.1f}% - focus on high-value inventory"
        elif direction == 'decreasing':
            return f"Premium markets contracting by {abs(change):.1f}% - diversify geographic reach"

        return f"Geographic distribution stable ({change:.1f}% change)"

    def _interpret_format_trend(self, format_type: str, trend: Dict[str, float]) -> str:
        """Interpreta tendencias de formato"""

        direction = trend['direction']
        change = trend['percent_change']

        if format_type == 'video':
            if direction == 'increasing':
                return f"Video ad share growing by {change:.1f}% - invest in video capabilities"
            elif direction == 'decreasing':
                return f"Video ad share declining by {abs(change):.1f}% - reassess video strategy"

        return f"{format_type.title()} format trend: {direction} ({change:.1f}%)"

# Uso del analizador
def run_market_analysis():
    """Ejecuta análisis completo de mercado"""

    analyzer = MarketTrendAnalyzer()

    # Analizar últimos 90 días
    results = analyzer.analyze_market_trends(lookback_days=90)

    # Publicar resultados
    print("Market Analysis Results:")
    print(json.dumps(results, indent=2, default=str))

    # Guardar en S3 para dashboard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    key = f"market_analysis/trend_analysis_{timestamp}.json"

    analyzer.s3_client.put_object(
        Bucket=analyzer.bucket,
        Key=key,
        Body=json.dumps(results, indent=2, default=str),
        ContentType='application/json'
    )

    print(f"Analysis saved to s3://{analyzer.bucket}/{key}")

    return results
```

## Optimización de Costos Avanzada

### Cost Intelligence Engine

**Sistema Inteligente de Optimización de Costos**:
```python
import boto3
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime, timedelta

class CostIntelligenceEngine:
    """Motor de inteligencia para optimización automática de costos"""

    def __init__(self, region='eu-west-1'):
        self.ce_client = boto3.client('ce', region_name='us-east-1')  # Cost Explorer solo en us-east-1
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.autoscaling = boto3.client('application-autoscaling', region_name=region)
        self.region = region

    def analyze_cost_optimization_opportunities(self) -> Dict[str, Any]:
        """Analiza oportunidades de optimización de costos"""

        # Obtener datos de costos
        cost_data = self._get_cost_data()

        # Analizar utilización de recursos
        utilization_analysis = self._analyze_resource_utilization()

        # Identificar oportunidades de ahorro
        savings_opportunities = self._identify_savings_opportunities(cost_data, utilization_analysis)

        # Generar recomendaciones automatizadas
        recommendations = self._generate_cost_recommendations(savings_opportunities)

        # Calcular ROI de optimizaciones
        roi_analysis = self._calculate_optimization_roi(recommendations)

        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'current_costs': cost_data,
            'utilization_analysis': utilization_analysis,
            'savings_opportunities': savings_opportunities,
            'recommendations': recommendations,
            'roi_analysis': roi_analysis,
            'estimated_monthly_savings': self._calculate_total_savings(recommendations)
        }

    def _get_cost_data(self) -> Dict[str, Any]:
        """Obtiene datos de costos de AWS Cost Explorer"""

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)

        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'INSTANCE_TYPE'}
                ]
            )

            # Procesar datos de costos
            cost_breakdown = {}
            total_cost = 0

            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    service = group['Keys'][0]
                    instance_type = group['Keys'][1] if len(group['Keys']) > 1 else 'N/A'
                    cost = float(group['Metrics']['BlendedCost']['Amount'])

                    if service not in cost_breakdown:
                        cost_breakdown[service] = {}

                    cost_breakdown[service][instance_type] = cost_breakdown[service].get(instance_type, 0) + cost
                    total_cost += cost

            return {
                'total_monthly_cost': total_cost,
                'cost_breakdown_by_service': cost_breakdown,
                'analysis_period_days': 30
            }

        except Exception as e:
            print(f"Error fetching cost data: {str(e)}")
            # Retornar datos simulados para demo
            return {
                'total_monthly_cost': 3811.52,
                'cost_breakdown_by_service': {
                    'Amazon SageMaker': {'ml.c5.xlarge': 1334.52},
                    'Amazon S3': {'Standard': 46.50},
                    'Amazon Kinesis': {'Shard Hour': 1530.36},
                    'AWS Lambda': {'Request': 166.67}
                },
                'analysis_period_days': 30
            }

    def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analiza utilización de recursos de ML"""

        utilization_data = {}

        # Analizar utilización de SageMaker endpoints
        sagemaker_utilization = self._analyze_sagemaker_utilization()
        utilization_data['sagemaker_endpoints'] = sagemaker_utilization

        # Analizar utilización de Lambda
        lambda_utilization = self._analyze_lambda_utilization()
        utilization_data['lambda_functions'] = lambda_utilization

        # Analizar utilización de Kinesis
        kinesis_utilization = self._analyze_kinesis_utilization()
        utilization_data['kinesis_streams'] = kinesis_utilization

        return utilization_data

    def _analyze_sagemaker_utilization(self) -> Dict[str, Any]:
        """Analiza utilización de endpoints SageMaker"""

        try:
            # Obtener lista de endpoints
            endpoints = self.sagemaker.list_endpoints()['Endpoints']

            endpoint_utilization = {}

            for endpoint in endpoints:
                endpoint_name = endpoint['EndpointName']

                # Obtener métricas de CloudWatch
                metrics = self._get_endpoint_metrics(endpoint_name)

                utilization = self._calculate_endpoint_utilization(metrics)
                endpoint_utilization[endpoint_name] = utilization

            return endpoint_utilization

        except Exception as e:
            print(f"Error analyzing SageMaker utilization: {str(e)}")
            return {
                'yield-predictor-multi-model': {
                    'avg_invocations_per_minute': 150.0,
                    'avg_model_latency_ms': 35.0,
                    'utilization_percentage': 45.0,
                    'recommendation': 'DOWNSIZE'
                }
            }

    def _get_endpoint_metrics(self, endpoint_name: str) -> Dict[str, Any]:
        """Obtiene métricas de CloudWatch para un endpoint"""

        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)

        metrics = {}

        try:
            # Invocaciones
            invocations_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='Invocations',
                Dimensions=[
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': 'AllTraffic'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hora
                Statistics=['Average', 'Maximum']
            )

            metrics['invocations'] = invocations_response['Datapoints']

            # Latencia
            latency_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName='ModelLatency',
                Dimensions=[
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': 'AllTraffic'}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average', 'Maximum']
            )

            metrics['latency'] = latency_response['Datapoints']

        except Exception as e:
            print(f"Error getting metrics for {endpoint_name}: {str(e)}")

        return metrics

    def _calculate_endpoint_utilization(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula utilización de endpoint basada en métricas"""

        if not metrics.get('invocations'):
            return {'utilization_percentage': 0, 'recommendation': 'INSUFFICIENT_DATA'}

        # Calcular invocaciones promedio por minuto
        invocations_data = metrics['invocations']
        avg_invocations_per_hour = np.mean([dp['Average'] for dp in invocations_data])
        avg_invocations_per_minute = avg_invocations_per_hour / 60

        # Calcular latencia promedio
        latency_data = metrics.get('latency', [])
        avg_latency_ms = np.mean([dp['Average'] for dp in latency_data]) if latency_data else 0

        # Estimar utilización (basado en capacity teórica)
        # Asumir capacity máxima de 500 invocaciones/minuto por instancia ml.c5.xlarge
        theoretical_max_invocations = 500
        utilization_percentage = (avg_invocations_per_minute / theoretical_max_invocations) * 100

        # Generar recomendación
        recommendation = self._generate_utilization_recommendation(utilization_percentage, avg_latency_ms)

        return {
            'avg_invocations_per_minute': avg_invocations_per_minute,
            'avg_model_latency_ms': avg_latency_ms,
            'utilization_percentage': utilization_percentage,
            'recommendation': recommendation
        }

    def _generate_utilization_recommendation(self, utilization: float, latency: float) -> str:
        """Genera recomendación basada en utilización y latencia"""

        if utilization < 20:
            return 'DOWNSIZE'
        elif utilization > 80:
            return 'UPSIZE'
        elif latency > 100:  # >100ms latencia
            return 'UPSIZE'
        elif utilization < 40 and latency < 50:
            return 'CONSIDER_DOWNSIZE'
        else:
            return 'OPTIMAL'

    def _analyze_lambda_utilization(self) -> Dict[str, Any]:
        """Analiza utilización de funciones Lambda"""

        # Simulación de análisis Lambda
        return {
            'yield-predictor-integration': {
                'avg_duration_ms': 150,
                'avg_memory_used_mb': 200,
                'configured_memory_mb': 512,
                'memory_utilization_percentage': 39.1,
                'recommendation': 'REDUCE_MEMORY'
            },
            'drift-detection': {
                'avg_duration_ms': 4500,
                'avg_memory_used_mb': 800,
                'configured_memory_mb': 1024,
                'memory_utilization_percentage': 78.1,
                'recommendation': 'OPTIMAL'
            }
        }

    def _analyze_kinesis_utilization(self) -> Dict[str, Any]:
        """Analiza utilización de streams Kinesis"""

        # Simulación de análisis Kinesis
        return {
            'ssmas-impressions-stream': {
                'shard_count': 100,
                'avg_incoming_records_per_second': 578,
                'capacity_per_shard': 1000,
                'utilization_percentage': 57.8,
                'recommendation': 'OPTIMAL'
            }
        }

    def _identify_savings_opportunities(self, cost_data: Dict, utilization_data: Dict) -> List[Dict[str, Any]]:
        """Identifica oportunidades específicas de ahorro"""

        opportunities = []

        # Oportunidades en SageMaker
        sagemaker_endpoints = utilization_data.get('sagemaker_endpoints', {})
        for endpoint_name, utilization in sagemaker_endpoints.items():
            if utilization['recommendation'] == 'DOWNSIZE':
                opportunities.append({
                    'service': 'SageMaker',
                    'resource': endpoint_name,
                    'opportunity_type': 'INSTANCE_DOWNSIZE',
                    'current_cost_monthly': 1334.52 / len(sagemaker_endpoints),  # Distribución estimada
                    'potential_savings_monthly': 667.26,  # 50% savings
                    'confidence': 'HIGH',
                    'implementation_effort': 'LOW',
                    'description': f'Endpoint {endpoint_name} showing {utilization["utilization_percentage"]:.1f}% utilization - candidate for downsizing'
                })

        # Oportunidades en Lambda
        lambda_functions = utilization_data.get('lambda_functions', {})
        for function_name, utilization in lambda_functions.items():
            if utilization['recommendation'] == 'REDUCE_MEMORY':
                monthly_savings = 20.0  # Estimación
                opportunities.append({
                    'service': 'Lambda',
                    'resource': function_name,
                    'opportunity_type': 'MEMORY_OPTIMIZATION',
                    'current_cost_monthly': 83.34,  # 166.67 / 2 functions
                    'potential_savings_monthly': monthly_savings,
                    'confidence': 'MEDIUM',
                    'implementation_effort': 'LOW',
                    'description': f'Function {function_name} using {utilization["memory_utilization_percentage"]:.1f}% of allocated memory'
                })

        # Oportunidades en S3
        opportunities.append({
            'service': 'S3',
            'resource': 'Data Lake Storage',
            'opportunity_type': 'INTELLIGENT_TIERING',
            'current_cost_monthly': 46.50,
            'potential_savings_monthly': 13.95,  # 30% savings
            'confidence': 'HIGH',
            'implementation_effort': 'LOW',
            'description': 'Enable S3 Intelligent Tiering for automatic cost optimization'
        })

        # Oportunidades en Reserved Instances
        opportunities.append({
            'service': 'SageMaker',
            'resource': 'Production Endpoints',
            'opportunity_type': 'RESERVED_INSTANCES',
            'current_cost_monthly': 1334.52,
            'potential_savings_monthly': 533.81,  # 40% savings
            'confidence': 'HIGH',
            'implementation_effort': 'MEDIUM',
            'description': 'Purchase SageMaker Savings Plans for 40% cost reduction'
        })

        return opportunities

    def _generate_cost_recommendations(self, opportunities: List[Dict]) -> List[Dict[str, Any]]:
        """Genera recomendaciones accionables basadas en oportunidades"""

        recommendations = []

        for opp in opportunities:
            if opp['confidence'] == 'HIGH' and opp['potential_savings_monthly'] > 20:
                priority = 'HIGH'
            elif opp['potential_savings_monthly'] > 10:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'

            recommendation = {
                'id': f"rec_{hash(opp['resource'])}",
                'title': f"{opp['opportunity_type'].replace('_', ' ').title()} - {opp['resource']}",
                'description': opp['description'],
                'service': opp['service'],
                'resource': opp['resource'],
                'priority': priority,
                'estimated_monthly_savings': opp['potential_savings_monthly'],
                'implementation_effort': opp['implementation_effort'],
                'confidence_level': opp['confidence'],
                'action_items': self._generate_action_items(opp),
                'timeline': self._estimate_implementation_timeline(opp['implementation_effort'])
            }

            recommendations.append(recommendation)

        # Ordenar por savings potenciales
        recommendations.sort(key=lambda x: x['estimated_monthly_savings'], reverse=True)

        return recommendations

    def _generate_action_items(self, opportunity: Dict[str, Any]) -> List[str]:
        """Genera items de acción específicos para cada oportunidad"""

        action_items = []

        if opportunity['opportunity_type'] == 'INSTANCE_DOWNSIZE':
            action_items = [
                "Analyze current endpoint performance and utilization patterns",
                "Test endpoint with smaller instance type in staging environment",
                "Implement gradual traffic shifting to new instance type",
                "Monitor performance metrics during transition",
                "Update auto-scaling policies if necessary"
            ]

        elif opportunity['opportunity_type'] == 'MEMORY_OPTIMIZATION':
            action_items = [
                "Profile Lambda function memory usage over 1 week",
                "Update function configuration with optimized memory allocation",
                "Test function performance with new memory settings",
                "Monitor execution duration and cost changes"
            ]

        elif opportunity['opportunity_type'] == 'INTELLIGENT_TIERING':
            action_items = [
                "Enable S3 Intelligent Tiering on data lake buckets",
                "Configure lifecycle policies for automated transitions",
                "Monitor storage class transitions and cost impact",
                "Adjust policies based on access patterns"
            ]

        elif opportunity['opportunity_type'] == 'RESERVED_INSTANCES':
            action_items = [
                "Analyze historical usage patterns for consistent workloads",
                "Calculate optimal commitment level for Savings Plans",
                "Purchase SageMaker Savings Plans with appropriate term",
                "Monitor usage and adjust future purchases accordingly"
            ]

        return action_items

    def _estimate_implementation_timeline(self, effort: str) -> str:
        """Estima timeline de implementación"""

        timelines = {
            'LOW': '1-2 weeks',
            'MEDIUM': '2-4 weeks',
            'HIGH': '1-2 months'
        }

        return timelines.get(effort, '2-4 weeks')

    def _calculate_optimization_roi(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Calcula ROI de las optimizaciones recomendadas"""

        total_savings = sum(rec['estimated_monthly_savings'] for rec in recommendations)
        annual_savings = total_savings * 12

        # Estimar costos de implementación
        implementation_costs = self._estimate_implementation_costs(recommendations)

        # Calcular ROI
        if implementation_costs > 0:
            roi_percentage = ((annual_savings - implementation_costs) / implementation_costs) * 100
        else:
            roi_percentage = float('inf')

        payback_months = implementation_costs / total_savings if total_savings > 0 else float('inf')

        return {
            'estimated_monthly_savings': total_savings,
            'estimated_annual_savings': annual_savings,
            'estimated_implementation_costs': implementation_costs,
            'roi_percentage': roi_percentage,
            'payback_period_months': payback_months,
            'net_annual_benefit': annual_savings - implementation_costs
        }

    def _estimate_implementation_costs(self, recommendations: List[Dict]) -> float:
        """Estima costos de implementación de recomendaciones"""

        # Costos estimados basados en effort level
        effort_costs = {
            'LOW': 500,    # €500 en tiempo de engineering
            'MEDIUM': 2000,  # €2000 en tiempo de engineering
            'HIGH': 5000   # €5000 en tiempo de engineering
        }

        total_cost = 0
        for rec in recommendations:
            effort = rec['implementation_effort']
            total_cost += effort_costs.get(effort, 1000)

        return total_cost

    def _calculate_total_savings(self, recommendations: List[Dict]) -> float:
        """Calcula ahorros totales mensuales"""

        return sum(rec['estimated_monthly_savings'] for rec in recommendations)

    def implement_automatic_optimizations(self) -> Dict[str, Any]:
        """Implementa optimizaciones automáticas seguras"""

        implemented = []

        # Auto-scaling para SageMaker endpoints
        try:
            self._setup_sagemaker_autoscaling()
            implemented.append({
                'optimization': 'SageMaker Auto Scaling',
                'status': 'SUCCESS',
                'description': 'Enabled auto-scaling for production endpoints'
            })
        except Exception as e:
            implemented.append({
                'optimization': 'SageMaker Auto Scaling',
                'status': 'FAILED',
                'error': str(e)
            })

        # S3 Intelligent Tiering
        try:
            self._enable_s3_intelligent_tiering()
            implemented.append({
                'optimization': 'S3 Intelligent Tiering',
                'status': 'SUCCESS',
                'description': 'Enabled intelligent tiering on data lake'
            })
        except Exception as e:
            implemented.append({
                'optimization': 'S3 Intelligent Tiering',
                'status': 'FAILED',
                'error': str(e)
            })

        return {
            'implementation_timestamp': datetime.now().isoformat(),
            'implemented_optimizations': implemented,
            'total_optimizations': len(implemented),
            'success_rate': len([opt for opt in implemented if opt['status'] == 'SUCCESS']) / len(implemented) * 100
        }

    def _setup_sagemaker_autoscaling(self):
        """Configura auto-scaling para endpoints SageMaker"""

        # Configurar auto-scaling policy
        self.autoscaling.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId='endpoint/yield-predictor-multi-model/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=1,
            MaxCapacity=5
        )

        # Crear scaling policy
        self.autoscaling.put_scaling_policy(
            PolicyName='yield-predictor-scaling-policy',
            ServiceNamespace='sagemaker',
            ResourceId='endpoint/yield-predictor-multi-model/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': 70.0,  # 70% utilization target
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleOutCooldown': 300,  # 5 minutes
                'ScaleInCooldown': 300
            }
        )

    def _enable_s3_intelligent_tiering(self):
        """Habilita S3 Intelligent Tiering"""

        s3_client = boto3.client('s3', region_name=self.region)

        bucket = 'ssmas-yield-prod-datalake-eu-west-1'

        # Configurar Intelligent Tiering
        s3_client.put_bucket_intelligent_tiering_configuration(
            Bucket=bucket,
            Id='EntireBucketIT',
            IntelligentTieringConfiguration={
                'Id': 'EntireBucketIT',
                'Status': 'Enabled',
                'Filter': {
                    'Prefix': ''  # Aplicar a todo el bucket
                },
                'Tierings': [
                    {
                        'Days': 90,
                        'AccessTier': 'ARCHIVE_ACCESS'
                    },
                    {
                        'Days': 180,
                        'AccessTier': 'DEEP_ARCHIVE_ACCESS'
                    }
                ]
            }
        )

# Uso del Cost Intelligence Engine
def run_cost_optimization():
    """Ejecuta análisis completo de optimización de costos"""

    cost_engine = CostIntelligenceEngine()

    # Análisis de oportunidades
    analysis = cost_engine.analyze_cost_optimization_opportunities()

    print("Cost Optimization Analysis:")
    print(json.dumps(analysis, indent=2, default=str))

    # Implementar optimizaciones automáticas
    implementation = cost_engine.implement_automatic_optimizations()

    print("\nAutomatic Optimizations:")
    print(json.dumps(implementation, indent=2))

    return analysis, implementation
```

## Cronograma de la Fase III

### Planning Detallado (12 semanas)

#### Semanas 1-3: AutoML e Inteligencia de Modelos
**Semana 1: Setup AutoML**
- Configuración de SageMaker Autopilot
- Desarrollo de pipeline de experimentos automáticos
- Setup de model comparison framework
- Testing inicial de AutoML

**Semana 2: Multi-Objective Optimization**
- Implementación de optimización multi-objetivo
- Desarrollo de Pareto frontier analysis
- Integración con sistema de precios dinámicos
- Testing de balanceo CPM/Fill Rate/Revenue

**Semana 3: Model Explainability**
- Implementación de SHAP para interpretabilidad
- Desarrollo de business insights engine
- Creación de dashboards de explicabilidad
- Testing de insights generation

#### Semanas 4-6: Advanced Analytics
**Semana 4: Market Trend Analysis**
- Implementación de trend analysis engine
- Desarrollo de forecasting capabilities
- Setup de anomaly detection avanzada
- Integration con datos externos

**Semana 5: Predictive Analytics**
- Implementación de forecasting models
- Desarrollo de market intelligence
- Setup de competitive analysis
- Testing de prediction accuracy

**Semana 6: Business Intelligence**
- Desarrollo de advanced dashboards
- Implementación de automated reporting
- Setup de alert systems avanzados
- Integration con sistemas de negocio

#### Semanas 7-9: Cost Optimization
**Semana 7: Cost Intelligence**
- Implementación de cost analysis engine
- Desarrollo de automatic optimization
- Setup de ROI tracking
- Testing de cost recommendations

**Semana 8: Resource Optimization**
- Implementación de auto-scaling avanzado
- Desarrollo de capacity planning
- Setup de predictive scaling
- Testing de resource efficiency

**Semana 9: Financial Optimization**
- Implementación de Reserved Instance optimization
- Desarrollo de Savings Plans strategy
- Setup de cost allocation tracking
- Integration con finance systems

#### Semanas 10-12: Integration y Production
**Semana 10: System Integration**
- Integración de todos los componentes
- Testing end-to-end de nuevas capacidades
- Validation de performance improvements
- Documentation de nuevas features

**Semana 11: Production Deployment**
- Deployment gradual de nuevas capacidades
- Monitoring intensivo de impacto
- Ajustes basados en feedback inicial
- Training del equipo en nuevas herramientas

**Semana 12: Optimization y Handover**
- Fine-tuning de todos los sistemas
- Documentation completa de operación
- Handover al equipo de operaciones
- Planning de roadmap futuro

## Criterios de Éxito de la Fase III

### Métricas Técnicas Avanzadas
1. **AutoML Performance**:
   - Mejora > 5% en métricas de modelo vs. modelo base
   - Tiempo de experimentación reducido en 70%
   - Accuracy de model selection > 90%

2. **Multi-Objective Optimization**:
   - Balance óptimo demostrado entre CPM, Fill Rate y Revenue
   - Incremento en utility score > 15%
   - Pareto frontier analysis operativo

3. **Explicabilidad**:
   - Business insights generados automáticamente
   - SHAP values calculados para todas las predicciones
   - Dashboard de explicabilidad utilizado diariamente

### Métricas de Negocio Avanzadas
1. **Revenue Optimization**:
   - Incremento total de revenue > 8% vs. Fase II
   - Optimización multi-objetivo demostrada
   - ROI de inversión en Fase III > 300%

2. **Cost Efficiency**:
   - Reducción de costos de infraestructura > 25%
   - Automatic optimization funcionando
   - Cost per prediction reducido en 30%

3. **Market Intelligence**:
   - Forecasting accuracy > 85% para 7 días
   - Market trends identificados proactivamente
   - Competitive advantage demostrable

## Beneficios Empresariales de la Fase III

### Beneficios Inmediatos
- **Optimización Autónoma**: Sistema que mejora continuamente sin intervención manual
- **Insights Profundos**: Comprensión clara de drivers de CPM y comportamiento del mercado
- **Cost Efficiency**: Reducción significativa de costos operativos
- **Competitive Edge**: Capacidades avanzadas que diferencias a SSMAS en el mercado

### Beneficios a Largo Plazo
- **Innovation Platform**: Base para futuras innovaciones en AdTech
- **Market Leadership**: Posicionamiento como líder tecnológico en el sector
- **Scalability**: Capacidad de expansión sin degradación de performance
- **Business Intelligence**: Insights que informan estrategia de negocio

## Roadmap Post-Fase III

### Evoluciones Futuras
1. **Real-Time Bidding Optimization**: Optimización de estrategias de puja en tiempo real
2. **Cross-Platform Intelligence**: Expansión a otros canales publicitarios
3. **Blockchain Integration**: Transparencia y verificación de transacciones publicitarias
4. **AI-Driven Creative Optimization**: Optimización automática de creatividades publicitarias

### Expansión de Capacidades
1. **Multi-Modal AI**: Incorporación de análisis de imagen y video
2. **Edge Computing**: Procesamiento en edge para latencia ultra-baja
3. **Quantum Computing**: Exploración de algoritmos cuánticos para optimización
4. **Federated Learning**: Aprendizaje colaborativo preservando privacidad

## Conclusión de la Fase III

La Fase III del proyecto Yield Predictivo establece a SSMAS como pionero absoluto en la aplicación de inteligencia artificial avanzada al sector AdTech. A través de la implementación de capacidades de AutoML, optimización multi-objetivo, explicabilidad e inteligencia de costos, el sistema evoluciona de una herramienta de predicción a una plataforma completa de optimización publicitaria autoadaptativa.

Los beneficios de esta fase trascienden la optimización de revenue inmediato, estableciendo una ventaja competitiva sostenible que posiciona a SSMAS para liderar la próxima generación de tecnología publicitaria. La capacidad del sistema para aprender, evolucionar y optimizarse continuamente asegura que SSMAS mantenga su posición de vanguardia tecnológica mientras maximiza el valor para sus clientes.

La inversión en esta fase no solo completa la transformación digital de SSMAS en el ámbito de yield optimization, sino que establece las bases tecnológicas y metodológicas para futuras innovaciones que mantendrán a la empresa a la vanguardia del sector AdTech en un mercado cada vez más competitivo y tecnológicamente sofisticado.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Analizar documentos del caso de uso Yield Predictivo", "status": "completed", "activeForm": "Analizando documentos del caso de uso Yield Predictivo"}, {"content": "Generar documento Proyectos/caso1.md con an\u00e1lisis Yield Predictivo", "status": "completed", "activeForm": "Generando documento Proyectos/caso1.md con an\u00e1lisis Yield Predictivo"}, {"content": "Analizar documentos del caso de uso AdOps Aut\u00f3nomo", "status": "completed", "activeForm": "Analizando documentos del caso de uso AdOps Aut\u00f3nomo"}, {"content": "Generar documento Proyectos/caso2.md con an\u00e1lisis AdOps Aut\u00f3nomo", "status": "completed", "activeForm": "Generando documento Proyectos/caso2.md con an\u00e1lisis AdOps Aut\u00f3nomo"}, {"content": "Generar documento Proyectos/landingzone.md", "status": "completed", "activeForm": "Generando documento Proyectos/landingzone.md"}, {"content": "Generar arquitectura funcional de landing zone", "status": "completed", "activeForm": "Generando arquitectura funcional de landing zone"}, {"content": "Generar arquitectura t\u00e9cnica de landing zone", "status": "completed", "activeForm": "Generando arquitectura t\u00e9cnica de landing zone"}, {"content": "Generar documentaci\u00f3n y arquitectura Fase 1 caso 1", "status": "completed", "activeForm": "Generando documentaci\u00f3n y arquitectura Fase 1 caso 1"}, {"content": "Generar documentaci\u00f3n y arquitectura Fase 2 caso 1", "status": "completed", "activeForm": "Generando documentaci\u00f3n y arquitectura Fase 2 caso 1"}, {"content": "Generar documentaci\u00f3n y arquitectura Fase 3 caso 1", "status": "completed", "activeForm": "Generando documentaci\u00f3n y arquitectura Fase 3 caso 1"}]