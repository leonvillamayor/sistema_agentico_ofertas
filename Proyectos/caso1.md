# Caso de Uso 1: Yield Predictivo

## Introducción

El caso de uso Yield Predictivo representa una evolución estratégica fundamental para SSMAS, empresa líder en tecnología publicitaria (AdTech) en España. Esta solución transforma el modelo de optimización de ingresos publicitarios de reactivo a predictivo, permitiendo a SSMAS anticipar el valor de cada impresión publicitaria antes de que entre en el proceso de subasta.

SSMAS, primera empresa española en obtener la certificación Google Certified Publishing Partner (GCPP), gestiona actualmente más de 7.000 millones de impresiones publicitarias mensuales, logrando incrementos promedio del 37% en los ingresos de sus clientes durante los primeros meses de 2024.

## Descripción del Problema

### Situación Actual
SSMAS opera bajo un modelo de optimización reactiva que se basa únicamente en las condiciones actuales del mercado publicitario. Este enfoque presenta limitaciones significativas:

- **Precios mínimos estáticos**: Los precios de reserva (floor prices) se establecen de manera uniforme sin considerar las características específicas de cada oportunidad publicitaria
- **Falta de personalización**: No se aprovecha el potencial de ingresos diferenciado de cada impresión individual
- **Decisiones subóptimas**: La ausencia de predicciones precisas puede resultar en ventas por debajo del valor real de las impresiones

### Oportunidad de Mejora
Con una volumetría de 50 millones de registros diarios y acceso a datos históricos masivos, SSMAS tiene la oportunidad de implementar un sistema de optimización predictiva que maximice el yield (rendimiento) de cada impresión publicitaria.

## Objetivos del Caso de Uso

### Objetivo Principal
Desarrollar un modelo de machine learning avanzado que prediga el Coste Por Mil (CPM) esperado para cada oportunidad de anuncio individual, permitiendo la implementación de precios mínimos dinámicos y optimización de ingresos en tiempo real.

### Objetivos Específicos
1. **Maximización de ingresos totales**: Optimizar el yield mediante precios inteligentes y dinámicos
2. **Mejora de la tasa de relleno**: Equilibrar precio de venta con probabilidad de que el anuncio se muestre
3. **Ventaja competitiva**: Proporcionar a SSMAS una diferenciación tecnológica significativa en el mercado
4. **Decisiones basadas en datos**: Reemplazar la intuición con predicciones precisas basadas en patrones históricos

## Funcionamiento de la Solución

### 1. Preparación y Procesamiento de Datos
**Herramienta**: AWS Glue

La solución utiliza los enormes volúmenes de datos históricos de SSMAS como base fundamental:
- **Volumen de datos**: Más de 7.000 millones de impresiones mensuales
- **Datos históricos**: Registros detallados de subastas, CPMs, características de impresiones
- **Feature Engineering**: Extracción de variables relevantes incluyendo:
  - Tipo de contenido y formato de anuncio
  - Fuente del tráfico y características del usuario
  - Hora del día y patrón temporal
  - Geografía del usuario
  - Señales de demanda del mercado
  - Características del dispositivo y navegador

### 2. Modelo Predictivo
**Herramienta**: Amazon SageMaker con algoritmo XGBoost

El modelo utiliza XGBoost (eXtreme Gradient Boosting), una implementación de árboles de decisión potenciados por gradiente:
- **Algoritmo**: Ideal para datos tabulares y predicciones de alta precisión
- **Entrenamiento**: Aprendizaje de patrones históricos para predecir CPM probable
- **Input**: Características específicas de cada impresión
- **Output**: Predicción de CPM esperado para la subasta

### 3. Aplicación en Tiempo Real
**Implementación**: Precios mínimos dinámicos (Dynamic Floor Pricing)

La predicción del modelo se utiliza para:
- **CPM alto predicho**: Establecer precio mínimo elevado para asegurar valor óptimo
- **CPM bajo predicho**: Ajustar precio mínimo a la baja para mejorar fill rate
- **Decisión automática**: Optimización individual para cada una de los millones de impresiones gestionadas

## Datos Disponibles

### Volumetría
- **Registros diarios**: 50 millones
- **Impresiones mensuales**: 7.000 millones+
- **Fuente de datos**: Plataforma desplegada en Vercel
- **API de acceso**: Elastic share para consumo de datos

### Esquema de Datos Publicitarios
Los datos incluyen información detallada sobre:

**Métricas de rendimiento**:
- CPM (Costo por mil impresiones)
- CTR (Tasa de clics)
- Revenue (Ingresos totales)
- Impressions (Impresiones)
- Clicks (Clics)

**Características del inventario**:
- adUnit (Bloque de anuncios)
- format (Formato del anuncio)
- inventoryType (Tipo de inventario)
- deviceCategory (Categoría del dispositivo)

**Información contextual**:
- country (País)
- domain (Dominio)
- date (Fecha)
- bidder (Plataforma de puja)
- demandChannel (Canal de demanda)

### Esquema de Datos de Usuario
**Información técnica**:
- browser (Navegador)
- os (Sistema operativo)
- deviceMemory (Memoria del dispositivo)
- connectionDownlink (Ancho de banda)
- screenRes (Resolución de pantalla)

**Datos de comportamiento**:
- pageCount (Páginas vistas)
- referrer (URL de referencia)
- trafficSource (Fuente de tráfico)
- mobile (Dispositivo móvil)

### Datos de Eventos Publicitarios
**Información de la subasta**:
- winnerBidCpm (CPM ganador)
- winnerBidder (Postor ganador)
- advertiser (Anunciante)
- campaignId (ID de campaña)
- renderTime (Tiempo de renderizado)

## Arquitectura Técnica

### Flujo de Datos
1. **Ingesta**: Datos históricos de SSMAS → Amazon S3 Data Lake
2. **Procesamiento**: AWS Glue → Feature Engineering → Datos procesados
3. **Entrenamiento**: Amazon SageMaker → Modelo XGBoost → Artefacto del modelo
4. **Inferencia**: Tiempo real → SageMaker Endpoint → Predicción CPM
5. **Aplicación**: Sistema SSMAS → API Call → Floor price dinámico

### Componentes Clave
- **Amazon S3**: Almacenamiento de datos históricos (7B+ impresiones)
- **AWS Glue**: ETL y Feature Engineering
- **Amazon SageMaker**: Entrenamiento y despliegue del modelo
- **API Gateway**: Integración con sistema SSMAS
- **CloudWatch**: Monitorización y métricas

## Fases de Implementación

### Fase I: PoC y Desarrollo del MVP (4-6 semanas)
**Objetivo**: Validar viabilidad técnica en entorno controlado

**Actividades principales**:
- Configuración de entorno AWS aislado
- Ingesta y procesamiento de muestra de datos históricos
- Desarrollo del job de AWS Glue para feature engineering
- Entrenamiento del modelo XGBoost en SageMaker
- Evaluación de rendimiento (RMSE, MAE)
- Documentación completa del proceso

**Entregables**:
- Modelo de ML entrenado y validado
- Código fuente versionado
- Informe de métricas de rendimiento
- Presentación y validación del MVP

### Fase II: Operativización y MLOps (6-8 semanas)
**Objetivo**: Integrar MVP en entorno productivo escalable

**Actividades principales**:
- Diseño de arquitectura productiva con best practices
- Despliegue de endpoint de inferencia en tiempo real
- Implementación de pipelines de MLOps automatizados
- Integración con sistema de subastas SSMAS
- Configuración de monitorización avanzada

**Entregables**:
- Endpoint de inferencia seguro y escalable
- Pipeline CI/CD para ML completamente automatizado
- Dashboard de monitorización operativa
- Integración funcional con sistema existente

### Fase III: Optimización Continua (Opcional)
**Objetivo**: Mantenimiento y mejora continua del modelo

**Actividades principales**:
- Monitorización de deriva del modelo (Model Drift)
- Ciclos automatizados de reentrenamiento
- Optimización de costes AWS
- A/B Testing entre versiones del modelo
- Mejora continua basada en resultados

**Entregables**:
- Informes periódicos de rendimiento
- Nuevas versiones del modelo
- Reportes de optimización de costes

## Beneficios Empresariales

### Beneficio Principal
**Maximización de ingresos totales** mediante fijación de precios inteligentes y dinámicos. En lugar de aplicar precios mínimos estáticos, permite tomar decisiones de precios óptimas para cada impresión individual.

### Beneficios Específicos
1. **Incremento de ingresos**: Optimización del yield por impresión
2. **Ventaja competitiva**: Diferenciación tecnológica en el mercado AdTech
3. **Eficiencia operativa**: Automatización de decisiones de precios
4. **Escalabilidad**: Capacidad de procesar miles de millones de impresiones
5. **Precisión**: Decisiones basadas en machine learning vs. intuición

### Impacto en el Negocio
- **Refuerzo de la propuesta de valor**: Mejora la capacidad de SSMAS para generar más ingresos para sus clientes
- **Retención de clientes**: Mayor satisfacción por mejores resultados
- **Atracción de nuevos clientes**: Diferenciación competitiva clara
- **Escalabilidad del negocio**: Capacidad de manejar más volumen sin degradar el rendimiento

## Consideraciones Técnicas

### Requisitos de Rendimiento
- **Latencia**: Predicciones en milisegundos para subastas en tiempo real
- **Escalabilidad**: Capacidad para 50M+ predicciones diarias
- **Disponibilidad**: 99.9% uptime para no impactar ingresos
- **Precisión**: Métricas de error optimizadas continuamente

### Integración con Elastic
- **URL de conexión**: Disponible en documentación técnica
- **API de consumo**: Elastic share para acceso eficiente a datos
- **Optimización**: Desarrollo específico para manejar volumetría actual

### Arquitectura Serverless
- **Principio Serverless-First**: Minimizar carga operativa
- **Tiempo Real**: Diseño para baja latencia desde el inicio
- **Modular**: Componentes débilmente acoplados
- **Seguro**: Mejores prácticas de seguridad y gobernanza

## Métricas de Éxito

### Métricas Técnicas
- **RMSE/MAE**: Error de predicción del modelo
- **Latencia**: Tiempo de respuesta del endpoint
- **Throughput**: Predicciones por segundo
- **Uptime**: Disponibilidad del servicio

### Métricas de Negocio
- **Incremento de CPM promedio**: Mejora en precios obtenidos
- **Fill Rate**: Mantenimiento o mejora de tasa de relleno
- **Revenue Lift**: Incremento total de ingresos
- **ROI**: Retorno de inversión de la implementación

## Conclusión

El caso de uso Yield Predictivo representa una transformación fundamental en la estrategia de monetización de SSMAS, evolucionando de un enfoque reactivo a uno predictivo. La implementación de este sistema de machine learning avanzado proporcionará a SSMAS una ventaja competitiva significativa en el mercado AdTech, maximizando los ingresos de cada impresión publicitaria mediante decisiones de precios inteligentes y automáticas.

La solución aprovecha la vasta experiencia de SSMAS en el sector, su posición privilegiada como Google Certified Publishing Partner, y la enorme cantidad de datos históricos disponibles para crear un sistema que no solo mejora los resultados actuales, sino que establece las bases para futuras innovaciones en el campo de la optimización publicitaria predictiva.