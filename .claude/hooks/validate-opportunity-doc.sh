#!/bin/bash
# Hook para validar documentos de oportunidad antes de guardarlos
# Asegura que cumplan con los estándares de calidad y contenido

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Función para logging
log_info() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }

# Recibir ruta del documento a validar
DOC_PATH="${1:-}"

if [ -z "$DOC_PATH" ] || [ ! -f "$DOC_PATH" ]; then
    log_error "Documento no especificado o no existe: $DOC_PATH"
    exit 1
fi

echo "🔍 Validando documento de oportunidad: $(basename "$DOC_PATH")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

VALIDATION_SCORE=100
WARNINGS=()
ERRORS=()

# 1. Validar nombre del archivo
filename=$(basename "$DOC_PATH")
if ! [[ "$filename" =~ ^[0-9]+-[^-]+-[0-9]{8}\.md$ ]]; then
    ERRORS+=("Formato de nombre incorrecto. Esperado: ID-DESC-FECHA.md")
    VALIDATION_SCORE=$((VALIDATION_SCORE - 20))
fi

# 2. Validar estructura de secciones principales
echo "Verificando estructura del documento..."

REQUIRED_SECTIONS=(
    "# Oportunidad"
    "## Resumen Ejecutivo"
    "## Análisis de la Necesidad"
    "## Solución Propuesta AWS"
    "## Beneficios"
    "## Consideraciones de Implementación"
    "## Próximos Pasos"
)

for section in "${REQUIRED_SECTIONS[@]}"; do
    if ! grep -q "^$section" "$DOC_PATH"; then
        ERRORS+=("Sección requerida faltante: $section")
        VALIDATION_SCORE=$((VALIDATION_SCORE - 10))
    fi
done

# 3. Validar subsecciones importantes
IMPORTANT_SUBSECTIONS=(
    "### Contexto Empresarial"
    "### Requisitos Identificados"
    "### Arquitectura Recomendada"
    "### Servicios AWS"
    "### Fases de Implementación"
    "### Seguridad y Compliance"
)

for subsection in "${IMPORTANT_SUBSECTIONS[@]}"; do
    if ! grep -q "$subsection" "$DOC_PATH"; then
        WARNINGS+=("Subsección recomendada faltante: $subsection")
        VALIDATION_SCORE=$((VALIDATION_SCORE - 3))
    fi
done

# 4. Validar mención de servicios AWS específicos
echo "Verificando servicios AWS..."

AWS_SERVICES_FOUND=$(grep -oE "(EC2|S3|Lambda|RDS|DynamoDB|ECS|EKS|Fargate|CloudFormation|CDK|API Gateway|SageMaker|Bedrock|Athena|Glue|Kinesis|SNS|SQS|Route 53|CloudFront|WAF|Shield|GuardDuty|Security Hub|CloudWatch|X-Ray|Systems Manager|Secrets Manager|IAM|VPC|Direct Connect|Transit Gateway|Backup|EFS|FSx|ElastiCache|OpenSearch|EMR|Redshift|QuickSight|DataSync|Transfer Family|Step Functions|EventBridge|AppSync|Amplify|CodePipeline|CodeBuild|CodeDeploy|ECR|Control Tower|Organizations|Config|CloudTrail)" "$DOC_PATH" | sort -u | wc -l)

if [ "$AWS_SERVICES_FOUND" -lt 3 ]; then
    ERRORS+=("Servicios AWS insuficientes (encontrados: $AWS_SERVICES_FOUND, mínimo: 3)")
    VALIDATION_SCORE=$((VALIDATION_SCORE - 15))
elif [ "$AWS_SERVICES_FOUND" -lt 5 ]; then
    WARNINGS+=("Pocos servicios AWS mencionados (encontrados: $AWS_SERVICES_FOUND)")
    VALIDATION_SCORE=$((VALIDATION_SCORE - 5))
else
    log_info "Servicios AWS encontrados: $AWS_SERVICES_FOUND"
fi

# 5. Validar longitud mínima del contenido
echo "Verificando completitud del contenido..."

WORD_COUNT=$(wc -w < "$DOC_PATH")
MIN_WORDS=500

if [ "$WORD_COUNT" -lt "$MIN_WORDS" ]; then
    ERRORS+=("Documento muy corto ($WORD_COUNT palabras, mínimo: $MIN_WORDS)")
    VALIDATION_SCORE=$((VALIDATION_SCORE - 20))
elif [ "$WORD_COUNT" -lt 1000 ]; then
    WARNINGS+=("Documento podría ser más detallado ($WORD_COUNT palabras)")
    VALIDATION_SCORE=$((VALIDATION_SCORE - 5))
else
    log_info "Longitud del documento: $WORD_COUNT palabras"
fi

# 6. Validar menciones de mejores prácticas
echo "Verificando mejores prácticas..."

BEST_PRACTICES_KEYWORDS=("Well-Architected" "seguridad" "escalabilidad" "disponibilidad" "resiliencia" "costo" "optimización" "monitoreo" "automatización")
BEST_PRACTICES_FOUND=0

for keyword in "${BEST_PRACTICES_KEYWORDS[@]}"; do
    if grep -qi "$keyword" "$DOC_PATH"; then
        ((BEST_PRACTICES_FOUND++))
    fi
done

if [ "$BEST_PRACTICES_FOUND" -lt 3 ]; then
    WARNINGS+=("Pocas referencias a mejores prácticas (encontradas: $BEST_PRACTICES_FOUND)")
    VALIDATION_SCORE=$((VALIDATION_SCORE - 5))
else
    log_info "Referencias a mejores prácticas: $BEST_PRACTICES_FOUND"
fi

# 7. Validar presencia de métricas o KPIs
if ! grep -qE "(métrica|KPI|indicador|medición|objetivo)" "$DOC_PATH"; then
    WARNINGS+=("No se mencionan métricas o KPIs")
    VALIDATION_SCORE=$((VALIDATION_SCORE - 5))
fi

# 8. Validar estimaciones o consideraciones de costo
if ! grep -qiE "(costo|precio|inversión|presupuesto|ROI|TCO)" "$DOC_PATH"; then
    WARNINGS+=("No se mencionan consideraciones de costo")
    VALIDATION_SCORE=$((VALIDATION_SCORE - 5))
fi

# 9. Validar formato Markdown
echo "Verificando formato Markdown..."

# Verificar uso de listas
if ! grep -qE "^[\*\-\+] " "$DOC_PATH"; then
    WARNINGS+=("No se utilizan listas con viñetas")
fi

# Verificar uso de negritas
if ! grep -q "\*\*.*\*\*" "$DOC_PATH"; then
    WARNINGS+=("No se utiliza formato en negrita")
fi

# 10. Calcular puntuación final
if [ "$VALIDATION_SCORE" -lt 0 ]; then
    VALIDATION_SCORE=0
fi

# Mostrar resultados
echo ""
echo "📊 Resultado de Validación"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ${#ERRORS[@]} -gt 0 ]; then
    echo "❌ ERRORES ENCONTRADOS:"
    for error in "${ERRORS[@]}"; do
        echo "   • $error"
    done
    echo ""
fi

if [ ${#WARNINGS[@]} -gt 0 ]; then
    echo "⚠️  ADVERTENCIAS:"
    for warning in "${WARNINGS[@]}"; do
        echo "   • $warning"
    done
    echo ""
fi

echo "📈 Puntuación de Calidad: $VALIDATION_SCORE/100"

# Determinar estado final
STATUS="success"
MESSAGE="Documento válido y completo"
SHOULD_BLOCK="false"

if [ "$VALIDATION_SCORE" -lt 60 ]; then
    STATUS="error"
    MESSAGE="Documento no cumple estándares mínimos de calidad"
    SHOULD_BLOCK="true"
    echo ""
    log_error "DOCUMENTO RECHAZADO - Requiere mejoras significativas"
elif [ "$VALIDATION_SCORE" -lt 80 ]; then
    STATUS="warning"
    MESSAGE="Documento aceptable pero con oportunidades de mejora"
    echo ""
    log_warn "DOCUMENTO ACEPTADO CON RESERVAS"
else
    echo ""
    log_info "DOCUMENTO APROBADO"
fi

# Generar recomendaciones
if [ "$VALIDATION_SCORE" -lt 100 ]; then
    echo ""
    echo "💡 RECOMENDACIONES DE MEJORA:"

    if [ "$AWS_SERVICES_FOUND" -lt 5 ]; then
        echo "   • Incluir más servicios AWS específicos para la solución"
    fi

    if [ "$WORD_COUNT" -lt 1000 ]; then
        echo "   • Expandir el contenido con más detalles técnicos"
    fi

    if [ "$BEST_PRACTICES_FOUND" -lt 3 ]; then
        echo "   • Agregar más referencias a AWS Well-Architected Framework"
    fi

    if [ ${#ERRORS[@]} -gt 0 ]; then
        echo "   • Corregir errores estructurales identificados"
    fi
fi

echo ""
echo "═══════════════════════════════════════════"

# Retornar JSON para Claude Code
cat << EOF
{
  "status": "$STATUS",
  "validationScore": $VALIDATION_SCORE,
  "errors": ${#ERRORS[@]},
  "warnings": ${#WARNINGS[@]},
  "awsServicesFound": $AWS_SERVICES_FOUND,
  "wordCount": $WORD_COUNT,
  "shouldBlock": $SHOULD_BLOCK,
  "message": "$MESSAGE"
}
EOF

# Exit con código apropiado
if [ "$SHOULD_BLOCK" == "true" ]; then
    exit 1
fi

exit 0