#!/bin/bash
# Hook post-procesamiento para verificar completitud del análisis de leads
# Se ejecuta después del procesamiento de cada lead

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para logging
log_info() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_debug() { echo -e "${BLUE}ℹ${NC} $1"; }

# Recibir parámetros del lead procesado
LEAD_FILE="${1:-}"
EMPRESA="${2:-}"
ID_OPORTUNIDAD="${3:-}"

if [ -z "$LEAD_FILE" ] || [ -z "$EMPRESA" ] || [ -z "$ID_OPORTUNIDAD" ]; then
    log_error "Parámetros faltantes: LEAD_FILE, EMPRESA, ID_OPORTUNIDAD"
    exit 1
fi

echo "🔍 Verificando procesamiento de: $LEAD_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Verificar que el directorio de la empresa existe
EMPRESA_DIR="oportunidades/$EMPRESA"
if [ ! -d "$EMPRESA_DIR" ]; then
    log_error "Directorio de empresa no creado: $EMPRESA_DIR"
    exit 1
fi
log_info "Directorio de empresa existe: $EMPRESA_DIR"

# 2. Verificar archivo descripcion.md
DESCRIPCION_FILE="$EMPRESA_DIR/descripcion.md"
if [ ! -f "$DESCRIPCION_FILE" ]; then
    log_error "Archivo descripcion.md no encontrado"
    exit 1
fi

# Verificar contenido mínimo de descripcion.md
MIN_SIZE=500
ACTUAL_SIZE=$(wc -c < "$DESCRIPCION_FILE")
if [ "$ACTUAL_SIZE" -lt "$MIN_SIZE" ]; then
    log_warn "descripcion.md parece incompleto (< 500 caracteres)"
else
    log_info "Archivo descripcion.md creado correctamente"
fi

# 3. Verificar documentos de oportunidades generados
FECHA=$(date +%d%m%Y)
OPPORTUNITY_DOCS=$(find "$EMPRESA_DIR" -name "${ID_OPORTUNIDAD}-*-${FECHA}.md" -type f | wc -l)

if [ "$OPPORTUNITY_DOCS" -eq 0 ]; then
    log_error "No se encontraron documentos de oportunidad para ID: $ID_OPORTUNIDAD"
    exit 1
fi
log_info "Documentos de oportunidad generados: $OPPORTUNITY_DOCS"

# 4. Validar estructura de documentos de oportunidad
echo "Validando estructura de documentos..."
VALIDATION_ERRORS=0

for doc in "$EMPRESA_DIR"/${ID_OPORTUNIDAD}-*-${FECHA}.md; do
    if [ -f "$doc" ]; then
        filename=$(basename "$doc")

        # Verificar secciones requeridas
        REQUIRED_SECTIONS=(
            "# Oportunidad"
            "## Resumen Ejecutivo"
            "## Análisis de la Necesidad"
            "## Solución Propuesta AWS"
            "## Beneficios"
            "## Próximos Pasos"
        )

        for section in "${REQUIRED_SECTIONS[@]}"; do
            if ! grep -q "$section" "$doc"; then
                log_warn "Sección faltante en $filename: $section"
                ((VALIDATION_ERRORS++))
            fi
        done

        # Verificar mención de servicios AWS
        if ! grep -qE "(EC2|S3|Lambda|RDS|DynamoDB|ECS|EKS|CloudFormation|API Gateway)" "$doc"; then
            log_warn "No se mencionan servicios AWS específicos en $filename"
        fi

        # Verificar tamaño mínimo (2KB)
        DOC_SIZE=$(wc -c < "$doc")
        if [ "$DOC_SIZE" -lt 2048 ]; then
            log_warn "Documento $filename parece incompleto (< 2KB)"
            ((VALIDATION_ERRORS++))
        fi
    fi
done

if [ "$VALIDATION_ERRORS" -eq 0 ]; then
    log_info "Estructura de documentos validada correctamente"
else
    log_warn "Se encontraron $VALIDATION_ERRORS advertencias de validación"
fi

# 5. Verificar que el archivo fue movido a procesados
PROCESSED_FILE="leads/procesados/$(basename "$LEAD_FILE")"
if [ ! -f "$PROCESSED_FILE" ]; then
    log_warn "Archivo no movido a procesados. Moviendo ahora..."
    mv "leads/pendientes/$(basename "$LEAD_FILE")" "$PROCESSED_FILE" 2>/dev/null || true
fi

if [ -f "$PROCESSED_FILE" ]; then
    log_info "Archivo movido correctamente a procesados"
else
    log_error "Error al mover archivo a procesados"
fi

# 6. Generar estadísticas
echo ""
echo "📊 Estadísticas del Procesamiento"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Contar servicios AWS mencionados
AWS_SERVICES_COUNT=0
for doc in "$EMPRESA_DIR"/${ID_OPORTUNIDAD}-*-${FECHA}.md; do
    if [ -f "$doc" ]; then
        services=$(grep -oE "(EC2|S3|Lambda|RDS|DynamoDB|ECS|EKS|CloudFormation|API Gateway|SageMaker|Bedrock|Athena|Glue|Kinesis|SNS|SQS|Route 53|CloudFront|WAF)" "$doc" | sort -u | wc -l)
        AWS_SERVICES_COUNT=$((AWS_SERVICES_COUNT + services))
    fi
done

echo "  • Documentos generados: $OPPORTUNITY_DOCS"
echo "  • Servicios AWS únicos propuestos: $AWS_SERVICES_COUNT"
echo "  • Tamaño total documentación: $(du -sh "$EMPRESA_DIR" | cut -f1)"

# 7. Verificar integridad general
FINAL_STATUS="success"
FINAL_MESSAGE="Lead procesado correctamente"

if [ "$VALIDATION_ERRORS" -gt 0 ]; then
    FINAL_STATUS="warning"
    FINAL_MESSAGE="Lead procesado con advertencias"
fi

# 8. Crear registro de procesamiento
LOG_FILE="oportunidades/.processing_log.txt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Procesado: $LEAD_FILE | Empresa: $EMPRESA | Docs: $OPPORTUNITY_DOCS | Status: $FINAL_STATUS" >> "$LOG_FILE"

echo ""
echo "═══════════════════════════════════════════"
echo "✅ VERIFICACIÓN POST-PROCESAMIENTO COMPLETA"
echo "═══════════════════════════════════════════"
echo "  Estado: $FINAL_MESSAGE"
echo "  Lead: $(basename "$LEAD_FILE")"
echo "  Empresa: $EMPRESA"
echo "  Documentos: $OPPORTUNITY_DOCS"
echo "═══════════════════════════════════════════"

# Retornar JSON para Claude Code
cat << EOF
{
  "status": "$FINAL_STATUS",
  "leadFile": "$LEAD_FILE",
  "empresa": "$EMPRESA",
  "documentosGenerados": $OPPORTUNITY_DOCS,
  "serviciosAWS": $AWS_SERVICES_COUNT,
  "validationErrors": $VALIDATION_ERRORS,
  "message": "$FINAL_MESSAGE"
}
EOF

exit 0