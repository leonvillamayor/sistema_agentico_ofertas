#!/bin/bash
# Hook para validar la generación de diagramas draw.io
# Se ejecuta después de que el diagramador procesa las oportunidades

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

echo "🎨 Verificando generación de diagramas..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Variables para tracking
TOTAL_MDS=0
TOTAL_FUNC_DIAGRAMS=0
TOTAL_TECH_DIAGRAMS=0
MISSING_FUNC=()
MISSING_TECH=()
VALIDATION_ERRORS=0

# Buscar todas las carpetas de empresas en oportunidades
for empresa_dir in oportunidades/*/; do
    if [ -d "$empresa_dir" ]; then
        empresa=$(basename "$empresa_dir")
        echo ""
        echo "📁 Procesando empresa: $empresa"
        echo "────────────────────────────"

        # Contar archivos markdown de oportunidades (excluyendo descripcion.md)
        md_files=0
        func_diagrams=0
        tech_diagrams=0

        for md_file in "$empresa_dir"*.md; do
            if [ -f "$md_file" ]; then
                filename=$(basename "$md_file" .md)

                # Saltar descripcion.md
                if [ "$filename" == "descripcion" ]; then
                    continue
                fi

                ((md_files++))
                ((TOTAL_MDS++))

                # Verificar diagrama funcional
                func_diagram="${empresa_dir}${filename}_arq_func.drawio"
                if [ -f "$func_diagram" ]; then
                    ((func_diagrams++))
                    ((TOTAL_FUNC_DIAGRAMS++))

                    # Validar contenido mínimo del diagrama funcional
                    if ! grep -q "Arquitectura Funcional" "$func_diagram" 2>/dev/null; then
                        log_warn "Diagrama funcional sin título correcto: $(basename "$func_diagram")"
                        ((VALIDATION_ERRORS++))
                    fi

                    # Verificar tamaño mínimo (500 bytes)
                    size=$(wc -c < "$func_diagram")
                    if [ "$size" -lt 500 ]; then
                        log_warn "Diagrama funcional muy pequeño: $(basename "$func_diagram") ($size bytes)"
                        ((VALIDATION_ERRORS++))
                    fi
                else
                    MISSING_FUNC+=("$empresa/$filename")
                    log_warn "Falta diagrama funcional: ${filename}_arq_func.drawio"
                fi

                # Verificar diagrama técnico
                tech_diagram="${empresa_dir}${filename}_arq_tec.drawio"
                if [ -f "$tech_diagram" ]; then
                    ((tech_diagrams++))
                    ((TOTAL_TECH_DIAGRAMS++))

                    # Validar contenido mínimo del diagrama técnico
                    if ! grep -q "AWS Cloud" "$tech_diagram" 2>/dev/null; then
                        log_warn "Diagrama técnico sin elementos AWS: $(basename "$tech_diagram")"
                        ((VALIDATION_ERRORS++))
                    fi

                    # Verificar servicios AWS en el diagrama
                    if ! grep -qE "(EC2|Lambda|S3|RDS|DynamoDB|VPC|ECS|EKS)" "$tech_diagram" 2>/dev/null; then
                        log_warn "Diagrama técnico sin servicios AWS específicos"
                        ((VALIDATION_ERRORS++))
                    fi

                    # Verificar tamaño mínimo (1000 bytes)
                    size=$(wc -c < "$tech_diagram")
                    if [ "$size" -lt 1000 ]; then
                        log_warn "Diagrama técnico muy pequeño: $(basename "$tech_diagram") ($size bytes)"
                        ((VALIDATION_ERRORS++))
                    fi
                else
                    MISSING_TECH+=("$empresa/$filename")
                    log_warn "Falta diagrama técnico: ${filename}_arq_tec.drawio"
                fi
            fi
        done

        # Resumen por empresa
        if [ "$md_files" -gt 0 ]; then
            echo "  📄 Documentos de oportunidad: $md_files"
            echo "  🎨 Diagramas funcionales: $func_diagrams/$md_files"
            echo "  🔧 Diagramas técnicos: $tech_diagrams/$md_files"

            if [ "$func_diagrams" -eq "$md_files" ] && [ "$tech_diagrams" -eq "$md_files" ]; then
                log_info "Todos los diagramas generados correctamente"
            else
                log_warn "Faltan diagramas por generar"
            fi
        fi
    fi
done

# Validación de estructura XML de diagramas
echo ""
echo "🔍 Validando estructura XML de diagramas..."
for drawio_file in oportunidades/*/*.drawio; do
    if [ -f "$drawio_file" ]; then
        # Verificar que es XML válido
        if ! grep -q "<?xml version" "$drawio_file"; then
            log_error "Archivo no es XML válido: $(basename "$drawio_file")"
            ((VALIDATION_ERRORS++))
        fi

        # Verificar estructura mxfile
        if ! grep -q "<mxfile" "$drawio_file"; then
            log_error "Estructura mxfile faltante: $(basename "$drawio_file")"
            ((VALIDATION_ERRORS++))
        fi

        # Verificar que tiene al menos un diagrama
        if ! grep -q "<diagram" "$drawio_file"; then
            log_error "Sin diagramas en archivo: $(basename "$drawio_file")"
            ((VALIDATION_ERRORS++))
        fi
    fi
done

# Generar estadísticas finales
echo ""
echo "📊 Estadísticas de Diagramación"
echo "═══════════════════════════════════════════"
echo "  📄 Total documentos de oportunidad: $TOTAL_MDS"
echo "  🎨 Diagramas funcionales creados: $TOTAL_FUNC_DIAGRAMS"
echo "  🔧 Diagramas técnicos creados: $TOTAL_TECH_DIAGRAMS"
echo "  ⚠️  Errores de validación: $VALIDATION_ERRORS"
echo ""

# Calcular completitud
if [ "$TOTAL_MDS" -gt 0 ]; then
    FUNC_PERCENTAGE=$((TOTAL_FUNC_DIAGRAMS * 100 / TOTAL_MDS))
    TECH_PERCENTAGE=$((TOTAL_TECH_DIAGRAMS * 100 / TOTAL_MDS))
    echo "  📈 Completitud funcional: ${FUNC_PERCENTAGE}%"
    echo "  📈 Completitud técnica: ${TECH_PERCENTAGE}%"
fi

# Determinar estado final
FINAL_STATUS="success"
FINAL_MESSAGE="Validación de diagramas completada"

if [ ${#MISSING_FUNC[@]} -gt 0 ] || [ ${#MISSING_TECH[@]} -gt 0 ]; then
    FINAL_STATUS="warning"
    FINAL_MESSAGE="Faltan diagramas por generar"

    if [ ${#MISSING_FUNC[@]} -gt 0 ]; then
        echo ""
        echo "⚠️ Diagramas funcionales faltantes:"
        for missing in "${MISSING_FUNC[@]}"; do
            echo "   - $missing"
        done
    fi

    if [ ${#MISSING_TECH[@]} -gt 0 ]; then
        echo ""
        echo "⚠️ Diagramas técnicos faltantes:"
        for missing in "${MISSING_TECH[@]}"; do
            echo "   - $missing"
        done
    fi
fi

if [ "$VALIDATION_ERRORS" -gt 5 ]; then
    FINAL_STATUS="error"
    FINAL_MESSAGE="Múltiples errores de validación detectados"
fi

# Crear registro de validación
LOG_FILE="oportunidades/.diagrams_validation_log.txt"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Validación: Docs:$TOTAL_MDS Func:$TOTAL_FUNC_DIAGRAMS Tech:$TOTAL_TECH_DIAGRAMS Errors:$VALIDATION_ERRORS Status:$FINAL_STATUS" >> "$LOG_FILE"

echo ""
echo "═══════════════════════════════════════════"
if [ "$FINAL_STATUS" == "success" ]; then
    log_info "✅ $FINAL_MESSAGE"
elif [ "$FINAL_STATUS" == "warning" ]; then
    log_warn "⚠️ $FINAL_MESSAGE"
else
    log_error "❌ $FINAL_MESSAGE"
fi
echo "═══════════════════════════════════════════"

# Retornar JSON para Claude Code
cat << EOF
{
  "status": "$FINAL_STATUS",
  "totalDocuments": $TOTAL_MDS,
  "functionalDiagrams": $TOTAL_FUNC_DIAGRAMS,
  "technicalDiagrams": $TOTAL_TECH_DIAGRAMS,
  "missingFunctional": ${#MISSING_FUNC[@]},
  "missingTechnical": ${#MISSING_TECH[@]},
  "validationErrors": $VALIDATION_ERRORS,
  "message": "$FINAL_MESSAGE"
}
EOF

# Exit code basado en estado
if [ "$FINAL_STATUS" == "error" ]; then
    exit 1
fi

exit 0