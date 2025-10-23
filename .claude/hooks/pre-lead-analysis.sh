#!/bin/bash
# Hook de pre-ejecución para análisis de leads
# Se ejecuta antes de iniciar el procesamiento de leads

set -e

echo "🔍 Ejecutando validaciones pre-análisis..."

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para logging
log_info() { echo -e "${GREEN}✓${NC} $1"; }
log_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }

# 1. Verificar estructura de directorios
echo "Verificando estructura de directorios..."

REQUIRED_DIRS=("leads/pendientes" "leads/procesados" "oportunidades")
MISSING_DIRS=()

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        MISSING_DIRS+=("$dir")
    fi
done

if [ ${#MISSING_DIRS[@]} -gt 0 ]; then
    log_warn "Directorios faltantes detectados. Creando..."
    for dir in "${MISSING_DIRS[@]}"; do
        mkdir -p "$dir"
        log_info "Creado: $dir"
    done
else
    log_info "Estructura de directorios correcta"
fi

# 2. Verificar archivos pendientes
PENDING_FILES=$(find leads/pendientes -type f -name "*.txt" 2>/dev/null | wc -l)

if [ "$PENDING_FILES" -eq 0 ]; then
    log_warn "No hay archivos pendientes en leads/pendientes/"
    echo "ℹ️ Agregue archivos con el formato: ID-EMPRESA-OPORTUNIDAD.txt"
    exit 1
else
    log_info "Archivos pendientes encontrados: $PENDING_FILES"
fi

# 3. Validar formato de nombres de archivo
echo "Validando formato de archivos..."
INVALID_FILES=()

for file in leads/pendientes/*.txt; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .txt)
        # Verificar formato: ID-EMPRESA-OPORTUNIDAD
        if ! [[ "$filename" =~ ^[0-9]+-[^-]+-[^-]+$ ]]; then
            INVALID_FILES+=("$file")
        fi
    fi
done

if [ ${#INVALID_FILES[@]} -gt 0 ]; then
    log_error "Archivos con formato inválido:"
    for file in "${INVALID_FILES[@]}"; do
        echo "  - $(basename "$file")"
    done
    echo "ℹ️ Formato esperado: ID-EMPRESA-OPORTUNIDAD.txt"
    exit 1
else
    log_info "Todos los archivos tienen formato válido"
fi

# 4. Verificar contenido mínimo de archivos
echo "Verificando contenido de archivos..."
INCOMPLETE_FILES=()

for file in leads/pendientes/*.txt; do
    if [ -f "$file" ]; then
        # Verificar que contenga al menos "Nombre Empresa:" y "Necesidad"
        if ! grep -q "Nombre Empresa:" "$file" || ! grep -q "Necesidad" "$file"; then
            INCOMPLETE_FILES+=("$file")
        fi
    fi
done

if [ ${#INCOMPLETE_FILES[@]} -gt 0 ]; then
    log_warn "Archivos con contenido incompleto:"
    for file in "${INCOMPLETE_FILES[@]}"; do
        echo "  - $(basename "$file")"
    done
    echo "ℹ️ Asegúrese de incluir 'Nombre Empresa:' y al menos una 'Necesidad'"
fi

# 5. Verificar permisos de escritura
echo "Verificando permisos..."
if [ ! -w "oportunidades" ]; then
    log_error "Sin permisos de escritura en directorio oportunidades/"
    exit 1
fi
log_info "Permisos de escritura verificados"

# 6. Verificar espacio en disco
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
MIN_SPACE=102400  # 100MB mínimo

if [ "$AVAILABLE_SPACE" -lt "$MIN_SPACE" ]; then
    log_error "Espacio insuficiente en disco (mínimo 100MB)"
    exit 1
fi
log_info "Espacio en disco adecuado"

# 7. Crear backup de leads pendientes
echo "Creando backup de seguridad..."
BACKUP_DIR=".backups/leads/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r leads/pendientes/* "$BACKUP_DIR/" 2>/dev/null || true
log_info "Backup creado en $BACKUP_DIR"

# Resumen final
echo ""
echo "═══════════════════════════════════════════"
echo "📊 VALIDACIÓN PRE-ANÁLISIS COMPLETADA"
echo "═══════════════════════════════════════════"
echo "  ✅ Estructura de directorios: OK"
echo "  ✅ Archivos pendientes: $PENDING_FILES"
echo "  ✅ Formato de archivos: OK"
echo "  ✅ Permisos: OK"
echo "  ✅ Espacio en disco: OK"
echo "  ✅ Backup creado: OK"
echo "═══════════════════════════════════════════"
echo ""
echo "✅ Sistema listo para procesar leads"

# Retornar JSON para Claude Code
cat << EOF
{
  "status": "success",
  "pendingFiles": $PENDING_FILES,
  "ready": true,
  "message": "Validaciones completadas exitosamente"
}
EOF

exit 0