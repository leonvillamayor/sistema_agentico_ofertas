---
name: analisis_lead
description: Analiza leads pendientes y genera documentación de oportunidades AWS
params:
  - name: modo
    description: Modo de ejecución (completo|verificar|reporte)
    default: completo
---

# Comando: Análisis de Leads AWS

## Descripción
Procesa automáticamente los leads pendientes, analiza las necesidades empresariales y genera documentación completa de oportunidades aplicando las mejores prácticas de AWS.

## Sintaxis
```
/analisis_lead [modo]
```

## Parámetros
- **modo** (opcional):
  - `completo`: Ejecuta el análisis completo (por defecto)
  - `verificar`: Solo verifica el estado de los leads
  - `reporte`: Genera reporte del último procesamiento

## Flujo de Ejecución

### 1. Inicialización
```bash
# Verificar estructura de directorios
if [ ! -d "leads/pendientes" ] || [ ! -d "leads/procesados" ]; then
    echo "⚠️ Creando estructura de directorios..."
    mkdir -p leads/{pendientes,procesados}
    mkdir -p oportunidades
fi
```

### 2. Activación del Orquestador
El comando activa el subagente orquestador que coordina todo el proceso:

```bash
# Invocar al orquestador con el output-style configurado
claude task --agent orquestador --output-style analisis-leads << 'EOF'
MISIÓN: Coordinar el análisis completo de leads pendientes

1. VERIFICACIÓN INICIAL
   - Confirmar existencia de carpetas leads/pendientes y leads/procesados
   - Verificar disponibilidad de herramientas AWS (aws-docs, aws-knowledge)
   - Contar archivos en leads/pendientes

2. ACTIVACIÓN DEL ARQUITECTO DE SOLUCIONES
   Invocar al subagente arquitecto_soluciones_leads con la siguiente instrucción:

   "Analizar TODOS los archivos en leads/pendientes siguiendo el flujo establecido:
    - Para cada archivo, extraer ID, empresa y URL
    - Crear estructura en oportunidades/[EMPRESA]/
    - Generar descripcion.md si no existe
    - Para cada necesidad, generar documento completo con arquitectura AWS
    - Aplicar mejores prácticas y Well-Architected Framework
    - Mover archivo procesado a leads/procesados/"

3. SUPERVISIÓN CONTINUA
   - Monitorear progreso del arquitecto
   - Verificar que cada necesidad genera un documento
   - Validar nomenclatura de archivos
   - Confirmar movimiento de archivos procesados

4. VALIDACIÓN DE CALIDAD
   Para cada documento generado verificar:
   - Estructura completa según plantilla
   - Servicios AWS específicos incluidos
   - Arquitectura claramente definida
   - Beneficios y métricas identificadas
   - Próximos pasos concretos

5. GENERACIÓN DE REPORTE FINAL
   Crear resumen con:
   - Total de leads procesados
   - Documentos generados por empresa
   - Servicios AWS propuestos
   - Tiempo de procesamiento
   - Validaciones completadas

IMPORTANTE: No finalizar hasta que TODOS los leads estén procesados y movidos.
EOF
```

### 3. Proceso Principal

El orquestador ejecutará:

1. **Fase de Análisis**:
   - El arquitecto_soluciones_leads analiza cada lead
   - Busca información de la empresa via WebFetch
   - Consulta documentación AWS para cada necesidad
   - Genera documentos de oportunidad estructurados

2. **Fase de Validación**:
   - Verifica completitud de documentación
   - Valida aplicación de mejores prácticas
   - Confirma estructura y nomenclatura

3. **Fase de Diagramación**:
   - El diagramador crea diagramas funcionales y técnicos
   - Genera archivos draw.io para cada oportunidad
   - Valida completitud de diagramas

4. **Fase de Finalización**:
   - Mueve archivos procesados
   - Genera reporte de procesamiento
   - Confirma integridad del proceso

## Estructura de Salida Esperada

```
oportunidades/
├── [EMPRESA1]/
│   ├── descripcion.md
│   ├── [ID]-[DESC1]-[FECHA].md
│   └── [ID]-[DESC2]-[FECHA].md
└── [EMPRESA2]/
    ├── descripcion.md
    └── [ID]-[DESC]-[FECHA].md
```

## Hooks de Verificación

El comando integra hooks para asegurar la calidad:

1. **Pre-ejecución**: Valida estructura de directorios
2. **Durante proceso**: Verifica generación de documentos
3. **Post-ejecución**: Confirma movimiento de archivos

## Manejo de Errores

- Si no hay leads pendientes: Mensaje informativo
- Si falla análisis de un lead: Continuar con siguientes
- Si no se puede acceder a AWS docs: Usar conocimiento base
- Si falla movimiento de archivo: Registrar y continuar

## Ejemplo de Uso

```bash
# Análisis completo
/analisis_lead

# Solo verificación
/analisis_lead verificar

# Generar reporte
/analisis_lead reporte
```

## Salida Esperada

Al ejecutar el comando se mostrará:
1. Estado inicial y número de leads
2. Progreso de procesamiento por lead
3. Documentos generados para cada necesidad
4. Resumen final con estadísticas
5. Confirmación de archivos movidos

## Integración con Output Style

El comando utiliza automáticamente el output-style `analisis-leads` para formatear la salida con:
- Indicadores visuales de progreso
- Separadores claros entre secciones
- Resaltado de información importante
- Estadísticas finales estructuradas