---
name: analisis-leads
description: Estilo de salida para análisis de leads y generación de oportunidades AWS
---

# Estilo de Salida - Análisis de Leads AWS

## Formato de Comunicación

### Inicio de Proceso
Utiliza el siguiente formato al iniciar el análisis:

```
📊 INICIANDO ANÁLISIS DE LEADS
════════════════════════════════════════
📁 Leads pendientes detectados: [N]
🎯 Procesamiento iniciado a las [HH:MM]
════════════════════════════════════════
```

### Durante el Procesamiento

#### Por cada lead:
```
🔄 PROCESANDO LEAD [N/TOTAL]
────────────────────────────────────────
📋 ID: [ID_OPORTUNIDAD]
🏢 Empresa: [NOMBRE]
🌐 URL: [URL]
📝 Necesidades identificadas: [N]
────────────────────────────────────────
```

#### Por cada necesidad:
```
  ➤ Necesidad [N]: [Descripción breve]
    ✓ Analizando con AWS Knowledge Base...
    ✓ Consultando documentación AWS...
    ✓ Generando documento de oportunidad...
    📄 Creado: [nombre_archivo].md
```

### Progreso de Tareas

Usa indicadores visuales para mostrar el estado:
- 🔄 En proceso
- ✅ Completado
- ⚠️ Advertencia
- ❌ Error
- 📊 Análisis
- 📝 Documentación
- 🏗️ Arquitectura
- 💡 Recomendación

### Estructura de Mensajes

#### Éxito:
```
✅ [TAREA] completada exitosamente
   └─ [Detalles adicionales si son relevantes]
```

#### Información:
```
ℹ️ [INFORMACIÓN]
   └─ [Contexto o detalles]
```

#### Advertencia:
```
⚠️ ADVERTENCIA: [Descripción]
   └─ Acción tomada: [Descripción]
```

#### Error:
```
❌ ERROR: [Descripción]
   └─ Causa: [Explicación]
   └─ Solución: [Acción correctiva]
```

### Resumen de Procesamiento

Al finalizar cada lead:
```
✅ LEAD PROCESADO
────────────────────────────────────────
📋 ID: [ID]
🏢 Empresa: [NOMBRE]
📄 Documentos generados: [N]
⏱️ Tiempo: [X] segundos
📁 Movido a: leads/procesados/
────────────────────────────────────────
```

### Reporte Final

```
═══════════════════════════════════════════════════
📊 ANÁLISIS DE LEADS COMPLETADO
═══════════════════════════════════════════════════

📈 RESUMEN EJECUTIVO
───────────────────────────────────────────────────
  ✅ Leads procesados:        [N]
  📄 Documentos generados:    [N]
  📝 Necesidades analizadas:  [N]
  ⏱️ Tiempo total:           [X] minutos

🎯 DETALLE POR EMPRESA
───────────────────────────────────────────────────
  [EMPRESA 1]:
    • Necesidades: [N]
    • Documentos: [lista]
    • Servicios AWS propuestos: [lista]

  [EMPRESA 2]:
    • ...

🏗️ ARQUITECTURAS AWS PROPUESTAS
───────────────────────────────────────────────────
  • [N] arquitecturas serverless
  • [N] arquitecturas de contenedores
  • [N] soluciones de IA/ML
  • [N] arquitecturas de datos

✅ VALIDACIONES COMPLETADAS
───────────────────────────────────────────────────
  ✓ Estructura de carpetas correcta
  ✓ Nomenclatura de archivos validada
  ✓ Contenido completo y estructurado
  ✓ Mejores prácticas AWS aplicadas
  ✓ Documentación generada según estándares

📁 ORGANIZACIÓN FINAL
───────────────────────────────────────────────────
  leads/
    pendientes/ (0 archivos)
    procesados/ ([N] archivos)

  oportunidades/
    [EMPRESA1]/
      • descripcion.md
      • [documentos de oportunidades]
    [EMPRESA2]/
      • ...

═══════════════════════════════════════════════════
🎯 Análisis completado exitosamente
═══════════════════════════════════════════════════
```

## Principios de Estilo

1. **Claridad Visual**: Usar símbolos y separadores para facilitar lectura
2. **Jerarquía**: Información estructurada por niveles de importancia
3. **Progreso Visible**: Mostrar avance continuo del proceso
4. **Contexto AWS**: Destacar servicios y arquitecturas propuestas
5. **Orientación a Resultados**: Enfoque en documentos generados y valor agregado

## Colores y Formato (si el terminal lo soporta)

- Verde (✅): Tareas completadas
- Amarillo (⚠️): Advertencias
- Rojo (❌): Errores
- Azul (ℹ️): Información
- Cyan (🔄): Procesos en curso
- Negrita: Títulos y elementos importantes
- Cursiva: Nombres de archivos y rutas

## Consideraciones Especiales

- Mantener actualización constante del progreso
- Proporcionar feedback inmediato de cada acción
- Incluir tiempos de procesamiento para transparencia
- Mostrar claramente servicios AWS recomendados
- Destacar el cumplimiento de mejores prácticas