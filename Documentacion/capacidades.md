# Claude Code - Documentación Completa de Funcionalidades (2025)

## Índice
1. [Introducción y Visión General](#introducción-y-visión-general)
2. [Instalación y Configuración Inicial](#instalación-y-configuración-inicial)
3. [Modos de Acceso y Plataformas](#modos-de-acceso-y-plataformas)
4. [Funcionalidades Principales de Codificación](#funcionalidades-principales-de-codificación)
5. [Sistema de Subagentes](#sistema-de-subagentes)
6. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
7. [Comandos CLI y Modo Interactivo](#comandos-cli-y-modo-interactivo)
8. [Slash Commands](#slash-commands)
9. [Hooks y Automatización](#hooks-y-automatización)
10. [Plugins y Skills](#plugins-y-skills)
11. [Output Styles y Personalización](#output-styles-y-personalización)
12. [Claude Agent SDK](#claude-agent-sdk)
13. [Integraciones con IDEs](#integraciones-con-ides)
14. [Deployment y Cloud](#deployment-y-cloud)
15. [CI/CD y Automatización](#cicd-y-automatización)
16. [Administración y Seguridad](#administración-y-seguridad)
17. [Monitoreo y Análisis](#monitoreo-y-análisis)
18. [Gestión de Contexto y Memoria](#gestión-de-contexto-y-memoria)
19. [Sandboxing y Aislamiento](#sandboxing-y-aislamiento)
20. [Testing y Debugging](#testing-y-debugging)
21. [Características Avanzadas](#características-avanzadas)
22. [Modelos Disponibles](#modelos-disponibles)
23. [Mejores Prácticas](#mejores-prácticas)

---

## Introducción y Visión General

Claude Code es la herramienta de codificación agéntica oficial de Anthropic que vive en tu terminal y te ayuda a convertir ideas en código más rápido que nunca. Es el mejor modelo de codificación del mundo, el más fuerte para construir agentes complejos y el mejor modelo para usar computadoras.

### Características Principales
- **Codificación Autónoma**: Maneja más de 30 horas de codificación autónoma
- **Navegación de Código**: Navega por tu base de código, edita múltiples archivos y ejecuta comandos para verificar su trabajo
- **Multi-Plataforma**: Disponible en terminal/CLI, VS Code, web y iOS
- **Coherencia**: Mantiene coherencia a través de bases de código masivas

---

## Instalación y Configuración Inicial

### Instalación via npm
```bash
npm install -g @anthropic-ai/claude-code
```

### Inicio Rápido
```bash
claude
```

### Requisitos del Sistema
- Node.js 14 o superior
- Acceso a terminal/consola
- Conexión a internet para autenticación

---

## Modos de Acceso y Plataformas

### 1. Terminal/CLI
- Instalación mediante npm
- Comando `claude` para iniciar sesión interactiva
- Soporte para modo headless con flag `-p`

### 2. VS Code Extension
- Extensión nativa disponible en el marketplace
- Lanzamiento rápido con Cmd+Esc (Mac) o Ctrl+Esc (Windows/Linux)
- Integración automática con el terminal integrado

### 3. Claude Code en la Web
- Beta/preview de investigación
- Ejecuta tareas múltiples en paralelo
- Infraestructura gestionada por Anthropic en la nube
- Creación automática de PRs

### 4. Aplicación iOS
- Disponible para desarrolladores móviles
- Permite explorar codificación con Claude sobre la marcha

---

## Funcionalidades Principales de Codificación

### Comprensión de Base de Código
- Obtener visión general rápida de la estructura del proyecto
- Encontrar archivos de código relevantes
- Rastrear flujos de ejecución de código
- Entender patrones de arquitectura
- Identificar modelos de datos clave

### Corrección de Bugs y Optimización
- Diagnosticar mensajes de error
- Sugerir correcciones de código
- Rastrear problemas desde front-end hasta backend
- Recomendar mejoras de rendimiento

### Refactorización de Código
- Identificar uso de APIs obsoletas
- Sugerir características modernas del lenguaje
- Mantener compatibilidad hacia atrás
- Mejorar código incrementalmente

### Asistencia al Desarrollo
- Crear pull requests
- Generar documentación
- Agregar/actualizar casos de prueba
- Manejar análisis de imágenes en desarrollo

---

## Sistema de Subagentes

Los subagentes son asistentes AI especializados que operan dentro de Claude Code, cada uno con su propia ventana de contexto, prompts del sistema personalizados y permisos específicos de herramientas.

### Características Clave
- Contexto separado para cada subagente
- Permisos y herramientas configurables
- Ejecución secuencial de llamadas a herramientas
- Prioridad de subagentes a nivel proyecto sobre nivel usuario

### Estructura de Configuración
```yaml
---
name: your-sub-agent-name
description: Descripción de cuándo debe invocarse este subagente
tools: tool1, tool2, tool3  # Opcional - hereda todas si se omite
model: sonnet  # Opcional - especificar modelo
---
```

### Uso Recomendado
- Problemas complejos que requieren investigación especializada
- Tareas que necesitan diferentes conjuntos de permisos
- Preservación de contexto en conversaciones largas
- Verificación de detalles o investigación de preguntas específicas

---

## Model Context Protocol (MCP)

Claude Code puede conectarse a cientos de herramientas y fuentes de datos externas a través del Model Context Protocol, un estándar de código abierto para integraciones AI-herramienta.

### Beneficios Clave
- **Interfaz Unificada**: API estandarizada para interacciones
- **Integración Sin Costuras**: Se integra fácilmente en flujos de trabajo existentes
- **Soporte Multi-fuente**: Recupera información de múltiples fuentes

### Métodos de Transporte
1. **HTTP** (recomendado para servidores remotos)
2. **SSE** (Server-Sent Events)
3. **Stdio** (comunicación local)

### Instalación de Servidores MCP
```bash
# Transporte HTTP (recomendado)
claude mcp add --transport http <name> <url>

# Con autenticación
claude mcp add --transport http secure-api https://api.example.com/mcp \
  --header "Authorization: Bearer your-token"
```

### Integraciones Populares
- **GitHub**: Operaciones completas de GitHub (issues, PRs, repositorios)
- **Context7**: Documentación actualizada y específica de versión
- **Plaid**: Análisis y optimización de integraciones bancarias
- **Square/Stripe**: Procesamiento de pagos
- **Puppeteer**: Automatización del navegador
- **SQLite**: Gestión de bases de datos

### Gestión de Tokens
- Umbral de advertencia: 10,000 tokens
- Límite configurable mediante MAX_MCP_OUTPUT_TOKENS
- Límite predeterminado: 25,000 tokens

---

## Comandos CLI y Modo Interactivo

### Comandos Esenciales
- `claude`: Iniciar sesión interactiva
- `claude -c`: Reanudar última conversación
- `claude -p "consulta"`: Modo de consulta rápida sin interacción
- `claude --model sonnet`: Usar modelo Sonnet más reciente
- `claude --model opus`: Usar modelo Opus más reciente

### Flags de Línea de Comandos
- `--add-dir ../frontend`: Dar acceso a directorios específicos
- `--output-format stream-json`: Salida JSON en streaming
- `-p`: Habilitar modo headless con prompt

### Modo Interactivo
- Interfaz conversacional con interacción AI en tiempo real
- Ejecución de herramientas y respuestas en streaming
- Soporte para referencias con @ y comandos shell con !

### Referencias Especiales
- **@archivo**: Referencia directa a archivos con autocompletado
- **@agente**: Invocación de agentes específicos
- **@recurso**: Acceso a recursos MCP
- **!comando**: Ejecución de comandos shell inline

### Modos de Permisos
- **Default Mode**: Requiere permiso antes de ejecutar cambios
- **Vibe Coder Mode**: Trabajo autónomo en código sin esperar permisos
- **Configuración**: `/permissions` para especificar comandos permitidos
- **Flag peligroso**: `--dangerously-skip-permissions` para saltar todos los permisos

---

## Slash Commands

### Comandos Integrados
- `/help`: Muestra todos los comandos disponibles
- `/init`: Escanea proyecto y crea archivo CLAUDE.md inicial
- `/clear`: Limpia historial de conversación
- `/compact`: Resume conversación manteniendo partes importantes
- `/config`: Abre menú de configuración
- `/agents`: Modifica acceso a herramientas interactivamente

### Comandos Personalizados
Los usuarios pueden crear comandos slash personalizados colocando archivos Markdown en directorios `.claude/commands/`:

#### Ubicaciones
- **Proyecto**: `.claude/commands/` - Accesible via `/project:command-name`
- **Personal**: `~/.claude/commands/` - Accesible via `/user:command-name`

#### Namespacing
- `.claude/commands/example.md` → `/example`
- `.claude/commands/frontend/component.md` → `/frontend:component`

---

## Hooks y Automatización

Los hooks permiten adjuntar comandos shell a eventos del ciclo de vida de Claude Code.

### Eventos Disponibles
- `SubagentStop`: Cuando un subagente termina
- `Stop`: Cuando termina la sesión principal
- Eventos personalizados definidos por el usuario

### Configuración
Los hooks se configuran en archivos de configuración y pueden:
- Ejecutar comandos shell
- Imprimir a STDOUT
- Mostrar siguientes pasos en el transcript de Claude

### Casos de Uso
- Validación automática de código
- Notificaciones de finalización
- Activación de pipelines CI/CD
- Logging personalizado

---

## Plugins y Skills

### Sistema de Plugins
Los plugins son una forma ligera de empaquetar y compartir:
- **Slash commands**: Atajos personalizados para operaciones frecuentes
- **Subagentes**: Agentes especializados para tareas de desarrollo
- **Servidores MCP**: Conexión a herramientas y fuentes de datos
- **Hooks**: Personalización del comportamiento de Claude Code

### Skills
Las skills extienden Claude Code con la experiencia y flujos de trabajo de tu equipo:
- Instalables vía plugins desde el marketplace anthropics/skills
- Claude las carga automáticamente cuando son relevantes
- Personalizables para flujos de trabajo específicos del equipo

### Marketplace de Plugins
- Directorio de plugins creados por la comunidad
- Más de 60 subagentes especializados disponibles
- Organizados en dominios (Desarrollo, Infraestructura, Negocios, etc.)

---

## Output Styles y Personalización

### Sistema de Output Styles
Output Styles permiten personalizar completamente la personalidad de Claude Code manteniendo todas sus capacidades poderosas. Reemplazan la personalidad central mientras preservan el ecosistema completo de herramientas.

### Estilos Integrados
- **Default**: Diseñado para tareas eficientes de ingeniería de software
- **Explanatory**: Proporciona "Insights" educativos mientras ayuda con tareas
- **Learning**: Modo colaborativo con marcadores TODO(human) para implementación

### Características de Output Styles
- Preservan sistema de contexto CLAUDE.md
- Mantienen ecosistema completo de herramientas
- Soportan delegación a subagentes
- Compatible con integraciones MCP
- Gestión de contexto y automatización
- Operaciones del sistema de archivos

### Creación y Gestión
```bash
# Ver estilos disponibles
/output-style

# Crear nuevo estilo
/output-style:new

# Cambiar estilo via config
/config
```

### Ubicación de Archivos
- **Nivel Usuario**: `~/.claude/output-styles`
- **Nivel Proyecto**: `.claude/output-styles`
- **Configuración**: `.claude/settings.local.json`

### Estructura de Archivo de Estilo
```yaml
---
name: nombre-del-estilo
description: Descripción del estilo
---

# Instrucciones personalizadas
[Contenido específico del dominio y comportamiento]
```

### Status Line Personalizable
Monitoreo configurable de:
- Uso de ventana de contexto
- Tipo de modelo
- Rutas de directorio
- Barras de progreso
- Porcentajes de completación

### Formatos de Salida
- **HTML**: Para contenido web y presentaciones
- **Markdown**: Para documentación profesional
- **JSON**: Para integración con sistemas
- **Texto plano**: Para salida estándar

### Herramientas de la Comunidad
- **ccoutputstyles**: CLI tool con más de 15 plantillas pre-construidas
- **ccstatusline**: Displays personalizables de información
- **claude-powerline**: Powerline estilo vim con tracking en tiempo real

---

## Claude Agent SDK

### Visión General
El Claude Agent SDK proporciona los bloques de construcción utilizados para hacer Claude Code. Es un framework poderoso que permite a desarrolladores construir, personalizar y extender las capacidades de Claude.

### Características Principales

#### Infraestructura Lista para Producción
- Manejo de errores integrado
- Gestión de sesiones
- Monitoreo y observabilidad desde el día uno
- Caché automático de prompts
- Optimizaciones de rendimiento

#### Gestión de Contexto
- Ventanas de contexto grandes
- Compactación y resumen automático de contexto
- Gestión de tokens detrás de escenas
- Mantenimiento de porciones sustanciales de código

### Métodos de Autenticación
- Claude API key directo
- Amazon Bedrock
- Google Vertex AI
- Variables de entorno configurables

### Integración MCP
- Conexión de herramientas personalizadas
- Integración con bases de datos
- APIs via Model Context Protocol
- SDKs oficiales de TypeScript

### Capacidades de Desarrollo

#### Herramientas Disponibles
- Búsqueda de archivos apropiados
- Escritura y edición de archivos
- Linting de código
- Ejecución y debugging
- Acciones iterativas hasta éxito

#### Python SDK
```python
# Método simple
response = agent.query("tu consulta aquí")

# APIs avanzadas con soporte de herramientas
result = agent.execute_with_tools(...)
```

### Casos de Uso

#### Agentes de Codificación
- Agentes SRE para problemas de producción
- Bots de revisión de seguridad
- Asistentes de ingeniería oncall
- Agentes de revisión de código

#### Agentes de Negocio
- Asistentes legales
- Asesores financieros
- Agentes de soporte al cliente
- Automatización de procesos

### Características de Seguridad
- Autonomía con barreras de seguridad
- Control de modos de permisos
- Especificación de herramientas permitidas/prohibidas
- Hooks para seguridad adicional
- PreToolUseHook para bloquear comandos peligrosos

### Rendimiento y Escalabilidad
- Reducción de latencia y costo
- Mejora de throughput
- Optimizaciones automáticas
- Gestión eficiente de recursos

---

## Integraciones con IDEs

### VS Code
#### Características
- Extensión nativa en el marketplace
- Lanzamiento rápido: Cmd+Esc (Mac) o Ctrl+Esc (Windows/Linux)
- Los cambios de código se muestran en el visor de diferencias del IDE
- Instalación automática al ejecutar `claude` desde terminal integrado

#### Funcionalidades
- Contexto de selección compartido automáticamente
- Insertar referencias de archivos: Cmd+Option+K (Mac) o Alt+Ctrl+K (Linux/Windows)
- Errores de diagnóstico compartidos automáticamente
- Integración con visor de diferencias nativo

### JetBrains IDEs
#### IDEs Soportados
- IntelliJ IDEA
- PyCharm
- Android Studio
- WebStorm
- PhpStorm
- GoLand

#### Instalación
Settings/Preferences → Plugins → Marketplace → buscar "Claude Code [Beta]"

#### Claude Agent Nativo
- Ejecuta directamente dentro del chat AI de JetBrains
- Capacidades completas del IDE
- No requiere plugins adicionales para suscriptores JetBrains AI

### Configuración IDE
- Comando `/ide` para conectar Claude Code a tu IDE
- Comando `/config` para personalizar vista de diferencias
- Soporte para desarrollo remoto

---

## Deployment y Cloud

### Amazon Bedrock
#### Características
- Enterprise-ready sin salida a proveedores externos
- Requiere Claude 3.7 Sonnet y Claude 3.5 Haiku
- Región predeterminada: us-east-1

#### Configuración
1. Habilitar Amazon Bedrock y solicitar acceso a modelos
2. Crear proveedor IAM OIDC para GitLab si es necesario
3. Crear rol IAM con políticas apropiadas
4. Configurar AmazonBedrockInferenceProfile

### Google Vertex AI
#### Características
- Seguridad a nivel empresarial
- Roles IAM para control de acceso
- Integración con Cloud Audit Logs
- Soporte para Workload Identity Federation

#### Requisitos
- Habilitar servicios relevantes en GCP
- Configurar variables de entorno (ANTHROPIC_VERTEX_PROJECT_ID, etc.)
- Configurar cuentas de servicio con permisos necesarios

### Consideraciones Enterprise
- Enrutamiento a través de proxies corporativos
- LLM Gateway central para gestión de acceso
- Control de uso y límites de gasto
- Pago basado en tokens consumidos

---

## CI/CD y Automatización

### GitHub Actions
#### Características
- Acción general para PRs e issues de GitHub
- Detección inteligente de modo de ejecución
- Soporte para múltiples métodos de autenticación
- Análisis y revisión de código automatizada

#### Capacidades
- Responder a menciones @claude
- Analizar cambios de PR y sugerir mejoras
- Implementar correcciones y nuevas características
- Crear PRs automáticamente

### GitLab CI/CD
#### Características
- Construido sobre Claude Code CLI y SDK
- Enterprise-ready con opciones de residencia de datos
- Creación instantánea de MRs
- Implementación automatizada desde issues

#### Funcionalidades
- Descripción de cambios → MR completo
- Seguimiento de directrices CLAUDE.md
- Ejecución en GitLab runners propios
- Respeto a protección de ramas y aprobaciones

### Automatización de Pipeline
- Opera independientemente en pipeline CI/CD
- Analiza, resume y revisa pull requests
- Modifica código después del envío
- Automatización de pasos completos del flujo de trabajo

---

## Administración y Seguridad

### Gestión de Asientos
- Autoservicio para gestión de asientos
- Asientos estándar para web app Claude.ai
- Asientos premium para desarrolladores con Claude Code
- Control granular de límites de gasto

### Control de Acceso (RBAC)
- Disponible para Claude Team Pro en Q4 2025
- Segregación de responsabilidades
- Minimización de riesgo interno
- Acceso solo a características necesarias

### Single Sign-On (SSO)
- Soporte para SAML 2.0 y OIDC
- Captura de dominio para inscripción automatizada
- Aprovisionamiento Just-in-Time (JIT)
- Integración con Okta, Azure AD, Ping Identity

### Seguridad
#### Cifrado
- TLS 1.2+ para todas las solicitudes de red
- AES-256 para logs almacenados y salidas
- KMS-backed provider-managed keys
- BYOK (Bring Your Own Key) próximamente en H1 2026

#### Clasificadores de Seguridad
- Construidos con U.S. National Nuclear Security Administration
- Detección de prompts de alto riesgo
- Alertas webhook para prompts peligrosos
- Bloqueo de instrucciones relacionadas con armas

### Auditoría
- Capacidades alineadas con SOC 2 Type II
- Seguimiento de inicios de sesión y sesiones
- Registro de uso de tokens API
- Retención de logs por 30 días
- Exportación a SIEM (Splunk, Datadog, Elastic)

---

## Monitoreo y Análisis

### Herramientas de Monitoreo

#### OpenTelemetry
- Captura de métricas, logs y trazas
- Feed detallado en tiempo real de actividad
- Integración con Prometheus y Grafana
- Dashboards personalizados y alertas

#### Datadog Integration
- Datos normalizados en formato FOCUS
- Dashboards y reportes CCM
- Monitoreo de costos por modelo
- Análisis de uso de caché y sesiones

### Herramientas de la Comunidad

#### Claude-Code-Usage-Monitor
- Monitoreo en tiempo real de uso de tokens
- Análisis avanzado con predicciones ML
- Seguimiento de consumo y tasa de quemado
- Análisis de costos

#### Raycast Extension (ccusage)
- Estadísticas en tiempo real
- Historial de sesiones
- Proyecciones mensuales de costos
- Estadísticas por modelo (Opus, Sonnet, Haiku)

#### CC Usage
- Monitoreo de bloques de 5 horas
- Dashboard en vivo
- Desglose de costos por modelo
- Filtrado por fecha con exportación JSON
- Integración MCP

### Gestión de Costos
- Límite de gasto total del workspace
- Seguimiento con LiteLLM (herramienta open-source)
- Análisis de tier de servicio (Standard, Batch, Priority)
- Optimización de uso de caché

---

## Gestión de Contexto y Memoria

### CLAUDE.md Files
#### Propósito
- Documentar etiqueta del repositorio
- Configuración del entorno de desarrollo
- Comportamientos específicos del proyecto
- Carga automática en contexto de Claude Code

#### Uso
- Crear en raíz del proyecto
- Hacer check-in en git para compartir con equipo
- Usar CLAUDE.local.md para configuración personal
- Mantener conciso y legible

### Gestión de Ventana de Contexto
- Comando `/compact` para resumir conversaciones
- Subagentes con contextos separados
- Preservación automática de información importante
- Optimización de consumo de tokens

### Checkpoints
- Guardan progreso y permiten retroceder
- Retroceso instantáneo a estado anterior
- Útil para exploración segura de cambios
- Gestión de sesiones complejas

---

## Sandboxing y Aislamiento

### Características de Seguridad
- Aislamiento de sistema de archivos
- Aislamiento de red
- Reducción de prompts de permisos en 84%
- Ejecución más autónoma de Claude

### Implementación

#### Sistema de Archivos
- Claude solo accede a directorios específicos
- Prevención de modificación de archivos del sistema
- Permisos Read/Edit configurables
- Reglas de denegación dentro de rutas permitidas

#### Red
- Conexión solo a servidores aprobados
- Prevención de fuga de información sensible
- Proxy server con restricciones de dominio
- Confirmación de usuario para nuevos dominios

### Tecnologías
- **macOS**: Seatbelt
- **Linux**: Bubblewrap
- Librería open source: anthropic-experimental/sandbox-runtime

### Development Containers
- Interacciones en contenedores con reglas estrictas
- Permisos con alcance de workspace
- Características de seguridad personalizables
- Ideal para trabajo con clientes seguros

---

## Testing y Debugging

### Capacidades de Testing
- Ejecución automática de suites de prueba
- Corrección de casos de prueba fallidos
- Aplicación de reglas de linting para calidad de código
- Integración con frameworks de testing populares
- Generación automática de casos de prueba

### Características de Debugging

#### Identificación de Problemas
- Claude Code identifica issues y sugiere correcciones
- Resolución de dependencias faltantes
- Detección de cuellos de botella de rendimiento
- Análisis de mensajes de error
- Trazado de stack traces

#### Integración con Git
- Creación de commits descriptivos
- Resolución de conflictos de merge
- Búsqueda en historial de commits via lenguaje natural
- Gestión de branches y tags
- Análisis de diferencias entre versiones

### Sistema de Checkpoints

#### Funcionamiento
- Guardado automático del estado del código antes de cada cambio
- Retroceso instantáneo a versiones previas
- Activación con Esc dos veces o comando `/rewind`
- Historial completo de cambios realizados

#### Casos de Uso
- Exploración segura de cambios
- Recuperación de errores
- Comparación de implementaciones
- Prueba de diferentes enfoques

### Herramientas de Análisis

#### Análisis Estático
- Linting automático del código
- Verificación de tipos
- Detección de code smells
- Análisis de complejidad ciclomática

#### Análisis Dinámico
- Profiling de rendimiento
- Detección de memory leaks
- Análisis de cobertura de código
- Monitoreo de recursos en tiempo real

### Debugging Iterativo
- Acciones iterativas hasta que el código tenga éxito
- Corrección automática de errores comunes
- Sugerencias de optimización
- Refactorización guiada

### Integración con IDEs
- Errores de diagnóstico compartidos automáticamente
- Visualización de problemas en el editor
- Quick fixes disponibles
- Navegación a ubicación del error

---

## Características Avanzadas

### Modo Headless
- Para contextos no interactivos (CI, hooks, scripts)
- Flag `-p` con prompt para habilitar
- `--output-format stream-json` para salida JSON
- Conversaciones multi-turno soportadas

### Modo Plan
- Análisis de código seguro y solo lectura
- No realiza cambios en el sistema
- Ideal para exploración y planificación
- Investigación antes de implementación

### Extended Thinking
- Palabras clave especiales activan pensamiento extendido
- Niveles: "think" < "think hard" < "think harder" < "ultrathink"
- Mayor tiempo de computación para evaluación
- Mejor resolución de problemas complejos

### Ejecución Paralela
- Múltiples tareas en paralelo desde interfaz única
- Diferentes repositorios simultáneamente
- Creación automática de PRs
- Resúmenes claros de cambios

### Web Browsing y Acceso a Internet
- Navegación web controlada para investigación
- Extracción de información de documentación online
- Análisis de contenido web
- Restricciones de seguridad configurables

### Sistema de Archivos Avanzado
- Operaciones completas de lectura/escritura
- Navegación de directorios
- Creación y eliminación de archivos
- Permisos granulares por directorio
- Soporte para archivos binarios e imágenes

### Procesamiento de Imágenes
- Lectura y análisis de imágenes (PNG, JPG, etc.)
- Extracción de texto de imágenes
- Análisis visual para desarrollo UI
- Soporte para screenshots y mockups

### LLM Gateway
- Configuración con soluciones de gateway LLM
- Soporte para LiteLLM
- Seguimiento de uso y gestión de presupuesto
- Integración con vault y generación JWT

---

## Modelos Disponibles

### Claude Sonnet 4.5 (Octubre 2025)
- Modelo principal para Claude Code
- $3/$15 por millón de tokens
- Mejor rendimiento general de codificación
- Capacidad mejorada para tareas largas y complejas

### Claude Haiku 4.5 (Octubre 2025)
- 90% del rendimiento de Sonnet 4.5
- 2x más rápido, 3x más económico
- $1/$5 por millón de tokens
- Óptimo para agentes ligeros con invocación frecuente

### Claude Opus 4.1
- Modelo premium para tareas complejas
- Mayor capacidad de razonamiento
- Ideal para problemas arquitectónicos complejos

---

## Mejores Prácticas

### Uso de Subagentes
- Usar para problemas complejos que requieren investigación
- Especialmente útil al inicio de conversaciones
- Preserva disponibilidad de contexto
- Verificar detalles sin perder eficiencia

### Modo de Pensamiento
- Usar "think" para activar pensamiento extendido
- Aplicar niveles progresivos según complejidad
- Permitir tiempo adicional para problemas difíciles
- Evaluar alternativas más exhaustivamente

### Gestión de Contexto
- Mantener CLAUDE.md actualizado y conciso
- Usar `/compact` cuando sea necesario
- Aprovechar subagentes para tareas especializadas
- Monitorear uso de tokens con herramientas

### Seguridad
- Nunca incluir credenciales hardcodeadas
- Usar variables de entorno para secretos
- Configurar permisos mínimos necesarios
- Revisar logs de auditoría regularmente

### Integración CI/CD
- Implementar validación automática
- Usar modo headless para automatización
- Configurar hooks para eventos críticos
- Mantener pipelines simples y mantenibles

### Desarrollo en Equipo
- Compartir CLAUDE.md en repositorio
- Estandarizar comandos personalizados
- Documentar flujos de trabajo específicos
- Usar plugins para funcionalidad compartida

---

## Conclusión

Claude Code representa una evolución significativa en herramientas de desarrollo asistidas por IA, ofreciendo capacidades desde codificación autónoma hasta integración empresarial completa. Con su ecosistema de subagentes, MCP, plugins y características de seguridad, proporciona una plataforma completa y extensible para desarrollo moderno.

La plataforma continúa evolucionando con nuevas características y mejoras planificadas para 2026, manteniendo su posición como la herramienta de codificación AI líder en el mercado.

---

## Funcionalidades Adicionales

### Soporte Multi-idioma
- Comprensión y generación de código en múltiples lenguajes de programación
- Traducción de código entre lenguajes
- Adaptación a convenciones específicas del lenguaje
- Documentación multilingüe

### Gestión de Dependencias
- Análisis de dependencias del proyecto
- Actualización automática de paquetes
- Detección de vulnerabilidades
- Resolución de conflictos de versiones

### Análisis de Rendimiento
- Profiling de aplicaciones
- Identificación de cuellos de botella
- Sugerencias de optimización
- Benchmarking de código

### Documentación Automática
- Generación de documentación API
- Creación de README automáticos
- Comentarios de código inteligentes
- Diagramas de arquitectura

### Colaboración en Equipo
- Sincronización de configuraciones de proyecto
- Compartición de subagentes personalizados
- Plantillas de equipo
- Estándares de codificación compartidos

---

*Última actualización: Octubre 2025*
*Versión de documentación: 1.1.0*
*Basado en documentación oficial de Anthropic y recursos de la comunidad*
*Documento generado mediante análisis exhaustivo de la documentación de Claude Code*