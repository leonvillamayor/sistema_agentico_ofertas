# Servidores MCP de IA y Machine Learning Instalados

## Resumen de Instalación

Se han instalado exitosamente **7 servidores MCP** de IA y Machine Learning de AWS Labs. Todos los servidores están configurados para ejecutarse usando `uvx` con las versiones más recientes.

## Servidores Instalados

### 1. Amazon Bedrock Knowledge Bases Retrieval
- **Nombre del servidor:** `bedrock-kb-retrieval`
- **Paquete:** `awslabs.bedrock-kb-retrieval-mcp-server@latest`
- **Descripción:** Consultar bases de conocimiento empresariales con soporte de citación
- **Workflows:** Conversational Assistants
- **Variables de entorno requeridas:**
  - `AWS_PROFILE`: Perfil de AWS a utilizar
  - `AWS_REGION`: Región de AWS
  - `KB_INCLUSION_TAG_KEY`: Clave de etiqueta para inclusión de KB

### 2. Amazon Kendra Index
- **Nombre del servidor:** `kendra-index`
- **Paquete:** `awslabs.amazon-kendra-index-mcp-server@latest`
- **Descripción:** Búsqueda empresarial y mejora de RAG
- **Workflows:** Conversational Assistants
- **Variables de entorno requeridas:**
  - `AWS_REGION`: Región de AWS
  - `KEND_INDEX_ID`: ID del índice de Kendra
  - `KEND_ROLE_ARN`: ARN del rol de Kendra

### 3. Amazon Q Business Anonymous
- **Nombre del servidor:** `q-business`
- **Paquete:** `awslabs.amazon-qbusiness-anonymous-mcp-server@latest`
- **Descripción:** Asistente de IA basado en base de conocimiento con acceso anónimo
- **Workflows:** Conversational Assistants
- **Variables de entorno requeridas:**
  - `QBUSINESS_APP_ID`: ID de la aplicación Q Business
  - `QBUSINESS_USER_ID`: ID del usuario Q Business
  - `AWS_PROFILE`: Perfil de AWS a utilizar
  - `AWS_REGION`: Región de AWS

### 4. Amazon Q Index
- **Nombre del servidor:** `q-index`
- **Paquete:** `awslabs.amazon-qindex-mcp-server@latest`
- **Descripción:** Accesores de datos para buscar a través del índice Q empresarial
- **Workflows:** Conversational Assistants
- **Variables de entorno requeridas:**
  - `AWS_REGION`: Región de AWS
  - `QINDEX_ID`: ID del índice Q

### 5. Amazon Nova Canvas
- **Nombre del servidor:** `nova-canvas`
- **Paquete:** `awslabs.nova-canvas-mcp-server@latest`
- **Descripción:** Generación de imágenes con IA con guía de texto y color
- **Workflows:** Conversational Assistants
- **Variables de entorno requeridas:**
  - `AWS_PROFILE`: Perfil de AWS a utilizar
  - `AWS_REGION`: Región de AWS

### 6. Amazon Bedrock Data Automation
- **Nombre del servidor:** `bedrock-data-automation`
- **Paquete:** `awslabs.aws-bedrock-data-automation-mcp-server@latest`
- **Descripción:** Analizar documentos, imágenes, videos y archivos de audio
- **Workflows:** Conversational Assistants
- **Variables de entorno requeridas:**
  - `AWS_PROFILE`: Perfil de AWS a utilizar
  - `AWS_REGION`: Región de AWS
  - `AWS_BUCKET_NAME`: Nombre del bucket S3
  - `BASE_DIR`: Directorio base

### 7. Amazon Bedrock Custom Model Import
- **Nombre del servidor:** `bedrock-custom-model`
- **Paquete:** `awslabs.aws-bedrock-custom-model-import-mcp-server@latest`
- **Descripción:** Gestionar modelos personalizados en Bedrock para inferencia bajo demanda
- **Workflows:** Autonomous Background Agents, Conversational Assistants, Vibe Coding & Development
- **Variables de entorno requeridas:**
  - `AWS_PROFILE`: Perfil de AWS a utilizar
  - `AWS_REGION`: Región de AWS
  - `BEDROCK_MODEL_IMPORT_S3_BUCKET`: Bucket S3 para importación de modelos

## Configuración Actual

Los servidores están instalados en la configuración local de Claude Code (`~/.claude.json`) sin variables de entorno configuradas. Para utilizar estos servidores, necesitarás:

### Pasos de Configuración

1. **Configurar AWS CLI:**
   ```bash
   aws configure
   ```

2. **Configurar variables de entorno específicas para cada servidor** editando el archivo `~/.claude.json` y agregando las variables requeridas en la sección `env` de cada servidor.

3. **Ejemplo de configuración con variables de entorno:**
   ```json
   "bedrock-kb-retrieval": {
     "type": "stdio",
     "command": "uvx",
     "args": ["awslabs.bedrock-kb-retrieval-mcp-server@latest"],
     "env": {
       "AWS_PROFILE": "tu-perfil-aws",
       "AWS_REGION": "us-east-1",
       "KB_INCLUSION_TAG_KEY": "tu-clave-etiqueta"
     }
   }
   ```

## Verificar Instalación

Para verificar que los servidores están disponibles, puedes usar:

```bash
claude mcp list
```

## Próximos Pasos

1. Configurar las credenciales de AWS apropiadas
2. Obtener los IDs de recursos necesarios (índices de Kendra, aplicaciones Q Business, etc.)
3. Configurar las variables de entorno específicas para cada servidor
4. Probar la funcionalidad de cada servidor
5. Implementar casos de uso específicos según las necesidades del proyecto

## Recursos Adicionales

- [Documentación oficial AWS MCP Servers](https://awslabs.github.io/mcp/)
- [Repositorio GitHub AWS Labs MCP](https://github.com/awslabs/mcp)
- [Guía de instalación completa](https://awslabs.github.io/mcp/installation)

---

*Instalación completada el: 21 de octubre de 2025*
*Total de servidores instalados: 7*
*Estado: Listos para configuración*