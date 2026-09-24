# Guía de despliegue con la CLI `orb`

## Flujo básico

El ciclo de vida de un despliegue en Orbital tiene cuatro pasos. Todos se ejecutan desde la raíz del repositorio del servicio, donde debe existir el archivo `orbital.yaml`.

1. **Validar**: `orb validate` comprueba la sintaxis del manifiesto y las políticas de la organización (límites de recursos, etiquetas obligatorias, imagen base permitida). Es gratuito y no toca el clúster.
2. **Construir**: `orb build` genera la imagen Docker usando el `Dockerfile` del repositorio y la sube al registro interno. El tag resultante tiene la forma `<servicio>:<hash-commit>-<entorno>`.
3. **Desplegar**: `orb deploy --env <dev|stg|prod>` aplica el manifiesto en el clúster del entorno indicado. Por defecto utiliza una estrategia de rolling update con `maxUnavailable: 0` y `maxSurge: 25%`.
4. **Verificar**: `orb status` muestra el estado de los pods, la versión activa y los últimos eventos. `orb logs --tail 200` imprime los logs recientes.

## Estrategias de despliegue

Orbital soporta tres estrategias, configurables en el campo `strategy` del manifiesto:

- `rolling` (por defecto): reemplaza los pods gradualmente. Recomendado para la mayoría de los servicios.
- `canary`: envía primero un porcentaje del tráfico a la nueva versión. El porcentaje inicial se define con `canary.initialWeight` (valor por defecto: 10) y la promoción completa se hace con `orb promote`. Si las métricas de error superan el umbral `canary.maxErrorRate` (por defecto 2%), Orbital revierte automáticamente.
- `bluegreen`: levanta la nueva versión completa en paralelo y conmuta el tráfico de golpe con `orb switch`. Duplica el consumo de recursos durante la transición.

## Rollback

Para volver a la versión anterior se usa `orb rollback --env <entorno>`. Orbital conserva las últimas 10 revisiones de cada servicio. Se puede volver a una revisión específica con `orb rollback --to <número-de-revisión>`. El rollback en producción no requiere aprobación adicional, pero queda registrado en la auditoría y envía un aviso a `#orbital-deploys`.

## Aprobaciones en producción

Todo `orb deploy --env prod` crea una solicitud de aprobación. Un revisor del mismo equipo, distinto de quien lanzó el despliegue, debe aprobarla desde el panel web o con `orb approve <id>`. Las solicitudes expiran a las 2 horas si nadie las aprueba. Los servicios marcados como `tier: critical` en el manifiesto necesitan dos aprobaciones en lugar de una.

## Ejemplo de manifiesto mínimo

```yaml
name: pagos-api
team: cobros
tier: critical
image:
  dockerfile: ./Dockerfile
resources:
  cpu: "500m"
  memory: "512Mi"
replicas:
  min: 2
  max: 10
strategy: canary
canary:
  initialWeight: 5
  maxErrorRate: 1
```
