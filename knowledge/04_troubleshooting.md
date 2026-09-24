# Resolución de problemas frecuentes

## El despliegue queda en estado `Pending`

Causas habituales:

1. **Sin recursos en el clúster**: el pod pide más CPU o memoria de la disponible. Revisar con `orb status --verbose`; el evento dirá `Insufficient cpu` o `Insufficient memory`. Solución: bajar `resources` o pedir ampliación del nodo pool al equipo de Plataforma.
2. **Imagen no encontrada**: el tag no existe en `registry.orbital.internal`. Suele pasar cuando se ejecuta `orb deploy` sin haber corrido `orb build` para ese commit. Solución: correr `orb build` y volver a desplegar.
3. **Secreto inexistente en Vault**: el manifiesto referencia `vault:alguna_clave` que no fue cargada en el entorno destino. El evento muestra `SecretNotFound`. Solución: `orb secret set ALGUNA_CLAVE --env <entorno>`.

## El pod reinicia en bucle (`CrashLoopBackOff`)

Casi siempre es un error de la aplicación al arrancar. Pasos:

1. `orb logs --previous` para ver los logs del contenedor que falló.
2. Verificar que las variables de entorno declaradas en `env` y `secrets` estén completas.
3. Comprobar que el endpoint `/healthz` responda 200 dentro de los 30 segundos iniciales. Si la aplicación tarda más en arrancar, aumentar `health.initialDelaySeconds` (por defecto 30, máximo 300).

## Error `POLICY_VIOLATION` en `orb validate`

Orbital aplica políticas organizacionales antes de desplegar. Las más comunes:

- `IMAGE_BASE_NOT_ALLOWED`: el `Dockerfile` usa una imagen base que no está en la lista aprobada. Las imágenes permitidas son las de `registry.orbital.internal/base/*` (python, node, go y java).
- `MISSING_TEAM_LABEL`: falta el campo `team` en el manifiesto.
- `RESOURCES_EXCEED_LIMIT`: se pidieron más de 4 CPU u 8 GiB por pod.
- `PROD_WINDOW_CLOSED`: se intentó desplegar a producción fuera del horario permitido sin la bandera `--hotfix`.

## El canary revirtió solo

Cuando la tasa de errores HTTP 5xx de la nueva versión supera `canary.maxErrorRate`, Orbital revierte y deja el evento `CANARY_ABORTED` con el porcentaje observado. Para investigar: `orb logs --revision <n>` sobre la revisión abortada. Si el umbral es demasiado estricto para un servicio con poco tráfico, se puede subir `canary.maxErrorRate` o aumentar `canary.minRequests` (número mínimo de requests antes de evaluar, por defecto 100).

## Latencia alta después de un despliegue

Revisar si el autoescalador aún no alcanzó las réplicas necesarias: `orb status` muestra `replicas: actual/deseadas`. Recordar que el HPA tarda 2 minutos de CPU sostenida por encima del 70% para escalar. Para picos previsibles (campañas, lanzamientos) se recomienda subir `replicas.min` con antelación en lugar de esperar al autoescalado.

## Cómo pedir ayuda

Si el problema persiste, abrir un hilo en `#orbital-soporte` incluyendo: nombre del servicio, entorno, salida de `orb status --verbose` y el identificador del despliegue (`deploy-id`) que muestra `orb deploy` al finalizar. Para incidentes en producción usar el comando `/orbital-incidente` en Slack, que pagea al ingeniero de guardia.
