# Configuración, variables de entorno y secretos

## Variables de entorno

Las variables de entorno no sensibles se declaran en la sección `env` del archivo `orbital.yaml`. Orbital las convierte en un ConfigMap con el nombre `<servicio>-config` y las inyecta en todos los pods del servicio.

```yaml
env:
  LOG_LEVEL: info
  MAX_CONNECTIONS: "50"
  FEATURE_NUEVO_CHECKOUT: "false"
```

Los valores deben ser siempre strings. Un número sin comillas hace fallar `orb validate` con el error `ENV_VALUE_NOT_STRING`.

## Secretos

Los secretos (contraseñas, llaves de API, certificados) nunca se escriben en `orbital.yaml` ni se suben al repositorio. Orbital se integra con HashiCorp Vault: cada servicio tiene una ruta propia en Vault con el formato `secret/orbital/<entorno>/<servicio>`.

Para referenciar un secreto en el manifiesto se usa la sintaxis `vault:` seguida del nombre de la clave:

```yaml
secrets:
  DATABASE_URL: vault:database_url
  STRIPE_API_KEY: vault:stripe_key
```

En tiempo de despliegue, el Orbital Control Plane lee los valores de Vault y los monta como variables de entorno. Los desarrolladores cargan o rotan secretos con `orb secret set <CLAVE> --env <entorno>`; el comando pide el valor de forma interactiva y nunca lo acepta como argumento para evitar que quede en el historial de la shell.

La rotación de secretos no reinicia los pods automáticamente. Después de rotar un secreto hay que ejecutar `orb restart --env <entorno>` para que la nueva versión sea leída.

## Límites de recursos y autoescalado

Cada servicio debe declarar `resources.cpu` y `resources.memory`. Si se omiten, `orb validate` falla. Los valores máximos permitidos por pod son 4 CPU y 8 GiB de memoria; pedir más requiere una excepción aprobada por el equipo de Plataforma.

El autoescalado horizontal se controla con `replicas.min` y `replicas.max`. El HorizontalPodAutoscaler que crea Orbital escala cuando el uso promedio de CPU supera el 70% durante 2 minutos, y reduce réplicas cuando baja del 30% durante 10 minutos. El objetivo de CPU se puede ajustar con `autoscaling.targetCPU`.

## Health checks

Orbital exige que todo servicio HTTP exponga dos endpoints:

- `GET /healthz`: liveness. Si responde algo distinto de 200 tres veces seguidas, Kubernetes reinicia el pod.
- `GET /readyz`: readiness. Mientras no responda 200, el pod no recibe tráfico.

Las rutas se pueden cambiar con `health.liveness` y `health.readiness`. El intervalo por defecto entre chequeos es de 10 segundos y el timeout de 3 segundos.
