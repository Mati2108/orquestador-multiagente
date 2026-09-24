# Orbital: plataforma interna de despliegue de microservicios

## Qué es Orbital

Orbital es la plataforma interna de la empresa para publicar, escalar y operar microservicios sobre un clúster de Kubernetes. Fue creada en 2023 por el equipo de Plataforma para reemplazar los despliegues manuales con `kubectl apply` y unificar la forma en que los equipos de producto llevan código a producción.

Orbital no reemplaza a Kubernetes: es una capa de conveniencia por encima. Cada servicio se describe con un único archivo `orbital.yaml` en la raíz del repositorio y la CLI `orb` se encarga de traducirlo a los objetos de Kubernetes necesarios (Deployment, Service, HorizontalPodAutoscaler, Ingress y ConfigMap).

## Componentes principales

- **CLI `orb`**: herramienta de línea de comandos que usan los desarrolladores. Se instala con `pip install orbital-cli`. Versión estable actual: 4.2.
- **Orbital Control Plane (OCP)**: servicio central que recibe los manifiestos, valida las políticas de la organización y aplica los cambios en el clúster. Corre en el namespace `orbital-system`.
- **Registro de imágenes**: Orbital publica las imágenes Docker en el registro interno `registry.orbital.internal`. Cada imagen se etiqueta con el hash corto del commit y con el nombre del entorno.
- **Panel web**: disponible en `https://orbital.internal`, muestra el estado de cada servicio, el historial de despliegues y los logs de los últimos 7 días.

## Entornos

Orbital gestiona tres entornos, cada uno en un clúster separado:

| Entorno | Nombre en la CLI | Uso |
|---------|------------------|-----|
| Desarrollo | `dev` | Pruebas individuales, se destruye cada noche a las 03:00 |
| Staging | `stg` | Réplica de producción con datos anonimizados |
| Producción | `prod` | Tráfico real. Requiere aprobación de un revisor |

Los despliegues a `prod` solo pueden hacerse de lunes a jueves entre las 09:00 y las 17:00 hora de Buenos Aires, salvo que se declare un hotfix con la bandera `--hotfix`, que exige una justificación escrita y notifica automáticamente al canal de guardia.

## Equipo responsable

El equipo de Plataforma mantiene Orbital. El canal de soporte es `#orbital-soporte` y el SLA de respuesta para incidentes en producción es de 15 minutos. Las solicitudes de nuevas funcionalidades se cargan en el tablero `PLAT` y se revisan en la reunión semanal de los martes.
