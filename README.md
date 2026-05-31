# Trabajo Práctico 
## Operaciones de Máquina I

#### Integrantes:
 - Britez, Leandro Adrian 
 - Orellana, César Andrés
 



##### Primera entrega:

Se armo la estructura inicial del proyecto y sus servicios a usar(Faltan mas servicios).

Gestión de Experimentos con  (MLflow): Utilizado como servidor central para el registro de parámetros, métricas y artefactos de los modelos desarrollados.

Almacenamiento y Persistencia (PostgreSQL y MinIO):

Postgres actúa como el Backend Store para el almacenamiento estructurado de metadatos de MLflow.

MinIO (S3) funciona como el Artifact Store, permitiendo el almacenamiento escalable y persistente de los modelos y archivos pesados en un bucket local.