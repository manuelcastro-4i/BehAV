# Docker TODO

## Completado

- [x] Crear `.env.example` sin la key real
- [x] Crear directorio `config/` con `behav_params.yaml` (ya existia)
- [x] Verificar `.gitignore` incluye `.env` y `models/FastSAM-x.pt`
- [x] Crear `.dockerignore` para optimizar build context
- [x] Revisar cambios en `instruction-to-behavioral-costs.py` y `landmarkdetector_fastsam.py` (CLI args para Docker)
- [x] Build y test: `instruction_decomposition`, `landmark_test`, `behav`
- [x] Configurar `Dockerfile.go2_sim` con Jazzy + Gazebo Harmonic + noVNC
- [x] Configurar `entrypoint_sim.sh` con modo headless/noVNC
- [x] Actualizar `docker-compose.yml` para go2_sim cross-platform
- [x] Añadir FastSAM + ultralytics a `Dockerfile.behav` (para `landmark_detector`)

## Pendiente (Docker / infra)

- [x] Build y test de `go2_sim` (headless, Gazebo Harmonic arranca, topics ROS2 publicados, healthcheck healthy)
- [x] Build y test de `landmark_detector` (imports OK: FastSAM, rclpy, torch CUDA, ultralytics, cv_bridge)

---

## Demo completo: robot navega por camino evitando obstáculos

### Estado actual del pipeline (Feb 2026)

**Funciona:**
- Robot Go2 aparece en Gazebo con cámara + LiDAR + odometría
- BehAV planner corre CLIPSeg y genera costmap (visible en http://localhost:8080)
- Director descompone instrucción natural → prompts CLIPSeg + goal
- `/robot1/cmd_vel` se publica con velocidades del planificador

**Gaps restantes:**

#### 1. Locomotion — robot no se mueve (BLOQUEANTE) — ~1-2 días
- `cmd_vel` se publica pero nada lo traduce a movimiento físico en Gazebo
- `gz_ros2_control` + joint controllers fueron eliminados (causaban SIGABRT)
- **Solución**: añadir `gz::sim::systems::VelocityControl` al trunk del robot en `gazebo.xacro`,
  o usar el plugin `DiffDrive` de Gazebo sobre un par de joints proxy
- Requiere rebuild de `go2_sim` y test de que el robot se desplaza con cmd_vel

#### 2. Mundo vacío — sin camino, obstáculos ni meta — ~0.5-1 día
- El mundo actual (`docker/empty.world`) es solo un plano gris uniforme
- Necesita:
  - Franja de camino con textura asfalto (modelo SDF inline o `include` de Fuel)
  - Césped a los lados (textura verde)
  - 2-3 obstáculos físicos (cajas/cilindros con colisión) en o cerca del camino
  - Marcador de destino visible (cilindro de color)
- El fichero montado en el contenedor es `docker/empty.world` → fácil de editar sin rebuild

#### 3. Texturas para CLIPSeg — ~0.5 día
- Con materiales planos (color puro) CLIPSeg no distingue "pavement" de "grass"
- Opciones:
  - a) Usar texturas `.png` de asfalto y césped real en los materiales Gazebo/Ogre2
  - b) Ajustar prompts a descriptores más neutrales ("gray surface", "green surface")
  - c) Combinar ambas
- Requiere test visual: ver costmap en el viewer con el mundo nuevo

#### 4. Integración y tuning final — ~0.5 día
- Ajustar goal coords al destino en el mundo nuevo
- Verificar que la instrucción "Stay on the path, avoid obstacles" genera los prompts correctos
- Tuning de `goal_radius`, `V_MAX`, parámetros MPC si el robot oscila o no converge

### Orden de implementación
1. [x] Locomotion: `VelocityControl` plugin en `gazebo.xacro` → robot se mueve
     - `docker/go2_gazebo.xacro`: añadido `gz::sim::systems::VelocityControl` topic=`/robot1/cmd_vel`
     - `docker/gz_pose_bridge.yaml`: bridge `ROS_TO_GZ` para cmd_vel
     - `docker/shared_bridge_reader.py`: relay cmd_vel Humble→`/shared/cmd_vel.json`
     - `docker/sim_adapters.py`: relay `cmd_vel.json`→Jazzy `/robot1/cmd_vel` (20Hz, con safety timeout)
     - `docker-compose.yml`: mount xacro en path instalado, bridge_reader rw
     - **Verificado**: robot en x=9.4m tras 3s a 0.3 m/s
2. [x] Mundo: crear `docker/behav_demo.world` con camino + obstáculos + meta
     - Césped verde (120×120m), asfalto gris (18×3m), líneas blancas centrales
     - 3 obstáculos de hormigón a lo largo del camino
     - Árboles decorativos, bordillos blancos
     - Marcador rojo+amarillo en x=13m (goal)
     - **Verificado**: robot avanzó x=0→12.7m en línea recta hasta la meta
3. [ ] Texturas: añadir materiales Ogre2 con imágenes reales o ajustar prompts
4. [ ] Test end-to-end: instrucción → navegación → meta alcanzada
