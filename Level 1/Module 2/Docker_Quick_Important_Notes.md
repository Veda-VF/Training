# Docker --- Quick Important Notes (Basic to Advanced)

These notes explain Docker in simple, point-to-point language, with a
focus on Python, FastAPI, and AI/ML applications.

## 1. What is Docker?

**Docker** is a platform for packaging and running applications in
isolated environments called **containers**.

### Why use Docker?

-   Helps avoid the "works on my machine" problem.
-   Packages application code with its dependencies.
-   Makes development, testing, and deployment more consistent.
-   Makes it easier to run multiple services together.

## 2. Core concepts

  -----------------------------------------------------------------------
  Concept                             Meaning
  ----------------------------------- -----------------------------------
  **Image**                           A read-only template used to create
                                      containers.

  **Container**                       A running instance of an image.

  **Dockerfile**                      A text file containing instructions
                                      to build an image.

  **Registry**                        A place to store and download
                                      images, such as Docker Hub.

  **Volume**                          Storage used to keep data beyond a
                                      container's lifecycle.

  **Network**                         Allows containers to communicate.

  **Docker Engine**                   The software that builds and runs
                                      containers.

  **Docker Compose**                  A tool for defining and running
                                      multi-service applications.
  -----------------------------------------------------------------------

### Docker vs. virtual machines

-   Containers isolate application processes and share the host OS
    kernel.
-   Virtual machines run a guest OS with its own kernel.
-   Containers are often lighter and faster to start; VMs provide a
    separate OS environment.

## 3. Basic workflow

``` text
Dockerfile  →  Image  →  Container
```

1.  Write a `Dockerfile`.
2.  Build an image from it.
3.  Run a container from the image.

Images have layers. Docker can reuse unchanged layers to make later
builds faster.

## 4. Example Dockerfile for FastAPI

Assume the project contains `main.py` and `requirements.txt`.

``` dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile instructions

  -----------------------------------------------------------------------
  Instruction                         Purpose
  ----------------------------------- -----------------------------------
  `FROM`                              Chooses the base image.

  `WORKDIR`                           Sets the working directory inside
                                      the image/container.

  `COPY`                              Copies files from the build context
                                      into the image.

  `RUN`                               Runs a command while building the
                                      image, often to install
                                      dependencies.

  `ENV`                               Sets environment variables.

  `EXPOSE`                            Documents the port the app expects
                                      to listen on; it does **not**
                                      publish the port by itself.

  `CMD`                               Provides the default command when a
                                      container starts.

  `ENTRYPOINT`                        Defines the main executable for the
                                      container.

  `USER`                              Selects the user for later commands
                                      or the running application.

  `HEALTHCHECK`                       Defines a command Docker can use to
                                      check container health.
  -----------------------------------------------------------------------

**Why copy `requirements.txt` first?** Dependencies often change less
frequently than source code. Docker may reuse the dependency layer when
only application code changes.

### `CMD` vs. `ENTRYPOINT`

-   `CMD` provides a default command or default arguments.
-   `ENTRYPOINT` sets the main executable.
-   They can be combined; behavior depends on whether exec form or shell
    form is used.

## 5. Build and run

Run these commands in the directory containing the `Dockerfile`.

``` bash
docker build -t my-app .
docker run -d -p 8000:8000 --name my-container my-app
docker ps
docker logs my-container
docker stop my-container
docker rm my-container
```

-   `docker build`: builds an image.
-   `-t my-app`: names/tags the image.
-   `.`: uses the current directory as the build context.
-   `docker run`: creates and starts a container.
-   `-d`: runs in the background.
-   `-p 8000:8000`: maps host port 8000 to container port 8000.
-   `--name`: gives the container a readable name.
-   `docker ps`: lists running containers.
-   `docker ps -a`: lists all containers, including stopped ones.
-   `docker logs`: displays logs.
-   `docker stop`: stops a container.
-   `docker rm`: removes a stopped container.

If the app listens on port 8000, open `http://localhost:8000`.

**Remember:** `EXPOSE 8000` documents a port; `-p` publishes/maps it to
the host.

## 6. Volumes and persistent data

Use a **volume** or **bind mount** for data that must survive container
replacement.

-   **Named volume:** managed by Docker.
-   **Bind mount:** connects a host path to a path inside the container.

Example:

``` bash
docker run -v my-data:/app/data my-app
```

This mounts the named volume `my-data` at `/app/data`.

-   Removing a container does not normally remove a separately managed
    named volume.
-   `docker compose down -v` also removes Compose-managed named volumes,
    which may delete stored database data. Use it carefully.

## 7. Networking and ports

Containers can communicate over Docker networks. In Compose, services on
the same network can usually reach each other by service name.

If the service is named `redis`, the API should usually connect to:

``` text
redis:6379
```

Do not normally use `localhost:6379` to reach Redis in another
container. Inside the API container, `localhost` refers to that API
container.

## 8. Docker Compose

**Docker Compose** defines and runs an application with multiple
services using a YAML file, commonly `compose.yaml` or
`docker-compose.yml`.

A typical project might contain:

``` text
FastAPI application
   ├── PostgreSQL database
   └── Redis cache
```

Compose can configure services, ports, environment variables, volumes,
networks, health checks, and restart behavior.

### Common commands

``` bash
docker compose up -d
docker compose up -d --build
docker compose logs -f
docker compose logs -f api
docker compose down
docker compose down -v
```

-   `up -d`: starts services in the background.
-   `up -d --build`: builds images and starts services.
-   `logs -f`: follows logs.
-   `logs -f api`: follows the `api` service logs.
-   `down`: stops and removes Compose containers and networks.
-   `down -v`: also removes Compose-managed named volumes; stored data
    may be deleted.

**`docker run` vs. Compose:** `docker run` is useful for an individual
container; Compose is useful when an application needs several connected
services.

## 9. Environment variables and `.env`

Environment variables configure an application without hard-coding
settings into source code or the image. Examples include database
host/port, cache URL, and application mode.

Security rules: - Do not commit `.env` files containing secrets to
Git. - Do not bake passwords, tokens, or API keys into images. - For
production, use a suitable secrets manager or deployment platform's
secret mechanism. - Environment variables are configuration, not a
complete secret-management system.

## 10. `.dockerignore`

A `.dockerignore` file prevents unnecessary files from being sent in the
Docker build context.

Example:

``` text
.git
.venv
__pycache__
*.pyc
.env
.pytest_cache
```

It can make builds smaller and faster and help prevent accidental
inclusion of local files or secrets.

**Important:** A file excluded by `.dockerignore` cannot be copied from
that build context by a matching `COPY` instruction.

## 11. Advanced concepts

  -----------------------------------------------------------------------
  Concept                             Why it matters
  ----------------------------------- -----------------------------------
  **Image layers/cache**              Reuses unchanged build steps to
                                      speed up rebuilds.

  **Multi-stage builds**              Separates build and runtime stages
                                      to help keep the final image
                                      smaller.

  **Non-root user**                   Reduces application privileges
                                      inside the container.

  **Health check**                    Indicates service health; it does
                                      not automatically fix every
                                      failure.

  **Restart policy**                  Can restart containers under
                                      specified conditions.

  **Registry**                        Stores images so other machines or
                                      deployment systems can pull them.

  **Docker SDK for Python**           Lets Python code interact with
                                      Docker, such as listing or managing
                                      containers.
  -----------------------------------------------------------------------

A **multi-stage build** can use one stage to prepare/build an
application and another, smaller stage to run it. Only the required
artifacts need to be copied to the final stage.

## 12. Docker for AI/ML

``` text
Train model
    ↓
Save model and preprocessing artifacts
    ↓
Create an API (e.g., FastAPI)
    ↓
Build a Docker image
    ↓
Run the container
    ↓
Send input data and receive predictions
```

Docker can package API code and software dependencies. Model files may
be included in the image or mounted/downloaded separately, depending on
the project.

Docker does **not** automatically make a model more accurate, guarantee
GPU access, make a large model fit in a small image, or replace model
serving and monitoring design. GPU access depends on the host, drivers,
runtime/tooling, and configuration.

## 13. Best practices

-   Use a trusted, suitable base image.
-   Use `.dockerignore`.
-   Copy dependency files first when it improves build caching.
-   Do not put secrets in images or source control.
-   Run as a non-root user where practical.
-   Use volumes for data that must persist.
-   Keep images as small as reasonably possible.
-   Pin dependency versions when reproducible builds matter.
-   Check logs and health status when debugging.
-   Understand which volumes cleanup commands will remove.

## 14. Quick command reference

  -----------------------------------------------------------------------
  Command                             Purpose
  ----------------------------------- -----------------------------------
  `docker --version`                  Shows the Docker CLI version.

  `docker pull IMAGE`                 Downloads an image.

  `docker images`                     Lists local images.

  `docker build -t NAME .`            Builds an image from the current
                                      build context.

  `docker run IMAGE`                  Creates and starts a container.

  `docker ps`                         Lists running containers.

  `docker ps -a`                      Lists all containers.

  `docker logs NAME`                  Shows container logs.

  `docker exec -it NAME sh`           Opens a shell inside a running
                                      container, if `sh` is available.

  `docker stop NAME`                  Stops a container.

  `docker rm NAME`                    Removes a container.

  `docker rmi IMAGE`                  Removes a local image if it is not
                                      in use.

  `docker compose up -d --build`      Builds and starts Compose services
                                      in the background.

  `docker compose down`               Stops and removes Compose
                                      containers and networks.
  -----------------------------------------------------------------------

## 15. Troubleshooting checks

1.  **Container exits:** inspect `docker logs CONTAINER_NAME`.
2.  **App is unreachable:** check that the app listens on `0.0.0.0`
    inside the container and that port mapping is correct.
3.  **Cannot reach database/cache:** check service name, port, network,
    and credentials. Avoid `localhost` for a different container.
4.  **Changes do not appear:** rebuild when required with
    `docker compose up -d --build`.
5.  **Data disappeared:** check whether data was in the container layer
    or a volume, and whether a cleanup command removed the volume.
6.  **Image is too large:** review the base image, `.dockerignore`,
    dependency installation, and whether a multi-stage build would help.

## 16. Important note for Python async apps

Calling `time.sleep()` inside `async def` blocks the event loop and
stops other async tasks from progressing during that wait.

Use:

``` python
import asyncio

await asyncio.sleep(1)
```

Prefer non-blocking async I/O libraries where possible. CPU-heavy work
may need a process pool, worker, or separate service rather than running
directly on the event loop.

## Final mental model

-   **Dockerfile** describes how to build.
-   **Image** is the packaged template.
-   **Container** is the running instance.
-   **Port mapping** makes a container service reachable from the host.
-   **Network** lets containers communicate.
-   **Volume** stores data beyond the container lifecycle.
-   **Compose** coordinates multiple connected services.

## Official documentation

-   [Docker Get Started](https://docs.docker.com/get-started/)
-   [Dockerfile
    reference](https://docs.docker.com/reference/dockerfile/)
-   [Docker Compose documentation](https://docs.docker.com/compose/)
