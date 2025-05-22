# Running a Kubernetes Cluster Locally with Kind

This guide walks you through setting up a local Kubernetes cluster using Kind (Kubernetes in Docker) and deploying a sample application.

## Prerequisites

### 1. Tool Installation

Ensure you have the following tools installed on your local machine:

- **Kind:** Installation Guide
- **kubectl:** Installation Guide
- **Docker Desktop (or Docker Engine):** Kind runs Kubernetes clusters using Docker containers. Docker Desktop is recommended for easily managing resources for your Kind cluster, especially on Windows and macOS.
  - **Tip:** For Docker Desktop users, consider installing the Kind extension from the settings and increasing allocated resources (CPU, memory) to prevent resource-related errors when running your cluster.

### 2. Project Files

Prepare the following files in your project directory:

- **`Dockerfile`**: This file defines how to build your application's Docker image. (Example: `../Dockerfile`)
- **`.env`**: This file should contain your application's secrets or environment-specific configurations (e.g., API keys, database URLs). It will be used to create Kubernetes Secrets.
  *Example `.env` file content:*
  ```env
  API_KEY=your_secret_api_key
  ANOTHER_VARIABLE=some_value
  ```
- **`kind-deployment.yaml`**: This file will contain your Kubernetes Deployment and Service definitions. You will create this file with the content provided in the steps below.

## Deploying the Application on Kind

Follow these steps to build your application, set up a Kind cluster, and deploy your application.

### Step 1: Build Your Docker Image

Navigate to the directory containing your `Dockerfile` and build your Docker image. Replace `cancer-cls:latest` with your desired image name and tag.

```bash
docker build -t cancer-cls:latest .
```

### Step 2: Create a Kind Cluster

Create a local Kubernetes cluster using Kind. If no name is provided, Kind defaults to `kind`.

```bash
# You can choose any name for your cluster
kind create cluster --name my-cluster
```

> **💡 Important for Docker Desktop Users:**
> If you haven't already, ensure Docker Desktop is running and has sufficient resources allocated (CPU, Memory).

### Step 3: Load Docker Image into the Kind Cluster

Make your locally built Docker image available to the Kind cluster.

```bash
kind load docker-image cancer-cls:latest --name my-cluster
```

### Step 4: Prepare Kubernetes Manifest (`kind-deployment.yaml`)

Create a file named `kind-deployment.yaml` in your project directory and add the following content. This manifest defines a Deployment to run your application and a Service to expose it.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cancer-cls # Name of the Deployment
spec:
  replicas: 1      # Number of desired application instances (pods)
  # revisionHistoryLimit: 0 # Uncomment to limit the number of old ReplicaSets to retain
  selector:
    matchLabels:
      app: cancer-cls # This selector links the Deployment to its Pods
  template:
    metadata:
      labels:
        app: cancer-cls # Pods created by this Deployment will have this label
    spec:
      containers:
      - name: cancer-cls # Name of the container within the Pod
        image: cancer-cls:latest # The Docker image to run
        imagePullPolicy: Never   # Instructs Kubernetes to use the locally loaded image and not pull from a remote registry
        ports:
        - containerPort: 8000    # The port your application listens on inside the container
        envFrom:
        - secretRef:
            name: cancer-cls-secrets # Reference to the Kubernetes Secret containing environment variables
---
apiVersion: v1
kind: Service
metadata:
  name: cancer-cls # Name of the Service
spec:
  selector:
    app: cancer-cls # This selector links the Service to Pods with the label 'app: cancer-cls'
  ports:
  - protocol: TCP
    port: 80        # The port on which the Service will be accessible within the cluster
    targetPort: 8000 # The port on the Pods to which traffic will be forwarded
  # type: LoadBalancer # For local Kind clusters, NodePort or port-forwarding is more common for external access
```

### Step 5: Create Kubernetes Secrets

Create Kubernetes Secrets from your `.env` file. These secrets will be mounted as environment variables into your application pods.

First, it's good practice to delete any existing secret with the same name to ensure a clean state (optional, but useful for re-runs):

```bash
kubectl delete secret cancer-cls-secrets --ignore-not-found
```

Now, create the secret:

```bash
kubectl create secret generic cancer-cls-secrets --from-env-file=.env
```

### Step 6: Apply Kubernetes Manifests

Deploy your application to the Kind cluster using the `kind-deployment.yaml` file.

```bash
kubectl apply -f kind-deployment.yaml
```

### Step 7: Verify the Deployment

Check the status of your deployment, service, and pods:

```bash
# Check the Deployment status
kubectl get deployment cancer-cls

# Check the Service status
kubectl get service cancer-cls

# Check the Pods managed by your application (filtered by label)
kubectl get pods -l app=cancer-cls

# Get all pods in all namespaces to check overall cluster health (optional)
kubectl get pods -A
```

Wait until the `STATUS` of your pod is `Running`.

### Step 7.1: Check logs to see if setup is successful and server is already running

```bash
 kubectl logs <podame> -c <container name from -spec.containers.name>
```

example

```bash
 kubectl logs  cancer-cls-67f4b88c5c-n4j6s  -c cancer-cls-container -f
```

### Step 8: Access Your Application

To access your application from your local machine (outside the Kind cluster), you can use `kubectl port-forward`. This command forwards traffic from a local port on your machine to the port your service is listening on.

```bash
# Forward local port 8080 to port 80 on the 'cancer-cls' service
kubectl port-forward service/cancer-cls 8080:80
```

You should now be able to access your application at `http://localhost:8080`.

**Alternative ways to access services (more advanced):**

- **Ingress Controller:** For more robust and production-like access, set up an Ingress Controller (like Nginx Ingress) and an Ingress resource.
- **Service Type `NodePort`:** Change the Service type to `NodePort` in `kind-deployment.yaml` and use `extraPortMappings` in your Kind cluster configuration if you need a fixed port directly on your host.

## Managing and Troubleshooting

### Fetching Container Logs

To view the logs from your application container:

1. **Get the Pod name:**

   ```bash
   kubectl get pods -l app=cancer-cls
   ```

   (Or `kubectl get pods -A` to list all pods if you're unsure of the label)

2. **View logs:** Replace `<pod-name>` with the actual name of your pod from the previous command. The container name `-c cancer-cls` should match the `spec.containers.name` in your `kind-deployment.yaml`.

   ```bash
   kubectl logs <pod-name> -c cancer-cls
   ```

   To follow the logs in real-time (stream), add the `-f` flag:

   ```bash
   kubectl logs <pod-name> -c cancer-cls -f
   ```

### Describing Pods

To get detailed information about a pod, including events and configuration, which is useful for troubleshooting:

```bash
kubectl describe pod <pod-name>
```

## Updating Your Application (Re-deployment Process)

If you modify your application code or `Dockerfile`, follow these steps to re-deploy:

1. **Rebuild your Docker image** with the same tag (or a new one, and update `kind-deployment.yaml` accordingly):

   ```bash
   docker build -t cancer-cls:latest .
   ```

2. **Reload the updated image into your Kind cluster:**

   ```bash
   kind load docker-image cancer-cls:latest --name my-cluster
   ```

3. **Trigger a rolling restart of your Deployment's pods** to pick up the new image:

   ```bash
   kubectl rollout restart deployment cancer-cls
   ```

4. **Monitor the rollout status** and wait for the new pods to be `Running`:

   ```bash
   kubectl rollout status deployment cancer-cls
   # or
   kubectl get pods -l app=cancer-cls -w
   ```

5. **Re-establish port-forwarding** if you stopped the previous one:

   ```bash
   kubectl port-forward service/cancer-cls 8080:80
   ```

6. **Access your updated application** at `http://localhost:8080/`.
