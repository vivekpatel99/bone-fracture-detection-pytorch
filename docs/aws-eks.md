# HowTo run App in AWS EKS cluster

## Setup

- Install for AWS stuff will be done in docker container for convenience and easy setup and removal.
- verify files in `.devcontainer`

```
├── devcontainer.json
├── ensure-mount-sources
└── local-features
    └── copy-kube-config
        ├── copy-kube-config.sh
        ├── devcontainer-feature.json
        └── install.sh

```

- Build and Open in Dev Container:

  1. Press Ctrl+Shift+P (or Cmd+Shift+P on macOS).
  2. Type and select Dev Containers: Rebuild and Reopen in Container.
  3. This will build the Docker image for aws tools and start the development container. VS Code will automatically connect to it.

- verify everything

```bash
# Verify AWS CLI:
aws --version

# Verify kubectl:
kubectl version --client

# Verify eksctl
eksctl: eksctl version
```

- verify secretes in github actions

```bash
	AWS_ACCESS_KEY_ID
	AWS_SECRET_ACCESS_KEY
	AWS_S3_REGION_NAME
	AWS_ECR_REPO_NAME
```

- Execute CICD pipeline till the stage where we build and push image to ECR.
- Create an EKS cluster:

```bash
eksctl create cluster --name cancer-cls-cluster --region us-east-1 --nodegroup-name cancer-cls-nodes --node-type t3.small --nodes 1 --nodes-min 1 --nodes-max 1 --managed
```

- Update kubectl Config(Once the cluster is created, eksctl will automatically update your kubectl config file. However, you can verify and set it manually using:) This ensures your kubectl is pointing to the correct cluster.

```bash
aws eks --region us-east-1 update-kubeconfig --name cancer-cls-cluster
```

- Check EKS Cluster Configuration Ensure you can access your EKS cluster by running

```bash
    aws eks list-clusters
```

- Delete cluster(optional):

```bash
    eksctl delete cluster --name cancer-cls-cluster --region us-east-1
```

Also, verify cluster deletion:

```bash
    eksctl get cluster --region us-east-1
```

- Verify the cluster status:

```bash
    aws eks --region us-east-1 describe-cluster --name cancer-cls-cluster --query "cluster.status"
```

- Check cluster connectivity:

```bash
kubectl get nodes
```

- Check the namespaces:

```bash
kubectl get namespaces
```

- Verify the deployment:

```bash
kubectl get pods
kubectl get svc
```

- Deploy the app on EKS via CICD pipeline

  > > edit ci.yaml, deployment.yaml, dockerfile
  > > Also edit the security group for nodes and edit inbound rule for 8000 port

- Once the LoadBalancer service is up, get the external IP:

```bash
kubectl get svc cancer-cls-service
```

- Try externa-ip:8000 directly on url or on terminal : curl http://external-ip:8080

```bash
curl aea540af5e1e14a30920e9738f48d126-21096163.us-east-1.elb.amazonaws.com:8080
```

# Reference

- [YT-Tutorial- EKS Cluster Deployment ](https://www.youtube.com/watch?v=T4UGsVn0D_I&list=PLupK5DK91flV45dkPXyGViMLtHadRr6sp&index=44)
- https://github.com/vikashishere/YT-Capstone-Project/blob/main/projectflow.txt
