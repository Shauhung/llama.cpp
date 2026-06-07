# llama-server on Kubernetes

這個目錄放的是本地 Kubernetes / minikube 環境中啟動 `llama-server` 的設定。

主線是：

```text
Dockerfile build llama-server image
  -> minikube 掛載本機 GGUF models 到 /models
  -> Kubernetes Deployment 啟動 /app/llama-server
  -> Service 暴露 8080
  -> kubectl port-forward 到 localhost:8080
```

## 前置需求

- Docker
- minikube 或其他 Kubernetes cluster
- kubectl
- 本機已有 GGUF 模型，例如：

```text
/Users/username/code/cpp/llama.cpp/models/llama_quant/Llama3-8.0B-Q4_K_M.gguf
```

## 快速開始

### 1. 啟動 minikube 並掛載模型目錄

```bash
minikube start \
  --memory 16384 \
  --mount \
  --mount-string="/Users/username/code/cpp/llama.cpp/models/llama_quant:/models"
```

`k8s/manifest.yaml` 裡的 `hostPath` 使用 `/models`，這個路徑對應到 minikube VM
裡的掛載點。容器內會把 `/models` 掛到 `/data`。

### 2. Build image

在 llama.cpp repo root：

```bash
docker build -t my-llama-server:v1 -f Dockerfile .
```

如果 image 是 build 在本機 Docker daemon，而 cluster 是 minikube，記得載入：

```bash
minikube image load my-llama-server:v1
```

### 3. 部署 llama-server

```bash
kubectl apply -f k8s/manifest.yaml
kubectl get pods
```

等 pod 變成 `Running`：

```bash
kubectl logs -f deployment/llama-server-deployment
```

### 4. Port forward

```bash
kubectl port-forward service/llama-server-service 8080:8080
```

然後可以測：

```bash
curl http://localhost:8080/v1/models
```

或呼叫 OpenAI-compatible chat endpoint：

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "max_tokens": 32
  }'
```

## manifest 內容

[manifest.yaml](manifest.yaml) 目前包含兩個資源：

- `Deployment/llama-server-deployment`
- `Service/llama-server-service`

Deployment 使用 image：

```yaml
image: my-llama-server:v1
imagePullPolicy: Never
```

啟動參數：

```yaml
args:
  - "-m"
  - "/data/Llama3-8.0B-Q4_K_M.gguf"
  - "--host"
  - "0.0.0.0"
  - "--port"
  - "8080"
  - "-c"
  - "4096"
  - "-np"
  - "1"
```

模型 volume：

```yaml
hostPath:
  path: /models
```

容器內路徑：

```yaml
mountPath: /data
```

## 修改模型或資源

要換模型，改 `k8s/manifest.yaml` 的 `-m` 參數：

```yaml
- "-m"
- "/data/your-model.gguf"
```

要調整 context：

```yaml
- "-c"
- "4096"
```

要調整資源：

```yaml
resources:
  requests:
    memory: "6Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

## deploy.sh

可以用：

```bash
k8s/deploy.sh
```

它會：

1. build `my-llama-server:v1`
2. 若有 minikube，載入 image
3. `kubectl apply -f k8s/manifest.yaml`
4. 等 deployment ready
5. port-forward `localhost:8080 -> llama-server-service:8080`

## 常用指令

```bash
kubectl get deploy,svc,pods
kubectl describe pod -l app=llama-server
kubectl logs -f deployment/llama-server-deployment
kubectl rollout restart deployment/llama-server-deployment
kubectl delete -f k8s/manifest.yaml
```

## 清理

```bash
kubectl delete -f k8s/manifest.yaml
minikube stop
```
