# llama-bench on Kubernetes

在 Kubernetes 中運行 llama-bench，並通過 Web Dashboard 查看結果。

## 📋 前置要求

- Docker
- Kubernetes cluster (minikube, Docker Desktop K8s, 或真實 K8s)
- kubectl

## 🚀 快速開始

### 1️⃣ 構建 Docker Image

```bash
# 在 llama.cpp 目錄下
docker build -t llama-bench:latest -f Dockerfile .
```

如果用 minikube：
```bash
# 讓 minikube 使用本地 Docker daemon
eval $(minikube docker-env)
docker build -t llama-bench:latest -f Dockerfile .
```

### 2️⃣ 準備模型

你有幾個選擇：

#### 選項 A: 使用 HostPath（本地開發）
編輯 `k8s/manifest.yaml`，將 PVC 改為 HostPath：

```yaml
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: llama-models-pv
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadOnlyMany
  hostPath:
    path: /path/to/your/models  # 改成你的模型路徑
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llama-models
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 50Gi
  volumeName: llama-models-pv
```

#### 選項 B: 使用真實 Storage（生產環境）
使用 NFS、EBS、GCE Persistent Disk 等

### 3️⃣ 部署到 Kubernetes

```bash
# 應用 manifests
kubectl apply -f k8s/manifest.yaml

# 檢查狀態
kubectl get ns llama-bench
kubectl get all -n llama-bench

# 查看 Job 日誌
kubectl logs -n llama-bench -f job/llama-bench-job

# 查看 Dashboard pod
kubectl get pods -n llama-bench
kubectl logs -n llama-bench deployment/llama-bench-dashboard
```

### 4️⃣ 訪問 Web Dashboard

```bash
# 轉發端口
kubectl port-forward -n llama-bench svc/llama-bench-dashboard 8080:80

# 訪問
open http://localhost:8080
```

或者如果用 LoadBalancer：
```bash
kubectl get svc -n llama-bench llama-bench-dashboard
# 獲取 EXTERNAL-IP，然後訪問
```

## 📁 目錄結構

```
llama.cpp/
├── Dockerfile                    # 容器構建文件
├── web_dashboard.py             # Web Dashboard Flask 應用
├── llama_bench_runner.py        # llama-bench 運行器
└── k8s/
    ├── manifest.yaml            # K8s 資源清單
    └── entrypoint.sh            # 容器入口腳本
```

## 🔧 自定義配置

### 修改測試參數

編輯 `llama_bench_runner.py` 中的 `TEST_CONFIGS`：

```python
TEST_CONFIGS = {
    "your_test": {
        "description": "Your Test Name",
        "args": ["-n", "128", "-p", "512", "-ngl", "99"]
    }
}
```

### 修改 Job 資源限制

編輯 `k8s/manifest.yaml` 中的 `resources`：

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "4"
  limits:
    memory: "8Gi"
    cpu: "8"
```

## 📊 結果存儲

結果自動保存到 PVC `llama-bench-results` 中：

```bash
# 獲取 results pod
kubectl get pods -n llama-bench

# 進入 pod 查看結果
kubectl exec -it -n llama-bench <pod-name> -- bash
ls -la /results/
```

## 🐛 常見問題

### Q: 容器卡住了？
```bash
# 查看日誌
kubectl logs -n llama-bench job/llama-bench-job

# 重新啟動
kubectl delete -f k8s/manifest.yaml
kubectl apply -f k8s/manifest.yaml
```

### Q: 模型找不到？
- 確保 PVC 掛載正確
- 檢查模型文件名是否匹配 `llama_bench_runner.py` 中的配置

### Q: Web 顯示無數據？
```bash
# 確保 Dashboard pod 運行
kubectl get pods -n llama-bench

# 檢查 results PVC 中是否有 CSV 文件
kubectl exec -it -n llama-bench deployment/llama-bench-dashboard -- ls -la /results/
```

## 🎯 進階用法

### 使用 GPU

編輯 `k8s/manifest.yaml` 的 Job spec：

```yaml
containers:
- name: llama-bench
  # ... 其他配置 ...
  resources:
    requests:
      nvidia.com/gpu: 1
    limits:
      nvidia.com/gpu: 1
```

### 定期運行（CronJob）

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: llama-bench-daily
  namespace: llama-bench
spec:
  schedule: "0 2 * * *"  # 每天 2:00 AM
  jobTemplate:
    spec:
      template:
        spec:
          # ... 同 Job spec ...
```

## 📈 Dashboard 功能

- 🎯 **實時圖表** - 生成速度、模型大小對比
- 📊 **詳細表格** - 所有指標一覽
- 🔄 **自動刷新** - 每 10 秒更新一次
- 📱 **響應式設計** - 支持各種屏幕

## 🧹 清理

```bash
# 刪除所有資源
kubectl delete namespace llama-bench

# 或單獨刪除
kubectl delete -f k8s/manifest.yaml
```

## 📚 相關資源

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [llama.cpp Repository](https://github.com/ggml-org/llama.cpp)
- [Docker Documentation](https://docs.docker.com/)

## 💡 提示

1. **本地測試** - 用 `minikube` 或 Docker Desktop 先測試
2. **監控** - 用 `kubectl top` 監控資源使用
3. **持久化** - 結果存在 PVC 中，Pod 重啟不會丟失
4. **並行運行** - 可以創建多個 Job，各自測試不同配置

祝你學習 K8s 愉快! 🎉
