# docker部署

## 1 构建 mofe-server 镜像

PS:相关依赖都已就国内源处理

```shell
cd mofe/server
docker build -t mofe-server:latest .  
```

## 2 启动后端

```shell
cd mofe/docker
cp .env.example .env
docker compose up -d
```
