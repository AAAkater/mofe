# 后端文件夹

## 1 下载依赖

```shell
code mofe/server
uv sync
source ./.venv/bin/activate
```

## 2 启动后端

复制`.env`文件

```shell
cd mofe/server
cp .env.example .env
```

在`.env`里配置pg与minio相关参数

```env
# PGSQL
POSTGRESQL_USER=postgres
POSTGRESQL_PASSWORD=
POSTGRESQL_PROT=5432
POSTGRESQL_SERVER=localhost
POSTGRESQL_DB=mofe

...

# MINIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_SERVER_URL=localhost:9000
MINIO_CONSOLE_URL=localhost:9001
MINIO_BUCKET_NAME=mofe
MINIO_SECURE=false
```

两种启动方式

- 点击`F5`启动,点击`shift+F5`停止运行.
- 或使用`scripts`文件下的 shell.

  ```shell
    cd server
    bash ./scripts/launch.sh
  ```

## 3 模型开发

因为是`workspace`的方式开发,需要什么依赖就去对应`packages`加

如:

```shell
cd packages/PConv
uv add xxx #下载xxx依赖
```
