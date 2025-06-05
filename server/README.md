# 后端文件夹

## 1 下载依赖

```shell
code mofe/server
uv sync
source ./.venv/bin/activate
```

## 2 启动后端

两种方式

- 点击`F5`启动,点击`shift+F5`停止运行.
- 使用`scripts`文件下的 shell.

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
