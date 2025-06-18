# AI修复神器 - 基于深度学习的老照片修复平台

一个现代化的AI照片修复平台，采用先进的深度学习技术，帮助用户修复模糊、损坏的老照片。

## ✨ 主要特性

- 🤖 **AI智能修复** - 基于深度学习算法，自动修复模糊、损坏的照片
- 📸 **多种上传方式** - 支持文件上传、拖拽上传、URL链接上传
- 🔄 **实时对比预览** - 鼠标滑动查看修复前后对比效果
- 📦 **批量处理** - 一次性上传并处理多张图片（最多30张）
- 💾 **一键下载** - 支持单张或批量下载修复后的图片
- 📱 **响应式设计** - 完美适配桌面端和移动端
- 🎨 **现代化UI** - 基于Tailwind CSS的美观界面设计

## 🛠️ 技术栈

- **前端框架**: React 19 + Vite
- **路由管理**: React Router 7
- **样式框架**: Tailwind CSS 4
- **UI组件**: Arco Design
- **图标库**: React Icons
- **状态管理**: React Hooks

## 🚀 快速开始

### 环境要求

- Node.js >= 16.0.0
- npm 或 pnpm

### 安装依赖

```bash
# 使用npm
npm install

# 或使用pnpm
pnpm install
```

### 启动开发服务器

```bash
# 使用npm
npm run dev

# 或使用pnpm
pnpm dev
```

### 构建生产版本

```bash
# 使用npm
npm run build

# 或使用pnpm
pnpm build
```

## 📁 项目结构

```
src/
├── components/          # 可复用组件
│   ├── ImageUploader.jsx    # 图片上传组件
│   └── UploadModal.jsx      # 上传模态窗口
├── pages/              # 页面组件
│   ├── index.jsx           # 主页
│   └── _layout.jsx         # 布局组件
├── assets/             # 静态资源
├── index.css          # 全局样式
├── main.jsx           # 应用入口
└── router.ts          # 路由配置
```

## 🎯 核心功能

### 1. 图片上传
- **拖拽上传**: 直接拖拽图片文件到上传区域
- **点击上传**: 点击按钮选择本地图片文件
- **URL上传**: 输入图片URL链接进行上传
- **格式支持**: JPG、PNG、WEBP等主流图片格式

### 2. AI修复处理
- **智能算法**: 基于深度学习的图像修复技术
- **快速处理**: 通常在几秒内完成修复
- **质量保证**: 高质量的修复效果
- **批量处理**: 支持同时处理多张图片

### 3. 结果展示
- **对比预览**: 滑动查看修复前后对比
- **高清下载**: 下载修复后的高清图片
- **批量下载**: 一键下载所有修复结果

## 🎨 界面设计

### 主页布局
- **导航栏**: 佐糖品牌标识和功能导航
- **hero区域**: 产品介绍和主要操作按钮
- **对比展示**: 交互式的修复效果预览
- **统计数据**: 用户数量和服务统计

### 上传界面
- **模态窗口**: 优雅的弹窗式上传界面
- **进度显示**: 清晰的处理进度反馈
- **结果展示**: 修复前后的对比展示

## 🔧 自定义配置

### Tailwind CSS配置
项目使用Tailwind CSS 4，配置文件位于 `tailwind.config.js`

### Vite配置
构建配置位于 `vite.config.js`，包含：
- React插件配置
- Tailwind CSS集成
- 开发服务器设置

## 🤝 贡献指南

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🎉 致谢

- 感谢所有为这个项目做出贡献的开发者
- 感谢开源社区提供的优秀工具和库
- 特别感谢Unsplash提供的高质量示例图片

---

**享受AI修复的神奇效果，让您的老照片重新焕发生机！** ✨
