import { useState } from "react";
import { FaUpload, FaImages, FaArrowRight } from "react-icons/fa";
import UploadModal from "../components/UploadModal";

function Page() {
  const [isHovering, setIsHovering] = useState(false);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleSliderChange = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = (x / rect.width) * 100;
    setSliderPosition(Math.max(0, Math.min(100, percentage)));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <div className="max-w-7xl mx-auto px-4 py-16">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          {/* 左侧内容区域 */}
          <div className="space-y-8">
            <div className="space-y-6">
              <div className="inline-block">
                <span className="bg-gradient-to-r from-blue-500 to-purple-500 text-white px-4 py-2 rounded-full text-sm font-medium mb-4 inline-block">
                  🎓 工程实践项目
                </span>
              </div>
              <h1 className="text-5xl lg:text-6xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                智能图像修复系统
              </h1>
              <h2 className="text-2xl lg:text-3xl font-semibold bg-gradient-to-r from-gray-700 to-gray-900 bg-clip-text text-transparent">
                基于深度学习的图像增强技术
              </h2>
              <p className="text-lg text-gray-600 leading-relaxed max-w-md">
                采用先进的卷积神经网络和生成对抗网络，实现模糊图像的智能修复。支持多种图像格式，提供实时处理和批量操作功能。
              </p>
              <div className="flex flex-wrap gap-2 mt-4">
                <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">
                  CNN
                </span>
                <span className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">
                  GAN
                </span>
                <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">
                  超分辨率
                </span>
                <span className="bg-orange-100 text-orange-800 px-3 py-1 rounded-full text-sm">
                  图像去噪
                </span>
              </div>
            </div>

            {/* 按钮组 */}
            <div className="space-y-4">
              <button
                className="w-full max-w-sm bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 hover:from-blue-600 hover:via-purple-600 hover:to-pink-600 text-white font-semibold py-4 px-8 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 hover:scale-105 flex items-center justify-center space-x-3"
                onMouseEnter={() => setIsHovering(true)}
                onMouseLeave={() => setIsHovering(false)}
                onClick={() => setIsModalOpen(true)}
              >
                <FaUpload className="text-lg" />
                <div className="text-left">
                  <div className="text-lg">开始体验</div>
                  <div className="text-sm opacity-90">上传图片测试算法效果</div>
                </div>
                <FaArrowRight
                  className={`text-lg transition-transform duration-300 ${
                    isHovering ? "translate-x-2" : ""
                  }`}
                />
              </button>
              <div className="text-sm text-gray-500 max-w-sm">
                💡 支持 JPG、PNG、WebP 格式，最大 10MB
              </div>
            </div>
          </div>

          {/* 右侧图片对比区域 */}
          <div className="relative ">
            <div className="relative w-full max-w-lg mx-auto bg-white rounded-2xl shadow-2xl overflow-hidden">
              {/* 对比图片容器 */}
              <div
                className="relative w-full h-80 lg:h-96 cursor-crosshair"
                onMouseMove={handleSliderChange}
                onClick={handleSliderChange}
              >
                {/* 原图 */}
                <div className="absolute inset-0">
                  <img
                    src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=20"
                    alt="原图"
                    style={{
                      imageRendering: "pixelated",
                      filter: "blur(1px) contrast(1.1)",
                      transformOrigin: "center",
                    }}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-4 left-4 bg-black/50 text-white px-3 py-1 rounded-full text-sm font-medium">
                    原图
                  </div>
                </div>

                {/* 修复后的图片 */}
                <div
                  className="absolute inset-0 overflow-hidden"
                  style={{
                    clipPath: `polygon(${sliderPosition}% 0%, 100% 0%, 100% 100%, ${sliderPosition}% 100%)`,
                  }}
                >
                  <img
                    src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1000&q=80"
                    alt="修复后"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-4 right-4 bg-blue-600 text-white px-3 py-1 rounded-full text-sm font-medium">
                    修复后
                  </div>
                </div>

                {/* 滑动条 */}
                <div
                  className="absolute top-0 bottom-0 w-[1px] bg-white shadow-lg cursor-ew-resize flex items-center justify-center"
                  style={{
                    left: `${sliderPosition}%`,
                    transform: "translateX(-50%)",
                  }}
                >
                  <div className="w-6 h-6 bg-white rounded-full shadow-lg flex items-center justify-center">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 功能展示区域 */}
        <div className="mt-20 pt-12">
          {/* 用户统计数据 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-16">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">
                500万+
              </div>
              <div className="text-gray-600">图片已修复</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">98%</div>
              <div className="text-gray-600">用户满意度</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">
                150万+
              </div>
              <div className="text-gray-600">活跃用户</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">24/7</div>
              <div className="text-gray-600">全天候服务</div>
            </div>
          </div>

          {/* 功能标题 */}
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">
              满足您需求的理想工具
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              AI修复神器提供全方位的图片修复解决方案，适用于各种场景和需求
            </p>
          </div>

          {/* 功能特点 */}
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-6 max-w-6xl mx-auto mb-16">
            <div className="flex flex-col items-center text-center space-y-3">
              <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-white"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="text-sm text-gray-700">
                高清照片书籍和编辑印刷
              </div>
            </div>

            <div className="flex flex-col items-center text-center space-y-3">
              <div className="w-12 h-12 bg-purple-500 rounded-full flex items-center justify-center">
                <FaImages className="text-white text-xl" />
              </div>
              <div className="text-sm text-gray-700">社交媒体分享优化</div>
            </div>

            <div className="flex flex-col items-center text-center space-y-3">
              <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-white"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="text-sm text-gray-700">专业设计高清文档</div>
            </div>

            <div className="flex flex-col items-center text-center space-y-3">
              <div className="w-12 h-12 bg-orange-500 rounded-full flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-white"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8a2 2 0 11-4 0 2 2 0 014 0zM14 15a4 4 0 00-8 0v3h8v-3z" />
                </svg>
              </div>
              <div className="text-sm text-gray-700">人像照片修复</div>
            </div>

            <div className="flex flex-col items-center text-center space-y-3">
              <div className="w-12 h-12 bg-red-500 rounded-full flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-white"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" />
                </svg>
              </div>
              <div className="text-sm text-gray-700">批量处理支持</div>
            </div>

            <div className="flex flex-col items-center text-center space-y-3">
              <div className="w-12 h-12 bg-indigo-500 rounded-full flex items-center justify-center">
                <svg
                  className="w-6 h-6 text-white"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="text-sm text-gray-700">AI智能算法</div>
            </div>
          </div>

          {/* 用户评价 */}
          <div className="bg-gray-50 rounded-2xl p-8 mb-16">
            <h3 className="text-2xl font-bold text-center text-gray-800 mb-8">
              用户真实评价
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white rounded-lg p-6 shadow-sm">
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center text-white font-semibold">
                    张
                  </div>
                  <div className="ml-3">
                    <div className="font-semibold text-gray-800">张小明</div>
                    <div className="text-sm text-gray-600">摄影师</div>
                  </div>
                </div>
                <div className="flex text-yellow-400 mb-2">★★★★★</div>
                <p className="text-gray-600 text-sm">
                  "修复效果太棒了！以前模糊的老照片现在清晰如新，操作简单快捷，强烈推荐！"
                </p>
              </div>

              <div className="bg-white rounded-lg p-6 shadow-sm">
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center text-white font-semibold">
                    李
                  </div>
                  <div className="ml-3">
                    <div className="font-semibold text-gray-800">李设计师</div>
                    <div className="text-sm text-gray-600">平面设计师</div>
                  </div>
                </div>
                <div className="flex text-yellow-400 mb-2">★★★★★</div>
                <p className="text-gray-600 text-sm">
                  "作为设计师，这个工具帮我节省了大量时间，AI修复效果专业，客户都很满意！"
                </p>
              </div>

              <div className="bg-white rounded-lg p-6 shadow-sm">
                <div className="flex items-center mb-4">
                  <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center text-white font-semibold">
                    王
                  </div>
                  <div className="ml-3">
                    <div className="font-semibold text-gray-800">王女士</div>
                    <div className="text-sm text-gray-600">普通用户</div>
                  </div>
                </div>
                <div className="flex text-yellow-400 mb-2">★★★★★</div>
                <p className="text-gray-600 text-sm">
                  "家里的老照片终于可以变清晰了，操作很简单，连我这样的新手都能轻松使用！"
                </p>
              </div>
            </div>
          </div>

          {/* 产品展示卡片 */}
          <div className="mb-16">
            <h3 className="text-2xl font-bold text-center text-gray-800 mb-8">
              应用场景展示
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                <div className="h-40 bg-gradient-to-br from-blue-100 to-blue-200 flex items-center justify-center">
                  <img
                    src="https://images.unsplash.com/photo-1549298916-b41d501d3772?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80"
                    alt="电子商务产品"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="p-4">
                  <h4 className="text-lg font-semibold text-gray-800 mb-2">
                    电商产品图
                  </h4>
                  <p className="text-gray-600 text-sm">
                    提升商品图片清晰度，增加销量转化率
                  </p>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                <div className="h-40 bg-gradient-to-br from-orange-100 to-orange-200 flex items-center justify-center">
                  <img
                    src="https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80"
                    alt="房地产厨房"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="p-4">
                  <h4 className="text-lg font-semibold text-gray-800 mb-2">
                    房产展示
                  </h4>
                  <p className="text-gray-600 text-sm">
                    房产图片高清修复，提升展示效果
                  </p>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                <div className="h-40 bg-gradient-to-br from-green-100 to-green-200 flex items-center justify-center">
                  <img
                    src="https://images.unsplash.com/photo-1559526324-4b87b5e36e44?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80"
                    alt="艺术作品"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="p-4">
                  <h4 className="text-lg font-semibold text-gray-800 mb-2">
                    艺术收藏
                  </h4>
                  <p className="text-gray-600 text-sm">
                    珍贵艺术品数字化高清保存
                  </p>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                <div className="h-40 bg-gradient-to-br from-purple-100 to-purple-200 flex items-center justify-center">
                  <img
                    src="https://images.unsplash.com/photo-1606787366850-de6330128bfc?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80"
                    alt="家庭照片"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="p-4">
                  <h4 className="text-lg font-semibold text-gray-800 mb-2">
                    家庭回忆
                  </h4>
                  <p className="text-gray-600 text-sm">
                    老照片修复，重现珍贵回忆
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* 选择计划 */}
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">
              选择您的计划
            </h2>
            <p className="text-lg text-gray-600">
              选择最适合您需求的套餐，开始您的AI修复之旅
            </p>
          </div>

          {/* 定价计划 */}
          <div className=" w-1/3 gap-8 max-w-5xl mx-auto mb-16">
            {/* 免费版 */}
            <div className="bg-white rounded-2xl shadow-lg p-8 border border-gray-200 hover:shadow-xl transition-shadow duration-300">
              <div className="text-center">
                <h3 className="text-xl font-bold text-gray-800 mb-2">
                  免费体验
                </h3>
                <div className="text-3xl font-bold text-gray-800 mb-1">¥0</div>
                <div className="text-gray-600 mb-6">永久免费</div>
                <ul className="text-left space-y-3 mb-8">
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 text-green-500 mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    不限次数图片修复
                  </li>
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 text-green-500 mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    基础修复效果
                  </li>
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 text-green-500 mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    标准处理速度
                  </li>
                </ul>
                <button className="w-full bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-3 px-6 rounded-lg transition-colors duration-200">
                  开始使用
                </button>
              </div>
            </div>

            {/* 专业版 */}
            {/* <div className="bg-white rounded-2xl shadow-xl p-8 border-2 border-blue-500 relative hover:shadow-2xl transition-shadow duration-300 transform scale-105">
              <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                <span className="bg-blue-500 text-white px-4 py-1 rounded-full text-sm font-medium">
                  推荐
                </span>
              </div>
              <div className="text-center">
                <h3 className="text-xl font-bold text-gray-800 mb-2">专业版</h3>
                <div className="text-3xl font-bold text-blue-600 mb-1">¥29</div>
                <div className="text-gray-600 mb-6">每月</div>
                <ul className="text-left space-y-3 mb-8">
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 text-green-500 mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    每日100张图片
                  </li>
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 text-green-500 mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    高级AI修复
                  </li>
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 text-green-500 mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    优先处理队列
                  </li>
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 text-green-500 mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    批量处理功能
                  </li>
                </ul>
                <button className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors duration-200">
                  立即升级
                </button>
              </div>
            </div> */}

            {/* 企业版 */}
            {/* <div className="bg-white rounded-2xl shadow-lg p-8 border border-gray-200 hover:shadow-xl transition-shadow duration-300">
              <div className="text-center">
                <h3 className="text-xl font-bold text-gray-800 mb-2">企业版</h3>
                <div className="text-3xl font-bold text-gray-800 mb-1">
                  ¥299
                </div>
                <div className="text-gray-600 mb-6">每月</div>
                <ul className="text-left space-y-3 mb-8">
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 text-green-500 mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    无限制使用
                  </li>
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 text-green-500 mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    顶级AI算法
                  </li>
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 text-green-500 mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    API接口支持
                  </li>
                  <li className="flex items-center">
                    <svg
                      className="w-5 h-5 text-green-500 mr-3"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    专属客服支持
                  </li>
                </ul>
                <button className="w-full bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200">
                  联系销售
                </button>
              </div>
            </div> */}
          </div>
        </div>

        <UploadModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          mode="single"
        />
      </div>
    </div>
  );
}

export default Page;
