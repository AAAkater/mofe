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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="max-w-7xl mx-auto px-4 py-16">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          {/* 左侧内容区域 */}
          <div className="space-y-8">
            <div className="space-y-6">
              <h1 className="text-5xl lg:text-6xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                AI修复神器
              </h1>
              <h2 className="text-2xl lg:text-3xl font-semibold text-gray-800">
                一键照片修复
              </h2>
              <p className="text-lg text-gray-600 leading-relaxed max-w-md">
                利用先进的智能修复模糊图片技术，只需一键操作，无需等待，即刻体验模糊照片变清晰，重现美好瞬间。
              </p>
            </div>

            {/* 按钮组 */}
            <div className="">
              <button
                className="w-full max-w-sm bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white font-semibold py-4 px-8 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 flex items-center justify-center space-x-3"
                onMouseEnter={() => setIsHovering(true)}
                onMouseLeave={() => setIsHovering(false)}
                onClick={() => setIsModalOpen(true)}
              >
                <FaUpload className="text-lg" />
                <div className="text-left">
                  <div className="text-lg">上传图片</div>
                  <div className="text-sm opacity-90">
                    可传入文件、粘贴图片或URL
                  </div>
                </div>
                <FaArrowRight
                  className={`text-lg transition-transform duration-300 ${
                    isHovering ? "translate-x-1" : ""
                  }`}
                />
              </button>
            </div>
          </div>

          {/* 右侧图片对比区域 */}
          <div className="relative">
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
                    className="w-full h-full object-cover filter blur-sm brightness-75"
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
                  <div className="absolute top-4 right-4 bg-green-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                    修复后
                  </div>
                </div>

                {/* 滑动条 */}
                <div
                  className="absolute top-0 bottom-0 w-1 bg-white shadow-lg cursor-ew-resize flex items-center justify-center"
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

        {/* 底部统计数据 */}
        <div className="mt-20 pt-12 border-t border-gray-200">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
            <div className="space-y-2">
              <div className="text-3xl font-bold text-blue-600">1M+</div>
              <div className="text-gray-600">用户数量</div>
            </div>
            <div className="space-y-2">
              <div className="text-3xl font-bold text-green-600">500k+</div>
              <div className="text-gray-600">图片修复</div>
            </div>
            <div className="space-y-2">
              <div className="text-3xl font-bold text-purple-600">99%</div>
              <div className="text-gray-600">满意度</div>
            </div>
            <div className="space-y-2">
              <div className="text-3xl font-bold text-orange-600">24/7</div>
              <div className="text-gray-600">在线服务</div>
            </div>
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
