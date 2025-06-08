import { useState } from "react";
import { FaTimes, FaCog, FaDownload } from "react-icons/fa";
import ImageUploader from "./ImageUploader";

const UploadModal = ({ isOpen, onClose, mode = "single" }) => {
  // mode: 'single' or 'batch'
  const [uploadedImages, setUploadedImages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedImages, setProcessedImages] = useState([]);

  const handleImageUpload = (images) => {
    setUploadedImages(images);
  };

  const handleStartProcessing = () => {
    if (uploadedImages.length === 0) return;

    setIsProcessing(true);

    // 模拟AI处理过程
    setTimeout(() => {
      const processed = uploadedImages.map((img) => ({
        ...img,
        processedUrl: img.preview || img.url, // 实际项目中这里会是AI处理后的图片URL
        isProcessed: true,
      }));
      setProcessedImages(processed);
      setIsProcessing(false);
    }, 3000);
  };

  const handleDownload = (image) => {
    // 实际项目中会下载处理后的图片
    console.log("下载图片:", image);
  };

  const handleDownloadAll = () => {
    processedImages.forEach((img) => handleDownload(img));
  };

  const handleClose = () => {
    setUploadedImages([]);
    setProcessedImages([]);
    setIsProcessing(false);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80  flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden">
        {/* 头部 */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div>
            <h2 className="text-2xl font-bold text-gray-800">
              {mode === "single" ? "单张图片修复" : "批量图片修复"}
            </h2>
            <p className="text-gray-600 mt-1">
              {mode === "single"
                ? "上传一张图片进行AI修复"
                : `批量上传最多30张图片`}
            </p>
          </div>
          <button
            onClick={handleClose}
            className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-gray-100 transition-colors duration-200"
          >
            <FaTimes className="text-gray-500" />
          </button>
        </div>

        {/* 内容区域 */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
          {!isProcessing && processedImages.length === 0 && (
            <ImageUploader
              onImageUpload={handleImageUpload}
              maxFiles={mode === "single" ? 1 : 30}
            />
          )}

          {/* 处理中状态 */}
          {isProcessing && (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-6">
                <div className="w-full h-full border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
              </div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">
                AI正在处理您的图片
              </h3>
              <p className="text-gray-600">
                正在使用深度学习技术修复图片，请稍候...
              </p>
              <div className="mt-6 bg-gray-200 rounded-full h-2 w-64 mx-auto">
                <div className="bg-blue-600 h-2 rounded-full w-1/2 animate-pulse"></div>
              </div>
            </div>
          )}

          {/* 处理完成状态 */}
          {processedImages.length > 0 && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-semibold text-gray-800">
                  处理完成 ({processedImages.length}张)
                </h3>
                <button
                  onClick={handleDownloadAll}
                  className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors duration-200"
                >
                  <FaDownload />
                  <span>全部下载</span>
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {processedImages.map((image) => (
                  <div key={image.id} className="bg-gray-50 rounded-xl p-4">
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      {/* 原图 */}
                      <div>
                        <p className="text-sm font-medium text-gray-600 mb-2">
                          原图
                        </p>
                        <div className="aspect-square bg-white rounded-lg overflow-hidden">
                          <img
                            src={image.preview || image.url}
                            alt="原图"
                            className="w-full h-full object-cover filter blur-sm brightness-75"
                          />
                        </div>
                      </div>

                      {/* 修复后 */}
                      <div>
                        <p className="text-sm font-medium text-gray-600 mb-2">
                          修复后
                        </p>
                        <div className="aspect-square bg-white rounded-lg overflow-hidden">
                          <img
                            src={image.processedUrl}
                            alt="修复后"
                            className="w-full h-full object-cover"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">
                        {image.name}
                      </span>
                      <button
                        onClick={() => handleDownload(image)}
                        className="flex items-center space-x-1 text-blue-600 hover:text-blue-700 font-medium text-sm"
                      >
                        <FaDownload />
                        <span>下载</span>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 底部按钮 */}
        {uploadedImages.length > 0 &&
          !isProcessing &&
          processedImages.length === 0 && (
            <div className="flex items-center justify-between p-6 border-t border-gray-200 bg-gray-50">
              <div className="text-sm text-gray-600">
                已选择 {uploadedImages.length} 张图片
              </div>
              <div className="flex items-center space-x-3">
                <button
                  onClick={handleClose}
                  className="px-6 py-2 text-gray-600 hover:text-gray-800 font-medium transition-colors duration-200"
                >
                  取消
                </button>
                <button
                  onClick={handleStartProcessing}
                  className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors duration-200"
                >
                  <FaCog />
                  <span>开始修复</span>
                </button>
              </div>
            </div>
          )}
      </div>
    </div>
  );
};

export default UploadModal;
