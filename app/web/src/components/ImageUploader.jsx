import { useState, useRef, useEffect } from "react";
import { FaUpload, FaImage, FaLink, FaTimes, FaDownload, FaSpinner } from "react-icons/fa";
import { downloadImage, uploadImage } from "../service/index.js";

const ImageUploader = ({ onImageUpload, maxFiles = 1,uploadedImages,setUploadedImages }) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadMode, setUploadMode] = useState("file"); // 'file', 'url'
  const [urlInput, setUrlInput] = useState("");
  const [uploading, setUploading] = useState(false);
  const [previewImageUrl, setPreviewImageUrl] = useState(null);
  const [previewOriginImage, setPreviewOriginImage] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [repairedImageUrls, setRepairedImageUrls] = useState({}); // {id: url}
  const [repairStatus, setRepairStatus] = useState({}); // {id: 'pending' | 'processing' | 'completed' | 'failed'}
  const [pollingIntervals, setPollingIntervals] = useState({}); // 存储轮询定时器
  const [retryCount, setRetryCount] = useState({}); // 记录重试次数

  const fileInputRef = useRef(null);

  // 保存到历史记录的辅助函数
  const saveToHistory = (historyItem) => {
    try {
      const existingHistory = JSON.parse(localStorage.getItem('repairHistory') || '[]');
      // 检查是否已存在相同file_id的记录，如果存在则更新
      const existingIndex = existingHistory.findIndex(item => item.file_id === historyItem.file_id);
      if (existingIndex !== -1) {
        existingHistory[existingIndex] = historyItem;
      } else {
        existingHistory.unshift(historyItem); // 添加到开头
      }
      
      // 限制历史记录数量，最多保存100条
      if (existingHistory.length > 100) {
        existingHistory.splice(100);
      }
      
      // 将imageBlob转换为base64存储
      if (historyItem.imageBlob) {
        const reader = new FileReader();
        reader.onload = () => {
          const historyItemWithBase64 = {
            ...historyItem,
            imageData: reader.result,
          };
          delete historyItemWithBase64.imageBlob;
          
          const updatedHistory = existingIndex !== -1 
            ? existingHistory.map((item, index) => 
                index === existingIndex ? historyItemWithBase64 : item
              )
            : [historyItemWithBase64, ...existingHistory.slice(1)];
          
          localStorage.setItem('repairHistory', JSON.stringify(updatedHistory));
        };
        reader.readAsDataURL(new Blob([historyItem.imageBlob]));
      } else {
        localStorage.setItem('repairHistory', JSON.stringify(existingHistory));
      }
    } catch (error) {
      console.error('保存历史记录失败:', error);
    }
  };

  // 轮询检查图片修复状态
  const pollRepairStatus = async (imageId) => {
    try {
      const response = await downloadImage(imageId);
      const url = window.URL.createObjectURL(new Blob([response.data]));
      setRepairedImageUrls((prev) => ({ ...prev, [imageId]: url }));
      setRepairStatus((prev) => ({ ...prev, [imageId]: 'completed' }));
      
      // 保存到历史记录
      const historyItem = {
        file_id: imageId,
        filename: uploadedImages.find(img => img.id === imageId)?.name || `image_${imageId}`,
        status: 'completed',
        created_at: new Date().toISOString(),
        imageBlob: response.data, // 保存图片数据
      };
      saveToHistory(historyItem);
      
      // 清理轮询定时器
      if (pollingIntervals[imageId]) {
        clearInterval(pollingIntervals[imageId]);
        setPollingIntervals((prev) => {
          const copy = { ...prev };
          delete copy[imageId];
          return copy;
        });
      }
    } catch (error) {
      // 如果还是404或其他错误，说明还在处理中，继续轮询
      if (error.response?.status === 404) {
        // 继续轮询，不做任何操作
        return;
      } else {
        // 其他错误，标记为失败
        setRepairStatus((prev) => ({ ...prev, [imageId]: 'failed' }));
        
        // 保存失败记录到历史
        const historyItem = {
          file_id: imageId,
          filename: uploadedImages.find(img => img.id === imageId)?.name || `image_${imageId}`,
          status: 'failed',
          created_at: new Date().toISOString(),
        };
        saveToHistory(historyItem);
        
        if (pollingIntervals[imageId]) {
          clearInterval(pollingIntervals[imageId]);
          setPollingIntervals((prev) => {
            const copy = { ...prev };
            delete copy[imageId];
            return copy;
          });
        }
      }
    }
  };

  // 开始轮询
  const startPolling = (imageId) => {
    setRepairStatus((prev) => ({ ...prev, [imageId]: 'processing' }));
    setRetryCount((prev) => ({ ...prev, [imageId]: 0 }));
    
    // 保存处理中状态到历史记录
    const historyItem = {
      file_id: imageId,
      filename: uploadedImages.find(img => img.id === imageId)?.name || `image_${imageId}`,
      status: 'processing',
      created_at: new Date().toISOString(),
    };
    saveToHistory(historyItem);
    
    const intervalId = setInterval(() => {
      pollRepairStatus(imageId);
    }, 2000); // 每2秒检查一次
    
    setPollingIntervals((prev) => ({ ...prev, [imageId]: intervalId }));

    // 设置最大轮询时间为60秒，防止无限等待
    setTimeout(() => {
      if (pollingIntervals[imageId] && repairStatus[imageId] === 'processing') {
        clearInterval(pollingIntervals[imageId]);
        setPollingIntervals((prev) => {
          const copy = { ...prev };
          delete copy[imageId];
          return copy;
        });
        setRepairStatus((prev) => ({ ...prev, [imageId]: 'failed' }));
        
        // 超时失败也要更新历史记录
        const timeoutHistoryItem = {
          file_id: imageId,
          filename: uploadedImages.find(img => img.id === imageId)?.name || `image_${imageId}`,
          status: 'failed',
          created_at: new Date().toISOString(),
        };
        saveToHistory(timeoutHistoryItem);
      }
    }, 60000);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFiles(files);
    }
  };

  const handleFiles = async (files) => {
    if (uploadedImages.length + files.length > maxFiles) {
      alert(`最多只能上传 ${maxFiles} 张图片`);
      return;
    }

    setUploading(true);
    try {
      for (const file of files) {
        const formData = new FormData();
        formData.append("file", file);

        const response = await uploadImage(formData);
        console.log(response)
        const newImage = {
          id: response.data.data.id,
          name: file.name,
          size: file.size,
          preview: URL.createObjectURL(file),
          url: response.data.data.id,
          bucket: response.data.data.bucket,
        };

        setUploadedImages((prev) => {
          const updated = [...prev, newImage];
          // onImageUpload && onImageUpload(updated);
          return updated;
        });

        // 开始轮询检查修复状态
        startPolling(newImage.id);
      }
    } catch (error) {
      console.error("上传失败:", error);
    } finally {
      setUploading(false);
    }
  };

  const handleFileInput = (e) => {
    const files = e.target.files;
    if (files) {
      handleFiles(files);
    }
  };

  const handleUrlSubmit = async () => {
    if (uploadedImages.length >= maxFiles) {
      alert(`最多只能上传 ${maxFiles} 张图片`);
      return;
    }

    if (!urlInput.trim()) return;

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("url_file", urlInput);

      const response = await uploadImage(formData);

      const newImage = {
        id: response.data.id,
        name: urlInput.split("/").pop() || "未命名图片",
        url: response.data.id,
        bucket: response.data.bucket,
      };

      setUploadedImages((prev) => {
        const updated = [...prev, newImage];
        // onImageUpload && onImageUpload(updated);
        return updated;
      });

      // 开始轮询检查修复状态
      startPolling(newImage.id);

      setUrlInput("");
    } catch (error) {
      console.error("URL上传失败:", error);
    } finally {
      setUploading(false);
    }
  };

  const handleDownload = async (image) => {
    console.log(image)
    try {
      const response = await downloadImage(image.id);
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", image.name);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
    } catch (error) {
      console.error("下载图片失败:", error);
    }
  };

  const removeImage = (id) => {
    // 清理轮询定时器
    if (pollingIntervals[id]) {
      clearInterval(pollingIntervals[id]);
      setPollingIntervals((prev) => {
        const copy = { ...prev };
        delete copy[id];
        return copy;
      });
    }

    // 清理修复状态
    setRepairStatus((prev) => {
      const copy = { ...prev };
      delete copy[id];
      return copy;
    });

    // 清理重试计数
    setRetryCount((prev) => {
      const copy = { ...prev };
      delete copy[id];
      return copy;
    });

    // 清理修复后图片URL
    if (repairedImageUrls[id]) {
      window.URL.revokeObjectURL(repairedImageUrls[id]);
      setRepairedImageUrls((prev) => {
        const copy = { ...prev };
        delete copy[id];
        return copy;
      });
    }

    setUploadedImages((prev) => {
      const updated = prev.filter((img) => img.id !== id);
      // onImageUpload && onImageUpload(updated);
      return updated;
    });
  };
  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  const handlePreview = async (image) => {
    setPreviewLoading(true);
    setPreviewOriginImage(image);
    
    if (repairStatus[image.id] === 'completed' && repairedImageUrls[image.id]) {
      // 如果已经有修复后的图片，直接使用
      setPreviewImageUrl(repairedImageUrls[image.id]);
    } else {
      // 否则尝试获取
      try {
        const response = await downloadImage(image.id);
        const url = window.URL.createObjectURL(new Blob([response.data]));
        setPreviewImageUrl(url);
      } catch (error) {
        console.error("预览图片失败:", error);
        setPreviewImageUrl(null);
      }
    }
    setPreviewLoading(false);
  };

  // 清理组件卸载时的定时器
  useEffect(() => {
    return () => {
      Object.values(pollingIntervals).forEach(clearInterval);
    };
  }, [pollingIntervals]);

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* 上传模式切换 */}
      <div className="flex items-center justify-center mb-6">
        <div className="bg-gray-100 rounded-lg p-1 flex">
          <button
            onClick={() => setUploadMode("file")}
            disabled={uploading}
            className={`px-4 py-2 rounded-md font-medium transition-all duration-200 ${
              uploadMode === "file"
                ? "bg-white text-blue-600 shadow-sm"
                : "text-gray-600 hover:text-blue-600"
            } ${uploading ? "opacity-50 cursor-not-allowed" : ""}`}
          >
            <FaImage className="inline mr-2" />
            文件上传
          </button>
          <button
            onClick={() => setUploadMode("url")}
            disabled={uploading}
            className={`px-4 py-2 rounded-md font-medium transition-all duration-200 ${
              uploadMode === "url"
                ? "bg-white text-blue-600 shadow-sm"
                : "text-gray-600 hover:text-blue-600"
            } ${uploading ? "opacity-50 cursor-not-allowed" : ""}`}
          >
            <FaLink className="inline mr-2" />
            URL输入
          </button>
        </div>
      </div>
      {/* 文件上传区域 */}
      { uploadMode === "file" && (
        <div
          className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
            dragActive
              ? "border-blue-500 bg-blue-50"
              : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
          } ${uploading ? "opacity-50 cursor-not-allowed" : ""}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple={maxFiles > 1}
            accept="image/*"
            onChange={handleFileInput}
            disabled={uploading}
            className="hidden"
          />

          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
              <FaUpload
                className={`text-2xl text-blue-600 ${
                  uploading ? "animate-pulse" : ""
                }`}
              />
            </div>

            <div>
              <p className="text-lg font-semibold text-gray-800 mb-2">
                {uploading ? (
                  "正在上传..."
                ) : (
                  <>
                    拖拽图片到此处，或
                    <button
                      onClick={openFileDialog}
                      disabled={uploading}
                      className="text-blue-600 hover:text-blue-700 underline ml-1"
                    >
                      点击上传
                    </button>
                  </>
                )}
              </p>
              <p className="text-sm text-gray-500">
                支持 JPG、PNG、WEBP 格式，
                {maxFiles > 1 ? `最多 ${maxFiles} 张图片` : "单张图片"}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* URL输入区域 */}
      { uploadMode === "url" && (
        <div className="space-y-4">
          <div className="flex space-x-2">
            <input
              type="url"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="请输入图片URL地址..."
              disabled={uploading}
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
              onKeyPress={(e) =>
                e.key === "Enter" && !uploading && handleUrlSubmit()
              }
            />
            <button
              onClick={handleUrlSubmit}
              disabled={!urlInput.trim() || uploading}
              className={`px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white font-medium rounded-lg transition-colors duration-200 ${
                uploading ? "opacity-50 cursor-not-allowed" : ""
              }`}
            >
              {uploading ? "上传中..." : "添加"}
            </button>
          </div>
          <p className="text-sm text-gray-500 text-center">
            请确保URL指向有效的图片文件
          </p>
        </div>
      )}

      {/* 已上传图片预览 */}
      {uploadedImages.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            已上传图片 ({uploadedImages.length})
          </h3>
          <div className="space-y-4">
            {uploadedImages.map((image) => (
              <div key={image.id} className="relative bg-white rounded-xl shadow-md border border-gray-200 p-4 group">
                <div className="flex flex-col sm:flex-row gap-4 items-center">
                  {/* 原图 */}
                  <div className="flex flex-col items-center flex-1">
                    <div className="w-32 h-32 sm:w-40 sm:h-40 bg-gray-50 rounded-lg flex items-center justify-center overflow-hidden border">
                      <img
                        src={image.preview || image.url}
                        alt={image.name}
                        className="max-w-full max-h-full object-contain"
                        onError={(e) => {
                          e.target.src =
                            "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtc2l6ZT0iMTgiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIiBmaWxsPSIjOTk5Ij7lm77niYfliqDovb3lpLHotKU8L3RleHQ+PC9zdmc+";
                        }}
                      />
                    </div>
                    <span className="mt-2 px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-md font-medium">原图</span>
                  </div>

                  {/* 分隔符 */}
                  <div className="hidden sm:block w-px h-20 bg-gray-200"></div>
                  <div className="sm:hidden w-20 h-px bg-gray-200"></div>

                  {/* 修复后图 */}
                  <div className="flex flex-col items-center flex-1 relative">
                    <div className="w-32 h-32 sm:w-40 sm:h-40 bg-gray-50 rounded-lg flex items-center justify-center overflow-hidden border relative group/hover">
                      {repairStatus[image.id] === 'processing' ? (
                        <div className="flex flex-col items-center justify-center text-blue-600">
                          <FaSpinner className="text-2xl animate-spin mb-2" />
                          <span className="text-xs">修复中...</span>
                        </div>
                      ) : repairStatus[image.id] === 'failed' ? (
                        <div className="flex flex-col items-center justify-center text-red-600 cursor-pointer" onClick={() => startPolling(image.id)}>
                          <FaTimes className="text-2xl mb-2" />
                          <span className="text-xs">修复失败</span>
                          <span className="text-xs text-blue-600 underline mt-1">点击重试</span>
                        </div>
                      ) : repairStatus[image.id] === 'completed' && repairedImageUrls[image.id] ? (
                        <>
                          <img
                            src={repairedImageUrls[image.id]}
                            alt="修复后"
                            className="max-w-full max-h-full object-contain"
                          />
                          {/* 下载覆盖层 */}
                          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center opacity-0 group-hover/hover:opacity-100 transition-opacity duration-200 rounded-lg">
                            <button
                              onClick={() => {
                                const link = document.createElement("a");
                                link.href = repairedImageUrls[image.id];
                                link.setAttribute("download", image.name ? `repaired_${image.name}` : "repaired_image.png");
                                document.body.appendChild(link);
                                link.click();
                                link.parentNode.removeChild(link);
                              }}
                              className="flex flex-col items-center text-white hover:text-blue-200 focus:outline-none bg-blue-600 px-3 py-2 rounded-lg"
                            >
                              <FaDownload className="text-lg mb-1" />
                              <span className="text-xs">下载</span>
                            </button>
                          </div>
                        </>
                      ) : (
                        <div className="flex flex-col items-center justify-center text-gray-400">
                          <FaSpinner className="text-2xl animate-spin mb-2" />
                          <span className="text-xs">准备修复...</span>
                        </div>
                      )}
                    </div>
                    <span className={`mt-2 px-2 py-1 text-xs rounded-md font-medium ${
                      repairStatus[image.id] === 'processing' ? 'bg-blue-100 text-blue-700' :
                      repairStatus[image.id] === 'failed' ? 'bg-red-100 text-red-700' :
                      repairStatus[image.id] === 'completed' ? 'bg-green-100 text-green-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {repairStatus[image.id] === 'processing' ? '修复中' :
                       repairStatus[image.id] === 'failed' ? '修复失败' :
                       repairStatus[image.id] === 'completed' ? '修复完成' :
                       '等待修复'}
                    </span>
                  </div>
                </div>

                {/* 文件信息 */}
                <div className="mt-4 pt-3 border-t border-gray-100 flex justify-between items-center text-sm">
                  <span className="text-gray-700 font-medium truncate">{image.name}</span>
                  {image.size && (
                    <span className="text-gray-500 text-xs ml-2">
                      {(image.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                  )}
                </div>

                {/* 删除按钮 */}
                <button
                  onClick={() => removeImage(image.id)}
                  className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center shadow-lg border-2 border-white"
                >
                  <FaTimes className="text-xs" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 预览图片弹窗（对比） */}
      {previewImageUrl && previewOriginImage && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-70">
          <div className="relative bg-white rounded-lg shadow-lg p-4 max-w-full max-h-full flex flex-col items-center">
            <button
              className="absolute top-2 right-2 text-gray-600 hover:text-red-500 text-2xl"
              onClick={() => {
                window.URL.revokeObjectURL(previewImageUrl);
                setPreviewImageUrl(null);
                setPreviewOriginImage(null);
              }}
            >
              <FaTimes />
            </button>
            <div className="flex flex-row gap-6 items-center justify-center max-w-[80vw] max-h-[80vh]">
              {/* 原图 */}
              <div className="flex flex-col items-center">
                <img
                  src={previewOriginImage.preview || previewOriginImage.url}
                  alt="原图"
                  className="max-w-[35vw] max-h-[70vh] object-contain rounded border"
                />
                <span className="mt-2 text-sm text-gray-600">原图</span>
              </div>
              {/* 修复后图（带下载层） */}
              <div className="flex flex-col items-center group relative">
                {repairStatus[previewOriginImage.id] === 'processing' ? (
                  <div className="max-w-[35vw] max-h-[70vh] flex items-center justify-center rounded border bg-gray-50 p-8">
                    <div className="flex flex-col items-center text-blue-600">
                      <FaSpinner className="text-4xl animate-spin mb-4" />
                      <span className="text-lg">修复中...</span>
                    </div>
                  </div>
                                 ) : repairStatus[previewOriginImage.id] === 'failed' ? (
                   <div className="max-w-[35vw] max-h-[70vh] flex items-center justify-center rounded border bg-gray-50 p-8 cursor-pointer" onClick={() => startPolling(previewOriginImage.id)}>
                     <div className="flex flex-col items-center text-red-600">
                       <FaTimes className="text-4xl mb-4" />
                       <span className="text-lg">修复失败</span>
                       <span className="text-sm text-blue-600 underline mt-2">点击重试</span>
                     </div>
                   </div>
                ) : repairStatus[previewOriginImage.id] === 'completed' && previewImageUrl ? (
                  <>
                    <img
                      src={previewImageUrl}
                      alt="修复后图片预览"
                      className="max-w-[35vw] max-h-[70vh] object-contain rounded border"
                    />
                    {/* 下载覆盖层 */}
                    <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center opacity-0 group-hover:opacity-80 transition-opacity duration-200 z-10">
                      <button
                        onClick={() => {
                          // 直接下载修复后图片
                          const link = document.createElement("a");
                          link.href = previewImageUrl;
                          link.setAttribute("download", previewOriginImage.name ? `repaired_${previewOriginImage.name}` : "repaired_image.png");
                          document.body.appendChild(link);
                          link.click();
                          link.parentNode.removeChild(link);
                        }}
                        className="flex flex-col items-center text-white hover:text-blue-200 focus:outline-none"
                      >
                        <FaDownload className="text-2xl mb-1" />
                        <span className="text-xs">下载</span>
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="max-w-[35vw] max-h-[70vh] flex items-center justify-center rounded border bg-gray-50 p-8">
                    <div className="flex flex-col items-center text-gray-400">
                      <FaSpinner className="text-4xl animate-spin mb-4" />
                      <span className="text-lg">准备修复...</span>
                    </div>
                  </div>
                )}
                <span className="mt-2 text-sm text-gray-600">
                  {repairStatus[previewOriginImage.id] === 'processing' ? '修复中' :
                   repairStatus[previewOriginImage.id] === 'failed' ? '修复失败' :
                   repairStatus[previewOriginImage.id] === 'completed' ? '修复完成' :
                   '等待修复'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
