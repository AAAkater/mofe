import { useState } from "react";
import {
  FaTimes,
  FaSearch,
  FaDownload,
  FaEye,
  FaTrash,
  FaCalendarAlt,
  FaImage,
} from "react-icons/fa";

const HistoryModal = ({ isOpen, onClose }) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");
  const [sortBy, setSortBy] = useState("newest");

  // 模拟历史数据
  const historyData = [
    {
      id: 1,
      fileName: "家庭合照.jpg",
      originalSize: "2.3 MB",
      processedSize: "2.8 MB",
      status: "completed",
      date: "2024-01-15 14:30",
      quality: "高清修复",
      originalUrl:
        "https://images.unsplash.com/photo-1549298916-b41d501d3772?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=20",
      processedUrl:
        "https://images.unsplash.com/photo-1549298916-b41d501d3772?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
    },
    {
      id: 2,
      fileName: "毕业照片.png",
      originalSize: "1.8 MB",
      processedSize: "2.1 MB",
      status: "completed",
      date: "2024-01-14 09:15",
      quality: "标准修复",
      originalUrl:
        "https://images.unsplash.com/photo-1606787366850-de6330128bfc?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=20",
      processedUrl:
        "https://images.unsplash.com/photo-1606787366850-de6330128bfc?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
    },
    {
      id: 3,
      fileName: "风景照.jpg",
      originalSize: "3.1 MB",
      processedSize: "3.5 MB",
      status: "processing",
      date: "2024-01-14 16:45",
      quality: "高清修复",
      originalUrl:
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=20",
      processedUrl: null,
    },
    {
      id: 4,
      fileName: "老照片修复.jpg",
      originalSize: "1.2 MB",
      processedSize: "1.6 MB",
      status: "completed",
      date: "2024-01-13 11:20",
      quality: "专业修复",
      originalUrl:
        "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=20",
      processedUrl:
        "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=80",
    },
    {
      id: 5,
      fileName: "产品图片.png",
      originalSize: "2.7 MB",
      processedSize: "3.2 MB",
      status: "failed",
      date: "2024-01-12 15:30",
      quality: "高清修复",
      originalUrl:
        "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&q=20",
      processedUrl: null,
    },
  ];

  // 过滤和排序逻辑
  const filteredHistory = historyData
    .filter((item) => {
      const matchesSearch = item.fileName
        .toLowerCase()
        .includes(searchTerm.toLowerCase());
      const matchesFilter =
        filterStatus === "all" || item.status === filterStatus;
      return matchesSearch && matchesFilter;
    })
    .sort((a, b) => {
      if (sortBy === "newest") {
        return new Date(b.date) - new Date(a.date);
      } else if (sortBy === "oldest") {
        return new Date(a.date) - new Date(b.date);
      } else if (sortBy === "name") {
        return a.fileName.localeCompare(b.fileName);
      }
      return 0;
    });

  const getStatusBadge = (status) => {
    const statusConfig = {
      completed: { color: "bg-green-100 text-green-800", text: "已完成" },
      processing: { color: "bg-blue-100 text-blue-800", text: "处理中" },
      failed: { color: "bg-red-100 text-red-800", text: "失败" },
    };

    const config = statusConfig[status] || statusConfig.completed;
    return (
      <span
        className={`px-2 py-1 rounded-full text-xs font-medium ${config.color}`}
      >
        {config.text}
      </span>
    );
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col">
        {/* 头部 */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
              <FaImage className="text-white text-lg" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-gray-800">历史修复</h2>
              <p className="text-sm text-gray-600">查看您的所有修复记录</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-gray-100 transition-colors duration-200"
          >
            <FaTimes className="text-gray-500 text-lg" />
          </button>
        </div>

        {/* 工具栏 */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between space-y-4 md:space-y-0">
            {/* 搜索框 */}
            <div className="relative flex-1 max-w-md">
              <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="搜索文件名..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200"
              />
            </div>

            {/* 过滤器 */}
            <div className="flex items-center space-x-4">
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
              >
                <option value="all">所有状态</option>
                <option value="completed">已完成</option>
                <option value="processing">处理中</option>
                <option value="failed">失败</option>
              </select>

              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
              >
                <option value="newest">最新优先</option>
                <option value="oldest">最旧优先</option>
                <option value="name">文件名排序</option>
              </select>
            </div>
          </div>
        </div>

        {/* 列表内容 */}
        <div className="flex-1 overflow-y-auto p-6">
          {filteredHistory.length === 0 ? (
            <div className="text-center py-20">
              <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <FaImage className="text-gray-400 text-2xl" />
              </div>
              <h3 className="text-lg font-medium text-gray-800 mb-2">
                没有找到相关记录
              </h3>
              <p className="text-gray-600">尝试调整搜索条件或过滤器</p>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredHistory.map((item) => (
                <div
                  key={item.id}
                  className="bg-white border border-gray-200 rounded-xl p-4 hover:shadow-md transition-shadow duration-200"
                >
                  <div className="flex items-center space-x-4">
                    {/* 预览图 */}
                    <div className="flex-shrink-0">
                      <div className="relative">
                        <img
                          src={item.originalUrl}
                          alt={item.fileName}
                          className="w-16 h-16 object-cover rounded-lg"
                        />
                        {item.status === "processing" && (
                          <div className="absolute inset-0 bg-blue-500 bg-opacity-20 rounded-lg flex items-center justify-center">
                            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* 文件信息 */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="text-lg font-medium text-gray-800 truncate">
                          {item.fileName}
                        </h4>
                        {getStatusBadge(item.status)}
                      </div>
                      <div className="flex items-center space-x-4 text-sm text-gray-600">
                        <div className="flex items-center space-x-1">
                          <FaCalendarAlt className="text-gray-400" />
                          <span>{item.date}</span>
                        </div>
                        <div>质量: {item.quality}</div>
                        <div>原始: {item.originalSize}</div>
                        {item.processedSize && (
                          <div>修复后: {item.processedSize}</div>
                        )}
                      </div>
                    </div>

                    {/* 操作按钮 */}
                    <div className="flex items-center space-x-2">
                      {item.status === "completed" && (
                        <>
                          <button
                            className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors duration-200"
                            title="预览"
                          >
                            <FaEye className="text-lg" />
                          </button>
                          <button
                            className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors duration-200"
                            title="下载"
                          >
                            <FaDownload className="text-lg" />
                          </button>
                        </>
                      )}
                      <button
                        className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors duration-200"
                        title="删除"
                      >
                        <FaTrash className="text-lg" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 底部统计 */}
        <div className="p-6 border-t border-gray-200 bg-gray-50">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <div>共 {filteredHistory.length} 条记录</div>
            <div className="flex items-center space-x-4">
              <span>
                已完成:{" "}
                {
                  filteredHistory.filter((item) => item.status === "completed")
                    .length
                }
              </span>
              <span>
                处理中:{" "}
                {
                  filteredHistory.filter((item) => item.status === "processing")
                    .length
                }
              </span>
              <span>
                失败:{" "}
                {
                  filteredHistory.filter((item) => item.status === "failed")
                    .length
                }
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HistoryModal;
