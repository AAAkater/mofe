import { Menu, Drawer, Button } from "@arco-design/web-react";
import { IconMenu, IconDown } from "@arco-design/web-react/icon";
import { useState, useEffect } from "react";
import { Outlet } from "react-router-dom";
import HistoryModal from "../components/HistoryModal";

const MenuItem = Menu.Item;
const SubMenu = Menu.SubMenu;

const Layout = () => {
  const [visible, setVisible] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [historyModalOpen, setHistoryModalOpen] = useState(false);

  const menuItems = [
    {
      key: "history",
      label: "历史修复",
    },
  ];

  // 检测设备是否为移动端
  useEffect(() => {
    const checkIsMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    checkIsMobile();
    window.addEventListener("resize", checkIsMobile);

    return () => {
      window.removeEventListener("resize", checkIsMobile);
    };
  }, []);

  const handleMenuClick = (key) => {
    if (key === "history") {
      setHistoryModalOpen(true);
      setVisible(false); // 关闭移动端菜单
    }
  };

  // 移动端菜单内容
  const mobileMenuContent = (
    <div className="py-6">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-blue-600">导航菜单</h2>
      </div>
      <Menu mode="vertical" style={{ border: "none", width: "100%" }}>
        {menuItems.map((item) => (
          <MenuItem
            key={item.key}
            className="text-lg py-3"
            onClick={() => handleMenuClick(item.key)}
          >
            {item.label}
          </MenuItem>
        ))}
      </Menu>
    </div>
  );

  return (
    <div className="w-full max-w-screen bg-white overflow-x-hidden">
      {/* 导航栏 */}
      <div className="w-full bg-white border-b border-gray-100 fixed shadow-lg">
        {isMobile ? (
          <div className="flex items-center justify-between px-4 py-4">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">Mofe</span>
              </div>
              <span className="text-xl font-bold text-gray-800">Mofe</span>
            </div>

            <Button
              icon={<IconMenu />}
              shape="circle"
              type="text"
              size="large"
              onClick={() => setVisible(true)}
              className="text-gray-700"
            />

            <Drawer
              width="100%"
              height="100%"
              visible={visible}
              placement="top"
              onCancel={() => setVisible(false)}
              footer={null}
              headerStyle={{ display: "none" }}
            >
              {mobileMenuContent}
            </Drawer>
          </div>
        ) : (
          <div className="flex items-center justify-between px-6 py-4 ">
            <div className="flex items-center space-x-8">
              <div className="flex items-center space-x-2">
                <div className="scale-150">✨</div>
                <span className="text-2xl font-bold text-gray-800">Mofe</span>
              </div>

              {/* 导航菜单 */}
              <div className="flex items-center space-x-6">
                {menuItems.map((item) => (
                  <div key={item.key} className="relative group">
                    <button
                      onClick={() => handleMenuClick(item.key)}
                      className="flex items-center space-x-1 text-gray-600 hover:text-blue-600 font-medium transition-colors duration-200"
                    >
                      <span>{item.label}</span>
                      {item.hasDrcopdown && (
                        <IconDown className="w-4 h-4 text-gray-400 group-hover:text-blue-600" />
                      )}
                    </button>
                    {/* 下拉菜单占位 */}
                    {item.hasDropdown && (
                      <div className="absolute top-full left-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-100 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                        <div className="py-2">
                          <div className="px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 cursor-pointer">
                            功能开发中...
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* 右侧：登录注册按钮 */}
            {/* <div className="flex items-center space-x-4">
              <button className="text-gray-600 hover:text-blue-600 font-medium transition-colors duration-200">
                登录
              </button>
              <span className="text-gray-300">/</span>
              <button className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-medium px-6 py-2 rounded-full transition-all duration-200 shadow-md hover:shadow-lg">
                注册
              </button>
            </div> */}
          </div>
        )}
      </div>

      {/* 主要内容区域 */}
      <div className="w-full h-full overflow-auto pt-10 mx-auto">
        <Outlet />
      </div>

      {/* 历史修复Modal */}
      <HistoryModal
        isOpen={historyModalOpen}
        onClose={() => setHistoryModalOpen(false)}
      />
    </div>
  );
};

export default Layout;
