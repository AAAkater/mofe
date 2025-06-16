// 图片上传服务
export const uploadImage = async (file) => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("上传失败");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("图片上传出错:", error);
    throw error;
  }
};

// URL图片上传
export const uploadImageByUrl = async (imageUrl) => {
  try {
    const response = await fetch("/api/upload/url", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url: imageUrl }),
    });

    if (!response.ok) {
      throw new Error("URL图片上传失败");
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("URL图片上传出错:", error);
    throw error;
  }
};
