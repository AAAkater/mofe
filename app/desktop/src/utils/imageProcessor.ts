export interface ProcessedImage {
  originalFile: File
  processedDataUrl: string
  timestamp: number
  id: string
}

export const processImage = async (file: File): Promise<string> => {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    const img = new Image()

    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height

      // 绘制原图
      ctx!.drawImage(img, 0, 0)

      // 获取图像数据
      const imageData = ctx!.getImageData(0, 0, canvas.width, canvas.height)
      const pixels = imageData.data

      // 转换为黑白
      for (let i = 0; i < pixels.length; i += 4) {
        const gray =
          pixels[i] * 0.299 + pixels[i + 1] * 0.587 + pixels[i + 2] * 0.114
        pixels[i] = gray // red
        pixels[i + 1] = gray // green
        pixels[i + 2] = gray // blue
      }

      // 放回画布
      ctx!.putImageData(imageData, 0, 0)

      // 返回处理后的数据URL
      const processedDataUrl = canvas.toDataURL('image/png')
      resolve(processedDataUrl)
    }

    img.src = URL.createObjectURL(file)
  })
}

export const generateId = (): string => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2)
}
