import React, { useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  generateId,
  ProcessedImage,
  processImage,
} from '../utils/imageProcessor'
import './HomePage.css'

interface HomePageProps {
  history: ProcessedImage[]
  onAddToHistory: (item: ProcessedImage) => void
}

export const HomePage: React.FC<HomePageProps> = ({
  history,
  onAddToHistory,
}) => {
  const [isProcessing, setIsProcessing] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  const handleFileSelect = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]
    if (!file || !file.type.startsWith('image/')) {
      alert('请选择有效的图像文件')
      return
    }

    setIsProcessing(true)
    try {
      const processedDataUrl = await processImage(file)
      const processedItem: ProcessedImage = {
        originalFile: file,
        processedDataUrl,
        timestamp: Date.now(),
        id: generateId(),
      }

      onAddToHistory(processedItem)
      navigate(`/preview/${processedItem.id}`)
    } catch (error) {
      console.error('处理图像时出错:', error)
      alert('处理图像时出错，请重试')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleHistoryItemClick = (item: ProcessedImage) => {
    navigate(`/preview/${item.id}`)
  }

  return (
    <div className="home-page">
      <div className="main-content">
        <h1>图像修复工具</h1>
        <p className="subtitle">选择图像进行黑白化处理</p>

        <div className="upload-section">
          <button
            className="upload-btn"
            onClick={handleFileSelect}
            disabled={isProcessing}
          >
            {isProcessing ? '处理中...' : '选择图像'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>
      </div>

      {history.length > 0 && (
        <div className="history-section">
          <h2>处理历史</h2>
          <div className="history-grid">
            {history.map((item) => (
              <div
                key={item.id}
                className="history-item"
                onClick={() => handleHistoryItemClick(item)}
              >
                <img src={item.processedDataUrl} alt="处理后的图像" />
                <div className="history-item-info">
                  <p className="filename">{item.originalFile.name}</p>
                  <p className="timestamp">
                    {new Date(item.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
