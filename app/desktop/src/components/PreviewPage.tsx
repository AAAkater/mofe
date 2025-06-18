import { save } from '@tauri-apps/plugin-dialog'
import { writeFile } from '@tauri-apps/plugin-fs'
import React, { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ProcessedImage } from '../utils/imageProcessor'
import './PreviewPage.css'

interface PreviewPageProps {
  history: ProcessedImage[]
}

export const PreviewPage: React.FC<PreviewPageProps> = ({ history }) => {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [currentImage, setCurrentImage] = useState<ProcessedImage | null>(null)
  const [comparisonMode, setComparisonMode] = useState<'button' | 'slider'>(
    'button'
  )
  const [showOriginal, setShowOriginal] = useState(false)
  const [sliderPosition, setSliderPosition] = useState(50)
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const [originalImageUrl, setOriginalImageUrl] = useState<string>('')

  useEffect(() => {
    const image = history.find((item) => item.id === id)
    if (image) {
      setCurrentImage(image)
      // 创建原图的URL
      const url = URL.createObjectURL(image.originalFile)
      setOriginalImageUrl(url)

      return () => {
        URL.revokeObjectURL(url)
      }
    }
  }, [id, history])

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (comparisonMode !== 'slider') return
    setIsDragging(true)
    updateSliderPosition(e)
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging || comparisonMode !== 'slider') return
    e.preventDefault()
    updateSliderPosition(e)
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const updateSliderPosition = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setSliderPosition(percentage)
  }

  const handleSave = async () => {
    if (!currentImage) return

    try {
      const fileName = `processed_${
        currentImage.originalFile.name.split('.')[0]
      }.png`
      const filePath = await save({
        filters: [
          {
            name: 'Image',
            extensions: ['png'],
          },
        ],
        defaultPath: fileName,
      })

      if (filePath) {
        // 将base64数据转换为Uint8Array
        const base64Data = currentImage.processedDataUrl.split(',')[1]
        const binaryString = atob(base64Data)
        const bytes = new Uint8Array(binaryString.length)
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i)
        }

        await writeFile(filePath, bytes)
        alert('图像保存成功！')
      }
    } catch (error) {
      console.error('保存图像时出错:', error)
      alert('保存图像时出错，请重试')
    }
  }

  if (!currentImage) {
    return (
      <div className="preview-page">
        <div className="error-message">
          <h2>图像未找到</h2>
          <button onClick={() => navigate('/')}>返回首页</button>
        </div>
      </div>
    )
  }

  return (
    <div className="preview-page">
      <div className="preview-header">
        <button className="back-btn" onClick={() => navigate('/')}>
          ← 返回首页
        </button>
        <h1>{currentImage.originalFile.name}</h1>
        <div className="preview-controls">
          <div className="mode-toggle">
            <button
              className={comparisonMode === 'button' ? 'active' : ''}
              onClick={() => setComparisonMode('button')}
            >
              按钮对比
            </button>
            <button
              className={comparisonMode === 'slider' ? 'active' : ''}
              onClick={() => setComparisonMode('slider')}
            >
              滑动对比
            </button>
          </div>
          <button className="save-btn" onClick={handleSave}>
            保存图像
          </button>
        </div>
      </div>

      <div className="preview-content">
        {comparisonMode === 'button' ? (
          <div className="button-comparison">
            <div className="image-container">
              <img
                src={
                  showOriginal
                    ? originalImageUrl
                    : currentImage.processedDataUrl
                }
                alt={showOriginal ? '原图' : '处理后'}
                draggable={false}
              />
              <div className="image-label">
                {showOriginal ? '原图' : '处理后'}
              </div>
            </div>
            <button
              className="compare-btn"
              onMouseDown={() => setShowOriginal(true)}
              onMouseUp={() => setShowOriginal(false)}
              onMouseLeave={() => setShowOriginal(false)}
            >
              按住查看原图
            </button>
          </div>
        ) : (
          <div
            className="slider-comparison"
            ref={containerRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            <div className="comparison-container">
              <img
                className="original-image"
                src={originalImageUrl}
                alt="原图"
                draggable={false}
              />
              <div
                className="processed-overlay"
                style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
              >
                <img
                  src={currentImage.processedDataUrl}
                  alt="处理后"
                  draggable={false}
                />
              </div>
              <div
                className="slider-line"
                style={{ left: `${sliderPosition}%` }}
              >
                <div className="slider-handle"></div>
              </div>
              <div className="image-labels">
                <span className="label-original">原图</span>
                <span className="label-processed">处理后</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
