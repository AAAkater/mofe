export interface ProcessedImage {
  originalFile: File
  processedDataUrl: string
  timestamp: number
  id: string
}

const baseURL = 'http://127.0.0.1:8000' // API base URL

export const processImage = async (file: File): Promise<string> => {
  const formData = new FormData()
  formData.append('file', file)

  const uploadResponse = await fetch(`${baseURL}/upload/image`, {
    method: 'POST',
    body: formData,
  })

  if (!uploadResponse.ok) {
    const errorText = await uploadResponse.text()
    throw new Error(
      `Image upload failed: ${uploadResponse.status} ${errorText}`
    )
  }

  const uploadResult = await uploadResponse.json()
  const fileId = uploadResult?.data?.id

  if (!fileId) {
    throw new Error('Failed to get file ID from upload response')
  }

  // eslint-disable-next-line no-constant-condition
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 2000))

    try {
      const downloadResponse = await fetch(
        `${baseURL}/download/image?file_id=${fileId}`
      )

      if (downloadResponse.ok) {
        const imageBlob = await downloadResponse.blob()
        if (imageBlob.type.startsWith('image/')) {
          return URL.createObjectURL(imageBlob)
        }
        // If the response is not an image, it might be an error encoded in JSON.
        // We will continue polling.
      } else if (downloadResponse.status !== 404) {
        const errorText = await downloadResponse.text()
        throw new Error(
          `Failed to download processed image. Status: ${downloadResponse.status}. Body: ${errorText}`
        )
      }
      // If status is 404, we continue polling.
    } catch (e) {
      console.error('Polling failed, retrying...', e)
    }
  }
}

export const generateId = (): string => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2)
}
