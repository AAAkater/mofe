import { useState } from 'react'
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom'
import './App.css'
import { HomePage } from './components/HomePage'
import { PreviewPage } from './components/PreviewPage'
import { ProcessedImage } from './utils/imageProcessor'

function App() {
  const [history, setHistory] = useState<ProcessedImage[]>([])

  const handleAddToHistory = (item: ProcessedImage) => {
    setHistory((prev) => [item, ...prev])
  }

  return (
    <Router>
      <div className="app">
        <Routes>
          <Route
            path="/"
            element={
              <HomePage history={history} onAddToHistory={handleAddToHistory} />
            }
          />
          <Route
            path="/preview/:id"
            element={<PreviewPage history={history} />}
          />
        </Routes>
      </div>
    </Router>
  )
}

export default App
