import { useState, useEffect } from 'react'
import useTrainingSocket from './hooks/useTrainingSocket'
import LossChart from './components/LossChart'
import BatchLossChart from './components/BatchLossChart'
import AccuracyChart from './components/AccuracyChart'
import GradientChart from './components/GradientChart'
import ClassAccuracyChart from './components/ClassAccuracyChart'
import LRChart from './components/LRChart'
import AnomalyPanel from './components/AnomalyPanel'
import ProgressBar from './components/ProgressBar'
import SummaryCard from './components/SummaryCard'
import PreviousRuns from './components/PreviousRuns'
import './App.css'

const DEFAULT_CONFIG = { epochs: 10, batch_size: 64, learning_rate: 0.001, model: 'simple_cnn' }

const MODEL_LABELS = {
  simple_cnn: 'Simple CNN',
  resnet9:    'ResNet-9',
}

export default function App() {
  const [config, setConfig] = useState(DEFAULT_CONFIG)
  const [theme, setTheme]   = useState(() => localStorage.getItem('theme') || 'dark')

  const {
    isConnected, status, currentModel,
    epochData, batchData, gradNorms, anomalies,
    classAccuracies, lrHistory,
    progress, lastCheckpoint, duration,
    startTraining, stopTraining,
  } = useTrainingSocket()

  // Persist theme and sync body attribute so background follows the toggle
  useEffect(() => {
    localStorage.setItem('theme', theme)
    document.body.setAttribute('data-theme', theme)
  }, [theme])

  const isTraining = status === 'training'
  const isDone     = status === 'done' || status === 'stopped'
  const hasData    = epochData.length > 0 || batchData.length > 0
  const modelLabel = currentModel ? (MODEL_LABELS[currentModel] || currentModel) : MODEL_LABELS[config.model]

  const handleExport = () => {
    const payload = { epochData, batchData, classAccuracies, lrHistory, anomalies, config, duration }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href     = url
    a.download = `training-run-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className={`app ${theme}`}>
      <header className="app-header">
        <div className="header-left">
          <h1>ML Training Inspector</h1>
          <span className="subtitle">CIFAR-10 · {modelLabel}</span>
        </div>
        <div className="header-right">
          <span className={`dot ${isConnected ? 'dot-green' : 'dot-red'}`} />
          <span className="connection-label">{isConnected ? 'Connected' : 'Connecting…'}</span>
          {status !== 'idle' && (
            <span className={`status-badge status-${status}`}>{status}</span>
          )}
          <button className="theme-toggle" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? '☀' : '☾'}
          </button>
        </div>
      </header>

      <div className="controls-card">
        <div className="controls-fields">
          <label className="field">
            <span>Model</span>
            <select
              value={config.model}
              onChange={(e) => setConfig({ ...config, model: e.target.value })}
              disabled={isTraining}
            >
              <option value="simple_cnn">Simple CNN</option>
              <option value="resnet9">ResNet-9</option>
            </select>
          </label>
          <label className="field">
            <span>Epochs</span>
            <input
              type="number" min="1" max="50"
              value={config.epochs}
              onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 10 })}
              disabled={isTraining}
            />
          </label>
          <label className="field">
            <span>Batch Size</span>
            <input
              type="number" min="16" max="512" step="16"
              value={config.batch_size}
              onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) || 64 })}
              disabled={isTraining}
            />
          </label>
          <label className="field">
            <span>Learning Rate</span>
            <input
              type="number" min="0.00001" max="1" step="0.0001"
              value={config.learning_rate}
              onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) || 0.001 })}
              disabled={isTraining}
            />
          </label>
        </div>
        <div className="controls-buttons">
          <button
            className="start-btn"
            onClick={() => startTraining(config)}
            disabled={isTraining || !isConnected}
          >
            {isTraining ? 'Training…' : 'Start Training'}
          </button>
          {isTraining && (
            <button className="stop-btn" onClick={stopTraining}>Stop</button>
          )}
        </div>
      </div>

      {isTraining && <ProgressBar progress={progress} />}

      {lastCheckpoint && (
        <div className="checkpoint-notice">
          Checkpoint saved → <code>{lastCheckpoint}</code>
        </div>
      )}

      {anomalies.length > 0 && <AnomalyPanel anomalies={anomalies} />}

      {isDone && epochData.length > 0 && (
        <SummaryCard
          epochData={epochData}
          anomalies={anomalies}
          duration={duration}
          onExport={handleExport}
        />
      )}

      <div className="charts-grid">
        <div className="chart-card">
          <h2>Epoch Loss</h2>
          <LossChart epochData={epochData} />
        </div>
        <div className="chart-card">
          <h2>Accuracy</h2>
          <AccuracyChart epochData={epochData} />
        </div>
        <div className="chart-card">
          <h2>Live Batch Loss</h2>
          <BatchLossChart batchData={batchData} />
        </div>
        <div className="chart-card">
          <h2>Gradient Norms</h2>
          <GradientChart gradNorms={gradNorms} />
        </div>
      </div>

      <div className="charts-row-2">
        <div className="chart-card chart-wide">
          <h2>Class Accuracy</h2>
          <ClassAccuracyChart classAccuracies={classAccuracies} />
        </div>
        <div className="chart-card">
          <h2>LR Schedule</h2>
          <LRChart lrHistory={lrHistory} />
        </div>
      </div>

      {!hasData && !isTraining && (
        <div className="empty-hint">
          Configure the run above and click <strong>Start Training</strong> to begin.
          <br />
          <span>First run downloads CIFAR-10 (~170 MB) — takes a minute.</span>
        </div>
      )}

      <PreviousRuns />
    </div>
  )
}
