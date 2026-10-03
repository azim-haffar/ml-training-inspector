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

const DEFAULT_CONFIG = { epochs: 2, batch_size: 32, learning_rate: 0.001, model: 'simple_cnn', dataset: 'synthetic', seed: 42 }

const MODEL_LABELS = {
  simple_cnn: 'Simple CNN',
  resnet9:    'ResNet-9',
}

export default function App() {
  const [config, setConfig] = useState(DEFAULT_CONFIG)
  const [theme, setTheme]   = useState(() => localStorage.getItem('theme') || 'dark')

  const {
    isConnected, status, currentModel, currentDataset, error, streamNotice,
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

  const isTraining = ['starting', 'training', 'stopping'].includes(status)
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
          <span className="subtitle">{(currentDataset || config.dataset) === 'synthetic' ? 'Synthetic CPU demo' : 'CIFAR-10'} · {modelLabel}</span>
        </div>
        <div className="header-right">
          <span className={`dot ${isConnected ? 'dot-green' : 'dot-red'}`} />
          <span role="status" className="connection-label">{isConnected ? 'Connected' : 'Connecting…'}</span>
          {status !== 'idle' && (
            <span className={`status-badge status-${status}`}>{status}</span>
          )}
          <button aria-label="Toggle light or dark theme" className="theme-toggle" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? '☀' : '☾'}
          </button>
        </div>
      </header>

      <div className="controls-card">
        <div className="controls-fields">
          <label className="field"><span>Dataset</span>
            <select value={config.dataset} disabled={isTraining} onChange={e => setConfig({ ...config, dataset: e.target.value })}>
              <option value="synthetic">Synthetic demo</option><option value="cifar10">CIFAR-10</option>
            </select>
          </label>
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

      {error && <div className="error-notice" role="alert">{error}</div>}
      {streamNotice && <p role="status">{streamNotice}</p>}
      {status === 'starting' && <p role="status">Preparing data and model… CIFAR-10 may need an initial download.</p>}
      <p className="metric-note">CPU · Seed {config.seed} · One shared run per server. Synthetic results demonstrate the pipeline, not real-world model quality. Signals use fixed heuristic thresholds.</p>
      {isTraining && <ProgressBar progress={progress} />}

      {lastCheckpoint && (
        <div className="checkpoint-notice">
          Model snapshot saved (resume unavailable) → <code>{lastCheckpoint}</code>
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
          <h2>Sampled Batch Loss</h2>
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
          <h2>Learning Rate Used</h2>
          <LRChart lrHistory={lrHistory} />
        </div>
      </div>

      {!hasData && !isTraining && (
        <div className="empty-hint">
          Configure the run above and click <strong>Start Training</strong> to begin.
          <br />
          <span>Synthetic demo runs without downloading data. CIFAR-10 downloads about 170 MB on first use.</span>
        </div>
      )}

      {epochData.length > 0 && <details className="metric-table"><summary>Received epoch metrics (accessible table)</summary>
        <table className="runs-table"><caption>Same server values used by loss and accuracy charts</caption>
          <thead><tr><th>Epoch</th><th>Train loss</th><th>Validation loss</th><th>Train accuracy %</th><th>Validation accuracy %</th></tr></thead>
          <tbody>{epochData.map(e => <tr key={e.epoch}><td>{e.epoch}</td><td>{e.train_loss}</td><td>{e.val_loss}</td><td>{e.train_acc}</td><td>{e.val_acc}</td></tr>)}</tbody>
        </table></details>}
      <PreviousRuns />
    </div>
  )
}
