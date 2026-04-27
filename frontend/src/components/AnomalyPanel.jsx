import { useState } from 'react'

const SEVERITY_STYLES = {
  error:   { color: '#ef4444', bg: '#1f0d0d', border: '#7f1d1d' },
  warning: { color: '#f97316', bg: '#1a0f00', border: '#7c2d12' },
  info:    { color: '#60a5fa', bg: '#0d1520', border: '#1e3a5f' },
}

const TYPE_LABELS = {
  vanishing_gradient: 'Vanishing Gradient',
  exploding_gradient: 'Exploding Gradient',
  overfitting:        'Overfitting',
  loss_plateau:       'Loss Plateau',
  accuracy_plateau:   'Accuracy Plateau',
  high_loss:          'High Loss',
}

// For gradient anomalies the layer name is in the message — extract it so we
// can group by type+layer instead of producing one row per epoch.
function groupKey(anomaly) {
  if (anomaly.type === 'vanishing_gradient' || anomaly.type === 'exploding_gradient') {
    const match = anomaly.message.match(/in '([^']+)'/)
    const layer = match ? match[1] : 'unknown'
    return `${anomaly.type}::${layer}`
  }
  return anomaly.type
}

function epochRange(epochs) {
  const sorted = [...new Set(epochs)].sort((a, b) => a - b)
  if (sorted.length === 1) return `epoch ${sorted[0]}`
  return `epochs ${sorted[0]}–${sorted[sorted.length - 1]}`
}

// Collapse repeated anomalies into one row per type+layer
function buildGroups(anomalies) {
  const map = new Map()
  for (const a of anomalies) {
    const key = groupKey(a)
    if (!map.has(key)) {
      map.set(key, { key, type: a.type, severity: a.severity, message: a.message, epochs: [], ids: [] })
    }
    const g = map.get(key)
    g.epochs.push(a.epoch)
    g.ids.push(a.id)
  }
  return [...map.values()]
}

export default function AnomalyPanel({ anomalies }) {
  const [dismissedKeys, setDismissedKeys] = useState(new Set())

  const groups = buildGroups(anomalies).filter(g => !dismissedKeys.has(g.key))

  if (groups.length === 0) return null

  const dismiss = (key) => setDismissedKeys(prev => new Set([...prev, key]))
  const dismissAll = () => setDismissedKeys(new Set(buildGroups(anomalies).map(g => g.key)))

  return (
    <div className="anomaly-panel">
      <div className="anomaly-header">
        <h3>⚠ Anomalies Detected ({groups.length})</h3>
        <button className="dismiss-all-btn" onClick={dismissAll}>Dismiss all</button>
      </div>
      <div className="anomaly-list">
        {groups.map((g) => {
          const s = SEVERITY_STYLES[g.severity] || SEVERITY_STYLES.info
          return (
            <div
              key={g.key}
              className="anomaly-item"
              style={{ background: s.bg, borderLeft: `3px solid ${s.border}` }}
            >
              <span className="anomaly-badge" style={{ color: s.color }}>
                {TYPE_LABELS[g.type] || g.type}
              </span>
              <span className="anomaly-message">{g.message}</span>
              <span className="anomaly-epoch">{epochRange(g.epochs)}</span>
              {g.epochs.length > 1 && (
                <span className="anomaly-count" style={{ color: s.color }}>
                  ×{g.epochs.length}
                </span>
              )}
              <button
                className="anomaly-dismiss"
                onClick={() => dismiss(g.key)}
                aria-label="dismiss"
              >
                ×
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )
}
