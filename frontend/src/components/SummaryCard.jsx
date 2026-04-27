function valAccColor(acc) {
  if (acc == null) return '#8892a4'
  if (acc >= 65) return '#22c55e'
  if (acc >= 50) return '#f59e0b'
  return '#ef4444'
}

function formatDuration(seconds) {
  if (seconds == null) return '—'
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

export default function SummaryCard({ epochData, anomalies, duration, onExport }) {
  if (epochData.length === 0) return null

  const last = epochData[epochData.length - 1]
  const bestEpoch = epochData.reduce((best, e) =>
    e.val_acc > best.val_acc ? e : best, epochData[0])

  const valAcc = last.val_acc
  const color  = valAccColor(valAcc)

  return (
    <div className="summary-card">
      <div className="summary-header">
        <h2>Run Summary</h2>
        <button className="export-btn" onClick={onExport}>Export JSON</button>
      </div>
      <div className="summary-stats">
        <div className="stat">
          <span className="stat-label">Train Acc</span>
          <span className="stat-value">{last.train_acc.toFixed(1)}%</span>
        </div>
        <div className="stat">
          <span className="stat-label">Val Acc</span>
          <span className="stat-value" style={{ color }}>{valAcc.toFixed(1)}%</span>
        </div>
        <div className="stat">
          <span className="stat-label">Best Epoch</span>
          <span className="stat-value">{bestEpoch.epoch} <span className="stat-sub">({bestEpoch.val_acc.toFixed(1)}%)</span></span>
        </div>
        <div className="stat">
          <span className="stat-label">Anomalies</span>
          <span className="stat-value">{anomalies.length}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Duration</span>
          <span className="stat-value">{formatDuration(duration)}</span>
        </div>
      </div>
    </div>
  )
}
