export default function ProgressBar({ progress }) {
  const { epoch, batch, totalBatches, totalEpochs } = progress

  if (!totalEpochs) return null

  const epochPct = totalEpochs ? Math.round((epoch / totalEpochs) * 100) : 0
  const batchPct = totalBatches && batch ? Math.round((batch / totalBatches) * 100) : 0

  return (
    <div className="progress-section">
      <div className="progress-row">
        <span className="progress-label">Epoch {epoch}/{totalEpochs}</span>
        <div className="progress-track">
          <div className="progress-fill epoch-fill" style={{ width: `${epochPct}%` }} />
        </div>
        <span className="progress-pct">{epochPct}%</span>
      </div>
      <div className="progress-row">
        <span className="progress-label">Batch {batch}/{totalBatches}</span>
        <div className="progress-track">
          <div className="progress-fill batch-fill" style={{ width: `${batchPct}%` }} />
        </div>
        <span className="progress-pct">{batchPct}%</span>
      </div>
    </div>
  )
}
