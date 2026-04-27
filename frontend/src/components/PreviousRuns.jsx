import { useState, useEffect } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function formatDate(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
}

function valColor(acc) {
  if (acc == null) return '#8892a4'
  if (acc >= 65) return '#22c55e'
  if (acc >= 50) return '#f59e0b'
  return '#ef4444'
}

export default function PreviousRuns() {
  const [open, setOpen]       = useState(false)
  const [runs, setRuns]       = useState([])
  const [loading, setLoading] = useState(false)

  // Fetch once when the section is first opened
  useEffect(() => {
    if (!open || runs.length > 0) return
    setLoading(true)
    fetch(`${API_URL}/api/history`)
      .then(r => r.json())
      .then(data => setRuns((data.history || []).reverse())) // newest first
      .catch(() => setRuns([]))
      .finally(() => setLoading(false))
  }, [open])

  return (
    <div className="prev-runs">
      <button className="prev-runs-toggle" onClick={() => setOpen(o => !o)}>
        <span>Previous Runs</span>
        <span className="toggle-arrow">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="prev-runs-body">
          {loading && <p className="prev-runs-empty">Loading…</p>}
          {!loading && runs.length === 0 && (
            <p className="prev-runs-empty">No runs recorded yet.</p>
          )}
          {!loading && runs.length > 0 && (
            <table className="runs-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Model</th>
                  <th>Epochs</th>
                  <th>LR</th>
                  <th>Val Acc</th>
                  <th>Anomalies</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r, i) => (
                  <tr key={i}>
                    <td>{formatDate(r.timestamp)}</td>
                    <td>{r.model || 'simple_cnn'}</td>
                    <td>{r.epochs}</td>
                    <td>{r.lr}</td>
                    <td style={{ color: valColor(r.final_val_acc), fontWeight: 600 }}>
                      {r.final_val_acc != null ? `${r.final_val_acc.toFixed(1)}%` : '—'}
                    </td>
                    <td>{r.anomaly_count ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  )
}
