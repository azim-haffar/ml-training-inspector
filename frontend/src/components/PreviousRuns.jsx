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
  const [error, setError] = useState(null)
  const [open, setOpen]       = useState(false)
  const [runs, setRuns]       = useState([])
  const [loading, setLoading] = useState(false)

  // Fetch once when the section is first opened
  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    setError(null)
    setLoading(true)
    fetch(`${API_URL}/api/history`, { signal: controller.signal })
      .then(r => { if (!r.ok) throw new Error('History unavailable'); return r.json() })
      .then(data => setRuns((data.history || []).reverse())) // newest first
      .catch(e => { if (e.name !== 'AbortError') setError('Could not load previous runs. Close and reopen to retry.') })
      .finally(() => setLoading(false))
    return () => controller.abort()
  }, [open])

  return (
    <div className="prev-runs">
      <button aria-expanded={open} className="prev-runs-toggle" onClick={() => setOpen(o => !o)}>
        <span>Previous Runs</span>
        <span className="toggle-arrow">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="prev-runs-body">
          {loading && <p className="prev-runs-empty">Loading…</p>}
          {error && <p role="alert">{error}</p>}
          {!loading && !error && runs.length === 0 && (
            <p className="prev-runs-empty">No runs recorded yet.</p>
          )}
          {!loading && runs.length > 0 && (
            <table className="runs-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Model / Dataset</th>
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
                    <td>{r.model || 'simple_cnn'} / {r.dataset || 'cifar10'}</td>
                    <td>{r.completed_epochs ?? r.epochs}/{r.epochs} ({r.status || "unknown"})</td>
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
