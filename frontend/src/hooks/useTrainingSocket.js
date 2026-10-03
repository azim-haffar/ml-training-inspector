import { epochPoint, batchPoint, terminalStatus } from './metricPoints'
import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const MAX_BATCH_POINTS = 120

export default function useTrainingSocket() {
  const [error, setError] = useState(null)
  const [streamNotice, setStreamNotice] = useState(null)
  const [isConnected, setIsConnected]         = useState(false)
  const [status, setStatus]                   = useState('idle') // idle|training|done|stopped|error
  const [epochData, setEpochData]             = useState([])
  const [batchData, setBatchData]             = useState([])
  const [gradNorms, setGradNorms]             = useState({})
  const [anomalies, setAnomalies]             = useState([])
  const [classAccuracies, setClassAccuracies] = useState({})
  const [lrHistory, setLrHistory]             = useState([])
  const [progress, setProgress]               = useState({ epoch: 0, batch: 0, totalBatches: 0, totalEpochs: 0 })
  const [lastCheckpoint, setLastCheckpoint]   = useState(null)
  const [duration, setDuration]               = useState(null) // seconds
  const [currentModel, setCurrentModel]       = useState(null)

  const runRef = useRef(null)
  const [currentDataset, setCurrentDataset] = useState(null)
  const wsRef         = useRef(null)
  const reconnectRef  = useRef(null)
  const startTimeRef  = useRef(null)

  useEffect(() => {
    let destroyed = false

    const connect = () => {
      if (destroyed) return
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen  = () => { if (!destroyed) setIsConnected(true) }
      ws.onerror = () => ws.close()
      ws.onclose = () => {
        if (destroyed) return
        setIsConnected(false)
        reconnectRef.current = setTimeout(connect, 2000)
      }

      ws.onmessage = (event) => {
        let data
        try { data = JSON.parse(event.data) } catch { setError('Received an invalid server message'); return }
        if (data.type === 'status') {
          setStatus(data.status)
          setLastCheckpoint(data.checkpoint)
          setError(data.error)
          if (data.training_start) {
            if (runRef.current !== data.training_start.run_id) {
              setEpochData([]); setBatchData([]); setAnomalies([]); setGradNorms({}); setClassAccuracies({}); setLrHistory([])
              runRef.current = data.training_start.run_id
            }
            setCurrentDataset(data.training_start.dataset)
            setCurrentModel(data.training_start.model)
            setProgress(p => ({ ...p, totalEpochs: data.training_start.total_epochs, totalBatches: data.training_start.total_batches }))
            setStreamNotice('Connected to an existing run. Earlier chart points are not replayed; charts show received samples only.')
          }
          return
        }

        if (data.type === 'heartbeat' || data.type === 'ping') return

        if (data.type === 'training_start') {
          runRef.current = data.run_id
          setCurrentDataset(data.dataset)
          setStatus('training')
          setError(null)
          setStreamNotice(null)
          setEpochData([])
          setBatchData([])
          setAnomalies([])
          setGradNorms({})
          setClassAccuracies({})
          setLrHistory([])
          setLastCheckpoint(null)
          setDuration(null)
          setCurrentModel(data.model || null)
          startTimeRef.current = Date.now()
          setProgress({
            epoch: 0, batch: 0,
            totalBatches: data.total_batches,
            totalEpochs: data.total_epochs,
          })
        }

        else if (data.type === 'batch') {
          setProgress(p => ({ ...p, epoch: data.epoch, batch: data.batch }))
          setBatchData(prev => {
            const point = batchPoint(data)
            return [...prev, point].slice(-MAX_BATCH_POINTS)
          })
          if (data.grad_norms) setGradNorms(data.grad_norms)
        }

        else if (data.type === 'epoch') {
          setEpochData(prev => [...prev, epochPoint(data)])
          if (data.class_accuracies) setClassAccuracies(data.class_accuracies)
          if (data.lr != null) setLrHistory(prev => [...prev, { epoch: data.epoch, lr: data.lr }])
          if (data.anomalies?.length > 0) {
            setAnomalies(prev => [
              ...prev,
              ...data.anomalies.map(a => ({
                ...a,
                epoch: data.epoch,
                id: `${a.type}-${data.epoch}-${Date.now()}`,
              })),
            ])
          }
        }

        else if (data.type === 'stopped') {
          setStatus('stopped')
          if (data.checkpoint) setLastCheckpoint(data.checkpoint)
          if (startTimeRef.current) setDuration(Math.round((Date.now() - startTimeRef.current) / 1000))
        }

        else if (data.type === 'checkpoint_saved') {
          setLastCheckpoint(data.checkpoint)
        }

        else if (data.type === 'done') {
          setStatus(terminalStatus(data))
          if (startTimeRef.current) setDuration(Math.round((Date.now() - startTimeRef.current) / 1000))
        }

        else if (data.type === 'error') {
          setStatus('error')
          setError(data.message)
        }
      }
    }

    connect()
    return () => {
      destroyed = true
      clearTimeout(reconnectRef.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [])

  const startTraining = useCallback(async (config) => {
    setStatus('starting')
    setError(null)
    try {
      const res = await fetch(`${API_URL}/api/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      const result = await res.json()
      if (!res.ok || result.error) { setError(typeof result.detail === 'string' ? result.detail : 'Invalid training settings or server error'); setStatus('error'); return false }
      return true
    } catch (err) {
      setError('Cannot reach the backend. Check that the server is running.'); setStatus('error')
      return false
    }
  }, [])

  const stopTraining = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/stop`, { method: 'POST' })
      const result = await res.json()
      if (!res.ok || result.error) setError(result.detail || result.error)
      else setStatus('stopping')
    } catch (err) {
      setError('Cannot reach the backend to stop training. Reconnect and try again.')
    }
  }, [])

  return {
    isConnected, status, currentModel, currentDataset, error, streamNotice,
    epochData, batchData, gradNorms, anomalies,
    classAccuracies, lrHistory,
    progress, lastCheckpoint, duration,
    startTraining, stopTraining,
  }
}
