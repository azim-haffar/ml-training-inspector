import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const MAX_BATCH_POINTS = 120

export default function useTrainingSocket() {
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
        const data = JSON.parse(event.data)

        if (data.type === 'heartbeat' || data.type === 'ping') return

        if (data.type === 'training_start') {
          setStatus('training')
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
            const point = { label: `${data.epoch}-${data.batch}`, loss: data.loss, accuracy: data.accuracy }
            return [...prev, point].slice(-MAX_BATCH_POINTS)
          })
          if (data.grad_norms) setGradNorms(data.grad_norms)
        }

        else if (data.type === 'epoch') {
          setEpochData(prev => [...prev, {
            epoch:      data.epoch,
            train_loss: data.train_loss,
            val_loss:   data.val_loss,
            train_acc:  data.train_acc,
            val_acc:    data.val_acc,
          }])
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
          setStatus('done')
          if (startTimeRef.current) setDuration(Math.round((Date.now() - startTimeRef.current) / 1000))
        }

        else if (data.type === 'error') {
          setStatus('error')
          console.error('Training error:', data.message)
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
    try {
      const res = await fetch(`${API_URL}/api/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      const result = await res.json()
      if (result.error) { console.error('Could not start:', result.error); return false }
      return true
    } catch (err) {
      console.error('Failed to reach backend:', err)
      return false
    }
  }, [])

  const stopTraining = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/stop`, { method: 'POST' })
      const result = await res.json()
      if (result.error) console.error('Could not stop:', result.error)
    } catch (err) {
      console.error('Failed to reach backend:', err)
    }
  }, [])

  return {
    isConnected, status, currentModel,
    epochData, batchData, gradNorms, anomalies,
    classAccuracies, lrHistory,
    progress, lastCheckpoint, duration,
    startTraining, stopTraining,
  }
}
