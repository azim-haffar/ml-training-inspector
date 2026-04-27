import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'

export default function LRChart({ lrHistory }) {
  if (lrHistory.length === 0) {
    return <div className="chart-empty">LR schedule appears after first epoch…</div>
  }

  // Format the LR value nicely — it can be very small after cosine decay
  const formatLR = (v) => {
    if (v === 0) return '0'
    if (v < 0.0001) return v.toExponential(2)
    return v.toFixed(5).replace(/\.?0+$/, '')
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={lrHistory} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2433" />
        <XAxis
          dataKey="epoch"
          label={{ value: 'Epoch', position: 'insideBottom', offset: -10, fill: '#4a5568' }}
          tick={{ fill: '#4a5568', fontSize: 12 }}
        />
        <YAxis
          tickFormatter={formatLR}
          tick={{ fill: '#4a5568', fontSize: 10 }}
          width={55}
        />
        <Tooltip
          contentStyle={{ background: '#141824', border: '1px solid #2d3748', borderRadius: 6 }}
          labelStyle={{ color: '#8892a4' }}
          formatter={(v) => [formatLR(v), 'Learning Rate']}
        />
        <Line
          type="monotone"
          dataKey="lr"
          stroke="#a78bfa"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
