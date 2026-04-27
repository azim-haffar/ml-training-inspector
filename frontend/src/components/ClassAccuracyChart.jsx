import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer,
} from 'recharts'

function barColor(acc) {
  if (acc >= 60) return '#22c55e'
  if (acc >= 40) return '#f59e0b'
  return '#ef4444'
}

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const { name, acc } = payload[0].payload
  return (
    <div style={{ background: '#141824', border: '1px solid #2d3748', borderRadius: 6, padding: '6px 10px', fontSize: '0.8rem' }}>
      <strong style={{ color: '#e2e8f0' }}>{name}</strong>
      <div style={{ color: barColor(acc) }}>{acc.toFixed(1)}%</div>
    </div>
  )
}

export default function ClassAccuracyChart({ classAccuracies }) {
  const data = Object.entries(classAccuracies).map(([name, acc]) => ({ name, acc }))

  if (data.length === 0) {
    return <div className="chart-empty">Per-class accuracy appears after first epoch…</div>
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 30, bottom: 0, left: 70 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2433" horizontal={false} />
        <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fill: '#4a5568', fontSize: 11 }} />
        <YAxis type="category" dataKey="name" tick={{ fill: '#8892a4', fontSize: 12 }} width={65} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: '#ffffff08' }} />
        <Bar dataKey="acc" radius={[0, 3, 3, 0]} isAnimationActive={false}>
          {data.map((entry, i) => (
            <Cell key={i} fill={barColor(entry.acc)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
