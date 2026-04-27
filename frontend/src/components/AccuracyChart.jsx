import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

export default function AccuracyChart({ epochData }) {
  if (epochData.length === 0) {
    return <div className="chart-empty">Waiting for epoch data…</div>
  }

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={epochData} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2433" />
        <XAxis
          dataKey="epoch"
          label={{ value: 'Epoch', position: 'insideBottom', offset: -10, fill: '#4a5568' }}
          tick={{ fill: '#4a5568', fontSize: 12 }}
        />
        <YAxis
          domain={[0, 100]}
          unit="%"
          tick={{ fill: '#4a5568', fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{ background: '#1a1f2e', border: '1px solid #2d3748', borderRadius: 6 }}
          labelStyle={{ color: '#8892a4' }}
          formatter={(v) => `${v}%`}
        />
        <Legend wrapperStyle={{ paddingTop: 8 }} />
        <Line type="monotone" dataKey="train_acc" stroke="#fbbf24" strokeWidth={2} dot={false} name="Train Acc" />
        <Line type="monotone" dataKey="val_acc" stroke="#f97316" strokeWidth={2} dot={false} name="Val Acc" />
      </LineChart>
    </ResponsiveContainer>
  )
}
