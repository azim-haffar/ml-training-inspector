import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'

export default function BatchLossChart({ batchData }) {
  if (batchData.length === 0) {
    return <div className="chart-empty">Waiting for live batch data…</div>
  }

  return (
    <ResponsiveContainer width="100%" height={240}>
      <AreaChart data={batchData} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
        <defs>
          <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2433" />
        <XAxis
          dataKey="label"
          tick={false}
          label={{ value: '← batch history (last 120)', position: 'insideBottom', offset: -10, fill: '#4a5568', fontSize: 11 }}
        />
        <YAxis tick={{ fill: '#4a5568', fontSize: 12 }} />
        <Tooltip
          contentStyle={{ background: '#1a1f2e', border: '1px solid #2d3748', borderRadius: 6 }}
          labelStyle={{ color: '#8892a4' }}
          labelFormatter={(l) => `step ${l}`}
        />
        <Area
          type="monotone"
          dataKey="loss"
          stroke="#818cf8"
          strokeWidth={1.5}
          fill="url(#lossGrad)"
          name="Batch Loss"
          dot={false}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
