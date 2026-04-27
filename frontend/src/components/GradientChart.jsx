import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

// Shorten parameter names so they fit on the axis
function shortenName(name) {
  return name
    .replace('features.', 'f.')
    .replace('classifier.', 'cls.')
    .replace('.weight', '.w')
    .replace('.bias', '.b')
}

// Colour bars red when gradient norm looks suspiciously small
function barColor(norm) {
  if (norm < 1e-5) return '#ef4444'
  if (norm < 1e-3) return '#f97316'
  return '#818cf8'
}

export default function GradientChart({ gradNorms }) {
  const data = Object.entries(gradNorms).map(([name, norm]) => ({
    name: shortenName(name),
    norm,
    fullName: name,
  }))

  if (data.length === 0) {
    return <div className="chart-empty">Gradient data appears after first batch…</div>
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 5, right: 20, bottom: 60, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e2433" vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ fill: '#4a5568', fontSize: 11 }}
          angle={-35}
          textAnchor="end"
          interval={0}
        />
        <YAxis tick={{ fill: '#4a5568', fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: '#1a1f2e', border: '1px solid #2d3748', borderRadius: 6 }}
          labelStyle={{ color: '#8892a4' }}
          formatter={(v, _, props) => [v.toExponential(3), props.payload.fullName]}
        />
        <Bar dataKey="norm" name="Gradient Norm" isAnimationActive={false}>
          {data.map((entry, i) => (
            <Cell key={i} fill={barColor(entry.norm)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
