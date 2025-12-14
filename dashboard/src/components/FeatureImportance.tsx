'use client'

import { motion } from 'framer-motion'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

interface Feature {
  feature: string
  importance: number
}

interface FeatureImportanceProps {
  features: Feature[]
}

export default function FeatureImportance({ features }: FeatureImportanceProps) {
  // Normalize and sort features
  const maxImportance = Math.max(...features.map(f => f.importance))
  const chartData = features
    .slice(0, 10)
    .map(f => ({
      name: f.feature.replace(/_/g, ' ').slice(0, 15),
      value: (f.importance / maxImportance) * 100,
      raw: f.importance,
    }))
    .sort((a, b) => b.value - a.value)

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: { name: string; raw: number } }> }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white/95 border border-ocean-500/20 rounded-lg p-2 backdrop-blur-xl shadow-lg">
          <p className="text-xs font-mono text-slate-800">{payload[0].payload.name}</p>
          <p className="text-xs font-mono text-ocean-400">
            Importance: {payload[0].payload.raw.toFixed(4)}
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className="data-card rounded-2xl p-4 h-[380px] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-display text-lg text-slate-800 tracking-wide">MODEL INSIGHTS</h2>
          <p className="text-xs font-mono text-slate-500">Top predictive features</p>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-ocean-400">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
          <span>XGBoost</span>
        </div>
      </div>

      {/* Chart */}
      <motion.div
        className="flex-1"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
          >
            <XAxis
              type="number"
              domain={[0, 100]}
              tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              axisLine={{ stroke: 'rgba(0,184,230,0.1)' }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              axisLine={false}
              tickLine={false}
              width={100}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(0,184,230,0.05)' }} />
            <Bar
              dataKey="value"
              radius={[0, 4, 4, 0]}
              maxBarSize={20}
            >
              {chartData.map((_, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={`rgba(0, 184, 230, ${1 - index * 0.08})`}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </motion.div>
    </div>
  )
}
