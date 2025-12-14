'use client'

import { motion } from 'framer-motion'

interface DispatchWindow {
  port: string
  state: string
  optimal_hours: string
  risk_score: number
  expected_trucks: number
  recommendation: string
}

interface DispatchWindowsProps {
  windows: DispatchWindow[]
}

export default function DispatchWindows({ windows }: DispatchWindowsProps) {
  const getRiskColor = (score: number) => {
    if (score >= 0.7) return 'text-signal-red'
    if (score >= 0.4) return 'text-signal-yellow'
    return 'text-signal-green'
  }

  const getBgColor = (score: number) => {
    if (score >= 0.7) return 'bg-signal-red/10 border-signal-red/20'
    if (score >= 0.4) return 'bg-signal-yellow/10 border-signal-yellow/20'
    return 'bg-signal-green/10 border-signal-green/20'
  }

  return (
    <div className="data-card rounded-2xl p-4 h-[380px] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-display text-lg text-slate-800 tracking-wide">DISPATCH WINDOWS</h2>
          <p className="text-xs font-mono text-slate-500">Optimal scheduling</p>
        </div>
        <div className="flex items-center gap-1 text-xs font-mono text-ocean-400">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>TODAY</span>
        </div>
      </div>

      {/* Windows List */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1">
        {windows.slice(0, 6).map((window, index) => (
          <motion.div
            key={`${window.port}-${index}`}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className={`
              border rounded-xl p-3
              ${getBgColor(window.risk_score)}
            `}
          >
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="font-mono text-sm text-slate-800 font-medium">
                  {window.port}
                </h3>
                <span className="text-[10px] font-mono text-slate-500">{window.state}</span>
              </div>
              <div className={`text-right ${getRiskColor(window.risk_score)}`}>
                <div className="text-lg font-display">{(window.risk_score * 100).toFixed(0)}%</div>
                <div className="text-[10px] font-mono opacity-70">RISK</div>
              </div>
            </div>

            <div className="flex items-center justify-between text-xs font-mono">
              <div className="flex items-center gap-2">
                <svg className="w-3.5 h-3.5 text-ocean-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-ocean-300">{window.optimal_hours}</span>
              </div>
              <div className="flex items-center gap-2 text-slate-400">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                </svg>
                <span>{window.expected_trucks.toLocaleString()} trucks</span>
              </div>
            </div>

            <div className="mt-2 text-[10px] font-mono text-slate-500 truncate">
              {window.recommendation}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
