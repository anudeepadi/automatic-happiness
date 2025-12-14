'use client'

import { motion } from 'framer-motion'

interface Terminal {
  port: string
  state: string
  utilization_pct: number
  status: string
}

interface TerminalUtilizationProps {
  terminals: Terminal[]
}

export default function TerminalUtilization({ terminals }: TerminalUtilizationProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'CRITICAL':
        return {
          text: 'text-signal-red',
          bg: 'bg-signal-red',
          border: 'border-signal-red/30',
        }
      case 'HIGH':
        return {
          text: 'text-cargo-orange',
          bg: 'bg-cargo-orange',
          border: 'border-cargo-orange/30',
        }
      case 'NORMAL':
        return {
          text: 'text-signal-yellow',
          bg: 'bg-signal-yellow',
          border: 'border-signal-yellow/30',
        }
      default:
        return {
          text: 'text-signal-green',
          bg: 'bg-signal-green',
          border: 'border-signal-green/30',
        }
    }
  }

  const sortedTerminals = [...terminals]
    .sort((a, b) => b.utilization_pct - a.utilization_pct)
    .slice(0, 8)

  return (
    <div className="data-card rounded-2xl p-4 h-[380px] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-display text-lg text-slate-800 tracking-wide">TERMINAL LOAD</h2>
          <p className="text-xs font-mono text-slate-500">Capacity utilization</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-signal-red">
            {terminals.filter(t => t.status === 'CRITICAL').length} CRITICAL
          </span>
        </div>
      </div>

      {/* Terminal List */}
      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {sortedTerminals.map((terminal, index) => {
          const colors = getStatusColor(terminal.status)
          return (
            <motion.div
              key={`${terminal.port}-${index}`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className="group"
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2 min-w-0">
                  <div className={`w-2 h-2 rounded-full ${colors.bg}`} />
                  <span className="font-mono text-xs text-slate-800 truncate">
                    {terminal.port}
                  </span>
                  <span className="text-[10px] font-mono text-slate-600">
                    {terminal.state}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-display ${colors.text}`}>
                    {terminal.utilization_pct.toFixed(0)}%
                  </span>
                  <span className={`
                    text-[9px] font-mono px-1.5 py-0.5 rounded
                    ${colors.text} bg-slate-100 border ${colors.border}
                  `}>
                    {terminal.status}
                  </span>
                </div>
              </div>

              {/* Progress bar */}
              <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
                <motion.div
                  className={`h-full rounded-full ${colors.bg}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.min(terminal.utilization_pct, 100)}%` }}
                  transition={{ delay: index * 0.05 + 0.2, duration: 0.8, ease: 'easeOut' }}
                  style={{
                    boxShadow: terminal.status === 'CRITICAL'
                      ? '0 0 10px rgba(239,68,68,0.5)'
                      : 'none',
                  }}
                />
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
