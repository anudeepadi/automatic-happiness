'use client'

import { motion } from 'framer-motion'

interface Prediction {
  port: string
  surge_probability: number
  risk_level: string
  predicted_calls_1d: number
}

interface SurgeAlertsProps {
  predictions: Prediction[]
  onPortSelect: (port: string) => void
}

export default function SurgeAlerts({ predictions, onPortSelect }: SurgeAlertsProps) {
  // Sort by surge probability descending
  const sortedPredictions = [...predictions]
    .sort((a, b) => b.surge_probability - a.surge_probability)
    .slice(0, 8)

  const getRiskStyles = (level: string) => {
    switch (level) {
      case 'HIGH':
        return {
          bg: 'bg-signal-red/10',
          border: 'border-signal-red/30',
          text: 'text-signal-red',
          glow: 'shadow-[0_0_15px_rgba(239,68,68,0.2)]',
        }
      case 'MEDIUM':
        return {
          bg: 'bg-signal-yellow/10',
          border: 'border-signal-yellow/30',
          text: 'text-signal-yellow',
          glow: '',
        }
      default:
        return {
          bg: 'bg-signal-green/10',
          border: 'border-signal-green/30',
          text: 'text-signal-green',
          glow: '',
        }
    }
  }

  return (
    <div className="data-card rounded-2xl p-4 h-[500px] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-display text-lg text-slate-800 tracking-wide">SURGE ALERTS</h2>
          <p className="text-xs font-mono text-slate-500">24h forecast</p>
        </div>
        <div className="flex items-center gap-2">
          <motion.div
            className="w-2 h-2 rounded-full bg-signal-red"
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 1, repeat: Infinity }}
          />
          <span className="text-xs font-mono text-signal-red">
            {predictions.filter(p => p.risk_level === 'HIGH').length} CRITICAL
          </span>
        </div>
      </div>

      {/* Alert List */}
      <div className="flex-1 overflow-y-auto space-y-2 pr-1">
        {sortedPredictions.map((pred, index) => {
          const styles = getRiskStyles(pred.risk_level)
          return (
            <motion.div
              key={pred.port}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => onPortSelect(pred.port)}
              className={`
                ${styles.bg} ${styles.border} ${styles.glow}
                border rounded-xl p-3 cursor-pointer
                hover:scale-[1.02] transition-all duration-200
              `}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    {pred.risk_level === 'HIGH' && (
                      <motion.div
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{ duration: 0.5, repeat: Infinity }}
                      >
                        <svg className="w-4 h-4 text-signal-red" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                      </motion.div>
                    )}
                    <h3 className="font-mono text-sm text-slate-800 font-medium truncate">
                      {pred.port}
                    </h3>
                  </div>
                  <div className="flex items-center gap-3 mt-2">
                    <span className={`text-xs font-mono ${styles.text}`}>
                      {pred.surge_probability.toFixed(0)}% SURGE
                    </span>
                    <span className="text-xs font-mono text-slate-500">
                      ~{pred.predicted_calls_1d.toFixed(0)} calls
                    </span>
                  </div>
                </div>
                <div className={`
                  px-2 py-1 rounded text-[10px] font-mono font-bold
                  ${styles.bg} ${styles.text} border ${styles.border}
                `}>
                  {pred.risk_level}
                </div>
              </div>

              {/* Progress bar */}
              <div className="mt-2 h-1 bg-slate-200 rounded-full overflow-hidden">
                <motion.div
                  className={`h-full rounded-full ${
                    pred.risk_level === 'HIGH' ? 'bg-signal-red' :
                    pred.risk_level === 'MEDIUM' ? 'bg-signal-yellow' : 'bg-signal-green'
                  }`}
                  initial={{ width: 0 }}
                  animate={{ width: `${pred.surge_probability}%` }}
                  transition={{ delay: index * 0.05 + 0.2, duration: 0.5 }}
                />
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
