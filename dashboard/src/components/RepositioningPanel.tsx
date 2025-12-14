'use client'

import { motion } from 'framer-motion'

interface Repositioning {
  from_terminal: string
  to_terminal: string
  trucks: number
  urgency: string
  reason: string
}

interface RepositioningPanelProps {
  repositioning: Repositioning[]
}

export default function RepositioningPanel({ repositioning }: RepositioningPanelProps) {
  const getUrgencyStyles = (urgency: string) => {
    switch (urgency) {
      case 'high':
        return {
          bg: 'bg-signal-red/10',
          border: 'border-signal-red/30',
          text: 'text-signal-red',
          badge: 'bg-signal-red/20 text-signal-red border-signal-red/30',
        }
      case 'medium':
        return {
          bg: 'bg-cargo-orange/10',
          border: 'border-cargo-orange/30',
          text: 'text-cargo-orange',
          badge: 'bg-cargo-orange/20 text-cargo-orange border-cargo-orange/30',
        }
      default:
        return {
          bg: 'bg-ocean-500/10',
          border: 'border-ocean-500/30',
          text: 'text-ocean-400',
          badge: 'bg-ocean-500/20 text-ocean-400 border-ocean-500/30',
        }
    }
  }

  if (repositioning.length === 0) {
    return (
      <div className="data-card rounded-2xl p-4">
        <div className="flex items-center gap-3 mb-4">
          <h2 className="font-display text-lg text-white tracking-wide">TRUCK REPOSITIONING</h2>
          <span className="text-xs font-mono text-signal-green px-2 py-1 bg-signal-green/10 rounded border border-signal-green/20">
            NO ACTION NEEDED
          </span>
        </div>
        <p className="text-sm font-mono text-slate-500">
          Fleet is optimally positioned. No repositioning recommendations at this time.
        </p>
      </div>
    )
  }

  return (
    <div className="data-card rounded-2xl p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <h2 className="font-display text-lg text-white tracking-wide">TRUCK REPOSITIONING</h2>
          <span className="text-xs font-mono text-cargo-orange px-2 py-1 bg-cargo-orange/10 rounded border border-cargo-orange/20">
            {repositioning.length} MOVES
          </span>
        </div>
        <div className="text-xs font-mono text-slate-500">
          Total: {repositioning.reduce((sum, r) => sum + r.trucks, 0)} trucks
        </div>
      </div>

      {/* Repositioning Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {repositioning.map((rec, index) => {
          const styles = getUrgencyStyles(rec.urgency)
          return (
            <motion.div
              key={`${rec.from_terminal}-${rec.to_terminal}-${index}`}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.05 }}
              className={`
                ${styles.bg} border ${styles.border}
                rounded-xl p-4 relative overflow-hidden
              `}
            >
              {/* Urgency indicator */}
              <div className="absolute top-0 right-0 w-16 h-16 pointer-events-none">
                <div className={`
                  absolute top-0 right-0 w-16 h-16
                  ${rec.urgency === 'high' ? 'opacity-30' : 'opacity-10'}
                `}>
                  <svg viewBox="0 0 100 100" className={styles.text}>
                    <polygon points="100,0 100,100 0,0" fill="currentColor" />
                  </svg>
                </div>
              </div>

              {/* From/To */}
              <div className="flex items-center gap-2 mb-3">
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] font-mono text-slate-500 mb-0.5">FROM</div>
                  <div className="font-mono text-sm text-white truncate">
                    {rec.from_terminal}
                  </div>
                </div>

                <motion.div
                  animate={{ x: [0, 5, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                  className={styles.text}
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                </motion.div>

                <div className="flex-1 min-w-0 text-right">
                  <div className="text-[10px] font-mono text-slate-500 mb-0.5">TO</div>
                  <div className="font-mono text-sm text-white truncate">
                    {rec.to_terminal}
                  </div>
                </div>
              </div>

              {/* Stats */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                  </svg>
                  <span className={`font-display text-lg ${styles.text}`}>
                    {rec.trucks}
                  </span>
                  <span className="text-xs font-mono text-slate-500">trucks</span>
                </div>

                <span className={`
                  text-[10px] font-mono font-bold uppercase px-2 py-1 rounded border
                  ${styles.badge}
                `}>
                  {rec.urgency}
                </span>
              </div>

              {/* Reason */}
              <div className="mt-3 pt-3 border-t border-slate-700/50">
                <p className="text-[10px] font-mono text-slate-400 line-clamp-2">
                  {rec.reason}
                </p>
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
