'use client'

import { motion } from 'framer-motion'

interface StatsBarProps {
  summary: {
    total_ports: number
    avg_drayage_distance_km: number
    avg_drayage_time_min: number
    avg_drayage_cost_usd: number
    high_surge_ports: number
  }
  optimization: {
    high_risk_ports: number
    critical_terminals: number
    total_trucks_to_reposition: number
    avg_utilization_pct: number
  }
}

export default function StatsBar({ summary, optimization }: StatsBarProps) {
  const stats = [
    {
      label: 'ACTIVE PORTS',
      value: summary.total_ports,
      suffix: '',
      color: 'ocean',
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      ),
    },
    {
      label: 'HIGH RISK',
      value: optimization.high_risk_ports,
      suffix: '',
      color: 'red',
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      ),
    },
    {
      label: 'TRUCKS TO MOVE',
      value: optimization.total_trucks_to_reposition,
      suffix: '',
      color: 'orange',
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
        </svg>
      ),
    },
    {
      label: 'AVG DRAYAGE',
      value: Math.round(summary.avg_drayage_distance_km),
      suffix: 'km',
      color: 'cyan',
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      ),
    },
    {
      label: 'UTILIZATION',
      value: Math.round(optimization.avg_utilization_pct),
      suffix: '%',
      color: 'green',
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
    },
    {
      label: 'AVG COST',
      value: Math.round(summary.avg_drayage_cost_usd),
      suffix: 'USD',
      color: 'amber',
      icon: (
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
    },
  ]

  const colorMap: Record<string, string> = {
    ocean: 'text-ocean-400 bg-ocean-500/10 border-ocean-500/20',
    red: 'text-signal-red bg-signal-red/10 border-signal-red/20',
    orange: 'text-cargo-orange bg-cargo-orange/10 border-cargo-orange/20',
    cyan: 'text-ocean-300 bg-ocean-400/10 border-ocean-400/20',
    green: 'text-signal-green bg-signal-green/10 border-signal-green/20',
    amber: 'text-cargo-amber bg-cargo-amber/10 border-cargo-amber/20',
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mt-6">
      {stats.map((stat, index) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          className="data-card rounded-xl p-4"
        >
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg border ${colorMap[stat.color]}`}>
              {stat.icon}
            </div>
            <div>
              <div className="text-[10px] font-mono text-slate-500 tracking-wider">
                {stat.label}
              </div>
              <div className="flex items-baseline gap-1">
                <span className={`text-2xl font-display ${colorMap[stat.color].split(' ')[0]}`}>
                  {stat.value.toLocaleString()}
                </span>
                {stat.suffix && (
                  <span className="text-xs font-mono text-slate-500">{stat.suffix}</span>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  )
}
