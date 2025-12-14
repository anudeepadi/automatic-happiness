'use client'

import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

export default function Header() {
  const [mounted, setMounted] = useState(false)
  const [time, setTime] = useState<string>('--:--:--')
  const [date, setDate] = useState<string>('--- --- --')

  useEffect(() => {
    setMounted(true)
    const updateTime = () => {
      const now = new Date()
      setTime(now.toLocaleTimeString('en-US', { hour12: false }))
      setDate(now.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }))
    }
    updateTime()
    const interval = setInterval(updateTime, 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <header className="relative border-b border-ocean-500/10 bg-slate-950/80 backdrop-blur-xl sticky top-0 z-50">
      {/* Scan line effect */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          className="absolute h-[1px] w-full bg-gradient-to-r from-transparent via-ocean-400/40 to-transparent"
          animate={{ y: ['-100%', '4000%'] }}
          transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
        />
      </div>

      <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between">
          {/* Logo & Title */}
          <div className="flex items-center gap-4">
            <div className="relative">
              <motion.div
                className="w-12 h-12 rounded-lg bg-gradient-to-br from-ocean-500/20 to-ocean-700/20 border border-ocean-500/30 flex items-center justify-center"
                animate={{ boxShadow: ['0 0 20px rgba(0,184,230,0.1)', '0 0 40px rgba(0,184,230,0.2)', '0 0 20px rgba(0,184,230,0.1)'] }}
                transition={{ duration: 3, repeat: Infinity }}
              >
                <svg className="w-7 h-7 text-ocean-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                </svg>
              </motion.div>
            </div>
            <div>
              <h1 className="font-display text-xl sm:text-2xl tracking-wider text-white">
                PORT<span className="text-ocean-400">SURGE</span>
              </h1>
              <p className="text-[10px] sm:text-xs font-mono text-slate-500 tracking-widest uppercase">
                Rail Dispatch Command Center
              </p>
            </div>
          </div>

          {/* Status Indicators */}
          <div className="hidden md:flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="relative">
                <div className="w-2 h-2 rounded-full bg-signal-green" />
                <div className="absolute inset-0 w-2 h-2 rounded-full bg-signal-green animate-ping opacity-75" />
              </div>
              <span className="text-xs font-mono text-signal-green">LIVE</span>
            </div>

            <div className="h-8 w-px bg-slate-800" />

            <div className="text-right">
              <div className="font-mono text-lg text-white tracking-wider">{time}</div>
              <div className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">{date}</div>
            </div>
          </div>

          {/* Mobile Time */}
          <div className="md:hidden flex items-center gap-3">
            <div className="relative">
              <div className="w-2 h-2 rounded-full bg-signal-green" />
              <div className="absolute inset-0 w-2 h-2 rounded-full bg-signal-green animate-ping opacity-75" />
            </div>
            <span className="font-mono text-sm text-white">{time}</span>
          </div>
        </div>
      </div>
    </header>
  )
}
