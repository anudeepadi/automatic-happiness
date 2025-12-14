'use client'

import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import PortMap from '@/components/PortMap'
import SurgeAlerts from '@/components/SurgeAlerts'
import DispatchWindows from '@/components/DispatchWindows'
import TerminalUtilization from '@/components/TerminalUtilization'
import RepositioningPanel from '@/components/RepositioningPanel'
import FeatureImportance from '@/components/FeatureImportance'
import StatsBar from '@/components/StatsBar'
import Header from '@/components/Header'

interface DashboardData {
  summary: {
    total_ports: number
    total_terminals: number
    avg_drayage_distance_km: number
    avg_drayage_time_min: number
    avg_drayage_cost_usd: number
    high_surge_ports: number
    last_updated: string
  }
  ports: Array<{
    portid: string
    portname: string
    port_lat: number
    port_lon: number
    terminal_state: string
    distance_km: number
    daily_trucks_needed: number
    avg_calls: number | null
  }>
  surge_analysis: Array<{
    port: string
    surge_rate: number
    avg_calls: number
    max_calls: number
    max_zscore: number
    avg_import: number
  }>
  predictions: Array<{
    port: string
    surge_probability: number
    risk_level: string
    predicted_calls_1d: number
  }>
  optimization: {
    summary: {
      total_ports: number
      high_risk_ports: number
      critical_terminals: number
      total_trucks_to_reposition: number
      avg_utilization_pct: number
    }
    dispatch_windows: Array<{
      port: string
      state: string
      optimal_hours: string
      risk_score: number
      expected_trucks: number
      recommendation: string
    }>
    repositioning: Array<{
      from_terminal: string
      to_terminal: string
      trucks: number
      urgency: string
      reason: string
    }>
    terminal_status: Array<{
      port: string
      state: string
      utilization_pct: number
      status: string
    }>
  }
  feature_importance: Array<{
    feature: string
    importance: number
  }>
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedPort, setSelectedPort] = useState<string | null>(null)

  useEffect(() => {
    async function fetchDashboard() {
      try {
        const response = await fetch('http://localhost:8000/dashboard')
        if (!response.ok) throw new Error('Failed to fetch dashboard data')
        const result = await response.json()
        setData(result)
        setError(null)
      } catch (err) {
        console.error('Error fetching dashboard:', err)
        setError('Unable to connect to API. Make sure FastAPI is running on port 8000.')
      } finally {
        setLoading(false)
      }
    }

    fetchDashboard()
    const interval = setInterval(fetchDashboard, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center"
        >
          <div className="relative w-24 h-24 mx-auto mb-6">
            <motion.div
              className="absolute inset-0 border-4 border-ocean-500/30 rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
            />
            <motion.div
              className="absolute inset-2 border-4 border-t-ocean-400 border-r-transparent border-b-transparent border-l-transparent rounded-full"
              animate={{ rotate: -360 }}
              transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
            />
          </div>
          <h2 className="font-display text-2xl text-ocean-400 tracking-wider">INITIALIZING</h2>
          <p className="text-slate-400 mt-2 font-mono text-sm">Loading port intelligence...</p>
        </motion.div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center max-w-lg"
        >
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-cargo-orange/10 flex items-center justify-center">
            <svg className="w-10 h-10 text-cargo-orange" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h2 className="font-display text-2xl text-cargo-orange tracking-wider mb-3">CONNECTION LOST</h2>
          <p className="text-slate-400 font-mono text-sm mb-6">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-3 bg-ocean-600/20 hover:bg-ocean-600/30 border border-ocean-500/30 rounded-lg font-mono text-ocean-400 transition-all"
          >
            RETRY CONNECTION
          </button>
        </motion.div>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="min-h-screen pb-8">
      <Header />

      <main className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8">
        {/* Stats Bar */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <StatsBar summary={data.summary} optimization={data.optimization.summary} />
        </motion.div>

        {/* Main Grid */}
        <div className="grid grid-cols-12 gap-4 mt-6">
          {/* Port Map - Large */}
          <motion.div
            className="col-span-12 lg:col-span-8 xl:col-span-9"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <PortMap
              ports={data.ports}
              predictions={data.predictions}
              selectedPort={selectedPort}
              onPortSelect={setSelectedPort}
            />
          </motion.div>

          {/* Surge Alerts - Side Panel */}
          <motion.div
            className="col-span-12 lg:col-span-4 xl:col-span-3"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <SurgeAlerts
              predictions={data.predictions}
              onPortSelect={setSelectedPort}
            />
          </motion.div>

          {/* Dispatch Windows */}
          <motion.div
            className="col-span-12 md:col-span-6 xl:col-span-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <DispatchWindows windows={data.optimization.dispatch_windows} />
          </motion.div>

          {/* Terminal Utilization */}
          <motion.div
            className="col-span-12 md:col-span-6 xl:col-span-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <TerminalUtilization terminals={data.optimization.terminal_status} />
          </motion.div>

          {/* Feature Importance */}
          <motion.div
            className="col-span-12 xl:col-span-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
          >
            <FeatureImportance features={data.feature_importance} />
          </motion.div>

          {/* Repositioning Recommendations */}
          <motion.div
            className="col-span-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
          >
            <RepositioningPanel repositioning={data.optimization.repositioning} />
          </motion.div>
        </div>
      </main>
    </div>
  )
}
