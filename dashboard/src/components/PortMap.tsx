'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { useState, useMemo } from 'react'

interface Port {
  portid: string
  portname: string
  port_lat: number
  port_lon: number
  terminal_state: string
  distance_km: number
  daily_trucks_needed: number
  avg_calls: number | null
}

interface Prediction {
  port: string
  surge_probability: number
  risk_level: string
  predicted_calls_1d: number
}

interface PortMapProps {
  ports: Port[]
  predictions: Prediction[]
  selectedPort: string | null
  onPortSelect: (port: string | null) => void
}

export default function PortMap({ ports, predictions, selectedPort, onPortSelect }: PortMapProps) {
  const [hoveredPort, setHoveredPort] = useState<string | null>(null)

  // Convert lat/lon to SVG coordinates
  const latLonToSvg = (lat: number, lon: number) => {
    // US bounds approximately: lat 24-50, lon -125 to -66
    const x = ((lon + 125) / 60) * 900 + 50
    const y = ((50 - lat) / 26) * 450 + 25
    return { x, y }
  }

  // Merge port data with predictions
  const portData = useMemo(() => {
    return ports.map(port => {
      const prediction = predictions.find(p => p.port === port.portname)
      const coords = latLonToSvg(port.port_lat, port.port_lon)
      return {
        ...port,
        ...coords,
        prediction,
        risk: prediction?.risk_level || 'LOW',
        surgeProb: prediction?.surge_probability || 0,
      }
    }).filter(p => p.x > 0 && p.x < 1000 && p.y > 0 && p.y < 500)
  }, [ports, predictions])

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'HIGH': return '#ef4444'
      case 'MEDIUM': return '#eab308'
      default: return '#22c55e'
    }
  }

  const selectedPortData = portData.find(p => p.portname === (selectedPort || hoveredPort))

  return (
    <div className="data-card rounded-2xl p-4 h-[500px] relative overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="font-display text-lg text-slate-800 tracking-wide">PORT NETWORK</h2>
          <p className="text-xs font-mono text-slate-500">Real-time risk visualization</p>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-signal-green" />
            <span className="text-slate-400">LOW</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-signal-yellow" />
            <span className="text-slate-400">MEDIUM</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-signal-red" />
            <span className="text-slate-400">HIGH</span>
          </div>
        </div>
      </div>

      {/* Map Container */}
      <div className="relative h-[calc(100%-60px)] bg-slate-50 rounded-xl overflow-hidden border border-ocean-500/15">
        {/* Radar sweep effect */}
        <motion.div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: 'conic-gradient(from 0deg, transparent 0deg, rgba(0,184,230,0.05) 30deg, transparent 60deg)',
            transformOrigin: '30% 40%',
          }}
          animate={{ rotate: 360 }}
          transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
        />

        {/* Grid lines */}
        <div className="absolute inset-0 opacity-20">
          {[...Array(10)].map((_, i) => (
            <div key={`h-${i}`} className="absolute w-full h-px bg-ocean-500/30" style={{ top: `${i * 10}%` }} />
          ))}
          {[...Array(15)].map((_, i) => (
            <div key={`v-${i}`} className="absolute h-full w-px bg-ocean-500/30" style={{ left: `${i * 7}%` }} />
          ))}
        </div>

        {/* SVG Map */}
        <svg viewBox="0 0 1000 500" className="w-full h-full" preserveAspectRatio="xMidYMid meet">
          {/* US Coastline simplified */}
          <path
            d="M 150 180 L 180 160 L 220 150 L 280 140 L 340 130 L 400 125 L 460 130 L 520 140 L 580 155 L 640 165 L 700 180 L 760 200 L 820 230 L 850 260 L 870 300 L 880 340 L 870 380 L 840 400 L 800 410 L 750 400 L 700 380 L 650 370 L 600 365 L 550 370 L 500 380 L 450 385 L 400 380 L 350 370 L 300 355 L 250 340 L 200 320 L 170 290 L 150 250 L 145 210 Z"
            fill="rgba(0,184,230,0.03)"
            stroke="rgba(0,184,230,0.2)"
            strokeWidth="1"
          />

          {/* Port markers */}
          {portData.map((port, index) => (
            <g
              key={port.portid}
              className="cursor-pointer port-marker"
              onClick={() => onPortSelect(selectedPort === port.portname ? null : port.portname)}
              onMouseEnter={() => setHoveredPort(port.portname)}
              onMouseLeave={() => setHoveredPort(null)}
            >
              {/* Pulse ring for high risk */}
              {port.risk === 'HIGH' && (
                <motion.circle
                  cx={port.x}
                  cy={port.y}
                  r={12}
                  fill="none"
                  stroke={getRiskColor(port.risk)}
                  strokeWidth="2"
                  initial={{ opacity: 1, scale: 1 }}
                  animate={{ opacity: 0, scale: 2 }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              )}

              {/* Port dot */}
              <motion.circle
                cx={port.x}
                cy={port.y}
                r={selectedPort === port.portname ? 10 : hoveredPort === port.portname ? 8 : 6}
                fill={getRiskColor(port.risk)}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: index * 0.02 }}
                style={{
                  filter: `drop-shadow(0 0 ${port.risk === 'HIGH' ? '8' : '4'}px ${getRiskColor(port.risk)})`,
                }}
              />

              {/* Port label on hover/select */}
              <AnimatePresence>
                {(hoveredPort === port.portname || selectedPort === port.portname) && (
                  <motion.g
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -5 }}
                  >
                    <rect
                      x={port.x - 50}
                      y={port.y - 35}
                      width={100}
                      height={22}
                      rx={4}
                      fill="rgba(255,255,255,0.95)"
                      stroke={getRiskColor(port.risk)}
                      strokeWidth="1"
                    />
                    <text
                      x={port.x}
                      y={port.y - 20}
                      textAnchor="middle"
                      fill="#1e293b"
                      fontSize="10"
                      fontFamily="JetBrains Mono"
                    >
                      {port.portname.slice(0, 15)}
                    </text>
                  </motion.g>
                )}
              </AnimatePresence>
            </g>
          ))}
        </svg>

        {/* Selected port info panel */}
        <AnimatePresence>
          {selectedPortData && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              className="absolute top-4 right-4 w-64 bg-white/95 border border-ocean-500/20 rounded-xl p-4 backdrop-blur-xl shadow-lg"
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-display text-sm text-slate-800">{selectedPortData.portname}</h3>
                <button
                  onClick={() => onPortSelect(null)}
                  className="text-slate-400 hover:text-slate-800 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="space-y-2 text-xs font-mono">
                <div className="flex justify-between">
                  <span className="text-slate-500">State</span>
                  <span className="text-white">{selectedPortData.terminal_state}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Risk Level</span>
                  <span style={{ color: getRiskColor(selectedPortData.risk) }}>{selectedPortData.risk}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Surge Prob.</span>
                  <span className="text-ocean-400">{selectedPortData.surgeProb.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Daily Trucks</span>
                  <span className="text-cargo-orange">{selectedPortData.daily_trucks_needed.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Drayage</span>
                  <span className="text-slate-700">{selectedPortData.distance_km.toFixed(0)} km</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
