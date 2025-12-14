import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Maritime industrial palette
        slate: {
          950: '#0a0c10',
          900: '#0f1218',
          850: '#151922',
          800: '#1a1f2e',
        },
        ocean: {
          50: '#e6f9ff',
          100: '#b3edff',
          200: '#80e1ff',
          300: '#4dd5ff',
          400: '#1ac9ff',
          500: '#00b8e6',
          600: '#0094b8',
          700: '#00708a',
          800: '#004c5c',
          900: '#00282e',
        },
        cargo: {
          orange: '#ff6b35',
          amber: '#ffb800',
          rust: '#c45a3b',
        },
        signal: {
          green: '#22c55e',
          yellow: '#eab308',
          red: '#ef4444',
        }
      },
      fontFamily: {
        display: ['var(--font-display)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'monospace'],
        body: ['var(--font-body)', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan': 'scan 4s linear infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'slide-up': 'slideUp 0.5s ease-out',
        'fade-in': 'fadeIn 0.6s ease-out',
      },
      keyframes: {
        scan: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(0, 184, 230, 0.2), 0 0 10px rgba(0, 184, 230, 0.1)' },
          '100%': { boxShadow: '0 0 20px rgba(0, 184, 230, 0.4), 0 0 40px rgba(0, 184, 230, 0.2)' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
      backgroundImage: {
        'grid-pattern': 'linear-gradient(rgba(0,184,230,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,184,230,0.03) 1px, transparent 1px)',
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [],
}
export default config
