import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Port-to-Rail Surge Forecaster | Command Center',
  description: 'GPU-accelerated port surge prediction and rail dispatch optimization',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased" suppressHydrationWarning>
        <div className="noise-overlay" />
        <div className="grid-overlay min-h-screen">
          {children}
        </div>
      </body>
    </html>
  )
}
