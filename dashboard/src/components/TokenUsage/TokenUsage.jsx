import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts'
import styles from './TokenUsage.module.css'

function TokenUsage({ data }) {
  const [activeModel, setActiveModel] = useState(null)

  // Transform data for stacked area chart
  const chartData = data.map(model => ({
    name: model.name.split('/').pop().substring(0, 12),
    fullName: model.name,
    input: model.avgInputTokens || Math.round(model.totalTokens * 0.3),
    output: model.avgOutputTokens || Math.round(model.totalTokens * 0.7),
    total: model.totalTokens,
    temperature: model.temperature
  }))

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className={styles.tooltip}>
          <p className={styles.tooltipTitle}>{label}</p>
          {payload.map((entry, idx) => (
            <p key={idx} className={styles.tooltipRow}>
              <span
                className={styles.tooltipDot}
                style={{ background: entry.color }}
              />
              {entry.name}: <span>{entry.value.toLocaleString()}</span>
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <motion.div
      className={styles.container}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className={styles.header}>
        <div className={styles.titleGroup}>
          <h3 className={styles.title}>Token Usage by Model</h3>
          <p className={styles.subtitle}>Input vs output token distribution</p>
        </div>

        <div className={styles.stats}>
          <div className={styles.statItem}>
            <span className={styles.statValue}>
              {(chartData.reduce((sum, d) => sum + d.total, 0) / 1000).toFixed(1)}K
            </span>
            <span className={styles.statLabel}>Total Tokens</span>
          </div>
        </div>
      </div>

      <div className={styles.chartContainer}>
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="colorInput" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00f5d4" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#00f5d4" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorOutput" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#9b5de5" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#9b5de5" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="name"
              tick={{ fill: 'var(--text-tertiary)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
              axisLine={{ stroke: 'var(--border-subtle)' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(val) => val >= 1000 ? `${(val/1000).toFixed(0)}K` : val}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="input"
              stackId="1"
              stroke="#00f5d4"
              fill="url(#colorInput)"
              strokeWidth={2}
              name="Input"
            />
            <Area
              type="monotone"
              dataKey="output"
              stackId="1"
              stroke="#9b5de5"
              fill="url(#colorOutput)"
              strokeWidth={2}
              name="Output"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className={styles.legend}>
        <div className={styles.legendItem}>
          <span className={styles.legendLine} style={{ background: '#00f5d4' }} />
          <span>Input Tokens</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendLine} style={{ background: '#9b5de5' }} />
          <span>Output Tokens</span>
        </div>
      </div>
    </motion.div>
  )
}

export default TokenUsage
