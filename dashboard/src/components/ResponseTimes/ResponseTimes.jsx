import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid
} from 'recharts'
import styles from './ResponseTimes.module.css'

const MODEL_COLORS = {
  'claude': '#00f5d4',
  'gemini': '#9b5de5',
  'Qwen': '#f9c74f',
  'Mistral': '#ff6b6b',
  'Phi': '#4cc9f0'
}

function ResponseTimes({ data, models }) {
  const [showSynthetic, setShowSynthetic] = useState(false)

  // Group time series by model
  const modelNames = [...new Set(models.map(m => m.name.split('/').pop().split('-')[0]))]

  // Create chart data points (sample every nth point for performance)
  const sampleRate = Math.max(1, Math.floor(data.length / 50))
  const chartData = data
    .filter((_, i) => i % sampleRate === 0)
    .map((point, index) => {
      const modelKey = point.model.split('/').pop().split('-')[0]
      return {
        index,
        time: point.elapsed,
        model: modelKey,
        isSynthetic: point.isSynthetic
      }
    })

  // Prepare data for multi-line chart
  const lineData = chartData.reduce((acc, point) => {
    if (!acc[point.index]) {
      acc[point.index] = { index: point.index }
    }
    acc[point.index][point.model] = point.time
    return acc
  }, {})

  const finalData = Object.values(lineData)

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className={styles.tooltip}>
          <p className={styles.tooltipTitle}>Sample #{label}</p>
          {payload.map((entry, idx) => (
            <p key={idx} className={styles.tooltipRow}>
              <span
                className={styles.tooltipDot}
                style={{ background: entry.color }}
              />
              {entry.dataKey}: <span>{entry.value?.toFixed(2)}s</span>
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
          <h3 className={styles.title}>Response Times</h3>
          <p className={styles.subtitle}>Latency per model over time</p>
        </div>

        <button
          className={`${styles.toggle} ${showSynthetic ? styles.toggleActive : ''}`}
          onClick={() => setShowSynthetic(!showSynthetic)}
        >
          Synthetic Probes
        </button>
      </div>

      <div className={styles.chartContainer}>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart
            data={finalData}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="var(--border-subtle)"
              vertical={false}
            />
            <XAxis
              dataKey="index"
              tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }}
              axisLine={{ stroke: 'var(--border-subtle)' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(val) => `${val}s`}
            />
            <Tooltip content={<CustomTooltip />} />
            {modelNames.map(modelName => (
              <Line
                key={modelName}
                type="monotone"
                dataKey={modelName}
                stroke={MODEL_COLORS[modelName] || 'var(--text-tertiary)'}
                strokeWidth={2}
                dot={false}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className={styles.legend}>
        {modelNames.map(name => (
          <div key={name} className={styles.legendItem}>
            <span
              className={styles.legendLine}
              style={{ background: MODEL_COLORS[name] || 'var(--text-tertiary)' }}
            />
            <span>{name}</span>
          </div>
        ))}
      </div>
    </motion.div>
  )
}

export default ResponseTimes
