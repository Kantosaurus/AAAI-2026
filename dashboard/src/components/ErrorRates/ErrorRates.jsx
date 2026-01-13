import { motion } from 'framer-motion'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts'
import { AlertTriangle } from 'lucide-react'
import styles from './ErrorRates.module.css'

function ErrorRates({ data }) {
  const chartData = data.map(model => ({
    name: model.name.split('/').pop().substring(0, 10),
    fullName: model.name,
    errors: model.errorCount,
    success: model.successCount,
    errorRate: ((model.errorCount / (model.successCount + model.errorCount)) * 100).toFixed(1),
    retries: model.totalRetries || 0
  }))

  // Sort by error count descending
  chartData.sort((a, b) => b.errors - a.errors)

  const totalErrors = chartData.reduce((sum, d) => sum + d.errors, 0)

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload
      return (
        <div className={styles.tooltip}>
          <p className={styles.tooltipTitle}>{d.fullName}</p>
          <p className={styles.tooltipError}>
            {d.errors} errors <span>({d.errorRate}%)</span>
          </p>
          <p className={styles.tooltipMeta}>
            {d.success} successful | {d.retries} retries
          </p>
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
          <h3 className={styles.title}>Error Rates</h3>
          <p className={styles.subtitle}>Failures per model</p>
        </div>

        <div className={styles.statBadge}>
          <AlertTriangle className={styles.statIcon} />
          <span className={styles.statValue}>{totalErrors}</span>
          <span className={styles.statLabel}>total errors</span>
        </div>
      </div>

      <div className={styles.chartContainer}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 0, right: 20, left: 0, bottom: 0 }}
          >
            <XAxis
              type="number"
              tick={{ fill: 'var(--text-tertiary)', fontSize: 10 }}
              axisLine={{ stroke: 'var(--border-subtle)' }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: 'var(--text-secondary)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
              axisLine={false}
              tickLine={false}
              width={80}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'var(--bg-tertiary)' }} />
            <Bar
              dataKey="errors"
              radius={[0, 4, 4, 0]}
              barSize={16}
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.errors > 5 ? 'var(--accent-coral)' : 'var(--accent-yellow)'}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className={styles.legend}>
        <div className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: 'var(--accent-coral)' }} />
          <span>High (&gt;5)</span>
        </div>
        <div className={styles.legendItem}>
          <span className={styles.legendDot} style={{ background: 'var(--accent-yellow)' }} />
          <span>Low (&le;5)</span>
        </div>
      </div>
    </motion.div>
  )
}

export default ErrorRates
