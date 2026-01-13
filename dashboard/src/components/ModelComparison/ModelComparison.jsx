import { useState } from 'react'
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
import styles from './ModelComparison.module.css'

const COLORS = {
  claude: '#00f5d4',
  gemini: '#9b5de5',
  local: '#f9c74f'
}

function ModelComparison({ data }) {
  const [selectedTemp, setSelectedTemp] = useState(null)

  // Group by temperature
  const temps = [...new Set(data.map(d => d.temperature))]

  const filteredData = selectedTemp !== null
    ? data.filter(d => d.temperature === selectedTemp)
    : data

  const chartData = filteredData.map(model => ({
    name: model.name.split('/').pop().substring(0, 15),
    fullName: model.name,
    successRate: ((model.successCount / (model.successCount + model.errorCount)) * 100).toFixed(1),
    type: model.name.includes('claude') ? 'claude'
        : model.name.includes('gemini') ? 'gemini'
        : 'local',
    temperature: model.temperature
  }))

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload
      return (
        <div className={styles.tooltip}>
          <p className={styles.tooltipTitle}>{d.fullName}</p>
          <p className={styles.tooltipValue}>
            Success Rate: <span>{d.successRate}%</span>
          </p>
          <p className={styles.tooltipMeta}>
            Temperature: {d.temperature}
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
          <h3 className={styles.title}>Model Performance</h3>
          <p className={styles.subtitle}>Success rate comparison</p>
        </div>

        <div className={styles.toggleGroup}>
          <button
            className={`${styles.toggle} ${selectedTemp === null ? styles.toggleActive : ''}`}
            onClick={() => setSelectedTemp(null)}
          >
            All
          </button>
          {temps.map(temp => (
            <button
              key={temp}
              className={`${styles.toggle} ${selectedTemp === temp ? styles.toggleActive : ''}`}
              onClick={() => setSelectedTemp(temp)}
            >
              T={temp}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.chartContainer}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 10, right: 30, left: 0, bottom: 10 }}
          >
            <XAxis
              type="number"
              domain={[0, 100]}
              tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }}
              axisLine={{ stroke: 'var(--border-subtle)' }}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fill: 'var(--text-secondary)', fontSize: 11, fontFamily: 'var(--font-mono)' }}
              axisLine={false}
              tickLine={false}
              width={120}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'var(--bg-tertiary)' }} />
            <Bar
              dataKey="successRate"
              radius={[0, 6, 6, 0]}
              barSize={24}
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[entry.type]}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className={styles.legend}>
        {Object.entries(COLORS).map(([type, color]) => (
          <div key={type} className={styles.legendItem}>
            <span className={styles.legendDot} style={{ background: color }} />
            <span className={styles.legendLabel}>{type}</span>
          </div>
        ))}
      </div>
    </motion.div>
  )
}

export default ModelComparison
