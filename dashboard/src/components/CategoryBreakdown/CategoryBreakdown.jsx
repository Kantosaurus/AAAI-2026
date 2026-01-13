import { motion } from 'framer-motion'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import styles from './CategoryBreakdown.module.css'

const COLORS = ['#00f5d4', '#9b5de5', '#f9c74f', '#ff6b6b', '#4cc9f0']

function CategoryBreakdown({ data }) {
  const total = data.reduce((sum, d) => sum + d.count, 0)

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload
      return (
        <div className={styles.tooltip}>
          <p className={styles.tooltipTitle}>{d.category}</p>
          <p className={styles.tooltipValue}>{d.count} prompts</p>
          <p className={styles.tooltipPercent}>
            {((d.count / total) * 100).toFixed(1)}%
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
        <h3 className={styles.title}>Category Distribution</h3>
        <p className={styles.subtitle}>Prompt types analyzed</p>
      </div>

      <div className={styles.chartContainer}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={90}
              paddingAngle={3}
              dataKey="count"
            >
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                  stroke="var(--bg-card)"
                  strokeWidth={2}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>

        {/* Center stat */}
        <div className={styles.centerStat}>
          <span className={styles.centerValue}>{total}</span>
          <span className={styles.centerLabel}>Total</span>
        </div>
      </div>

      {/* Legend */}
      <div className={styles.legend}>
        {data.map((item, index) => (
          <div key={item.category} className={styles.legendItem}>
            <span
              className={styles.legendDot}
              style={{ background: COLORS[index % COLORS.length] }}
            />
            <span className={styles.legendLabel}>
              {item.category.replace(/_/g, ' ')}
            </span>
            <span className={styles.legendCount}>{item.count}</span>
          </div>
        ))}
      </div>
    </motion.div>
  )
}

export default CategoryBreakdown
