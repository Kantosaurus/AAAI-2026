import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import styles from './MetricCard.module.css'

function MetricCard({
  label,
  value,
  suffix = '',
  decimals = 0,
  icon: Icon,
  accentColor = 'var(--text-primary)',
  delay = 0
}) {
  const [displayValue, setDisplayValue] = useState(0)

  // Animate number counting up
  useEffect(() => {
    const duration = 1500
    const startTime = Date.now()
    const startValue = 0
    const endValue = value

    const animate = () => {
      const elapsed = Date.now() - startTime
      const progress = Math.min(elapsed / duration, 1)

      // Easing function (ease-out-cubic)
      const eased = 1 - Math.pow(1 - progress, 3)
      const current = startValue + (endValue - startValue) * eased

      setDisplayValue(current)

      if (progress < 1) {
        requestAnimationFrame(animate)
      }
    }

    const timer = setTimeout(() => {
      requestAnimationFrame(animate)
    }, delay * 1000)

    return () => clearTimeout(timer)
  }, [value, delay])

  const formatValue = (val) => {
    if (val >= 1000000) {
      return (val / 1000000).toFixed(1) + 'M'
    }
    if (val >= 1000) {
      return (val / 1000).toFixed(1) + 'K'
    }
    return val.toFixed(decimals)
  }

  return (
    <motion.div
      className={styles.card}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      style={{ '--accent-color': accentColor }}
    >
      <div className={styles.iconContainer}>
        {Icon && <Icon className={styles.icon} />}
      </div>

      <div className={styles.content}>
        <span className={styles.value}>
          {formatValue(displayValue)}
          {suffix && <span className={styles.suffix}>{suffix}</span>}
        </span>
        <span className={styles.label}>{label}</span>
      </div>

      {/* Accent line */}
      <div className={styles.accentLine} />
    </motion.div>
  )
}

export default MetricCard
