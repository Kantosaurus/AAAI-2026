import { motion } from 'framer-motion'
import { Brain, Calendar, Database } from 'lucide-react'
import styles from './Header.module.css'

function Header({ metadata }) {
  const formatDate = (isoString) => {
    if (!isoString) return 'N/A'
    return new Date(isoString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <header className={styles.header}>
      {/* Diagonal stripe accent */}
      <div className={styles.diagonalStripe} />

      <div className={styles.content}>
        <motion.div
          className={styles.titleSection}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className={styles.logoContainer}>
            <Brain className={styles.logoIcon} />
          </div>
          <div className={styles.titleText}>
            <h1 className={styles.title}>
              <span className={styles.titleMain}>Hallucination</span>
              <span className={styles.titleAccent}>Research</span>
            </h1>
            <p className={styles.subtitle}>
              LLM Security Analysis Dashboard
            </p>
          </div>
        </motion.div>

        <motion.div
          className={styles.metaSection}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className={styles.metaItem}>
            <Calendar className={styles.metaIcon} />
            <div className={styles.metaContent}>
              <span className={styles.metaLabel}>Run Date</span>
              <span className={styles.metaValue}>{formatDate(metadata?.start_time)}</span>
            </div>
          </div>

          <div className={styles.metaDivider} />

          <div className={styles.metaItem}>
            <Database className={styles.metaIcon} />
            <div className={styles.metaContent}>
              <span className={styles.metaLabel}>Dataset</span>
              <span className={styles.metaValue}>
                {metadata?.totalPrompts || 0} prompts
              </span>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Bottom gradient line */}
      <div className={styles.bottomLine} />
    </header>
  )
}

export default Header
