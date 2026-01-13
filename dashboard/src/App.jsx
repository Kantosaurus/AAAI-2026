import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import styles from './App.module.css'

import Header from './components/Header/Header'
import MetricCard from './components/MetricCard/MetricCard'
import ModelComparison from './components/ModelComparison/ModelComparison'
import CategoryBreakdown from './components/CategoryBreakdown/CategoryBreakdown'
import TokenUsage from './components/TokenUsage/TokenUsage'
import ResponseTimes from './components/ResponseTimes/ResponseTimes'
import ErrorRates from './components/ErrorRates/ErrorRates'

import { useResults } from './hooks/useResults'
import { Activity, CheckCircle, Clock, Zap, Upload, FileJson } from 'lucide-react'

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
}

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: 'easeOut' }
  }
}

/**
 * Get results source from URL parameters or return null
 */
function getResultsSource() {
  const params = new URLSearchParams(window.location.search)
  return params.get('results') || null
}

function App() {
  const [resultsSource, setResultsSource] = useState(getResultsSource)
  const { data, loading, error, reload } = useResults(resultsSource)

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files?.[0]
    if (file) {
      const url = URL.createObjectURL(file)
      setResultsSource(url)
    }
  }

  // No data source provided - show upload interface
  if (!resultsSource) {
    return (
      <div className={styles.app}>
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>
            <FileJson size={48} />
          </div>
          <h2 className={styles.emptyTitle}>Load Pilot Results</h2>
          <p className={styles.emptyText}>
            Upload a pilot results JSON file or provide a URL to visualize your experiment data.
          </p>

          <div className={styles.uploadSection}>
            <label className={styles.uploadButton}>
              <Upload size={20} />
              <span>Upload Results File</span>
              <input
                type="file"
                accept=".json"
                onChange={handleFileUpload}
                className={styles.fileInput}
              />
            </label>
          </div>

          <div className={styles.urlSection}>
            <p className={styles.urlHint}>
              Or pass a URL via query parameter:
            </p>
            <code className={styles.urlExample}>
              ?results=/path/to/pilot_results.json
            </code>
          </div>
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className={styles.loadingScreen}>
        <div className={styles.loadingSpinner} />
        <p>Loading results...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className={styles.app}>
        <div className={styles.errorScreen}>
          <h2>Error Loading Results</h2>
          <p className={styles.errorMessage}>{error.message}</p>
          <button className={styles.retryButton} onClick={() => setResultsSource(null)}>
            Try Different File
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.app}>
      <Header metadata={data.metadata} />

      <motion.main
        className={styles.main}
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Metric Cards Row */}
        <motion.section className={styles.metricsRow} variants={itemVariants}>
          <MetricCard
            label="Total Prompts"
            value={data.summary.totalPrompts}
            icon={Activity}
            delay={0}
          />
          <MetricCard
            label="Success Rate"
            value={data.summary.successRate}
            suffix="%"
            icon={CheckCircle}
            accentColor="var(--accent-cyan)"
            delay={0.1}
          />
          <MetricCard
            label="Avg Response"
            value={data.summary.avgResponseTime}
            suffix="s"
            decimals={2}
            icon={Clock}
            delay={0.2}
          />
          <MetricCard
            label="Total Tokens"
            value={data.summary.totalTokens}
            icon={Zap}
            accentColor="var(--accent-yellow)"
            delay={0.3}
          />
        </motion.section>

        {/* Main Charts - Asymmetrical Grid */}
        <motion.section className={styles.chartsGrid} variants={itemVariants}>
          <div className={styles.chartLarge}>
            <ModelComparison data={data.byModel} />
          </div>
          <div className={styles.chartSmall}>
            <CategoryBreakdown data={data.byCategory} />
          </div>
        </motion.section>

        {/* Secondary Charts Row */}
        <motion.section className={styles.chartsGridAlt} variants={itemVariants}>
          <div className={styles.chartMedium}>
            <ErrorRates data={data.byModel} />
          </div>
          <div className={styles.chartWide}>
            <ResponseTimes data={data.timeSeries} models={data.byModel} />
          </div>
        </motion.section>

        {/* Full Width Chart */}
        <motion.section className={styles.fullWidthSection} variants={itemVariants}>
          <TokenUsage data={data.byModel} />
        </motion.section>
      </motion.main>

      {/* Footer */}
      <footer className={styles.footer}>
        <span className={styles.footerText}>
          LLM Hallucination Research Framework
        </span>
        <span className={styles.footerDivider}>|</span>
        <span className={styles.footerAccent}>
          {data.metadata.totalPrompts} prompts analyzed
        </span>
      </footer>
    </div>
  )
}

export default App
