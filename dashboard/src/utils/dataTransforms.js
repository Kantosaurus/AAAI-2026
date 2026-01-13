/**
 * Transform raw pilot results JSON to dashboard-friendly format
 * @param {Object} pilotResults - Raw results from run_pilot.py
 * @returns {Object} Transformed data for dashboard components
 */
export function transformResults(pilotResults) {
  const { metadata, runs } = pilotResults

  // Calculate summary statistics
  let totalPrompts = 0
  let totalSuccess = 0
  let totalErrors = 0
  let totalTokens = 0
  let totalTime = 0

  const byModel = []
  const categoryMap = new Map()
  const timeSeries = []

  for (const run of runs) {
    const modelConfig = run.model_config
    const results = run.results || []

    let modelSuccess = 0
    let modelErrors = 0
    let modelTokens = 0
    let modelTime = 0
    let modelRetries = 0

    for (const result of results) {
      totalPrompts++

      if (result.error) {
        totalErrors++
        modelErrors++
      } else {
        totalSuccess++
        modelSuccess++
      }

      const tokens = result.tokens_used?.total || 0
      totalTokens += tokens
      modelTokens += tokens

      const elapsed = result.elapsed_seconds || 0
      totalTime += elapsed
      modelTime += elapsed

      modelRetries += result.retry_count || 0

      // Category breakdown
      const category = result.prompt_category || 'unknown'
      if (!categoryMap.has(category)) {
        categoryMap.set(category, { count: 0, success: 0, errors: 0 })
      }
      const cat = categoryMap.get(category)
      cat.count++
      if (result.error) {
        cat.errors++
      } else {
        cat.success++
      }

      // Time series data point
      timeSeries.push({
        timestamp: result.timestamp,
        model: modelConfig.name,
        elapsed: result.elapsed_seconds,
        isSynthetic: result.is_synthetic_probe
      })
    }

    byModel.push({
      name: modelConfig.name,
      type: modelConfig.type,
      temperature: modelConfig.temperature || 0,
      successCount: modelSuccess,
      errorCount: modelErrors,
      totalTokens: modelTokens,
      avgTime: results.length > 0 ? modelTime / results.length : 0,
      totalRetries: modelRetries,
      avgInputTokens: Math.round(modelTokens * 0.3 / Math.max(results.length, 1)),
      avgOutputTokens: Math.round(modelTokens * 0.7 / Math.max(results.length, 1))
    })
  }

  // Convert category map to array
  const byCategory = Array.from(categoryMap.entries()).map(([category, stats]) => ({
    category,
    count: stats.count,
    successRate: stats.count > 0 ? ((stats.success / stats.count) * 100).toFixed(1) : 0
  }))

  // Sort categories by count
  byCategory.sort((a, b) => b.count - a.count)

  return {
    metadata: {
      ...metadata,
      totalPrompts
    },
    summary: {
      totalPrompts,
      successRate: totalPrompts > 0 ? (totalSuccess / totalPrompts) * 100 : 0,
      avgResponseTime: totalPrompts > 0 ? totalTime / totalPrompts : 0,
      totalTokens
    },
    byModel,
    byCategory,
    timeSeries
  }
}

/**
 * Format large numbers with K/M suffix
 * @param {number} num - Number to format
 * @returns {string} Formatted string
 */
export function formatNumber(num) {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M'
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K'
  }
  return num.toString()
}

/**
 * Format seconds to human readable time
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string
 */
export function formatTime(seconds) {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`
  }
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return `${minutes}m ${remainingSeconds.toFixed(0)}s`
}
