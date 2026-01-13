import { useState, useEffect } from 'react'
import { transformResults } from '../utils/dataTransforms'

/**
 * Hook for loading and transforming pilot results data
 *
 * @param {string} source - URL or file path to fetch results from (required)
 * @returns {{ data: Object|null, loading: boolean, error: Error|null, reload: Function }}
 */
export function useResults(source) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const loadData = async () => {
    if (!source) {
      setLoading(false)
      setError(new Error('No data source provided. Please specify a results file path or URL.'))
      return
    }

    try {
      setLoading(true)
      setError(null)

      const response = await fetch(source)
      if (!response.ok) {
        throw new Error(`Failed to fetch results: ${response.status} ${response.statusText}`)
      }

      const rawData = await response.json()

      // Validate that this looks like pilot results
      if (!rawData.metadata || !rawData.runs) {
        throw new Error('Invalid results format. Expected pilot results with metadata and runs.')
      }

      // Transform raw pilot results to dashboard format
      const transformed = transformResults(rawData)
      setData(transformed)
    } catch (err) {
      console.error('Error loading results:', err)
      setError(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [source])

  return { data, loading, error, reload: loadData }
}

export default useResults
