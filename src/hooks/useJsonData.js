import { useEffect, useRef, useState } from "react";

/**
 * Fetch JSON data from a URL with optional polling.
 *
 * @param {string} url - URL to fetch
 * @param {number|null} refreshInterval - Polling interval in milliseconds. Disabled when null or <= 0.
 * @returns {{ data: any, loading: boolean, error: string|null }}
 */
export function useJsonData(url, refreshInterval = null) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const refreshOnFocusRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    let intervalId = null;

    async function fetchData() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const json = await response.json();
        if (!cancelled) {
          setData(json);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Failed to load data");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchData();

    if (refreshInterval && refreshInterval > 0) {
      intervalId = window.setInterval(fetchData, refreshInterval);
    }

    // Refresh immediately when the tab becomes visible again, but only if
    // the hook has already been polling (i.e. the user left the tab open).
    function handleVisibilityChange() {
      if (
        !document.hidden &&
        refreshInterval &&
        refreshInterval > 0 &&
        refreshOnFocusRef.current
      ) {
        fetchData();
      }
      refreshOnFocusRef.current = true;
    }

    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      cancelled = true;
      if (intervalId) {
        window.clearInterval(intervalId);
      }
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [url, refreshInterval]);

  return { data, loading, error };
}
