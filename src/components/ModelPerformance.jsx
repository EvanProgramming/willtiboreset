function formatNumber(value, decimals = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "N/A";
  }
  return value.toFixed(decimals);
}

function MetricCard({ label, value, suffix = "" }) {
  return (
    <div className="card">
      <div className="card-title">{label}</div>
      <div
        className="mono"
        style={{
          fontSize: "32px",
          color: "var(--accent)",
          lineHeight: 1,
        }}
      >
        {value}
        {suffix}
      </div>
    </div>
  );
}

function CalibrationBars({ bins }) {
  if (!Array.isArray(bins) || bins.length === 0) {
    return <div className="empty">No calibration data yet.</div>;
  }

  const maxCount = Math.max(...bins.map((b) => b.count || 0), 1);

  return (
    <div style={{ marginTop: "24px" }}>
      <div className="card-title" style={{ marginBottom: "12px" }}>
        Calibration bins (24h)
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(60px, 1fr))",
          gap: "8px",
          alignItems: "end",
          height: "120px",
        }}
      >
        {bins.map((bin, index) => {
          const heightPct = ((bin.count || 0) / maxCount) * 100;
          return (
            <div key={index} style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
              <div
                style={{
                  width: "100%",
                  height: `${Math.max(heightPct, 4)}%`,
                  backgroundColor: bin.count > 0 ? "var(--accent)" : "rgba(0, 240, 255, 0.15)",
                  minHeight: bin.count > 0 ? "4px" : "4px",
                }}
              />
              <span
                className="mono muted"
                style={{ fontSize: "10px", marginTop: "6px" }}
              >
                {Math.round(bin.bin_start * 100)}-{Math.round(bin.bin_end * 100)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function ModelPerformance({ performance }) {
  if (!performance) {
    return (
      <section className="section">
        <div className="section-title">Model Performance</div>
        <h2 className="section-heading">Calibration & accuracy</h2>
        <div className="empty">Performance data not available.</div>
      </section>
    );
  }

  const horizon24 = performance.horizons?.find((h) => h.horizon_hours === 24);

  return (
    <section className="section">
      <div className="section-title">Model Performance</div>
      <h2 className="section-heading">Calibration & accuracy</h2>
      <div className="grid-3">
        <MetricCard label="Total predictions" value={performance.total_predictions || 0} />
        <MetricCard
          label="Resolved predictions"
          value={performance.resolved_predictions || 0}
        />
        <MetricCard
          label="Overall accuracy"
          value={formatNumber(performance.overall_accuracy, 3)}
        />
        <MetricCard
          label="Overall Brier score"
          value={formatNumber(performance.overall_brier_score, 3)}
        />
        <MetricCard
          label="24h accuracy"
          value={formatNumber(horizon24?.accuracy, 3)}
        />
        <MetricCard
          label="24h calibration error"
          value={formatNumber(horizon24?.calibration_error, 3)}
        />
      </div>
      <CalibrationBars bins={horizon24?.bins} />
    </section>
  );
}
