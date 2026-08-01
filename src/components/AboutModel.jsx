const steps = [
  "RSS Sources",
  "LLM Signal Analysis",
  "Bayesian Evidence Model",
  "Prediction",
  "Calibration",
];

export default function AboutModel() {
  return (
    <section className="section">
      <div className="section-title">About Model</div>
      <h2 className="section-heading">How it works</h2>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: "8px",
          padding: "24px 0",
        }}
      >
        {steps.map((step, index) => (
          <span key={step} style={{ display: "flex", alignItems: "center" }}>
            <span className="flow-step">{step}</span>
            {index < steps.length - 1 && (
              <span className="flow-arrow">→</span>
            )}
          </span>
        ))}
      </div>
      <p style={{ color: "var(--muted)", maxWidth: "720px", fontSize: "15px" }}>
        The model gathers public signals from RSS feeds, extracts structured
        scores via an LLM, then combines evidence using a Bayesian-inspired
        approach with historical reset intervals. Predictions are continuously
        calibrated against observed outcomes.
      </p>
    </section>
  );
}
