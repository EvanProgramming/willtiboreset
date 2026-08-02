function ProbabilityCard({ label, probability }) {
  const pct = Math.round((probability || 0) * 100);

  return (
    <div className="card">
      <div className="card-title">{label}</div>
      <div
        className="mono"
        style={{
          fontSize: "clamp(48px, 8vw, 72px)",
          fontWeight: 500,
          color: "var(--accent)",
          lineHeight: 1,
        }}
      >
        {pct}%
      </div>
      <div className="indicator">
        <div className="indicator-bar" style={{ width: `${pct}%` }}></div>
      </div>
    </div>
  );
}

export default function PredictionHero({ prediction }) {
  return (
    <section className="section">
      <div className="section-title">Next Reset Probability</div>
      <h2 className="section-heading">When will Tibo reset?</h2>
      <div className="grid-3">
        <ProbabilityCard label="Within 5 hours" probability={prediction?.within_5h} />
        <ProbabilityCard label="Within 24 hours" probability={prediction?.within_24h} />
        <ProbabilityCard label="Within 48 hours" probability={prediction?.within_48h} />
      </div>
    </section>
  );
}
