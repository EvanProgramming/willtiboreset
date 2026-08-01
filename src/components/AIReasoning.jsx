function FactorRow({ factor, impact, score }) {
  const scoreValue = Math.min(Math.max(score || 0, 0), 1);

  return (
    <div style={{ marginBottom: "24px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: "8px",
          gap: "16px",
        }}
      >
        <span style={{ fontSize: "15px" }}>{factor}</span>
        <span className="mono accent" style={{ fontSize: "14px", whiteSpace: "nowrap" }}>
          {impact}
        </span>
      </div>
      <div className="indicator">
        <div className="indicator-bar" style={{ width: `${scoreValue * 100}%` }}></div>
      </div>
    </div>
  );
}

export default function AIReasoning({ mainFactors, reasons }) {
  const factors = Array.isArray(mainFactors) ? mainFactors : [];
  const reasonList = Array.isArray(reasons) ? reasons : [];

  return (
    <section className="section">
      <div className="section-title">Why AI predicts this?</div>
      <h2 className="section-heading">Main factors</h2>
      <div className="grid-2">
        <div>
          {factors.length === 0 ? (
            <div className="empty">No factor data available.</div>
          ) : (
            factors.map((item, index) => (
              <FactorRow
                key={index}
                factor={item.factor}
                impact={item.impact}
                score={item.score}
              />
            ))
          )}
        </div>
        <div className="card">
          <div className="card-title">Model reasoning</div>
          {reasonList.length === 0 ? (
            <div className="empty">No reasoning available.</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: "18px", color: "var(--muted)" }}>
              {reasonList.map((reason, index) => (
                <li key={index} style={{ marginBottom: "10px" }}>
                  {reason}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}
