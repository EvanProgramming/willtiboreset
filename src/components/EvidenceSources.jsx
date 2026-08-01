function classifySource(source) {
  if (!source) return { category: "Other", strength: 0.3 };
  const lower = String(source).toLowerCase();
  if (lower.startsWith("tibo")) return { category: "Tibo", strength: 1.0 };
  if (lower.startsWith("openai")) return { category: "OpenAI", strength: 0.9 };
  if (lower.startsWith("community")) return { category: "Community", strength: 0.5 };
  return { category: "Other", strength: 0.3 };
}

function strengthLabel(score) {
  if (score >= 0.9) return "High";
  if (score >= 0.6) return "Medium";
  return "Low";
}

function truncate(text, maxLength = 120) {
  if (!text) return "";
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength).trim() + "…";
}

function formatTime(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function EvidenceSources({ tweets }) {
  const list = Array.isArray(tweets) ? tweets : [];

  const latestByCategory = list.reduce((acc, tweet) => {
    const { category, strength } = classifySource(tweet.source);
    const existing = acc[category];
    if (!existing || new Date(tweet.timestamp) > new Date(existing.timestamp)) {
      acc[category] = { ...tweet, strength };
    }
    return acc;
  }, {});

  const categories = ["Tibo", "OpenAI", "Community"];
  const hasAny = categories.some((cat) => latestByCategory[cat]);

  return (
    <section className="section">
      <div className="section-title">Evidence Sources</div>
      <h2 className="section-heading">Current signals</h2>
      {hasAny ? (
        <div className="grid-3">
          {categories.map((category) => {
            const item = latestByCategory[category];
            if (!item) {
              return (
                <div key={category} className="card">
                  <div className="card-title">{category}</div>
                  <div className="empty">No recent signal</div>
                </div>
              );
            }
            return (
              <div key={category} className="card">
                <div className="card-title">{category}</div>
                <div style={{ marginBottom: "12px" }}>
                  <span className="mono accent" style={{ fontSize: "14px" }}>
                    @{item.author}
                  </span>
                  <span className="muted" style={{ fontSize: "13px", marginLeft: "12px" }}>
                    {formatTime(item.timestamp)}
                  </span>
                </div>
                <p
                  style={{
                    margin: "0 0 16px",
                    fontSize: "14px",
                    color: "var(--muted)",
                    lineHeight: 1.5,
                  }}
                >
                  {truncate(item.text)}
                </p>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    fontSize: "13px",
                  }}
                >
                  <span className="muted">Signal strength</span>
                  <span className="mono accent">{strengthLabel(item.strength)}</span>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="empty">No signal sources available.</div>
      )}
    </section>
  );
}
