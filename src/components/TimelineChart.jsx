const WIDTH = 800;
const HEIGHT = 300;
const PADDING = { top: 20, right: 30, bottom: 50, left: 50 };

function formatDate(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

export default function TimelineChart({ history }) {
  const list = Array.isArray(history) ? history : [];
  const sorted = [...list]
    .filter((h) => h?.prediction_time && h.prediction?.within_24h !== undefined)
    .sort((a, b) => new Date(a.prediction_time) - new Date(b.prediction_time));

  if (sorted.length < 2) {
    return (
      <section className="section">
        <div className="section-title">Prediction Timeline</div>
        <h2 className="section-heading">24-hour probability history</h2>
        <div className="empty">Not enough history to draw a chart.</div>
      </section>
    );
  }

  const minTime = new Date(sorted[0].prediction_time).getTime();
  const maxTime = new Date(sorted[sorted.length - 1].prediction_time).getTime();
  const timeRange = Math.max(maxTime - minTime, 1);

  const innerWidth = WIDTH - PADDING.left - PADDING.right;
  const innerHeight = HEIGHT - PADDING.top - PADDING.bottom;

  const points = sorted.map((entry) => {
    const t = new Date(entry.prediction_time).getTime();
    const x = PADDING.left + ((t - minTime) / timeRange) * innerWidth;
    const y =
      PADDING.top + innerHeight - entry.prediction.within_24h * innerHeight;
    return { x, y, entry };
  });

  const pathD = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`)
    .join(" ");

  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const xTicks = [0, Math.floor((sorted.length - 1) / 2), sorted.length - 1];

  return (
    <section className="section">
      <div className="section-title">Prediction Timeline</div>
      <h2 className="section-heading">24-hour probability history</h2>
      <div style={{ overflowX: "auto" }}>
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          style={{ width: "100%", minWidth: "600px", height: "auto" }}
          role="img"
          aria-label="24-hour reset probability over time"
        >
          {/* Y axis ticks and labels */}
          {yTicks.map((tick) => {
            const y = PADDING.top + innerHeight - tick * innerHeight;
            return (
              <g key={tick}>
                <line
                  x1={PADDING.left}
                  y1={y}
                  x2={WIDTH - PADDING.right}
                  y2={y}
                  stroke="rgba(255,255,255,0.1)"
                  strokeWidth={1}
                />
                <text
                  x={PADDING.left - 10}
                  y={y + 4}
                  textAnchor="end"
                  fill="#888888"
                  fontSize="12"
                  fontFamily="var(--font-mono)"
                >
                  {Math.round(tick * 100)}%
                </text>
              </g>
            );
          })}

          {/* X axis labels */}
          {xTicks.map((index) => {
            const p = points[index];
            if (!p) return null;
            return (
              <text
                key={index}
                x={p.x}
                y={HEIGHT - 15}
                textAnchor="middle"
                fill="#888888"
                fontSize="12"
                fontFamily="var(--font-mono)"
              >
                {formatDate(p.entry.prediction_time)}
              </text>
            );
          })}

          {/* Data line */}
          <path
            d={pathD}
            fill="none"
            stroke="#00f0ff"
            strokeWidth={2}
          />

          {/* Data points */}
          {points.map((p, i) => (
            <circle
              key={i}
              cx={p.x}
              cy={p.y}
              r={4}
              fill="#000000"
              stroke="#00f0ff"
              strokeWidth={2}
            />
          ))}
        </svg>
      </div>
    </section>
  );
}
