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

function formatCountdown(hoursUntil) {
  if (hoursUntil == null || isNaN(hoursUntil)) return null;
  const totalMinutes = Math.max(0, Math.floor(hoursUntil * 60));
  const days = Math.floor(totalMinutes / (24 * 60));
  const hours = Math.floor((totalMinutes % (24 * 60)) / 60);
  const mins = totalMinutes % 60;
  if (days > 0) return `${days}d ${hours}h ${mins}m`;
  return `${hours}h ${mins}m`;
}

function formatTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "—";
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  });
}

function NextResetCard({ nextReset }) {
  if (!nextReset?.expected_time) return null;
  const countdown = formatCountdown(nextReset.hours_until);
  const overdue = nextReset.status === "overdue";

  return (
    <div className="card next-reset-card" style={{ marginTop: 24 }}>
      <div className="card-title">
        {overdue ? "Expected reset window passed" : "Next estimated reset"}
      </div>
      <div
        className="mono"
        style={{
          fontSize: "clamp(28px, 4vw, 40px)",
          fontWeight: 500,
          color: overdue ? "var(--warn)" : "var(--text)",
          lineHeight: 1.1,
        }}
      >
        {formatTime(nextReset.expected_time)}
      </div>
      {countdown && (
        <div className="muted" style={{ marginTop: 8, fontSize: 14 }}>
          {overdue
            ? "No new reset confirmed yet — next estimated cycle "
            : "Countdown to expected reset — "}
          {countdown}
        </div>
      )}
    </div>
  );
}

export default function PredictionHero({ prediction, nextReset }) {
  return (
    <section className="section">
      <div className="section-title">Next Reset Probability</div>
      <h2 className="section-heading">When will Tibo reset?</h2>
      <div className="grid-3">
        <ProbabilityCard label="Within 5 hours" probability={prediction?.within_5h} />
        <ProbabilityCard label="Within 24 hours" probability={prediction?.within_24h} />
        <ProbabilityCard label="Within 48 hours" probability={prediction?.within_48h} />
      </div>
      <NextResetCard nextReset={nextReset} />
    </section>
  );
}
