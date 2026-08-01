export default function Header({ updatedAt }) {
  const formatted = updatedAt
    ? new Date(updatedAt).toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        timeZoneName: "short",
      })
    : "—";

  return (
    <header
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        flexWrap: "wrap",
        gap: "24px",
        paddingBottom: "48px",
      }}
    >
      <div>
        <h1
          style={{
            margin: "0 0 8px",
            fontSize: "32px",
            fontWeight: 600,
            letterSpacing: "-0.02em",
          }}
        >
          Will Tibo Reset
        </h1>
        <p
          style={{
            margin: 0,
            color: "var(--muted)",
            fontSize: "15px",
            maxWidth: "420px",
          }}
        >
          AI-powered prediction system for ChatGPT/Codex reset events
        </p>
      </div>
      <div style={{ textAlign: "right" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            fontSize: "14px",
            fontWeight: 500,
            marginBottom: "8px",
          }}
        >
          <span className="status-dot"></span>
          System Online
        </div>
        <div className="mono muted" style={{ fontSize: "13px" }}>
          Last update: {formatted}
        </div>
      </div>
    </header>
  );
}
