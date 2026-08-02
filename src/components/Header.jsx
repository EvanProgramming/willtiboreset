export default function Header({ updatedAt, alertEnabled, onToggleAlert }) {
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
        paddingBottom: "32px",
      }}
    >
      <div style={{ flex: "1 1 280px", minWidth: 0 }}>
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
            margin: "0 0 12px",
            color: "var(--muted)",
            fontSize: "15px",
            maxWidth: "420px",
          }}
        >
          AI-powered prediction system for ChatGPT/Codex reset events
        </p>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "16px",
            fontSize: "13px",
          }}
        >
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              fontWeight: 500,
            }}
          >
            <span className="status-dot"></span>
            System Online
          </span>
          <span className="mono muted">Last update: {formatted}</span>
        </div>
        <button
          onClick={onToggleAlert}
          style={{
            marginTop: "12px",
            padding: "6px 12px",
            fontSize: "13px",
            fontWeight: 500,
            color: alertEnabled ? "#0f0f0f" : "#e8e8e8",
            background: alertEnabled ? "#22c55e" : "rgba(255,255,255,0.08)",
            border: "1px solid",
            borderColor: alertEnabled ? "#22c55e" : "rgba(255,255,255,0.15)",
            borderRadius: "999px",
            cursor: "pointer",
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            transition: "background 0.2s ease, border-color 0.2s ease, color 0.2s ease",
          }}
          onMouseEnter={(e) => {
            if (!alertEnabled) {
              e.currentTarget.style.background = "rgba(255,255,255,0.14)";
            }
          }}
          onMouseLeave={(e) => {
            if (!alertEnabled) {
              e.currentTarget.style.background = "rgba(255,255,255,0.08)";
            }
          }}
        >
          <span style={{ fontSize: "14px" }}>{alertEnabled ? "🔔" : "🔕"}</span>
          {alertEnabled
            ? "Tibo Alarm is ON — I'll scream if reset looks likely"
            : "Enable Tibo Alarm (burn your Codex credits in time)"}
        </button>
      </div>
      <div style={{ flex: "0 0 auto" }}>
        <img
          src="/HOnd3JraIAAhSeg.jpeg"
          alt="Saint Tibo of OpenAI — Patron of safe token resets"
          style={{
            display: "block",
            maxWidth: "220px",
            maxHeight: "280px",
            width: "auto",
            height: "auto",
            borderRadius: "12px",
            objectFit: "contain",
            boxShadow: "0 4px 24px rgba(0,0,0,0.28)",
          }}
        />
      </div>
    </header>
  );
}
