export default function Footer() {
  return (
    <footer
      className="section"
      style={{
        borderTop: "1px solid var(--border)",
        paddingTop: "48px",
      }}
    >
      <div className="grid-2">
        <div>
          <h2 className="section-heading" style={{ fontSize: "22px" }}>
            Will Tibo Reset
          </h2>
          <p style={{ color: "var(--muted)", fontSize: "14px", maxWidth: "420px" }}>
            An open-source, AI-powered prediction dashboard that forecasts
            ChatGPT and Codex usage-limit resets. Updated every 20 minutes from
            public signals and historical reset data.
          </p>
        </div>
        <div>
          <h3
            style={{
              margin: "0 0 16px",
              fontSize: "14px",
              fontWeight: 600,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            Links
          </h3>
          <ul
            style={{
              listStyle: "none",
              margin: 0,
              padding: 0,
              fontSize: "14px",
              lineHeight: 2,
            }}
          >
            <li>
              <a
                href="https://github.com/EvanProgramming/willtiboreset"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub Repository
              </a>
            </li>
            <li>
              <a
                href="https://x.com/thsottiaux"
                target="_blank"
                rel="noopener noreferrer"
              >
                Tibo on X
              </a>
            </li>
            <li>
              <a href="/sitemap.xml">XML Sitemap</a>
            </li>
          </ul>
        </div>
      </div>
      <div
        className="muted"
        style={{
          marginTop: "48px",
          fontSize: "13px",
          textAlign: "center",
        }}
      >
        &copy; {new Date().getFullYear()} Will Tibo Reset. Not affiliated with
        OpenAI. Data is for informational purposes only.
      </div>
    </footer>
  );
}
