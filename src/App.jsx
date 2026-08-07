import { useJsonData } from "./hooks/useJsonData";
import useCodexAlert from "./hooks/useCodexAlert";
import Header from "./components/Header";
import PredictionHero from "./components/PredictionHero";
import AIReasoning from "./components/AIReasoning";
import EvidenceSources from "./components/EvidenceSources";
import TimelineChart from "./components/TimelineChart";
import ModelPerformance from "./components/ModelPerformance";
import AboutModel from "./components/AboutModel";
import Footer from "./components/Footer";

// The GitHub Actions prediction workflow runs every 20 minutes.
// Poll every 5 minutes so the dashboard picks up new data quickly
// without generating excessive requests on GitHub Pages.
const REFRESH_INTERVAL_MS = 5 * 60 * 1000;

export default function App() {
  const base = import.meta.env.BASE_URL || "/";

  const prediction = useJsonData(`${base}data/prediction.json`, REFRESH_INTERVAL_MS);
  const performance = useJsonData(`${base}data/model_performance.json`, REFRESH_INTERVAL_MS);
  const history = useJsonData(`${base}data/prediction_history.json`, REFRESH_INTERVAL_MS);
  const tweets = useJsonData(`${base}data/tweets.json`, REFRESH_INTERVAL_MS);

  const { enabled: alertEnabled, toggle: toggleAlert } = useCodexAlert(prediction.data);

  const isLoading =
    prediction.loading || performance.loading || history.loading || tweets.loading;
  const hasError =
    prediction.error || performance.error || history.error || tweets.error;

  if (isLoading) {
    return (
      <div className="app">
        <div className="loading">Loading dashboard data...</div>
      </div>
    );
  }

  if (hasError) {
    return (
      <div className="app">
        <div className="error">Failed to load dashboard data: {hasError}</div>
      </div>
    );
  }

  return (
    <div className="app">
      <Header
        updatedAt={prediction.data?.updated_at}
        alertEnabled={alertEnabled}
        onToggleAlert={toggleAlert}
      />
      <PredictionHero
        prediction={prediction.data?.prediction}
        nextReset={prediction.data?.next_reset}
      />
      <AIReasoning
        mainFactors={prediction.data?.main_factors}
        reasons={prediction.data?.reasons}
      />
      <EvidenceSources tweets={tweets.data} />
      <TimelineChart history={history.data} />
      <ModelPerformance performance={performance.data} />
      <AboutModel />
      <Footer />
    </div>
  );
}
