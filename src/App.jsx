import { useJsonData } from "./hooks/useJsonData";
import Header from "./components/Header";
import PredictionHero from "./components/PredictionHero";
import AIReasoning from "./components/AIReasoning";
import EvidenceSources from "./components/EvidenceSources";
import TimelineChart from "./components/TimelineChart";
import ModelPerformance from "./components/ModelPerformance";
import AboutModel from "./components/AboutModel";
import Footer from "./components/Footer";

export default function App() {
  const base = import.meta.env.BASE_URL || "/";

  const prediction = useJsonData(`${base}data/prediction.json`);
  const performance = useJsonData(`${base}data/model_performance.json`);
  const history = useJsonData(`${base}data/prediction_history.json`);
  const tweets = useJsonData(`${base}data/tweets.json`);

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
      <Header updatedAt={prediction.data?.updated_at} />
      <PredictionHero prediction={prediction.data?.prediction} />
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
