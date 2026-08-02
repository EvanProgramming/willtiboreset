import { useEffect, useRef, useState } from "react";

const STORAGE_KEY = "codexAlertEnabled";

function requestNotificationPermission() {
  if (!("Notification" in window)) return Promise.resolve("denied");
  return Notification.requestPermission();
}

function sendNotification(title, body) {
  if (Notification.permission === "granted") {
    new Notification(title, {
      body,
      icon: "/favicon.png",
      badge: "/favicon.png",
      tag: "codex-reset-alert",
      renotify: true,
    });
  }
}

export default function useCodexAlert(prediction) {
  const [enabled, setEnabled] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) === "true";
    } catch {
      return false;
    }
  });
  const lastStateRef = useRef({ fired5h: false, fired24h: false });

  useEffect(() => {
    if (!enabled || !prediction) return;

    const p5h = prediction.prediction?.within_5h ?? 0;
    const p24h = prediction.prediction?.within_24h ?? 0;

    if (p5h > 0.5) {
      if (!lastStateRef.current.fired5h) {
        sendNotification(
          "🚨 Code Red: Reset Imminent!",
          "5-hour reset chance just passed 50%. Stop reading this and go burn your Codex credits before Tibo hits the button!"
        );
        lastStateRef.current.fired5h = true;
      }
    } else {
      lastStateRef.current.fired5h = false;
    }

    if (p24h > 0.6) {
      if (!lastStateRef.current.fired24h) {
        sendNotification(
          "⚠️ Tibo is Circling...",
          "24-hour reset probability is above 60%. Your Codex tokens are in danger. Use them or lose them, legend."
        );
        lastStateRef.current.fired24h = true;
      }
    } else {
      lastStateRef.current.fired24h = false;
    }
  }, [enabled, prediction]);

  const toggle = async () => {
    if (!enabled) {
      const permission = await requestNotificationPermission();
      if (permission === "granted") {
        setEnabled(true);
        localStorage.setItem(STORAGE_KEY, "true");
        sendNotification(
          "🔔 Tibo Alarm Armed",
          "Sit back. I'll scream at you when a reset looks likely."
        );
      } else {
        sendNotification("Permission denied", "I can't warn you if you don't let me.");
      }
    } else {
      setEnabled(false);
      localStorage.removeItem(STORAGE_KEY);
    }
  };

  return { enabled, toggle };
}
