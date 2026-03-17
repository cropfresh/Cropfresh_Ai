export function bindDuplexControls({
  isRecording,
  isSocketOpen,
  onConnectToggle,
  onFeedbackDown,
  onFeedbackUp,
  onLanguageChange,
  onOrbToggle,
  query,
}) {
  query("#voiceOrb")?.addEventListener("click", () => {
    onOrbToggle(isRecording());
  });
  query("#btnConnectToggle")?.addEventListener("click", () => {
    onConnectToggle(isSocketOpen());
  });
  query("#langSelect")?.addEventListener("change", (event) => {
    onLanguageChange(event.target.value);
  });
  query("#btnFeedbackUp")?.addEventListener("click", onFeedbackUp);
  query("#btnFeedbackDown")?.addEventListener("click", onFeedbackDown);
}
