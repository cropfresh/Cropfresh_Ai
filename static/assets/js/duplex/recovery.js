const DEFAULT_RECOVERY_POLICY = {
  dead_peer_timeout_ms: 30000,
  ice_restart_enabled: false,
  network_change_recovery: true,
  reconnect_token_required: true,
  retry_backoff_ms: [1000, 2000, 4000, 8000, 16000],
};

function sanitizeRetryBackoff(delays) {
  if (!Array.isArray(delays)) {
    return DEFAULT_RECOVERY_POLICY.retry_backoff_ms;
  }

  const sanitized = delays.filter((value) => Number.isFinite(value) && value > 0);
  return sanitized.length > 0 ? sanitized : DEFAULT_RECOVERY_POLICY.retry_backoff_ms;
}

export function resolveRecoveryPolicy(bootstrap = {}) {
  const merged = {
    ...DEFAULT_RECOVERY_POLICY,
    ...(bootstrap.recovery || {}),
  };
  return {
    ...merged,
    retry_backoff_ms: sanitizeRetryBackoff(merged.retry_backoff_ms),
  };
}

export function getReconnectDelay(policy, attempt) {
  const delays = sanitizeRetryBackoff(policy?.retry_backoff_ms);
  return delays[Math.min(attempt, delays.length - 1)];
}

export function createHeartbeatWatchdog({ getTimeoutMs, onTimeout }) {
  let intervalId = null;
  let lastAckAt = 0;

  function markAck() {
    lastAckAt = Date.now();
  }

  function stop() {
    if (intervalId !== null) {
      clearInterval(intervalId);
      intervalId = null;
    }
  }

  function start() {
    stop();
    markAck();
    intervalId = setInterval(() => {
      if (Date.now() - lastAckAt > getTimeoutMs()) {
        onTimeout();
      }
    }, Math.max(1000, Math.floor(getTimeoutMs() / 3)));
  }

  return {
    markAck,
    start,
    stop,
  };
}
