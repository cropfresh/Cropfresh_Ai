import { loadConfig } from "./config.js";
import { buildApp } from "./app.js";
import { SharedRingBuffer, secondsToPcm16Bytes } from "./audio/ring-buffer.js";

const config = loadConfig();
const { app } = buildApp(config);

// The ring buffer is ready at process start so later worker-thread audio handoff can plug in cleanly.
const safetyBuffer = new SharedRingBuffer(secondsToPcm16Bytes(5));

app.listen(config.port, config.host, () => {
  // TODO: publish buffer utilization once the gateway starts relaying live PCM frames.
  console.info(
    `[voice-gateway] listening on http://${config.host}:${config.port} ` +
      `with ${safetyBuffer.snapshot().capacityBytes} bytes of ring-buffer capacity`,
  );
});
