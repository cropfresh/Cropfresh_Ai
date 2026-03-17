import { createDuplexApp } from "./duplex/app.js";

const duplexApp = createDuplexApp();

window.CropFreshDuplex = duplexApp;

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => duplexApp.init(), {
    once: true,
  });
} else {
  duplexApp.init();
}
