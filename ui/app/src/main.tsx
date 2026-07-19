import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { adoptTokenFromUrl } from "./lib/api";

// P7 task 4 (DEC-P7-2): adopt ?token=... into sessionStorage before first render
// so the live-demo URL authenticates the chat path (and the token leaves the bar).
adoptTokenFromUrl();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
