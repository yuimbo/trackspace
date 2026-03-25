import type Htmx from "htmx.org";

declare global {
  interface Window {
    htmx: typeof Htmx;
  }
}

export {};
