/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        surface: { 0: "var(--surface-0)", 1: "var(--surface-1)", 2: "var(--surface-2)", 3: "var(--surface-3)" },
        ink: { 1: "var(--text-primary)", 2: "var(--text-secondary)", 3: "var(--text-muted)" },
        line: "var(--border)",
        accent: "var(--accent)",
        good: "var(--status-good)", warning: "var(--status-warning)",
        serious: "var(--status-serious)", critical: "var(--status-critical)",
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "Segoe UI", "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "Cascadia Mono", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
