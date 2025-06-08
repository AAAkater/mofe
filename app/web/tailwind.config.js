/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        "task-gray": "#f1f2f4",
        "task-hover": "#e2e4e9",
      },
    },
  },
  plugins: [],
};
