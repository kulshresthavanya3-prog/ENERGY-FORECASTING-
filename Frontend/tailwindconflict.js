export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        navy: { 900: "#0f2027", 800: "#16213e", 700: "#1a1a2e" },
        cyan: { 400: "#00d4ff" },
      },
      fontFamily: { inter: ["Inter", "sans-serif"] },
    },
  },
  plugins: [],
};
