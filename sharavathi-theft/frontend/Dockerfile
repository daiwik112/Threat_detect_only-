# Web image (Vite dev server with /api + /ws proxy to VITE_API_URL)
FROM node:20-slim
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
EXPOSE 5173
CMD ["npx", "vite", "--host", "0.0.0.0", "--port", "5173"]
