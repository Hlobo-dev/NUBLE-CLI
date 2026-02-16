# ============================================================
# ROKET Production Dockerfile
# Multi-stage build: Frontend (SvelteKit static) + Backend (Node.js)
# ============================================================

# ---- Stage 1: Build Frontend ----
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

# Install dependencies first (layer caching)
COPY nuble-frontend/package.json nuble-frontend/package-lock.json* ./
RUN npm ci --legacy-peer-deps

# Copy source and build
# Note: static/pyodide/ already contains pre-downloaded .whl files,
# so we skip the pyodide:fetch step and run vite build directly.
COPY nuble-frontend/ ./
ENV NODE_OPTIONS="--max-old-space-size=4096"
RUN npx vite build

# ---- Stage 2: Production Backend ----
FROM node:20-alpine AS production

# Security: run as non-root
RUN addgroup -g 1001 -S roket && \
    adduser -S roket -u 1001 -G roket

WORKDIR /app

# Install dependencies first (layer caching)
COPY nuble-backend/package.json nuble-backend/package-lock.json* ./
RUN npm ci --omit=dev --prefer-offline && npm cache clean --force

# Copy backend source
COPY nuble-backend/server.js ./
COPY nuble-backend/nova-sonic-client.js ./
COPY nuble-backend/static/ ./static/

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/build ./frontend-build

# Create data directory (writable by roket user)
RUN mkdir -p /app/data && chown -R roket:roket /app/data

# Environment defaults
ENV NODE_ENV=production \
    PORT=3000 \
    SERVE_FRONTEND=true \
    FRONTEND_PATH=/app/frontend-build

# Switch to non-root user
USER roket

# Expose port
EXPOSE 3000

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# Start server
CMD ["node", "server.js"]
