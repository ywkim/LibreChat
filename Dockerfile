# Base node image
FROM node:19-alpine AS node

COPY . /app
WORKDIR /app

# Install call deps - Install curl for health check
RUN apk --no-cache add curl && \
    # We want to inherit env from the container, not the file
    # This will preserve any existing env file if it's already in souce
    # otherwise it will create a new one
    touch .env && \
    # Build deps in seperate
    npm ci

RUN npm install -g ts-node

# React client build
ENV NODE_OPTIONS="--max-old-space-size=2048"
RUN npm run frontend

# Node API setup
ENV HOST=0.0.0.0
CMD ["npm", "run", "backend:dev"]

# Optional: for client with nginx routing
# FROM nginx:stable-alpine AS nginx-client
# WORKDIR /usr/share/nginx/html
# COPY --from=node /app/client/dist /usr/share/nginx/html
# COPY client/nginx.conf /etc/nginx/conf.d/default.conf
# ENTRYPOINT ["nginx", "-g", "daemon off;"]
