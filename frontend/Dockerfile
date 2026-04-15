FROM node:22-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
# Empty for Vite: the app calls same-origin `/api/...`. Nginx substitutes ${VITE_API_BASE_URL} at container start.
RUN VITE_API_BASE_URL= npm run build

FROM nginx:alpine

COPY docker-entrypoint-frontend.sh /docker-entrypoint-frontend.sh
RUN chmod +x /docker-entrypoint-frontend.sh

COPY nginx/templates/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

ENV PORT=8080
EXPOSE 8080

ENTRYPOINT ["/docker-entrypoint-frontend.sh"]
CMD ["nginx", "-g", "daemon off;"]
