FROM node:22-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
# Empty for Vite: the app calls same-origin `/api/...`. Nginx substitutes ${VITE_API_BASE_URL} at container start.
RUN VITE_API_BASE_URL= npm run build

FROM nginx:alpine

COPY nginx/templates/default.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
