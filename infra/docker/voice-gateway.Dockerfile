FROM node:24-alpine

WORKDIR /app

COPY services/voice-gateway/package.json ./package.json
COPY services/voice-gateway/tsconfig.json ./tsconfig.json
RUN npm install

COPY services/voice-gateway/src ./src
RUN npm run build

EXPOSE 3101

CMD ["npm", "run", "start"]
