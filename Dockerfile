# Use Node.js base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the project files
COPY . .

# Expose default dev server port (e.g., Vite uses 5173, CRA uses 3000)
EXPOSE 5173

# Start the dev server
CMD ["npm", "run", "dev"]