# Use an official Node runtime as a parent image
FROM node:19-alpine as build
# Set the working directory to /app
WORKDIR /app
# Copy the package.json and package-lock.json (or yarn.lock if you use Yarn) to the container
COPY frontend/package*.json ./
# Install dependencies
RUN npm install
# Copy the rest of the application code to the container, excluding node_modules
COPY frontend ./
# Build the React app
RUN npm run build

# Use an official Nginx runtime as a parent image
FROM nginx:1.21.0-alpine
# Copy the nginx.conf to the container
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
# Copy the React app build files from the build stage to the container
COPY --from=build /app/dist /usr/share/nginx/html
# Expose port 80 for Nginx
EXPOSE 80
# Start Nginx when the container starts
CMD ["nginx", "-g", "daemon off;"]