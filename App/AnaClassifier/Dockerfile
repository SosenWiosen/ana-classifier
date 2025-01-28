# Use the official nginx image to serve static files
FROM nginx:alpine

# Remove the default Nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy the exported static files from the dist directory to Nginx's document root
COPY dist /usr/share/nginx/html

# Copy a custom Nginx configuration file
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose is more of a documentation feature; it doesn't actually publish the port.
EXPOSE 5000

# Start Nginx when the container starts
CMD ["nginx", "-g", "daemon off;"]