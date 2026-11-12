FROM nginx:alpine
COPY index.html /usr/share/nginx/html/index.html
COPY data /usr/share/nginx/html/data
EXPOSE 80
