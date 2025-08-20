#!/bin/bash

# Build and deploy to GitHub Pages
echo "🚀 Building and deploying to GitHub Pages..."

# Build the project
echo "📦 Building the project..."
npm run build

# Deploy to GitHub Pages
echo "🌐 Deploying to GitHub Pages..."
npm run deploy

echo "✅ Deployment complete! Your site will be available at:"
echo "https://nikki-13.github.io/pneumonia_detection"
