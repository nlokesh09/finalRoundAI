#!/bin/bash

# FinalRound AI Deployment Script for Railway
echo "🚀 Starting FinalRound AI deployment..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Please install it first:"
    echo "npm install -g @railway/cli"
    exit 1
fi

# Login to Railway
echo "🔐 Logging into Railway..."
railway login

# Initialize Railway project (if not already done)
if [ ! -f ".railway" ]; then
    echo "📁 Initializing Railway project..."
    railway init
fi

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

# Get the deployment URL
echo "🔗 Getting deployment URL..."
DEPLOYMENT_URL=$(railway status --json | jq -r '.deployment.url')

if [ "$DEPLOYMENT_URL" != "null" ] && [ "$DEPLOYMENT_URL" != "" ]; then
    echo "✅ Deployment successful!"
    echo "🌐 Your app is live at: $DEPLOYMENT_URL"
    echo ""
    echo "📝 Next steps:"
    echo "1. Set up environment variables in Railway dashboard:"
    echo "   - OPENAI_API_KEY: Your OpenAI API key"
    echo "2. Update the frontend config with your backend URL"
    echo "3. Test the application"
else
    echo "❌ Deployment failed or URL not found"
    echo "Check Railway dashboard for more details"
fi 