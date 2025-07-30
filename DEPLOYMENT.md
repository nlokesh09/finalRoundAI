# FinalRound AI - Deployment Guide

This guide will help you deploy the FinalRound AI application to Railway.

## 🚀 Quick Deployment

### Prerequisites
1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Railway CLI**: Install with `npm install -g @railway/cli`
3. **OpenAI API Key**: Get one from [OpenAI Platform](https://platform.openai.com/api-keys)

### Step 1: Prepare Your Environment
```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd finalRoundAI

# Install Railway CLI
npm install -g @railway/cli
```

### Step 2: Deploy to Railway
```bash
# Login to Railway
railway login

# Initialize Railway project
railway init

# Deploy the application
railway up
```

### Step 3: Configure Environment Variables
1. Go to your Railway dashboard
2. Navigate to your project
3. Go to the "Variables" tab
4. Add the following environment variables:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 4: Get Your Deployment URL
After deployment, Railway will provide you with a URL like:
`https://your-app-name.railway.app`

## 🔧 Manual Deployment Steps

### 1. Backend Configuration
The backend is configured to:
- Use environment variables for port and host
- Handle CORS for frontend requests
- Use SQLite database (persisted in Railway)

### 2. Frontend Configuration
The frontend automatically detects the environment:
- Development: Uses `http://localhost:8000`
- Production: Uses the Railway backend URL

### 3. Database Setup
The SQLite database is automatically created on first run.

## 🌐 Accessing Your Application

Once deployed, you can access:
- **Main Application**: `https://your-app-name.railway.app`
- **API Documentation**: `https://your-app-name.railway.app/docs`
- **Health Check**: `https://your-app-name.railway.app/api/health`

## 📝 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key for ChatGPT integration | Yes |
| `PORT` | Port for the backend server | No (default: 8000) |
| `HOST` | Host for the backend server | No (default: 0.0.0.0) |

## 🔍 Troubleshooting

### Common Issues

1. **Deployment Fails**
   - Check Railway logs in the dashboard
   - Ensure all dependencies are in requirements.txt
   - Verify environment variables are set

2. **API Key Issues**
   - Ensure OPENAI_API_KEY is set correctly
   - Test the API key in OpenAI dashboard

3. **CORS Errors**
   - The backend is configured to allow all origins in production
   - Check if the frontend is using the correct backend URL

4. **Database Issues**
   - SQLite database is automatically created
   - Check Railway logs for database errors

### Logs and Monitoring
- View logs in Railway dashboard
- Monitor application health at `/api/health`
- Check API status at `/docs`

## 🚀 Production Considerations

1. **Scaling**: Railway automatically scales based on traffic
2. **Database**: Consider migrating to PostgreSQL for production
3. **Security**: Add authentication if needed
4. **Monitoring**: Set up alerts and monitoring

## 📞 Support

If you encounter issues:
1. Check Railway logs
2. Verify environment variables
3. Test API endpoints individually
4. Check the health endpoint

## 🔄 Updates and Maintenance

To update your deployment:
```bash
# Make your changes
git add .
git commit -m "Update application"

# Deploy changes
railway up
```

---

**Happy Deploying! 🎉** 