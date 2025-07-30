# Backend Environment Setup

## Quick Setup

1. **Copy the environment file:**
   ```bash
   cp env.example .env
   ```

2. **Edit the `.env` file** with your specific values:
   ```bash
   nano .env
   # or
   code .env
   ```

3. **Start the server:**
   ```bash
   python main.py
   ```

## Environment Variables Explained

### Required Variables
- `SECRET_KEY` - Change this to a secure random string
- `DATABASE_URL` - Database connection string
- `ALLOWED_ORIGINS` - Frontend URLs allowed for CORS

### Optional Variables
- `DEBUG` - Set to `False` in production
- `LOG_LEVEL` - Logging verbosity
- `PORT` - Server port (default: 8000)

## Security Notes

⚠️ **Important:** Never commit your `.env` file to version control!

The `.env` file is already in `.gitignore` to prevent accidental commits.

## Production Setup

For production, make sure to:

1. Set `DEBUG=False`
2. Use a strong `SECRET_KEY`
3. Configure proper `DATABASE_URL`
4. Set `ALLOWED_ORIGINS` to your production domain
5. Use environment-specific settings

## Example Production .env

```env
APP_NAME=Final Round AI API
APP_VERSION=1.0.0
DEBUG=False
ENVIRONMENT=production

HOST=0.0.0.0
PORT=8000
RELOAD=False

DATABASE_URL=postgresql://user:password@localhost/production_db
SECRET_KEY=your-super-secure-production-secret-key

ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
ALLOWED_METHODS=GET,POST,PUT,DELETE,OPTIONS
ALLOWED_HEADERS=*

LOG_LEVEL=WARNING
``` 