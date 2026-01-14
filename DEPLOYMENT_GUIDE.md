# Deployment Guide: Customer Churn Predictor on Render

## 📋 Prerequisites

1. GitHub account
2. Render account (free tier available at https://render.com)
3. GROQ API key (from https://console.groq.com)

## 🚀 Step-by-Step Deployment

### Step 1: Prepare Your Project Structure

Your project should have this structure:

```
customer-churn-predictor/
├── app.py                    # Flask application (NEW)
├── requirements.txt          # Python dependencies (UPDATED)
├── render.yaml              # Render configuration (NEW)
├── tools/
│   ├── data_tools.py
│   ├── feature_tools.py
│   ├── model_tools.py
│   └── strategy_tools.py
├── templates/               # NEW folder
│   ├── index.html
│   └── results.html
├── data/
├── models/
├── outputs/
└── .env                     # DON'T commit this!
```

### Step 2: Create .gitignore

Create a `.gitignore` file:

```
venv/
__pycache__/
*.pyc
.env
data/customer_data.csv
models/*.pkl
outputs/*.csv
uploads/
*.log
```

### Step 3: Initialize Git Repository

```bash
# In your project folder
git init
git add .
git commit -m "Initial commit - Customer Churn Predictor"
```

### Step 4: Push to GitHub

1. Create a new repository on GitHub (don't initialize with README)
2. Push your code:

```bash
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-predictor.git
git branch -M main
git push -u origin main
```

### Step 5: Deploy on Render

1. **Go to Render Dashboard**
   - Visit https://dashboard.render.com
   - Click "New +" → "Web Service"

2. **Connect Your Repository**
   - Connect your GitHub account
   - Select your `customer-churn-predictor` repository
   - Click "Connect"

3. **Configure Your Service**
   - Name: `customer-churn-predictor`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Instance Type: `Free` (or paid for better performance)

4. **Add Environment Variables**
   - Click "Advanced" → "Add Environment Variable"
   - Add:
     - Key: `GROQ_API_KEY`
     - Value: Your GROQ API key (from console.groq.com)
   - Add:
     - Key: `PYTHON_VERSION`
     - Value: `3.11.0`

5. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - Your app will be live at: `https://your-app-name.onrender.com`

## 🔧 Configuration Options

### For Better Performance (Paid Plans)

If using a paid Render plan, update your `render.yaml`:

```yaml
services:
  - type: web
    name: customer-churn-predictor
    env: python
    plan: starter  # or standard, pro
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --workers 4 --timeout 300
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: SECRET_KEY
        generateValue: true
```

### Increase Timeout (for long-running analyses)

In your Render dashboard:
- Settings → Advanced → HTTP Request Timeout
- Set to 300 seconds (5 minutes) or more

## 📱 Using Your Deployed App

1. Visit your Render URL: `https://your-app-name.onrender.com`
2. Upload a customer CSV file
3. Click "Start AI Analysis"
4. Wait for the analysis to complete (5-10 minutes)
5. View and download results

## 🔍 Monitoring & Debugging

### View Logs

In Render Dashboard:
- Go to your service
- Click "Logs" tab
- View real-time logs

### Check Status

Your app has a health endpoint:
```
https://your-app-name.onrender.com/health
```

### Common Issues

**Issue: App keeps spinning/crashing**
- Check logs for errors
- Verify GROQ_API_KEY is set correctly
- Ensure all dependencies are in requirements.txt

**Issue: Analysis takes too long**
- Free tier has limited resources
- Consider upgrading to a paid plan
- Reduce dataset size for testing

**Issue: Out of memory**
- Upgrade to a paid plan with more RAM
- Or optimize data processing

## 💰 Cost Considerations

### Free Tier Limitations
- 750 hours/month (enough for hobby projects)
- Spins down after 15 min of inactivity
- Limited CPU/RAM
- May be slow for large datasets

### Paid Plans (Recommended for Production)
- **Starter ($7/month)**: 
  - No spin-down
  - Better performance
  - 512MB RAM
- **Standard ($25/month)**:
  - 2GB RAM
  - Better for larger datasets

## 🔐 Security Best Practices

1. **Never commit `.env` file**
2. **Use environment variables** for sensitive data
3. **Enable HTTPS** (Render does this automatically)
4. **Rotate API keys** regularly
5. **Add rate limiting** for production

## 🚀 Optional: Add Custom Domain

1. In Render Dashboard → Settings
2. Click "Add Custom Domain"
3. Follow DNS configuration instructions
4. Your app will be at: `https://your-domain.com`

## 📊 Performance Tips

1. **Use caching** for repeated analyses
2. **Optimize dataset size** (sample large datasets)
3. **Add a queue system** (e.g., Redis + Celery) for production
4. **Enable horizontal scaling** with paid plans

## 🔄 Updating Your App

```bash
# Make changes to your code
git add .
git commit -m "Update: description of changes"
git push origin main

# Render will automatically redeploy
```

## 📞 Support

- **Render Docs**: https://render.com/docs
- **GROQ Docs**: https://console.groq.com/docs
- **CrewAI Docs**: https://docs.crewai.com

---

**Your app is now live! 🎉**

Share your Render URL with others to let them use your Customer Churn Predictor!