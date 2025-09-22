# Garmin Connect API Setup Guide

This guide walks you through setting up Garmin Connect API access for HRV and sleep data integration.

## 🔐 Prerequisites

- Garmin Connect account with activity data
- A Garmin device that tracks HRV and sleep (Forerunner, Fenix, Venu, etc.)
- Python environment with required dependencies

## 📝 Step 1: Apply for Garmin Connect Developer Access

### 1.1 Developer Program Application

1. Visit the [Garmin Connect Developer Program](https://developer.garmin.com/connect-iq/)
2. Click **"Connect IQ App Development"** → **"Web API"**
3. Complete the application form with:

   **Application Type**: Third-party application
   **Application Name**: `Strava Supercompensation Tool`
   **Description**:
   ```
   A training analysis tool that integrates Strava activity data with Garmin wellness metrics
   (HRV, sleep quality, stress) to provide comprehensive supercompensation-based training
   recommendations. The application helps athletes optimize training load and recovery by
   combining activity data from Strava with physiological readiness indicators from Garmin devices.
   ```

   **Data Usage**:
   ```
   - Heart Rate Variability (HRV) data for recovery assessment
   - Sleep quality metrics for fatigue evaluation
   - Daily stress levels for training readiness
   - Body Battery data for energy management
   - Step count and activity summaries for load calculation

   All data is used locally for personalized training recommendations and is not shared
   with third parties.
   ```

### 1.2 Required Information

**Technical Details**:
- **Platform**: Web Application / CLI Tool
- **Target Users**: Individual athletes and coaches
- **Expected User Volume**: < 100 users (personal/small team use)
- **Data Storage**: Local SQLite database
- **OAuth Flow**: OAuth 1.0a (required by Garmin)

**Contact Information**:
- Provide accurate contact details
- Business email preferred
- Clear project description

### 1.3 Application Review

- Review process typically takes **2-4 weeks**
- Garmin may request additional information
- You'll receive OAuth 1.0a credentials upon approval

## 🔧 Step 2: Configure OAuth 1.0a Credentials

Once approved, you'll receive:
- **Consumer Key**
- **Consumer Secret**

### 2.1 Environment Setup

Create or update your `.env` file:

```bash
# Existing Strava credentials
STRAVA_CLIENT_ID=your_strava_client_id
STRAVA_CLIENT_SECRET=your_strava_client_secret

# Add Garmin credentials
GARMIN_CONSUMER_KEY=your_garmin_consumer_key
GARMIN_CONSUMER_SECRET=your_garmin_consumer_secret
```

### 2.2 Install Additional Dependencies

The Garmin integration requires OAuth 1.0a support:

```bash
pip install requests-oauthlib
```

## 🚀 Step 3: Authentication Flow

### 3.1 Initial Authentication

Run the Garmin authentication command:

```bash
strava-super garmin auth
```

This will:
1. Generate an authorization URL
2. Open your browser to Garmin Connect
3. Prompt for authorization code
4. Store OAuth tokens locally

### 3.2 Authentication Process

1. **Authorization URL**: The tool generates a Garmin Connect authorization URL
2. **User Consent**: You authorize the application on Garmin's website
3. **Verification Code**: Garmin provides a verification code
4. **Token Exchange**: The tool exchanges the code for access tokens
5. **Storage**: Tokens are securely stored in your local database

## 📊 Step 4: Data Synchronization

### 4.1 Available Data Types

The integration supports three main data categories:

**HRV Data**:
- Heart Rate Variability RMSSD
- HRV Score (0-100)
- HRV Status (balanced, unbalanced, low, high)
- Stress level and recovery advisor

**Sleep Data**:
- Total sleep time and sleep stages (deep, light, REM)
- Sleep score and efficiency
- Sleep timing (bedtime, wake time)
- Heart rate during sleep
- Recovery indicators

**Wellness Data**:
- Daily stress levels and duration
- Body Battery charged/drained
- Respiration rate
- Blood oxygen saturation (SpO2)
- Step count and calories
- Activity intensity minutes

### 4.2 Sync Commands

```bash
# Sync all data types (last 30 days)
strava-super garmin sync

# Sync specific data type
strava-super garmin sync --type hrv --days 7
strava-super garmin sync --type sleep --days 14
strava-super garmin sync --type wellness --days 30

# Check sync status
strava-super garmin status
```

### 4.3 Wellness Analysis

```bash
# View wellness trends and insights
strava-super garmin wellness --days 7

# Test API connection
strava-super garmin test
```

## 🏃‍♀️ Step 5: Enhanced Training Analysis

### 5.1 Integrated Recommendations

With Garmin data, the recommendation engine considers:

- **HRV trends** for recovery assessment
- **Sleep quality** for fatigue evaluation
- **Stress levels** for training readiness
- **Body Battery** for energy management

### 5.2 Example Integration

```bash
# Get enhanced recommendations with wellness data
strava-super recommend

# Analyze training with physiological context
strava-super analyze --days 30
```

The system will automatically incorporate:
- Poor sleep → Lower intensity recommendations
- High stress → Extended recovery periods
- Low HRV → Reduced training load
- Low Body Battery → Rest day recommendations

## 🔒 Security & Privacy

### 5.1 Data Storage

- All data stored locally in SQLite database
- OAuth tokens encrypted and stored securely
- No data transmission to external servers

### 5.2 Data Access

- Read-only access to wellness metrics
- No modification of Garmin data
- User maintains full control over data

### 5.3 Token Management

```bash
# Check authentication status
strava-super garmin status

# Re-authenticate if needed
strava-super garmin auth

# Revoke access (included in reset)
strava-super reset
```

## 🛠️ Troubleshooting

### Common Issues

**Authentication Errors**:
- Verify OAuth credentials in `.env` file
- Check Garmin Developer Console for API status
- Ensure browser can access Garmin Connect

**API Rate Limits**:
- Garmin limits API calls per day
- Sync spreads requests over time
- Reduce sync frequency if hitting limits

**Data Availability**:
- HRV requires compatible Garmin device
- Some metrics may not be available on older devices
- Sleep data requires overnight wear

**Connection Issues**:
```bash
# Test connection
strava-super garmin test

# Check detailed status
strava-super garmin status
```

### Device Compatibility

**HRV Support**:
- Forerunner 245/255/265/955/965
- Fenix 6/7 series
- Venu series
- Epix series

**Sleep Tracking**:
- Most modern Garmin devices with all-day wear
- Requires firmware updates for latest features

## 📈 Advanced Usage

### 5.1 Custom Analysis

The wellness data integrates into the existing analysis engine:

```python
from strava_supercompensation.api.garmin_client import get_garmin_client

client = get_garmin_client()
trend = client.get_wellness_trend(days=14)
latest_hrv = client.get_latest_hrv_score()
```

### 5.2 Data Export

All Garmin data is stored with full detail in the database and can be exported:

```bash
# View database tables
sqlite3 ~/.strava_supercompensation/data.db ".tables"

# Export HRV data
sqlite3 ~/.strava_supercompensation/data.db "SELECT * FROM hrv_data"
```

## 🆘 Support

### Developer Resources

- [Garmin Connect API Documentation](https://developer.garmin.com/connect-iq/api/)
- [OAuth 1.0a Specification](https://oauth.net/1/)
- [Connect IQ Developer Forum](https://forums.garmin.com/developer/)

### Application Support

- Check GitHub issues for common problems
- Review logs in `~/.strava_supercompensation/`
- Use `strava-super garmin status` for diagnostics

---

**Note**: This integration requires approval from Garmin's Developer Program. The application process may take 2-4 weeks. Ensure you have a legitimate use case and provide accurate information in your application.