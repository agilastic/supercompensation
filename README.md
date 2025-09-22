# Strava Supercompensation Tool

An automated Python tool that analyzes your Strava training data using the Banister Impulse-Response model to provide personalized training recommendations based on supercompensation principles.

## Features

- **Strava Integration**: Automatic synchronization of activities via OAuth2
- **Supercompensation Analysis**: Implementation of Banister's Fitness-Fatigue model
- **Training Recommendations**: Daily recommendations (Rest, Easy, Moderate, Hard) based on current form
- **Metrics Tracking**: Monitor Fitness (CTL), Fatigue (ATL), and Form (TSB) over time
- **CLI Interface**: Rich terminal interface for easy interaction
- **Data Persistence**: SQLite database for activity and metrics storage

## Installation

### Prerequisites

- Python 3.10 or higher
- Strava API application credentials

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/strava-supercompensation.git
cd strava-supercompensation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

For basic functionality:
```bash
pip install -r requirements-minimal.txt
```

For full features including visualization:
```bash
pip install -r requirements.txt
```

For development:
```bash
pip install -r requirements-dev.txt
```

Or install as a package:
```bash
pip install -e .  # Basic installation
pip install -e ".[dev]"  # With development dependencies
pip install -e ".[viz]"  # With visualization support
```

4. Configure Strava API credentials:
```bash
cp .env.example .env
```

Edit `.env` and add your Strava API credentials:
- `STRAVA_CLIENT_ID`: Your Strava application's Client ID
- `STRAVA_CLIENT_SECRET`: Your Strava application's Client Secret

To get these credentials:
1. Go to https://www.strava.com/settings/api
2. Create a new application
3. Copy the Client ID and Client Secret

## Usage

### Authentication

First, authenticate with Strava:
```bash
strava-super auth
```

This will open a browser window for Strava authorization.

### Sync Activities

Fetch your recent activities from Strava:
```bash
strava-super sync --days 30
```

### Analyze Training

Calculate fitness, fatigue, and form metrics:
```bash
strava-super analyze --days 90
```

### Get Recommendations

Get today's training recommendation:
```bash
strava-super recommend
```

### Check Status

View current authentication and sync status:
```bash
strava-super status
```

### Reset

Reset all data and authentication:
```bash
strava-super reset
```

## Model Parameters

The tool uses the Banister model with the following default parameters (configurable in `.env`):

- **Fitness Decay Rate**: 42 days (Chronic Training Load time constant)
- **Fatigue Decay Rate**: 7 days (Acute Training Load time constant)
- **Fitness Magnitude**: 1.0
- **Fatigue Magnitude**: 2.0

## Training Recommendations

The tool provides recommendations based on your current form (TSB):

- **REST**: Very high fatigue, complete rest needed
- **RECOVERY**: High fatigue, active recovery recommended
- **EASY**: Low form, easy aerobic training
- **MODERATE**: Neutral state, moderate intensity training
- **HARD**: Good form, high intensity training possible
- **PEAK**: Excellent form, perfect for race or time trial

## Technical Details

### Architecture

- **Authentication**: OAuth2 flow with token refresh
- **Database**: SQLite with SQLAlchemy ORM
- **Analysis**: NumPy-based implementation of Banister model
- **CLI**: Click framework with Rich formatting

### Project Structure

```
strava_supercompensation/
├── auth/           # OAuth2 authentication
├── api/            # Strava API client
├── db/             # Database models and management
├── analysis/       # Supercompensation model and recommendations
├── cli.py          # Command-line interface
└── config.py       # Configuration management
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License

## Disclaimer

This tool provides training recommendations based on mathematical models and should not replace professional coaching or medical advice. Always listen to your body and consult with professionals for personalized training guidance.