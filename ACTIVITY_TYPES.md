# Strava Activity Types - Complete Support

This document lists all officially supported Strava activity types and how they're handled by the supercompensation system.

## Activity Type Categories

### 🏃‍♀️ Running Activities
| Strava Type | Base TSS/hr | Adjustment | Final TSS/hr | Notes |
|-------------|-------------|------------|--------------|--------|
| `Run` | 100 | 1.0 | 100 | Standard road running |
| `TrailRun` | 100 | 1.15 | 115 | Higher due to terrain |
| `VirtualRun` | 100 | 1.0 | 100 | Treadmill running |

### 🚴‍♀️ Cycling Activities
| Strava Type | Base TSS/hr | Adjustment | Final TSS/hr | Notes |
|-------------|-------------|------------|--------------|--------|
| `Ride` | 80 | 1.0 | 80 | Standard road cycling |
| `VirtualRide` | 80 | 1.0 | 80 | Indoor trainer |
| `MountainBikeRide` | 80 | 1.2 | 96 | Higher due to terrain/technique |
| `GravelRide` | 80 | 1.1 | 88 | Moderate increase vs road |
| `EBikeRide` | 80 | 0.7 | 56 | Lower due to assistance |
| `EMountainBikeRide` | 80 | 0.8 | 64 | Lower than regular MTB |
| `HandCycle` | 80 | 1.0 | 80 | Upper body cycling |
| `Velomobile` | 80 | 1.0 | 80 | Enclosed cycling |

### 🏊‍♀️ Swimming Activities
| Strava Type | Base TSS/hr | Adjustment | Final TSS/hr | Notes |
|-------------|-------------|------------|--------------|--------|
| `Swim` | 120 | 1.0 | 120 | Pool swimming |
| `OpenSwim` | 120 | 1.1 | 132 | Higher due to conditions |
| `KiteSwim` | 120 | 1.3 | 156 | High technique demands |

### 🚣‍♀️ Rowing/Paddling Activities
| Strava Type | Base TSS/hr | Adjustment | Final TSS/hr | Notes |
|-------------|-------------|------------|--------------|--------|
| `Rowing` | 110 | 1.0 | 110 | Standard rowing |
| `Canoeing` | 110 | 1.0 | 110 | Canoe paddling |
| `Kayaking` | 110 | 1.0 | 110 | Kayak paddling |
| `StandUpPaddling` | 110 | 0.9 | 99 | Lower than kayaking |

### ⛸️ Skating/Winter Sports
| Strava Type | Base TSS/hr | Adjustment | Final TSS/hr | Notes |
|-------------|-------------|------------|--------------|--------|
| `InlineSkate` | 85 | 1.0 | 85 | Inline skating |
| `IceSkate` | 85 | 1.0 | 85 | Ice skating |
| `BackcountrySki` | 85 | 1.3 | 110.5 | High intensity, full body |
| `AlpineSki` | 85 | 0.9 | 76.5 | Lower than XC skiing |
| `NordicSki` | 85 | 1.1 | 93.5 | Classic endurance sport |
| `SkiteSki` | 85 | 1.0 | 85 | Skate skiing |
| `Snowboard` | 85 | 0.95 | 80.75 | Similar to alpine skiing |
| `Snowshoe` | 85 | 1.2 | 102 | Higher than hiking due to snow |

### 🥾 Hiking/Climbing Activities
| Strava Type | Base TSS/hr | Adjustment | Final TSS/hr | Notes |
|-------------|-------------|------------|--------------|--------|
| `Hike` | 40 | 1.0 | 40 | Standard hiking |
| `Walk` | 40 | 0.6 | 24 | Lower than hiking |
| `RockClimbing` | 40 | 1.0 | 40 | Technical climbing |
| `Climbing` | 40 | 1.0 | 40 | General climbing |

### 🏋️‍♀️ Strength/Conditioning
| Strava Type | Base TSS/hr | Adjustment | Final TSS/hr | Notes |
|-------------|-------------|------------|--------------|--------|
| `WeightTraining` | 70 | 0.8 | 56 | Anaerobic, lower TSS |
| `StrengthTraining` | 70 | 0.8 | 56 | Similar to weight training |
| `HIIT` | 70 | 1.5 | 105 | Very high intensity |
| `CrossTraining` | 70 | 1.1 | 77 | Mixed modalities |
| `Workout` | 70 | 1.0 | 70 | General workout |

### 🧘‍♀️ Low Intensity Activities
| Strava Type | Base TSS/hr | Adjustment | Final TSS/hr | Notes |
|-------------|-------------|------------|--------------|--------|
| `Yoga` | 70 | 0.4 | 28 | Very low intensity |
| `Pilates` | 70 | 0.5 | 35 | Low-moderate intensity |
| `Meditation` | 70 | 0.1 | 7 | Minimal physical load |
| `Barre` | 70 | 0.6 | 42 | Moderate intensity |

### ⚽ Sports Activities
| Strava Type | Base TSS/hr | Adjustment | Final TSS/hr | Notes |
|-------------|-------------|------------|--------------|--------|
| `Soccer` | 70 | 1.3 | 91 | High intermittent intensity |
| `Basketball` | 70 | 1.4 | 98 | Very high intermittent |
| `Tennis` | 70 | 1.2 | 84 | High intermittent |
| `Hockey` | 70 | 1.5 | 105 | Very high intensity |
| `Rugby` | 70 | 1.4 | 98 | High contact sport intensity |
| `Boxing` | 70 | 1.6 | 112 | Extremely high intensity |
| `MartialArts` | 70 | 1.3 | 91 | High intensity, technique |
| `Volleyball` | 70 | 1.0 | 70 | Moderate intensity |
| `Golf` | 70 | 0.5 | 35 | Low intensity |

## Recovery Demands by Sport Category

### High Recovery Demand (1.5x base recovery time)
- **Running** (all types) - High impact stress
- **Rowing/Paddling** - High power output demands

### Moderate Recovery Demand (1.0x base recovery time)
- **Cycling** (most types) - Lower impact
- **Swimming** - Non-impact, full body
- **Skating/Winter Sports** - Variable impact
- **Most Sports** - Intermittent demands

### Low Recovery Demand (0.5x base recovery time)
- **Hiking/Walking** - Low intensity, long duration
- **Yoga/Pilates** - Very low intensity
- **Golf** - Minimal physical stress

## Cross-Training Recommendations

The system provides sport-specific cross-training suggestions:

### After Running
- Swimming (non-impact cardio)
- Easy cycling (active recovery)
- Aqua jogging (running-specific, no impact)

### After Cycling
- Running (impact training for bone density)
- Swimming (upper body strength)
- Hiking (active recovery)

### After Swimming
- Running (weight-bearing exercise)
- Cycling (leg strength maintenance)
- Yoga (flexibility, shoulder mobility)

## Environmental Adjustments

The system also considers:
- **Elevation gain** (affects running more than cycling)
- **Heart rate intensity** (sport-specific zones)
- **Speed/pace** (sport-specific thresholds)
- **Race flag** (workout_type = 1) for extended recovery protocols

## Usage

All activity types are automatically detected from Strava and processed with appropriate sport-specific calculations for:
- Training load (TSS equivalent)
- Recovery time requirements
- Cross-training recommendations
- Intensity zone analysis