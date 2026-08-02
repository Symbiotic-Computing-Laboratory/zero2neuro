# Mosquito Dataset Preparation Scripts Guide

This document explains the differences between the various versions of the `prepare_data_prior` scripts in this repository. Because data cleaning and preprocessing heavily influence neural network performance, multiple versions have been created to handle missing data, autoregressive priors, and threshold filtering in different ways.

## 1. `prepare_data_prior_v2.py`
**The Baseline Prior Script (Clean & Strict)**
*   **Purpose:** To generate the base dataset while introducing the crucial `Culex.pipiens.prior` autoregressive feature (using the previous week's mosquito count as an input to predict the current week).
*   **Key Features:**
    *   Shifts the target variable by `t-1` (rolling along the epiweek axis).
    *   Explicitly zeroes out the first week of every year (Week 19) so that the final week of the previous year (Week 48) doesn't improperly bleed across the 6-month winter gap.
    *   **No Imputation:** Any missing weeks where a trap was not checked are strictly padded with `0.0`. It does NOT guess or fill in missing weather or mosquito data.

## 2. `prepare_data_prior_v5.py`
**The Imputation Script (Smooth Weather Features)**
*   **Purpose:** To prevent the neural network from experiencing "zero-padding shock." In `v2`, missing weather data drops suddenly to `0.0`, which can confuse the network into thinking there was a sudden freeze or zero-precipitation event.
*   **Key Features:**
    *   Introduces **Forward-Fill** and **Backward-Fill** exclusively for the input features (weather, bio, lands). If week 25 is missing, it intelligently borrows the weather from week 24 to smooth out the timeline.
    *   Safely ignores the target variable (`Culex.pipiens`). It zeroes out the target anywhere the `data_mask` is `0.0` to prevent artificial mosquito plateaus from being copied into missing weeks.

## 3. `prepare_data_prior_v6.py`
**The Threshold Filtering Script (Quality Control)**
*   **Purpose:** To strictly enforce data quality by removing entire years from specific locations if they do not meet a minimum number of valid trap checks.
*   **Key Features:**
    *   Introduces a global command-line argument: `--data_min_valid_epiweeks` (defaults to 15).
    *   Checks the number of actual, valid epiweeks per `(Location, Year)` combination.
    *   **The Masking Trick:** If a year fails the threshold, the script does *not* physically delete the rows (which would break the `(70, 30, 1)` tensor shape and crash downstream plotting scripts). Instead, it forces the `data_mask` for that entire year to `0.0`.
    *   Because `zero2neuro` uses `--data_weights data_mask`, the neural network mathematically multiplies the loss for that bad year by 0, completely ignoring it during training while keeping the array shapes perfectly intact.

---
### Which one should I use?
*   Use **`v2`** if you want the absolute rawest data and want to ensure the neural network is forced to learn how to handle sudden `0.0` drops naturally.
*   Use **`v5`** if you want the most stable weather inputs and believe that last week's weather is a good approximation for a missing trap check.
*   Use **`v6`** if you are analyzing locations with highly fragmented data and want to ensure the network only trains on robust, high-quality years that have a substantial number of trap checks.
