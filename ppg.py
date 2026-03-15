import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# STEP 1: Load and aggregate game-level data from SQLite database
print("Loading and aggregating game-level data from nba.sqlite database...")

try:
    conn = sqlite3.connect('nba.sqlite')
    
    # Get game data and aggregate to season-level player stats
    query = """
    SELECT 
        season_id,
        team_id_home as team_id,
        team_abbreviation_home as team_abbr,
        team_name_home as team_name,
        COUNT(*) as games_played,
        AVG(pts_home) as pts_per_game,
        AVG(fgm_home) as fgm_per_game,
        AVG(fga_home) as fga_per_game,
        AVG(fg_pct_home) as fg_pct,
        AVG(fg3m_home) as fg3m_per_game,
        AVG(fg3a_home) as fg3a_per_game,
        AVG(fg3_pct_home) as fg3_pct,
        AVG(ftm_home) as ftm_per_game,
        AVG(fta_home) as fta_per_game,
        AVG(ft_pct_home) as ft_pct,
        AVG(oreb_home) as oreb_per_game,
        AVG(dreb_home) as dreb_per_game,
        AVG(reb_home) as reb_per_game,
        AVG(ast_home) as ast_per_game,
        AVG(stl_home) as stl_per_game,
        AVG(blk_home) as blk_per_game,
        AVG(tov_home) as tov_per_game,
        AVG(pf_home) as pf_per_game,
        AVG(plus_minus_home) as plus_minus_per_game
    FROM game 
    WHERE season_type = 'Regular Season'
    GROUP BY season_id, team_id_home
    """
    
    df_all = pd.read_sql_query(query, conn)
    
    # Also get away team stats and combine
    query_away = """
    SELECT 
        season_id,
        team_id_away as team_id,
        team_abbreviation_away as team_abbr,
        team_name_away as team_name,
        COUNT(*) as games_played,
        AVG(pts_away) as pts_per_game,
        AVG(fgm_away) as fgm_per_game,
        AVG(fga_away) as fga_per_game,
        AVG(fg_pct_away) as fg_pct,
        AVG(fg3m_away) as fg3m_per_game,
        AVG(fg3a_away) as fg3a_per_game,
        AVG(fg3_pct_away) as fg3_pct,
        AVG(ftm_away) as ftm_per_game,
        AVG(fta_away) as fta_per_game,
        AVG(ft_pct_away) as ft_pct,
        AVG(oreb_away) as oreb_per_game,
        AVG(dreb_away) as dreb_per_game,
        AVG(reb_away) as reb_per_game,
        AVG(ast_away) as ast_per_game,
        AVG(stl_away) as stl_per_game,
        AVG(blk_away) as blk_per_game,
        AVG(tov_away) as tov_per_game,
        AVG(pf_away) as pf_per_game,
        AVG(plus_minus_away) as plus_minus_per_game
    FROM game 
    WHERE season_type = 'Regular Season'
    GROUP BY season_id, team_id_away
    """
    
    df_away = pd.read_sql_query(query_away, conn)
    
    # Combine home and away stats
    df_all = pd.concat([df_all, df_away], ignore_index=True)
    
    # Convert season_id to integer year
    df_all['Season'] = df_all['season_id'].str.extract('(\d{4})').astype(int)
    
    conn.close()
    
except Exception as e:
    print(f"Error reading from database: {e}")
    raise

print(f"Loaded {len(df_all)} team-season records")

# STEP 2: Clean and prepare data
# Convert numeric columns
non_numeric = ['season_id', 'team_id', 'team_abbr', 'team_name', 'Season']
numeric_cols = [col for col in df_all.columns if col not in non_numeric]
df_all[numeric_cols] = df_all[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Filter for teams with sufficient games played
df_all = df_all[df_all['games_played'] >= 20]  # At least 20 games played
print(f"After filtering for >=20 games: {len(df_all)} team-seasons")

# Sort by team and season
df_all = df_all.sort_values(['team_id', 'Season']).dropna(subset=['team_id', 'pts_per_game'])

# STEP 3: Feature Engineering - Use past seasons to predict current season
stats_to_lag = [col for col in numeric_cols if col not in ['Season', 'games_played', 'pts_per_game']]
X_features = pd.DataFrame(index=df_all.index)
y_target = df_all['pts_per_game']

# Create lagged features from the past two seasons
for col in stats_to_lag:
    # Stats from 1 season ago
    X_features[f'{col}_lag1'] = df_all.groupby('team_id')[col].transform(lambda s: s.shift(1))
    # Stats from 2 seasons ago
    X_features[f'{col}_lag2'] = df_all.groupby('team_id')[col].transform(lambda s: s.shift(2))

# STEP 4: Clean final dataset
final_df = pd.concat([X_features, y_target], axis=1).dropna()
if final_df.empty:
    raise ValueError("DataFrame is empty. Not enough sequential seasons per team.")

X = final_df.drop('pts_per_game', axis=1)
y = final_df['pts_per_game']

print(f"\nFinal dataset: {len(final_df)} team-seasons with complete historical data")

# STEP 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training on {X_train.shape[0]} team-seasons using historical data.")
print(f"Testing on {X_test.shape[0]} team-seasons.")
print(f"Number of features: {X.shape[1]}")

# STEP 6: Define Grid Search Parameters
print("\n" + "="*60)
print("GRID SEARCH HYPERPARAMETER TUNING")
print("="*60)

# Random Forest Grid Search
print("\n1. Random Forest Grid Search...")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

rf_grid_search.fit(X_train, y_train)
print(f"Best Random Forest parameters: {rf_grid_search.best_params_}")
print(f"Best Random Forest CV score: {rf_grid_search.best_score_:.3f}")

# XGBoost Grid Search
print("\n2. XGBoost Grid Search...")
try:
    from xgboost import XGBRegressor
    
    xgb_param_grid = {
        'n_estimators': [25, 50],  # Reduced from [50, 100, 200] - much faster!
        'max_depth': [3, 5],  # Reduced from [3, 5, 7, 10]
        'learning_rate': [0.1, 0.2],  # Reduced from [0.01, 0.1, 0.2]
        'subsample': [0.8, 1.0],  # Reduced from [0.8, 0.9, 1.0]
        'colsample_bytree': [0.8, 1.0],  # Reduced from [0.8, 0.9, 1.0]
        'min_child_weight': [1, 3]  # Reduced from [1, 3, 5]
    }
    
    xgb_grid_search = GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1),
        xgb_param_grid,
        cv=3,  # Reduced from 5 for speed
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    xgb_grid_search.fit(X_train, y_train)
    print(f"Best XGBoost parameters: {xgb_grid_search.best_params_}")
    print(f"Best XGBoost CV score: {xgb_grid_search.best_score_:.3f}")
    
except ImportError:
    print("XGBoost not installed. Skipping XGBoost grid search.")
    xgb_grid_search = None

# STEP 7: Model Comparison
print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

def evaluate_model(name, model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n{name} Results:")
    print(f"  R² Score: {r2:.3f}")
    print(f"  RMSE: {rmse:.2f} points per game")
    print(f"  MAE: {mae:.2f} points per game")
    
    return {
        'model': model,
        'predictions': y_pred,
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }

# Evaluate Random Forest
rf_results = evaluate_model("Random Forest (Tuned)", rf_grid_search.best_estimator_, X_test, y_test)

# Evaluate XGBoost if available
if xgb_grid_search is not None:
    xgb_results = evaluate_model("XGBoost (Tuned)", xgb_grid_search.best_estimator_, X_test, y_test)
else:
    xgb_results = None

# STEP 8: Feature Importance Comparison
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Random Forest Feature Importance
rf_importances = pd.Series(
    rf_grid_search.best_estimator_.feature_importances_, 
    index=X.columns
).sort_values(ascending=False)

print(f"\nTop 10 Random Forest Features:")
print(rf_importances.head(10))

# XGBoost Feature Importance
if xgb_results is not None:
    xgb_importances = pd.Series(
        xgb_grid_search.best_estimator_.feature_importances_, 
        index=X.columns
    ).sort_values(ascending=False)
    
    print(f"\nTop 10 XGBoost Features:")
    print(xgb_importances.head(10))

# STEP 9: Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Random Forest Feature Importance
axes[0, 0].barh(range(20), rf_importances.head(20))
axes[0, 0].set_yticks(range(20))
axes[0, 0].set_yticklabels(rf_importances.head(20).index)
axes[0, 0].set_title('Random Forest - Top 20 Feature Importances')
axes[0, 0].set_xlabel('Importance Score')

# XGBoost Feature Importance
if xgb_results is not None:
    axes[0, 1].barh(range(20), xgb_importances.head(20))
    axes[0, 1].set_yticks(range(20))
    axes[0, 1].set_yticklabels(xgb_importances.head(20).index)
    axes[0, 1].set_title('XGBoost - Top 20 Feature Importances')
    axes[0, 1].set_xlabel('Importance Score')

# Actual vs Predicted Comparison
axes[1, 0].scatter(y_test, rf_results['predictions'], alpha=0.6)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual PPG')
axes[1, 0].set_ylabel('Predicted PPG')
axes[1, 0].set_title('Random Forest: Actual vs Predicted')

if xgb_results is not None:
    axes[1, 1].scatter(y_test, xgb_results['predictions'], alpha=0.6)
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual PPG')
    axes[1, 1].set_ylabel('Predicted PPG')
    axes[1, 1].set_title('XGBoost: Actual vs Predicted')

plt.tight_layout()
plt.show()

# STEP 10: Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nBest Model: {'XGBoost' if xgb_results and xgb_results['r2'] > rf_results['r2'] else 'Random Forest'}")
print(f"Best R² Score: {max(rf_results['r2'], xgb_results['r2'] if xgb_results else 0):.3f}")

if xgb_results:
    print(f"\nModel Comparison:")
    print(f"  Random Forest R²: {rf_results['r2']:.3f}")
    print(f"  XGBoost R²: {xgb_results['r2']:.3f}")
    print(f"  Difference: {abs(rf_results['r2'] - xgb_results['r2']):.3f}")
