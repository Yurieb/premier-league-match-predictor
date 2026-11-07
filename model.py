"""Functions for the Poisson match prediction model and its evaluation.

The notebook imports these so that the same code can also be tested automatically (see the tests folder).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

MAX_GOALS = 10   # The score grid covers each team scoring from 0 to 10 goals

COLUMNS = ["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]

# The predictors compared when testing a season: {column prefix: name shown in tables}
PREDICTORS = {
    "book": "Bet365",
    "updated": "Updated model",
    "original": "Original model",
    "base": "Baseline",
}


def season_label(start_year):
    """Turn a season's starting year into its label, for example 2024 becomes "2024/25"."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def clean_season(raw, label):
    """Keep the needed columns from one season's raw file, fix the dates and sort by date.

    Older files write dates as dd/mm/yy and newer ones as dd/mm/yyyy, so the format is chosen per file.
    """
    season = raw.dropna(subset=["HomeTeam"]).copy()    # Some files end with a blank row
    season["Season"] = label
    season = season[COLUMNS]

    date_lengths = set(season["Date"].astype(str).str.len())
    if date_lengths == {8}:
        date_format = "%d/%m/%y"
    elif date_lengths == {10}:
        date_format = "%d/%m/%Y"
    else:
        raise ValueError(f"{label}: unexpected date formats with lengths {date_lengths}")
    season["Date"] = pd.to_datetime(season["Date"], format=date_format)

    # A blank row makes pandas read goals as decimals, so convert them back to whole numbers
    season = season.astype({"FTHG": int, "FTAG": int})

    return season.sort_values("Date").reset_index(drop=True)


def load_season(start_year):
    """Download and clean one Premier League season from football-data.co.uk."""
    code = f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"    # 2024 becomes "2425"
    raw = pd.read_csv(f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv", encoding="latin-1")
    return clean_season(raw, season_label(start_year))


def load_queries(path):
    """Read a .sql file and return its queries as a dictionary of {name: query}.

    Each query in the file starts with a line such as "-- name: goals_per_season".
    """
    queries = {}
    name = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("-- name:"):
            name = line.removeprefix("-- name:").strip()
            queries[name] = ""
        elif name is not None:
            queries[name] += line + "\n"
    return {name: query.strip() for name, query in queries.items()}


def calculate_team_strengths(matches):
    """Rate each team's home and away attack and defence against the league average.

    Returns the table of ratings, the league average home goals and the league average away goals.
    A rating of 1.0 means exactly average. For attack, higher is better. For defence, lower is better.
    """
    # League averages: the baselines that every team is compared against
    avg_home_goals = matches["FTHG"].mean()
    avg_away_goals = matches["FTAG"].mean()

    # Each team's averages in its home matches. Here FTHG is goals scored and FTAG is goals conceded.
    home_stats = matches.groupby("HomeTeam").agg(
        home_matches=("FTHG", "size"),
        scored_at_home=("FTHG", "mean"),
        conceded_at_home=("FTAG", "mean"),
    )

    # Each team's averages in its away matches. Now FTAG is goals scored and FTHG is goals conceded.
    away_stats = matches.groupby("AwayTeam").agg(
        away_matches=("FTAG", "size"),
        scored_away=("FTAG", "mean"),
        conceded_away=("FTHG", "mean"),
    )

    # Put the home and away numbers side by side, matching rows on the team name
    strengths = home_stats.join(away_stats).rename_axis("Team")

    # Divide by the league average to get the four ratings.
    # A home team concedes to away teams, so home defence is compared with average away goals, and vice versa.
    strengths["home_attack"] = strengths["scored_at_home"] / avg_home_goals
    strengths["home_defence"] = strengths["conceded_at_home"] / avg_away_goals
    strengths["away_attack"] = strengths["scored_away"] / avg_away_goals
    strengths["away_defence"] = strengths["conceded_away"] / avg_home_goals

    return strengths, avg_home_goals, avg_away_goals


def expected_goals(home_team, away_team, strengths, avg_home_goals, avg_away_goals):
    """Work out how many goals each team is expected to score in this match."""
    home = strengths.loc[home_team]
    away = strengths.loc[away_team]
    home_xg = home["home_attack"] * away["away_defence"] * avg_home_goals
    away_xg = away["away_attack"] * home["home_defence"] * avg_away_goals
    return home_xg, away_xg


def score_matrix(home_xg, away_xg):
    """Build a grid where row i, column j is the chance of the score i-j."""
    goals = np.arange(MAX_GOALS + 1)                # 0, 1, 2, ... 10
    home_probs = poisson.pmf(goals, home_xg)        # Chance of the home team scoring exactly each number
    away_probs = poisson.pmf(goals, away_xg)        # Chance of the away team scoring exactly each number
    matrix = np.outer(home_probs, away_probs)       # Multiply every home chance by every away chance
    return matrix / matrix.sum()                    # Scale up so the grid adds to exactly 1, as scores above 10 are left out


def result_probabilities(matrix):
    """Add up the grid to get the chances of a home win, a draw and an away win."""
    home_win = np.tril(matrix, -1).sum()    # Below the diagonal: home goals > away goals
    draw = np.trace(matrix)                 # On the diagonal: equal goals
    away_win = np.triu(matrix, 1).sum()     # Above the diagonal: away goals > home goals
    return home_win, draw, away_win


def predict_with_updating_ratings(matches_to_predict, all_matches):
    """Predict each match using ratings built only from matches played on earlier dates.

    matches_to_predict: the matches to predict.
    all_matches: every match available, which can include the matches being predicted.

    Matches on the same date are all predicted together, without using each other's results,
    because the model wouldn't know any of those results before kick-off.

    Returns a table with one row per match, in the same order as matches_to_predict.
    """
    rows = []

    for date, matches_on_date in matches_to_predict.groupby("Date"):
        # The strict < is what prevents data leakage: nothing from this date or later is used
        known_results = all_matches[all_matches["Date"] < date]
        strengths, avg_home_goals, avg_away_goals = calculate_team_strengths(known_results)

        for match in matches_on_date.itertuples():
            home_xg, away_xg = expected_goals(match.HomeTeam, match.AwayTeam, strengths, avg_home_goals, avg_away_goals)
            home_win, draw, away_win = result_probabilities(score_matrix(home_xg, away_xg))
            rows.append({
                "match": match.Index,
                "home_xg": home_xg,
                "away_xg": away_xg,
                "prob_H": home_win,
                "prob_D": draw,
                "prob_A": away_win,
            })

    # Put the rows back in the original order, labelled with the original row numbers
    return pd.DataFrame(rows).set_index("match").rename_axis(None).loc[matches_to_predict.index]


def evaluate_test_season(training, test):
    """Predict one test season with every predictor, and score each prediction.

    training: the matches the models learn from (all played before the test season).
    test: the season being predicted.

    Matches involving a team with no training data are skipped, because neither model can rate that team.
    Returns one row per predicted match, with each predictor's probabilities, Brier score and pick.
    """
    known_teams = set(training["HomeTeam"]) | set(training["AwayTeam"])
    is_predictable = test["HomeTeam"].isin(known_teams) & test["AwayTeam"].isin(known_teams)
    results = test[is_predictable].copy()

    # Original model: ratings fixed before the season starts
    strengths, avg_home_goals, avg_away_goals = calculate_team_strengths(training)
    original = []
    for match in results.itertuples():
        home_xg, away_xg = expected_goals(match.HomeTeam, match.AwayTeam, strengths, avg_home_goals, avg_away_goals)
        original.append(result_probabilities(score_matrix(home_xg, away_xg)))
    results[["original_H", "original_D", "original_A"]] = pd.DataFrame(original, index=results.index)

    # Updated model: ratings recalculated before each matchday, from earlier results only.
    # The whole test season is passed in, including unpredictable teams' matches, because they really happened.
    updated = predict_with_updating_ratings(results, pd.concat([training, test], ignore_index=True))
    results["updated_H"] = updated["prob_H"]
    results["updated_D"] = updated["prob_D"]
    results["updated_A"] = updated["prob_A"]

    # Bet365, with the margin removed
    results["book_H"], results["book_D"], results["book_A"], _ = remove_margin(
        results["B365H"], results["B365D"], results["B365A"]
    )

    # Baseline: the training seasons' result rates, the same for every match
    rates = training["FTR"].value_counts(normalize=True)
    for result in ["H", "D", "A"]:
        results[f"base_{result}"] = rates.get(result, 0.0)

    # Score every predictor in the same way
    for prefix in PREDICTORS:
        prob_home, prob_draw, prob_away = results[f"{prefix}_H"], results[f"{prefix}_D"], results[f"{prefix}_A"]
        results[f"brier_{prefix}"] = brier_per_match(prob_home, prob_draw, prob_away, results["FTR"])
        results[f"pick_{prefix}"] = most_likely_result(prob_home, prob_draw, prob_away)

    return results


def remove_margin(home_odds, draw_odds, away_odds):
    """Convert decimal odds to probabilities with the bookmaker's margin removed.

    Works on single numbers or whole columns. Returns the home, draw and away probabilities, and the margin.
    """
    # Implied probability = 1 / odds. For each match, these add up to more than 1.
    implied_home = 1 / home_odds
    implied_draw = 1 / draw_odds
    implied_away = 1 / away_odds

    # The amount above 1 is the margin. Dividing by the total makes the three add up to exactly 1.
    total = implied_home + implied_draw + implied_away
    return implied_home / total, implied_draw / total, implied_away / total, total - 1


def brier_per_match(prob_home, prob_draw, prob_away, results):
    """Work out the Brier score for each match. 0 is perfect and 2 is the worst possible."""
    total = 0
    for result, probability in [("H", prob_home), ("D", prob_draw), ("A", prob_away)]:
        happened = (results == result).astype(int)    # 1 if this was the result, otherwise 0
        total = total + (probability - happened) ** 2
    return total


def most_likely_result(prob_home, prob_draw, prob_away):
    """Find the result with the highest probability in each match, as H, D or A.

    If two results are tied, the earlier one in the order H, D, A is chosen.
    """
    return pd.DataFrame({"H": prob_home, "D": prob_draw, "A": prob_away}).idxmax(axis=1)
