"""Tests for model.py. These use small made-up numbers whose answers can be worked out by hand."""

import math

import numpy as np
import pandas as pd
import pytest

from model import (
    MAX_GOALS,
    brier_per_match,
    calculate_team_strengths,
    expected_goals,
    most_likely_result,
    predict_with_updating_ratings,
    remove_margin,
    result_probabilities,
    score_matrix,
)


@pytest.fixture
def two_matches():
    """A tiny league: A beat B 3-1 at home, then B and A drew 1-1 at B's ground."""
    return pd.DataFrame({
        "HomeTeam": ["A", "B"],
        "AwayTeam": ["B", "A"],
        "FTHG": [3, 1],
        "FTAG": [1, 1],
    })


def test_league_averages(two_matches):
    _, avg_home_goals, avg_away_goals = calculate_team_strengths(two_matches)
    assert avg_home_goals == 2      # (3 + 1) / 2
    assert avg_away_goals == 1      # (1 + 1) / 2


def test_team_ratings_match_hand_calculation(two_matches):
    strengths, _, _ = calculate_team_strengths(two_matches)
    assert strengths.loc["A", "home_attack"] == 1.5     # Scored 3 at home / league average 2
    assert strengths.loc["A", "home_defence"] == 1      # Conceded 1 at home / league average away goals 1
    assert strengths.loc["A", "away_attack"] == 1       # Scored 1 away / league average away goals 1
    assert strengths.loc["A", "away_defence"] == 0.5    # Conceded 1 away / league average home goals 2
    assert strengths.loc["B", "away_defence"] == 1.5    # Conceded 3 away / league average home goals 2


def test_weighted_average_of_ratings_is_one(two_matches):
    strengths, _, _ = calculate_team_strengths(two_matches)
    assert np.average(strengths["home_attack"], weights=strengths["home_matches"]) == pytest.approx(1)
    assert np.average(strengths["away_defence"], weights=strengths["away_matches"]) == pytest.approx(1)


def test_expected_goals_match_hand_calculation(two_matches):
    strengths, avg_home_goals, avg_away_goals = calculate_team_strengths(two_matches)
    home_xg, away_xg = expected_goals("A", "B", strengths, avg_home_goals, avg_away_goals)
    assert home_xg == pytest.approx(4.5)    # 1.5 x 1.5 x 2
    assert away_xg == pytest.approx(1.0)    # 1 x 1 x 1


def test_score_matrix_has_expected_shape_and_adds_to_one():
    matrix = score_matrix(1.5, 1.2)
    assert matrix.shape == (MAX_GOALS + 1, MAX_GOALS + 1)
    assert matrix.sum() == pytest.approx(1)


def test_score_matrix_matches_the_poisson_formula():
    """Compare the 2-1 cell with a calculation written out from the formula."""
    home_xg, away_xg = 1.5, 1.2
    home_scores_2 = home_xg**2 * math.exp(-home_xg) / math.factorial(2)
    away_scores_1 = away_xg * math.exp(-away_xg)
    # The grid is rescaled very slightly because scores above 10 are left out, so allow a small difference
    assert score_matrix(home_xg, away_xg)[2, 1] == pytest.approx(home_scores_2 * away_scores_1, abs=1e-5)


def test_result_probabilities_use_the_correct_part_of_the_grid():
    # Rows are home goals and columns are away goals
    matrix = np.array([
        [0.1, 0.2],     # 0-0 is a draw, and 0-1 is an away win
        [0.3, 0.4],     # 1-0 is a home win, and 1-1 is a draw
    ])
    home_win, draw, away_win = result_probabilities(matrix)
    assert home_win == pytest.approx(0.3)
    assert draw == pytest.approx(0.5)
    assert away_win == pytest.approx(0.2)


def test_result_probabilities_add_to_one():
    assert sum(result_probabilities(score_matrix(2.8, 0.5))) == pytest.approx(1)


def test_equal_teams_have_equal_win_chances():
    home_win, _, away_win = result_probabilities(score_matrix(1.4, 1.4))
    assert home_win == pytest.approx(away_win)


def test_higher_expected_goals_raise_win_chance():
    weaker_home_win, _, _ = result_probabilities(score_matrix(1.0, 1.2))
    stronger_home_win, _, _ = result_probabilities(score_matrix(2.5, 1.2))
    assert stronger_home_win > weaker_home_win


@pytest.fixture
def season():
    """Two earlier matches, then two on 1 March, then one on 8 March."""
    return pd.DataFrame(
        [
            ("2024-01-01", "A", "B", 3, 1),
            ("2024-02-01", "B", "A", 1, 1),
            ("2024-03-01", "A", "B", 2, 0),
            ("2024-03-01", "B", "A", 0, 2),
            ("2024-03-08", "A", "B", 1, 1),
        ],
        columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"],
    ).assign(Date=lambda table: pd.to_datetime(table["Date"]))


def test_updating_ratings_only_use_earlier_dates(season):
    """The 1 March match should get exactly the same prediction as ratings built from the first two matches."""
    strengths, avg_home_goals, avg_away_goals = calculate_team_strengths(season.iloc[:2])
    expected_home_xg, expected_away_xg = expected_goals("A", "B", strengths, avg_home_goals, avg_away_goals)

    prediction = predict_with_updating_ratings(season.iloc[[2]], season)

    assert prediction.loc[2, "home_xg"] == pytest.approx(expected_home_xg)
    assert prediction.loc[2, "away_xg"] == pytest.approx(expected_away_xg)


def test_updating_ratings_ignore_the_match_being_predicted(season):
    changed = season.copy()
    changed.loc[2, ["FTHG", "FTAG"]] = [9, 9]
    original = predict_with_updating_ratings(season.iloc[[2]], season)
    new = predict_with_updating_ratings(changed.iloc[[2]], changed)
    pd.testing.assert_frame_equal(original, new)


def test_updating_ratings_ignore_other_matches_on_the_same_date(season):
    changed = season.copy()
    changed.loc[3, ["FTHG", "FTAG"]] = [9, 9]     # Match 3 is on the same date as match 2
    original = predict_with_updating_ratings(season.iloc[[2]], season)
    new = predict_with_updating_ratings(changed.iloc[[2]], changed)
    pd.testing.assert_frame_equal(original, new)


def test_updating_ratings_ignore_future_matches(season):
    changed = season.copy()
    changed.loc[4, ["FTHG", "FTAG"]] = [9, 9]     # Match 4 is a week after match 2
    original = predict_with_updating_ratings(season.iloc[[2]], season)
    new = predict_with_updating_ratings(changed.iloc[[2]], changed)
    pd.testing.assert_frame_equal(original, new)


def test_updating_ratings_do_change_after_new_results(season):
    """A big result on 1 March should change the prediction for 8 March."""
    changed = season.copy()
    changed.loc[2, ["FTHG", "FTAG"]] = [9, 0]
    original = predict_with_updating_ratings(season.iloc[[4]], season)
    new = predict_with_updating_ratings(changed.iloc[[4]], changed)
    assert new.loc[4, "home_xg"] != pytest.approx(original.loc[4, "home_xg"])


def test_updating_ratings_keep_the_original_match_order(season):
    subset = season.iloc[[4, 2]]    # Deliberately not in date order
    prediction = predict_with_updating_ratings(subset, season)
    assert prediction.index.tolist() == [4, 2]


def test_remove_margin_matches_hand_calculation():
    # Odds of 1.60, 4.00 and 5.00 imply 62.5%, 25% and 20%, a total of 107.5%, so the margin is 7.5%
    home, draw, away, margin = remove_margin(1.60, 4.00, 5.00)
    assert margin == pytest.approx(0.075)
    assert home == pytest.approx(0.625 / 1.075)
    assert draw == pytest.approx(0.25 / 1.075)
    assert away == pytest.approx(0.20 / 1.075)
    assert home + draw + away == pytest.approx(1)


def test_remove_margin_works_on_columns():
    home, draw, away, _ = remove_margin(pd.Series([1.6, 2.5]), pd.Series([4.2, 3.4]), pd.Series([5.25, 2.9]))
    assert (home + draw + away).tolist() == pytest.approx([1, 1])


def test_brier_score_is_zero_when_perfect_and_two_when_completely_wrong():
    results = pd.Series(["H"])
    perfect = brier_per_match(pd.Series([1.0]), pd.Series([0.0]), pd.Series([0.0]), results)
    worst = brier_per_match(pd.Series([0.0]), pd.Series([0.0]), pd.Series([1.0]), results)
    assert perfect.iloc[0] == 0
    assert worst.iloc[0] == 2


def test_brier_score_matches_hand_calculation():
    # (1 - 0.5)^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38
    score = brier_per_match(pd.Series([0.5]), pd.Series([0.3]), pd.Series([0.2]), pd.Series(["H"]))
    assert score.iloc[0] == pytest.approx(0.38)


def test_most_likely_result_picks_the_highest_probability():
    picks = most_likely_result(pd.Series([0.5, 0.2]), pd.Series([0.3, 0.3]), pd.Series([0.2, 0.5]))
    assert picks.tolist() == ["H", "A"]


def test_most_likely_result_chooses_home_when_tied():
    picks = most_likely_result(pd.Series([0.4]), pd.Series([0.2]), pd.Series([0.4]))
    assert picks.tolist() == ["H"]
