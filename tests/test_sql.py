"""Tests for sql/analysis.sql. These run each query on four made-up matches with answers worked out by hand."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from model import load_queries

SQL_FILE = Path(__file__).parent.parent / "sql" / "analysis.sql"


@pytest.fixture
def queries():
    return load_queries(SQL_FILE)


@pytest.fixture
def conn():
    matches = pd.DataFrame(
        [
            # Season, HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H, B365D, B365A
            ("2023/24", "A", "B", 2, 0, "H", 1.5, 4.0, 6.0),    # The home favourite won
            ("2023/24", "B", "A", 1, 1, "D", 2.0, 3.0, 2.0),    # No clear favourite: home and away odds are equal
            ("2024/25", "A", "C", 0, 1, "A", 1.8, 3.5, 4.0),    # The home favourite lost
            ("2024/25", "C", "B", 1, 3, "A", 3.0, 3.5, 2.0),    # The away favourite won
        ],
        columns=["Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"],
    )
    connection = sqlite3.connect(":memory:")
    matches.to_sql("matches", connection, index=False)
    yield connection
    connection.close()


def test_file_contains_every_query(queries):
    assert set(queries) == {
        "result_percentages",
        "goals_per_season",
        "top_scoring_teams",
        "best_defensive_teams",
        "favourite_win_rate",
    }


def test_result_percentages(queries, conn):
    table = pd.read_sql_query(queries["result_percentages"], conn).set_index("result")
    assert table.loc["Away win", "matches"] == 2
    assert table.loc["Away win", "percentage"] == 50.0
    assert table.loc["Home win", "percentage"] == 25.0
    assert table.loc["Draw", "percentage"] == 25.0


def test_goals_per_season(queries, conn):
    table = pd.read_sql_query(queries["goals_per_season"], conn).set_index("Season")
    assert table.loc["2023/24", "total_goals"] == 4     # 2-0 and 1-1
    assert table.loc["2023/24", "goals_per_match"] == 2.0
    assert table.loc["2024/25", "total_goals"] == 5     # 0-1 and 1-3
    assert table.loc["2024/25", "goals_per_match"] == 2.5


def test_top_scoring_team_counts_home_and_away_goals(queries, conn):
    table = pd.read_sql_query(queries["top_scoring_teams"], conn)
    top = table.iloc[0]
    assert top["team"] == "B"       # 0 + 1 + 3 goals in 3 matches
    assert top["goals_scored"] == 4
    assert top["scored_per_match"] == 1.33


def test_best_defensive_team_counts_home_and_away_goals(queries, conn):
    table = pd.read_sql_query(queries["best_defensive_teams"], conn)
    best = table.iloc[0]
    assert best["team"] == "A"      # 0 + 1 + 1 conceded in 3 matches
    assert best["goals_conceded"] == 2
    assert best["conceded_per_match"] == 0.67


def test_favourite_win_rate_excludes_matches_with_no_clear_favourite(queries, conn):
    table = pd.read_sql_query(queries["favourite_win_rate"], conn).set_index("category")
    # Check for exactly these rows, as a match with no favourite must not form a row of its own
    assert table.index.tolist() == [
        "Away favourite",
        "Home favourite",
        "All clear favourites",
        "No clear favourite (excluded)",
    ]
    assert table.loc["All clear favourites", "matches"] == 3
    assert table.loc["All clear favourites", "favourite_won"] == 2
    assert table.loc["All clear favourites", "win_percentage"] == 66.7
    assert table.loc["Home favourite", "favourite_won"] == 1
    assert table.loc["Away favourite", "favourite_won"] == 1
    assert table.loc["No clear favourite (excluded)", "matches"] == 1
