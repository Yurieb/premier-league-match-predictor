-- Premier League analysis queries (SQLite).
-- All queries read from one table, matches, with one row per match:
--   Season, Date, HomeTeam, AwayTeam
--   FTHG, FTAG           full-time home and away goals
--   FTR                  full-time result: H (home win), D (draw) or A (away win)
--   B365H, B365D, B365A  Bet365's decimal odds for each result
--
-- Each "-- name:" line starts a new query. The notebook and the tests use these names to find them.


-- name: result_percentages
-- Question 1: What percentage of matches are home wins, draws and away wins?
-- GROUP BY makes one row per result. The inner SELECT gets the total number of matches.
-- 100.0 (not 100) forces decimal division, because whole-number division would drop the decimals.
SELECT
    CASE FTR
        WHEN 'H' THEN 'Home win'
        WHEN 'D' THEN 'Draw'
        WHEN 'A' THEN 'Away win'
    END AS result,
    COUNT(*) AS matches,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM matches), 1) AS percentage
FROM matches
GROUP BY FTR
ORDER BY matches DESC;


-- name: goals_per_season
-- Question 2: How many goals per match were scored in each season?
-- Total goals in a match = home goals + away goals. AVG then averages that across each season.
SELECT
    Season,
    COUNT(*) AS matches,
    SUM(FTHG + FTAG) AS total_goals,
    ROUND(AVG(FTHG + FTAG), 2) AS goals_per_match
FROM matches
GROUP BY Season
ORDER BY Season;


-- name: top_scoring_teams
-- Question 3a: Which teams scored the most goals per match?
-- Each match row holds two teams, so the WITH block (a CTE) first builds a temporary table
-- with one row per team per match. UNION ALL stacks the home view and the away view.
-- Teams are ranked per match because not every team played all four seasons, which makes totals unfair.
WITH team_matches AS (
    SELECT HomeTeam AS team, FTHG AS scored FROM matches
    UNION ALL
    SELECT AwayTeam AS team, FTAG AS scored FROM matches
)
SELECT
    team,
    COUNT(*) AS matches,
    SUM(scored) AS goals_scored,
    ROUND(1.0 * SUM(scored) / COUNT(*), 2) AS scored_per_match
FROM team_matches
GROUP BY team
ORDER BY 1.0 * SUM(scored) / COUNT(*) DESC
LIMIT 5;


-- name: best_defensive_teams
-- Question 3b: Which teams conceded the fewest goals per match?
-- Same idea as 3a, but a home team concedes the away goals and an away team concedes the home goals.
-- ASC puts the lowest values first, because fewer goals conceded is better.
WITH team_matches AS (
    SELECT HomeTeam AS team, FTAG AS conceded FROM matches
    UNION ALL
    SELECT AwayTeam AS team, FTHG AS conceded FROM matches
)
SELECT
    team,
    COUNT(*) AS matches,
    SUM(conceded) AS goals_conceded,
    ROUND(1.0 * SUM(conceded) / COUNT(*), 2) AS conceded_per_match
FROM team_matches
GROUP BY team
ORDER BY 1.0 * SUM(conceded) / COUNT(*) ASC
LIMIT 5;


-- name: favourite_win_rate
-- Question 4: How often did the bookmaker's favourite actually win?
-- The favourite is the outcome with the lowest odds, because low odds mean the bookmaker thinks it's most likely.
-- If two outcomes share the lowest odds, there is no single favourite, so the match is excluded and counted separately.
-- In SQLite, (favourite = FTR) gives 1 when true and 0 when false, so SUM counts the correct picks.
WITH favourites AS (
    SELECT
        FTR,
        CASE
            WHEN B365H < B365D AND B365H < B365A THEN 'H'
            WHEN B365A < B365H AND B365A < B365D THEN 'A'
            WHEN B365D < B365H AND B365D < B365A THEN 'D'
            ELSE 'None'
        END AS favourite
    FROM matches
)
SELECT
    CASE favourite
        WHEN 'H' THEN 'Home favourite'
        WHEN 'A' THEN 'Away favourite'
        WHEN 'D' THEN 'Draw favourite'
    END AS category,
    COUNT(*) AS matches,
    SUM(favourite = FTR) AS favourite_won,
    ROUND(100.0 * SUM(favourite = FTR) / COUNT(*), 1) AS win_percentage
FROM favourites
WHERE favourite <> 'None'
GROUP BY favourite

UNION ALL

SELECT
    'All clear favourites',
    COUNT(*),
    SUM(favourite = FTR),
    ROUND(100.0 * SUM(favourite = FTR) / COUNT(*), 1)
FROM favourites
WHERE favourite <> 'None'

UNION ALL

SELECT 'No clear favourite (excluded)', COUNT(*), NULL, NULL
FROM favourites
WHERE favourite = 'None';
