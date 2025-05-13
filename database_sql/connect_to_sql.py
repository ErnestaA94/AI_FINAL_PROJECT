
import pandas as pd
from sqlalchemy import create_engine


engine = create_engine("mysql+pymysql://ErnestA94:ErnestA94@localhost/nba_prediction")


try:
    with engine.connect() as conn:
        print("Successfully connected to MySQL!")
except Exception as e:
    print(f"Connection error: {e}")
 
    temp_engine =create_engine("mysql+pymysql://ErnestA94:ErnestA94@localhost/nba_prediction")
    with temp_engine.connect() as conn:
        conn.execute("CREATE DATABASE IF NOT EXISTS nba_prediction")
    engine = create_engine("mysql+pymysql://ErnestA94:ErnestA94@localhost/nba_prediction")


print("Loading database files...")
games = pd.read_csv('database/games_clean.csv')
games_details = pd.read_csv('database/games_details_clean.csv')
players = pd.read_csv('database/players_clean.csv')
teams = pd.read_csv('database/teams_clean.csv')
ranking = pd.read_csv('database/ranking_clean.csv')

print("Storing data in MySQL...")

teams.to_sql('teams', engine, if_exists='replace', index=False)
print(f"✓ Teams table created: {len(teams)} rows")


players.to_sql('players', engine, if_exists='replace', index=False)
print(f"✓ Players table created: {len(players)} rows")


games.to_sql('games', engine, if_exists='replace', index=False)
print(f"✓ Games table created: {len(games)} rows")


games_details.to_sql('games_details', engine, if_exists='replace', 
                    index=False, chunksize=10000)
print(f"✓ Games details table created: {len(games_details)} rows")


ranking.to_sql('ranking', engine, if_exists='replace', index=False)
print(f"✓ Ranking table created: {len(ranking)} rows")

print("\nDatabase setup complete!")


for table in ['teams', 'players', 'games', 'games_details', 'ranking']:
    count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", engine).iloc[0, 0]
    print(f"{table}: {count} rows")

##############################################################################################################################################

def get_player_data(player_id, conn):
    """Retrieve all game details for a specific player"""
    query = f"""
    SELECT * FROM games_details 
    WHERE PLAYER_ID = {player_id}
    ORDER BY GAME_DATE
    """
    return pd.read_sql_query(query, conn)

def get_player_performance_timeline(player_id, conn):
    """Retrieve chronological performance metrics for a player"""
    query = f"""
    SELECT gd.GAME_ID, g.GAME_DATE, gd.PTS, gd.REB, gd.AST, gd.STL, gd.BLK, gd.FG_PCT, gd.FG3_PCT, gd.FT_PCT, gd.MIN
    FROM games_details gd
    JOIN games g ON gd.GAME_ID = g.GAME_ID
    WHERE gd.PLAYER_ID = {player_id}
    ORDER BY g.GAME_DATE
    """
    return pd.read_sql_query(query, conn)



