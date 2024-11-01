import nfl_data_py as nfl
from datetime import datetime

current_year = datetime.now().year

# Fetch the schedule
schedule = nfl.import_schedules([current_year])

# Print the columns
print("Schedule Columns:", schedule.columns.tolist())
