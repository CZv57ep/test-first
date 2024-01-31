from datetime import datetime

tsn = datetime.now()
print(f"{tsn.year}.{tsn.toordinal()}.0")
