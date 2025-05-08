import pandas as pd

#dataframe = pd.read_csv('database.csv')

class Database:
  def __init__(self, csv_path):
    self.dataframe = pd.read_csv(csv_path)
    self.dataframe = self.dataframe.set_index('nama obat')

  def getData(self, line_name, column_name):
    return self.dataframe.loc[line_name, column_name]

# p1 = Person("John", 36)
# p1.myfunc() 
