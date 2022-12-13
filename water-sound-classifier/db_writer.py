import psycopg2

class DataSink():
    def __init__(self):
        self.conn = psycopg2.connect(
            database="classifier",
            user='classifier', 
            password='classifier123', 
            host='34.27.149.182', 
            port= '5432')
        self.conn.autocommit = True

    def __del__(self):
        self.conn.close()

    def insert(self, data_arr):
        for data in data_arr:
            start, end, label, duration = data
            cursor = self.conn.cursor()
            query = """ INSERT INTO stats (start_time, end_time, class, duration) VALUES (%s,%s,%s,%s)"""
            try:
                cursor.execute(query, (start, end, label, duration))
            except:
                print("db insertion")
            self.conn.commit()
        print(str(len(data_arr))+" Records inserted........")



# writer = DataSink()
# writer.insert([['2022-12-10 21:45:00.000000', '2022-12-10 21:49:00.000000', 'sink', 240],['2022-12-10 20:58:00.000000', '2022-12-10 20:59:00.000000', 'sink', 60]])
# writer.insert(['2022-12-10 19:12:00.000000', '2022-12-10 19:29:00.000000', 'shower', 1900])
